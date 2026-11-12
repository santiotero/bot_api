"""Estado de las tareas en Redis, compartido por la API y el worker.

Clave: task:{phone}. El numero de celular es el identificador de la tarea y, a
la vez, el thread_id del grafo.
"""

import time
from datetime import datetime, timezone

from redis.asyncio import Redis

PENDING = "PENDING"
RUNNING = "RUNNING"
WAITING_HUMAN = "WAITING_HUMAN"
APPROVING = "APPROVING"
DONE = "DONE"
FAILED = "FAILED"

# Compare-and-swap atomico: con approves concurrentes solo el primero gana.
_LUA_CAS = """
if redis.call('HGET', KEYS[1], 'status') == ARGV[1] then
    redis.call('HSET', KEYS[1], 'status', ARGV[2])
    return 1
end
return 0
"""


def _clave(phone: str) -> str:
    return f"task:{phone}"


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


async def set_status(
    redis: Redis, phone: str, status: str, ttl: int, response: str | None = None
) -> None:
    campos = {"phone": phone, "status": status, "updated_at": _ahora()}
    if response is not None:
        campos["response"] = response

    clave = _clave(phone)
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(clave, mapping=campos)
        pipe.expire(clave, ttl)
        await pipe.execute()


async def get_task(redis: Redis, phone: str) -> dict | None:
    datos = await redis.hgetall(_clave(phone))
    return datos or None


async def marcar_aprobando(redis: Redis, phone: str) -> bool:
    """WAITING_HUMAN -> APPROVING de forma atomica. False si no estaba en espera."""
    resultado = await redis.eval(_LUA_CAS, 1, _clave(phone), WAITING_HUMAN, APPROVING)
    return bool(resultado)


async def revertir_a_espera(redis: Redis, phone: str, ttl: int) -> None:
    """Si no se pudo encolar la reanudacion, vuelve a WAITING_HUMAN."""
    await set_status(redis, phone, WAITING_HUMAN, ttl)


# --------------------------------------------------------------------------- #
# Indice de actividad por hilo
#
# El checkpointer de SQLite no tiene TTL propio (el de Redis si lo tenia), asi
# que la expiracion de conversaciones inactivas se lleva aca: un sorted set con
# el timestamp de la ultima actividad de cada numero. No se puede usar el TTL de
# task:{phone} porque Redis borra esas claves en silencio y no se pueden
# enumerar una vez vencidas.
# --------------------------------------------------------------------------- #

_CLAVE_ACTIVIDAD = "threads:activity"


async def registrar_actividad(redis: Redis, phone: str) -> None:
    await redis.zadd(_CLAVE_ACTIVIDAD, {phone: time.time()})


async def hilos_vencidos(redis: Redis, ttl: int) -> list[str]:
    """Numeros sin actividad desde hace mas de ttl segundos."""
    return await redis.zrangebyscore(_CLAVE_ACTIVIDAD, 0, time.time() - ttl)


async def olvidar_hilos(redis: Redis, phones: list[str]) -> None:
    if phones:
        await redis.zrem(_CLAVE_ACTIVIDAD, *phones)
