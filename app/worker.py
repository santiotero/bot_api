"""Worker ARQ: ejecuta el grafo LangGraph. La API nunca lo ejecuta.

Dos funciones encolables:
  - execute_task:   mensaje nuevo, corre el grafo desde el inicio del turno.
  - reanudar_tarea: retoma el grafo tras una decision humana (HITL).
"""

import logging
import uuid
from pathlib import Path

from arq import cron
from arq.connections import RedisSettings
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from redis.asyncio import Redis

from app import tasks
from app.config import get_config, get_prompts, get_settings
from app.graph import build_graph
from app.providers import callback
from app.providers.llm_client import build_client
from app.state import obtener_respuesta_final
from app.tools.sql_query import catalogo_tiene_filas
from app.vectorstore import db, vec_store

logger = logging.getLogger(__name__)


def _config_hilo(phone: str) -> dict:
    return {"configurable": {"thread_id": phone}}


async def _finalizar(ctx: dict, phone: str, resultado: dict) -> None:
    """Interpreta el resultado del grafo y notifica a watbot_v2 si corresponde."""
    settings, config, redis = ctx["settings"], ctx["config"], ctx["redis"]
    ttl = config.task_ttl_seconds

    if resultado.get("__interrupt__"):
        await tasks.set_status(redis, phone, tasks.WAITING_HUMAN, ttl)
        logger.info("Tarea %s en espera de aprobacion humana", phone)
        return

    respuesta = obtener_respuesta_final(resultado) or ""

    # Respuesta vacia: sintesis decidio no responder porque ya se aviso antes
    # que no hay informacion. No se llama al callback para no gastar un mensaje.
    if not respuesta.strip():
        await tasks.set_status(redis, phone, tasks.DONE, ttl, response="")
        logger.info("Tarea %s finalizada sin respuesta (ya notificado)", phone)
        return

    enviado = await callback.enviar_respuesta(settings, phone, respuesta)
    estado = tasks.DONE if enviado else tasks.FAILED
    await tasks.set_status(redis, phone, estado, ttl, response=respuesta)
    logger.info("Tarea %s finalizada (%s)", phone, estado)


async def execute_task(ctx: dict, phone: str, message: str, message_id: str | None = None) -> None:
    config, redis, grafo = ctx["config"], ctx["redis"], ctx["grafo"]
    await tasks.set_status(redis, phone, tasks.RUNNING, config.task_ttl_seconds)
    # Marca el hilo como activo para que la limpieza periodica no lo expire.
    await tasks.registrar_actividad(redis, phone)

    try:
        # Si el catalogo esta vacio el supervisor nunca rutea al buscador: evita
        # una llamada al LLM que no podria aportar nada.
        catalog_available = await catalogo_tiene_filas()

        estado_inicial = {
            "messages": [HumanMessage(content=message)],
            # Unico por turno: aisla las contribuciones de turnos anteriores.
            "turno_id": message_id or uuid.uuid4().hex,
            "intentos_supervisor": 0,
            "instruccion_actual": "",
            "next_node": "",
            # Se reinician siempre: si no, una aprobacion previa haria que todos
            # los turnos siguientes salten directo a sintesis por la guardia 1.
            "approved": None,
            "approval_reason": None,
            "approval_timestamp": None,
            "catalog_available": catalog_available,
            # no_data_notified se omite a proposito: persiste del checkpoint.
        }

        resultado = await grafo.ainvoke(estado_inicial, config=_config_hilo(phone))
        await _finalizar(ctx, phone, resultado)

    except Exception:
        logger.exception("Fallo la tarea %s", phone)
        await tasks.set_status(redis, phone, tasks.FAILED, config.task_ttl_seconds)


async def reanudar_tarea(ctx: dict, phone: str, approved: bool, reason: str = "") -> None:
    config, redis, grafo = ctx["config"], ctx["redis"], ctx["grafo"]
    await tasks.set_status(redis, phone, tasks.RUNNING, config.task_ttl_seconds)

    try:
        resultado = await grafo.ainvoke(
            Command(resume={"approved": approved, "reason": reason}),
            config=_config_hilo(phone),
        )
        await _finalizar(ctx, phone, resultado)
    except Exception:
        logger.exception("Fallo la reanudacion de %s", phone)
        await tasks.set_status(redis, phone, tasks.FAILED, config.task_ttl_seconds)


async def limpiar_checkpoints(ctx: dict) -> None:
    """Expira las conversaciones inactivas. Corre por cron, no en el camino del usuario.

    El checkpointer de SQLite no tiene TTL propio: la antiguedad se consulta en
    el indice de actividad de Redis y los hilos vencidos se borran por su API.
    """
    config, redis, checkpointer = ctx["config"], ctx["redis"], ctx["checkpointer"]

    vencidos = await tasks.hilos_vencidos(redis, config.checkpoint_ttl_seconds)
    if not vencidos:
        return

    for phone in vencidos:
        try:
            await checkpointer.adelete_thread(phone)
        except Exception:
            logger.exception("No se pudo borrar el checkpoint de %s", phone)

    await tasks.olvidar_hilos(redis, vencidos)
    logger.info("Checkpoints expirados por inactividad: %d", len(vencidos))


async def startup(ctx: dict) -> None:
    settings = get_settings()
    config = get_config()
    get_prompts()

    logging.basicConfig(level=settings.LOG_LEVEL)

    db.init(settings.SQLITE_PATH)
    await vec_store.warmup()

    ctx["settings"] = settings
    ctx["config"] = config
    ctx["redis"] = Redis.from_url(settings.REDIS_URL, decode_responses=True)

    # Checkpoints en SQLite, no en Redis: el checkpointer de Redis exige los
    # modulos RediSearch y RedisJSON, que el redis-server de Ubuntu no trae.
    # Redis queda solo para la cola ARQ, que no necesita modulos.
    Path(settings.CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # from_conn_string es un context manager: se entra a mano porque su ciclo de
    # vida es el del worker, no el de un bloque.
    cm = AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_PATH)
    checkpointer = await cm.__aenter__()
    await checkpointer.setup()
    ctx["_checkpointer_cm"] = cm
    ctx["checkpointer"] = checkpointer

    ctx["grafo"] = build_graph(build_client(settings, config), config, checkpointer)
    logger.info("Worker listo")


async def shutdown(ctx: dict) -> None:
    cm = ctx.get("_checkpointer_cm")
    if cm is not None:
        await cm.__aexit__(None, None, None)
    redis = ctx.get("redis")
    if redis is not None:
        await redis.aclose()
    db.close()


class WorkerSettings:
    functions = [execute_task, reanudar_tarea]
    # Una vez por hora, en el minuto 7 para no coincidir con otros cron en punto.
    cron_jobs = [cron(limpiar_checkpoints, minute=7)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().REDIS_URL)
    job_timeout = get_config().job_timeout_seconds
    # El estado de la tarea vive en task:{phone}, no en el resultado de ARQ.
    # Con keep_result > 0 el job_id (= phone) quedaria reservado y bloquearia
    # los mensajes siguientes del mismo numero.
    keep_result = 0
    max_jobs = 2
