"""API REST: recibe mensajes de watbot_v2, encola y expone estado.

Esta capa nunca ejecuta el grafo: solo encola en ARQ. El worker lo ejecuta.
"""

import logging
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from redis.asyncio import Redis

from app import tasks
from app.config import ConfigParams, Settings, get_config, get_prompts, get_settings
from app.dependencies import get_params, get_queue, get_redis, verify_token
from app.schema.models import (
    ApprovalRequest,
    CheckDBResponse,
    IngestRequest,
    IngestResponse,
    MessageRequest,
    TaskResponse,
)
from app.vectorstore import db, ingest, vec_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    config = get_config()
    get_prompts()  # valida que el archivo exista y sea JSON valido

    logging.basicConfig(level=settings.LOG_LEVEL)

    db.init(settings.SQLITE_PATH)
    await vec_store.warmup()

    app.state.redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    app.state.queue = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
    logger.info("API lista")

    yield

    await app.state.queue.aclose()
    await app.state.redis.aclose()
    db.close()


app = FastAPI(
    title="watbot_api",
    description="Sistema de IA multi-agente para WhatsApp",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Mensajes
# --------------------------------------------------------------------------- #

@app.post(
    "/message",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_token)],
    summary="Recibe un mensaje de WhatsApp y lo encola",
)
async def post_message(
    payload: MessageRequest,
    redis: Redis = Depends(get_redis),
    queue: ArqRedis = Depends(get_queue),
    config: ConfigParams = Depends(get_params),
) -> TaskResponse:
    phone = payload.phone

    job = await queue.enqueue_job(
        "execute_task",
        phone,
        payload.message,
        payload.message_id,
        _job_id=phone,  # un job activo por numero
    )

    # enqueue_job devuelve None si ya hay un job con ese id: el mensaje anterior
    # del mismo numero todavia se esta procesando.
    if job is None:
        actual = await tasks.get_task(redis, phone)
        estado = (actual or {}).get("status", tasks.RUNNING)
        return TaskResponse(job_id=phone, status=estado, response=None)

    await tasks.set_status(redis, phone, tasks.PENDING, config.task_ttl_seconds)
    return TaskResponse(job_id=phone, status=tasks.PENDING, response=None)


@app.get(
    "/message/{phone}",
    response_model=TaskResponse,
    dependencies=[Depends(verify_token)],
    summary="Estado actual de la tarea de un numero",
)
async def get_message(phone: str, redis: Redis = Depends(get_redis)) -> TaskResponse:
    datos = await tasks.get_task(redis, phone)
    if datos is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tarea no encontrada")
    return TaskResponse(
        job_id=phone,
        status=datos.get("status", tasks.PENDING),
        response=datos.get("response") or None,
    )


@app.post(
    "/message/{phone}/approve",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_token)],
    summary="Aprueba o rechaza una tarea en espera humana",
)
async def approve(
    phone: str,
    payload: ApprovalRequest,
    redis: Redis = Depends(get_redis),
    queue: ArqRedis = Depends(get_queue),
    config: ConfigParams = Depends(get_params),
) -> dict:
    datos = await tasks.get_task(redis, phone)
    if datos is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tarea no encontrada")

    # CAS atomico: un segundo approve concurrente recibe 409.
    if not await tasks.marcar_aprobando(redis, phone):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"La tarea no esta en {tasks.WAITING_HUMAN}",
        )

    job = await queue.enqueue_job(
        "reanudar_tarea", phone, payload.approved, payload.reason, _job_id=f"{phone}:resume"
    )
    if job is None:
        await tasks.revertir_a_espera(redis, phone, config.task_ttl_seconds)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo encolar la reanudacion",
        )

    return {"job_id": phone, "status": tasks.APPROVING}


# --------------------------------------------------------------------------- #
# Ingesta
# --------------------------------------------------------------------------- #

@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_token)],
    summary="Indexa documentos en FTS5 + sqlite-vec",
)
async def post_ingest(payload: IngestRequest, background: BackgroundTasks) -> IngestResponse:
    documentos = [d.model_dump() for d in payload.documents]

    async def procesar() -> None:
        try:
            nuevos = await ingest.ingest(documentos)
            logger.info("Ingesta finalizada: %d fragmentos nuevos", nuevos)
        except Exception:
            logger.exception("Fallo la ingesta")

    # El chunking y los embeddings son costosos: se responde al instante y el
    # trabajo sigue en background. count son los documentos aceptados.
    background.add_task(procesar)
    return IngestResponse(status="ok", count=len(documentos))


@app.delete(
    "/ingest",
    dependencies=[Depends(verify_token)],
    summary="Borra todos los datos de ambas stores",
)
async def delete_ingest() -> dict:
    await ingest.clear()
    return {"status": "cleared"}


@app.get(
    "/admin/checkdb",
    response_model=CheckDBResponse,
    dependencies=[Depends(verify_token)],
    summary="Consistencia entre FTS5 y sqlite-vec",
)
async def checkdb() -> CheckDBResponse:
    return CheckDBResponse(**await ingest.checkdb())


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

@app.get("/health", summary="Liveness: el proceso esta vivo")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/ready", summary="Readiness: Redis responde")
async def health_ready(redis: Redis = Depends(get_redis)) -> dict:
    try:
        await redis.ping()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": {"redis": "error"}},
        )
    return {"status": "ready", "checks": {"redis": "ok"}}
