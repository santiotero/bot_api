"""Dependencias de FastAPI: autenticacion y acceso a Redis / cola ARQ."""

import secrets

from arq.connections import ArqRedis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis

from app.config import ConfigParams, Settings, get_config, get_settings

_bearer = HTTPBearer(auto_error=False)


async def verify_token(
    credenciales: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    """Valida el Bearer token. compare_digest evita timing attacks."""
    if credenciales is None or not secrets.compare_digest(
        credenciales.credentials, settings.API_TOKEN
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalido o ausente",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


def get_queue(request: Request) -> ArqRedis:
    return request.app.state.queue


def get_params() -> ConfigParams:
    return get_config()
