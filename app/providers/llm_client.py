"""Cliente LLM con cadena de fallback.

Prueba los proveedores en el orden de config.json (llamacpp -> gemini), con
max_intentos_llm intentos por proveedor y backoff lineal. Los nodos y agentes
no saben que proveedor respondio.
"""

import asyncio
import logging
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from app.config import ConfigParams, Settings
from app.providers import gemini, llamacpp

logger = logging.getLogger(__name__)

_BUILDERS = {"llamacpp": llamacpp.build, "gemini": gemini.build}


class AllProvidersFailedError(RuntimeError):
    """Ningun proveedor respondio tras agotar los reintentos."""


class LLMClient:
    def __init__(self, providers: list[tuple[str, BaseChatModel]], max_intentos: int):
        self._providers = providers
        self._max_intentos = max_intentos

    async def _con_fallback(self, accion: Callable[[BaseChatModel], Awaitable[Any]]) -> Any:
        ultimo_error: Exception | None = None

        for nombre, provider in self._providers:
            for intento in range(self._max_intentos):
                try:
                    return await accion(provider)
                except Exception as exc:
                    ultimo_error = exc
                    logger.warning(
                        "Fallo el proveedor %s (intento %d/%d): %s",
                        nombre, intento + 1, self._max_intentos, type(exc).__name__,
                    )
                    if intento + 1 < self._max_intentos:
                        await asyncio.sleep(intento * 2 + 1)

        raise AllProvidersFailedError(
            f"Todos los proveedores fallaron: {type(ultimo_error).__name__}"
        ) from ultimo_error

    async def ainvoke(self, messages: list) -> AIMessage:
        return await self._con_fallback(lambda p: p.ainvoke(messages))

    async def ainvoke_structured(self, messages: list, schema: type[BaseModel]) -> BaseModel:
        return await self._con_fallback(
            lambda p: p.with_structured_output(schema).ainvoke(messages)
        )

    async def ainvoke_tools(self, messages: list, tools: list) -> AIMessage:
        return await self._con_fallback(lambda p: p.bind_tools(tools).ainvoke(messages))


def build_client(settings: Settings, config: ConfigParams) -> LLMClient:
    providers: list[tuple[str, BaseChatModel]] = []

    for nombre in config.orden_proveedores:
        builder = _BUILDERS.get(nombre)
        if builder is None:
            raise ValueError(f"Proveedor desconocido en orden_proveedores: {nombre}")
        if nombre == "gemini" and not settings.GEMINI_API_KEY:
            logger.warning("Se omite gemini: falta GEMINI_API_KEY")
            continue
        providers.append((nombre, builder(settings)))

    if not providers:
        raise ValueError("No hay proveedores LLM configurados")

    return LLMClient(providers, config.max_intentos_llm)
