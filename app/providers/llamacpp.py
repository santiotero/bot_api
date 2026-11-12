"""llama.cpp server via API OpenAI-compatible. Proveedor primario.

temperature=0 es fijo por diseño: el sistema necesita decisiones
deterministas y no debe improvisar.
"""

from langchain_openai import ChatOpenAI

from app.config import Settings


def build(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.LLAMACPP_BASE_URL,
        api_key="not-needed",
        model=settings.LLAMACPP_MODEL,
        temperature=0,
        timeout=settings.LLAMACPP_TIMEOUT,
        max_retries=0,  # los reintentos los maneja LLMClient
    )
