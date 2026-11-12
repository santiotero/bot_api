"""Gemini como proveedor de respaldo."""

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings


def build(settings: Settings) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
        max_retries=0,  # los reintentos los maneja LLMClient
    )
