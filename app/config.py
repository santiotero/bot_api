"""Configuracion: secretos desde .env, comportamiento desde config.json, prompts desde prompts.json."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class Settings(BaseSettings):
    """Infraestructura y secretos. Nada de esto se versiona."""

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    API_TOKEN: str
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    REDIS_URL: str = "redis://localhost:6379"

    LLAMACPP_BASE_URL: str
    LLAMACPP_MODEL: str
    LLAMACPP_TIMEOUT: int = 60

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash-lite"

    SQLITE_PATH: str = "./app/vectorstore/data/watbot.db"

    # Se fija a proposito: el default de fastembed es /tmp/fastembed_cache, que
    # en Linux se vacia al reiniciar (y si /tmp es tmpfs, el modelo vive en RAM).
    EMBEDDING_CACHE_DIR: str = "./models"

    DB_HOST: str
    DB_NAME: str
    DB_BUSCADOR_USER: str
    DB_BUSCADOR_PASS: str

    WATBOT_CALLBACK_URL: str
    WATBOT_CALLBACK_TOKEN: str

    LOG_LEVEL: str = "INFO"


class ConfigParams(BaseModel):
    """Parametros de comportamiento. Una clave desconocida detiene el arranque."""

    model_config = ConfigDict(extra="forbid")

    orden_proveedores: list[str]
    max_intentos_llm: int = Field(ge=1)
    max_intentos_supervisor: int = Field(ge=1)
    max_iteraciones_react: int = Field(ge=1)
    k_semantico: int = Field(ge=1)
    k_lexical: int = Field(ge=1)
    ensemble_weights: list[float]
    score_threshold: float = Field(ge=0.0, le=1.0)
    chunk_size: int = Field(ge=100)
    chunk_overlap: int = Field(ge=0)
    embedding_model: str
    job_timeout_seconds: int = Field(ge=1)
    task_ttl_seconds: int = Field(ge=1)
    checkpoint_ttl_seconds: int = Field(ge=1)
    max_tokens_contexto: int = Field(ge=100)
    max_lineas_respuesta: int = Field(ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config() -> ConfigParams:
    with open(CONFIG_DIR / "config.json", encoding="utf-8") as f:
        return ConfigParams(**json.load(f))


@lru_cache
def get_prompts() -> dict[str, str]:
    with open(CONFIG_DIR / "prompts.json", encoding="utf-8") as f:
        return json.load(f)
