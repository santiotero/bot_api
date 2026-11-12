"""Agente buscador: SELECT sobre la tabla de catalogo. No puede buscar en RAG
ni calcular."""

from langchain_core.tools import StructuredTool

from app.agents.base import AgenteReAct
from app.config import ConfigParams, get_prompts
from app.providers.llm_client import LLMClient
from app.tools.sql_query import TABLA_PERMITIDA, sql_query

NOMBRE = "buscador"


def build(llm: LLMClient, config: ConfigParams) -> AgenteReAct:
    herramienta = StructuredTool.from_function(
        coroutine=sql_query,
        name="sql_query",
        description=(
            f"Ejecuta una consulta SELECT sobre la tabla {TABLA_PERMITIDA} "
            "(catalogo de productos y servicios). Cualquier otra tabla u "
            "operacion es rechazada. Unica herramienta disponible para este agente."
        ),
    )
    return AgenteReAct(
        nombre=NOMBRE,
        system_prompt=get_prompts()[NOMBRE],
        herramientas=[herramienta],
        llm=llm,
        max_iteraciones=config.max_iteraciones_react,
    )
