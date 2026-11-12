"""Agente investigador: RAG hibrido. No puede calcular ni consultar SQL."""

from langchain_core.tools import StructuredTool

from app.agents.base import AgenteReAct
from app.config import ConfigParams, get_prompts
from app.providers.llm_client import LLMClient
from app.tools.retriever import busqueda_hibrida

NOMBRE = "investigador"


def build(llm: LLMClient, config: ConfigParams) -> AgenteReAct:
    herramienta = StructuredTool.from_function(
        coroutine=busqueda_hibrida,
        name="busqueda_hibrida",
        description=(
            "Busca informacion en la base de conocimiento del negocio "
            "(documentos, politicas, preguntas frecuentes). Unica herramienta "
            "disponible para este agente."
        ),
    )
    return AgenteReAct(
        nombre=NOMBRE,
        system_prompt=get_prompts()[NOMBRE],
        herramientas=[herramienta],
        llm=llm,
        max_iteraciones=config.max_iteraciones_react,
    )
