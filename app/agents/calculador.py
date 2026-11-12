"""Agente calculador: aritmetica exacta. No puede buscar ni consultar SQL."""

from langchain_core.tools import StructuredTool

from app.agents.base import AgenteReAct
from app.config import ConfigParams, get_prompts
from app.providers.llm_client import LLMClient
from app.tools.calculadora import calculadora_ast

NOMBRE = "calculador"


def build(llm: LLMClient, config: ConfigParams) -> AgenteReAct:
    herramienta = StructuredTool.from_function(
        coroutine=calculadora_ast,
        name="calculadora_ast",
        description=(
            "Evalua una expresion matematica con precision decimal. Operadores: "
            "+ - * / % ** y parentesis. Unica herramienta disponible para este agente."
        ),
    )
    return AgenteReAct(
        nombre=NOMBRE,
        system_prompt=get_prompts()[NOMBRE],
        herramientas=[herramienta],
        llm=llm,
        max_iteraciones=config.max_iteraciones_react,
    )
