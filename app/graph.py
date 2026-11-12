"""Armado del StateGraph multi-agente.

    START -> supervisor
    supervisor -> investigador | calculador | buscador | aprobacion | sintesis
    investigador | calculador | buscador -> supervisor
    aprobacion -> (interrupt) -> supervisor
    sintesis -> END          (unico nodo conectado a END)
"""

from langgraph.graph import END, START, StateGraph

from app.agents import buscador, calculador, investigador
from app.config import ConfigParams
from app.nodes import (
    AGENTES,
    build_nodo_agente,
    build_sintesis,
    build_supervisor,
    decidir_ruta,
    nodo_aprobacion,
)
from app.providers.llm_client import LLMClient
from app.state import CustomState


def build_graph(llm: LLMClient, config: ConfigParams, checkpointer):
    constructores = {
        "investigador": investigador.build,
        "calculador": calculador.build,
        "buscador": buscador.build,
    }

    grafo = StateGraph(CustomState)
    grafo.add_node("supervisor", build_supervisor(llm, config))
    for nombre in AGENTES:
        grafo.add_node(nombre, build_nodo_agente(constructores[nombre](llm, config)))
    grafo.add_node("aprobacion", nodo_aprobacion)
    grafo.add_node("sintesis", build_sintesis(llm, config))

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        decidir_ruta,
        {n: n for n in (*AGENTES, "aprobacion", "sintesis")},
    )
    for nombre in AGENTES:
        grafo.add_edge(nombre, "supervisor")
    grafo.add_edge("aprobacion", "supervisor")
    grafo.add_edge("sintesis", END)

    return grafo.compile(checkpointer=checkpointer)
