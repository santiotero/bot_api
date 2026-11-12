"""Estado del grafo LangGraph y helpers de lectura.

`messages` (heredado de MessagesState) acumula el historial del hilo. El
checkpointer de Redis lo persiste por thread_id (= numero de celular), lo que
da continuidad multi-turno.
"""

import operator
from typing import Annotated, Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState


class CustomState(MessagesState):
    """Estado compartido por todos los nodos del grafo."""

    contribuciones: Annotated[list[dict], operator.add]
    turno_id: str
    intentos_supervisor: int
    instruccion_actual: str
    next_node: str
    approved: Optional[bool]
    approval_reason: Optional[str]
    approval_timestamp: Optional[str]
    no_data_notified: bool
    catalog_available: bool


def texto_plano(contenido: Any) -> str:
    """Normaliza el content de un mensaje a texto.

    Los proveedores devuelven content como str, como lista de bloques
    ([{"type": "text", "text": "..."}]) o como dict.
    """
    if contenido is None:
        return ""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, dict):
        if contenido.get("type", "text") == "text":
            return str(contenido.get("text", ""))
        return ""
    if isinstance(contenido, (list, tuple)):
        return "".join(texto_plano(bloque) for bloque in contenido)
    return str(contenido)


def contribuciones_del_turno(state: dict) -> list[dict]:
    """Contribuciones del turno actual, filtradas por turno_id."""
    turno = state.get("turno_id")
    return [c for c in state.get("contribuciones") or [] if c.get("turno_id") == turno]


def obtener_respuesta_final(state: dict) -> Optional[str]:
    """Texto del ultimo AIMessage con name='sintesis'."""
    for mensaje in reversed(state.get("messages") or []):
        if isinstance(mensaje, AIMessage) and mensaje.name == "sintesis":
            return texto_plano(mensaje.content)
    return None


def pregunta_actual(state: dict) -> str:
    """Texto del ultimo HumanMessage no vacio."""
    for mensaje in reversed(state.get("messages") or []):
        if isinstance(mensaje, HumanMessage):
            texto = texto_plano(mensaje.content).strip()
            if texto:
                return texto
    return ""


def recortar_contexto(messages: list, max_tokens: int) -> list:
    """Recorta el historial para no saturar el contexto del LLM.

    Estima 4 caracteres por token y conserva los mensajes mas recientes. El
    primer HumanMessage del turno actual nunca se descarta.
    """
    if not messages:
        return []

    limite_chars = max_tokens * 4
    acumulado = 0
    conservados: list = []

    for mensaje in reversed(messages):
        costo = len(texto_plano(getattr(mensaje, "content", "")))
        if conservados and acumulado + costo > limite_chars:
            break
        conservados.append(mensaje)
        acumulado += costo

    return list(reversed(conservados))
