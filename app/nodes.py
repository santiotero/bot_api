"""Nodos del grafo: supervisor, agentes, aprobacion, sintesis y routing.

Separacion de responsabilidades (invariante del sistema):
  - supervisor: SOLO decide el siguiente paso. Nunca responde al usuario.
  - agentes:    SOLO aportan contribuciones. Nunca responden al usuario.
  - sintesis:   UNICO nodo que responde. Solo con la informacion disponible.
  - Solo el supervisor puede decidir ir a sintesis, y sintesis es el unico
    nodo conectado a END.
"""

import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.agents.base import AgenteReAct
from app.config import ConfigParams, get_prompts
from app.providers.llm_client import LLMClient
from app.schema.models import Contribucion, SupervisorOutput
from app.state import (
    contribuciones_del_turno,
    pregunta_actual,
    recortar_contexto,
    texto_plano,
)

logger = logging.getLogger(__name__)

AGENTES = ("investigador", "calculador", "buscador")


def _es_inutil(contenido: str) -> bool:
    """Una contribucion no aporta si fallo o no encontro nada."""
    limpio = (contenido or "").strip().lower()
    return not limpio or limpio == "sin resultados" or limpio.startswith("error")


def _resumen_contribuciones(state: dict) -> str:
    contribuciones = contribuciones_del_turno(state)
    if not contribuciones:
        return "Todavia ningun agente aporto informacion."
    return "\n\n".join(f"[{c['agente']}]\n{c['contenido']}" for c in contribuciones)


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #

def build_supervisor(llm: LLMClient, config: ConfigParams):
    prompt_base = get_prompts()["supervisor"]

    async def supervisor(state: dict) -> dict:
        # Guardia 1: ya hay decision humana -> sintesis sin gastar una llamada.
        if state.get("approved") is not None:
            return {
                "next_node": "sintesis",
                "instruccion_actual": "",
                "intentos_supervisor": state.get("intentos_supervisor", 0),
            }

        # Guardia 2: se agotaron las llamadas permitidas en el turno.
        intentos = state.get("intentos_supervisor", 0)
        if intentos >= config.max_intentos_supervisor:
            logger.info("Supervisor alcanzo el maximo de intentos: fuerza sintesis")
            return {"next_node": "sintesis", "instruccion_actual": "", "intentos_supervisor": intentos}

        # Guardia 3: catalogo vacio -> buscador no es una opcion ofrecida.
        disponibles = [a for a in AGENTES if a != "buscador" or state.get("catalog_available")]

        # Guardia 4: un agente que ya aporto en este turno no se vuelve a
        # ofrecer. Sin esto el modelo reintenta al mismo agente hasta agotar
        # max_intentos_supervisor, gastando llamadas que no pueden aportar nada
        # nuevo (el resultado seria identico).
        ya_aportaron = {c["agente"] for c in contribuciones_del_turno(state)}
        disponibles = [a for a in disponibles if a not in ya_aportaron]

        # Sin agentes utiles que queden, la unica salida es responder.
        if not disponibles:
            logger.info("Todos los agentes utiles ya aportaron: va a sintesis")
            return {"next_node": "sintesis", "instruccion_actual": "", "intentos_supervisor": intentos}

        opciones = ", ".join(disponibles + ["aprobacion", "sintesis"])

        historial = recortar_contexto(state.get("messages") or [], config.max_tokens_contexto)
        mensajes = [
            SystemMessage(content=f"{prompt_base}\n\nAgentes habilitados en este turno: {opciones}."),
            *historial,
            HumanMessage(
                content=(
                    f"Contribuciones recibidas en este turno:\n{_resumen_contribuciones(state)}\n\n"
                    f"Elegi el siguiente paso entre: {opciones}."
                )
            ),
        ]

        try:
            decision: SupervisorOutput = await llm.ainvoke_structured(mensajes, SupervisorOutput)
        except Exception as exc:
            logger.error("Supervisor sin LLM disponible: %s", exc)
            return {"next_node": "sintesis", "instruccion_actual": "", "intentos_supervisor": intentos + 1}

        # El esquema de salida admite los cinco nodos siempre, asi que el modelo
        # puede elegir uno que las guardias no habilitaron. Se valida en codigo:
        # el prompt solo sugiere, esto decide.
        elegido = decision.next_node
        permitidos = set(disponibles) | {"aprobacion", "sintesis"}
        if elegido not in permitidos:
            logger.info("Supervisor eligio %s, no habilitado en este turno: va a sintesis", elegido)
            elegido = "sintesis"

        return {
            "next_node": elegido,
            "instruccion_actual": decision.instruccion,
            "intentos_supervisor": intentos + 1,
        }

    return supervisor


def decidir_ruta(state: dict) -> str:
    """Routing posterior al supervisor. Solo el supervisor llega a sintesis."""
    destino = state.get("next_node")
    if destino in AGENTES or destino in ("aprobacion", "sintesis"):
        return destino
    logger.warning("next_node invalido (%r): se envia a sintesis", destino)
    return "sintesis"


# --------------------------------------------------------------------------- #
# Agentes
# --------------------------------------------------------------------------- #

def build_nodo_agente(agente: AgenteReAct):
    async def nodo(state: dict) -> dict:
        contenido = await agente.ejecutar(
            consulta=pregunta_actual(state),
            instruccion=state.get("instruccion_actual") or "",
        )
        contribucion = Contribucion(
            agente=agente.nombre,
            turno_id=state.get("turno_id", ""),
            contenido=contenido,
        )
        return {"contribuciones": [contribucion.model_dump()]}

    return nodo


# --------------------------------------------------------------------------- #
# Aprobacion (HITL)
# --------------------------------------------------------------------------- #

async def nodo_aprobacion(state: dict) -> dict:
    """Pausa el grafo hasta la decision humana. Sin LLM.

    Al reanudar con Command(resume={...}), interrupt() devuelve ese valor.
    Siempre vuelve al supervisor, que por la guardia 1 ira a sintesis.
    """
    decision = interrupt(
        {
            "motivo": "la consulta requiere autorizacion humana",
            "consulta": pregunta_actual(state),
        }
    )
    decision = decision or {}
    return {
        "approved": bool(decision.get("approved")),
        "approval_reason": str(decision.get("reason") or ""),
        "approval_timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Sintesis
# --------------------------------------------------------------------------- #

def build_sintesis(llm: LLMClient, config: ConfigParams):
    prompt_base = get_prompts()["sintesis"]

    async def sintesis(state: dict) -> dict:
        contribuciones = contribuciones_del_turno(state)
        hubo_agentes = bool(contribuciones)
        sin_datos = hubo_agentes and all(_es_inutil(c["contenido"]) for c in contribuciones)

        # Ya se aviso antes que no hay informacion: no se vuelve a responder.
        # Un AIMessage vacio le indica al worker que no debe llamar al callback,
        # para no gastar un mensaje pago de WhatsApp en repetir lo mismo.
        if sin_datos and state.get("no_data_notified"):
            logger.info("Sin datos y ya notificado: no se envia respuesta")
            return {"messages": [AIMessage(content="", name="sintesis")]}

        decision_humana = ""
        if state.get("approved") is not None:
            estado = "APROBADA" if state["approved"] else "RECHAZADA"
            decision_humana = (
                f"\n\nDecision humana registrada: la solicitud fue {estado}. "
                f"Motivo: {state.get('approval_reason') or 'sin motivo indicado'}. "
                "Comunicala primero y no la contradigas."
            )

        mensajes = [
            SystemMessage(content=prompt_base.format(max_lineas=config.max_lineas_respuesta)),
            HumanMessage(
                content=(
                    f"<consulta_usuario>{pregunta_actual(state)}</consulta_usuario>\n\n"
                    f"Informacion aportada por los agentes:\n{_resumen_contribuciones(state)}"
                    f"{decision_humana}"
                )
            ),
        ]

        try:
            respuesta = await llm.ainvoke(mensajes)
            texto = texto_plano(respuesta.content).strip()
        except Exception as exc:
            logger.error("Sintesis sin LLM disponible: %s", exc)
            texto = ""

        if not texto:
            texto = "Por el momento no tengo informacion disponible para responder tu consulta."

        return {
            "messages": [AIMessage(content=texto, name="sintesis")],
            # El flag se activa solo si esta respuesta fue por falta de datos, y
            # se limpia cuando hubo informacion real.
            "no_data_notified": sin_datos,
        }

    return sintesis
