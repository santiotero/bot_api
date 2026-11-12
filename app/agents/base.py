"""Loop ReAct compartido por los agentes.

Principio de responsabilidad exclusiva: cada agente recibe UNA sola
herramienta y no puede ejecutar la tarea de otro.

Ningun agente responde al usuario: producen contribuciones para sintesis. Si el
LLM falla, degradan devolviendo "ERROR: ..." en lugar de propagar la excepcion,
y el supervisor decide si reintentar o sintetizar con lo disponible.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.providers.llm_client import LLMClient
from app.state import texto_plano

logger = logging.getLogger(__name__)


class AgenteReAct:
    def __init__(
        self,
        nombre: str,
        system_prompt: str,
        herramientas: list[BaseTool],
        llm: LLMClient,
        max_iteraciones: int,
    ):
        self.nombre = nombre
        self._system_prompt = system_prompt
        self._herramientas = herramientas
        self._por_nombre = {h.name: h for h in herramientas}
        self._llm = llm
        self._max_iteraciones = max_iteraciones

    async def _ejecutar_tool(self, llamada: dict) -> str:
        herramienta = self._por_nombre.get(llamada["name"])
        if herramienta is None:
            # El modelo invento una herramienta que este agente no tiene.
            return f"ERROR: {self.nombre} no dispone de la herramienta {llamada['name']}"
        try:
            return str(await herramienta.ainvoke(llamada["args"]))
        except Exception as exc:
            logger.error("Fallo la tool %s: %s", llamada["name"], exc)
            return f"ERROR: fallo la herramienta ({type(exc).__name__})"

    async def ejecutar(self, consulta: str, instruccion: str) -> str:
        mensajes: list = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=(
                    f"<consulta_usuario>{consulta}</consulta_usuario>\n\n"
                    f"Tarea asignada: {instruccion}"
                )
            ),
        ]
        resultados_tool: list[str] = []

        for _ in range(self._max_iteraciones):
            try:
                respuesta = await self._llm.ainvoke_tools(mensajes, self._herramientas)
            except Exception as exc:
                logger.error("Agente %s sin LLM disponible: %s", self.nombre, exc)
                return f"ERROR: {self.nombre} no pudo consultar el modelo"

            mensajes.append(respuesta)
            llamadas = getattr(respuesta, "tool_calls", None) or []

            if not llamadas:
                texto = texto_plano(respuesta.content).strip()
                if not texto:
                    break
                # Sin haber ejecutado nunca su herramienta, el agente no produjo
                # un dato: esta improvisando. Se marca como ERROR para que
                # sintesis no lo tome como informacion real y el supervisor no
                # lo vuelva a elegir.
                if not resultados_tool:
                    logger.warning(
                        "Agente %s respondio sin ejecutar su herramienta", self.nombre
                    )
                    return f"ERROR: {self.nombre} no ejecuto su herramienta ({texto[:150]})"
                return texto

            for llamada in llamadas:
                resultado = await self._ejecutar_tool(llamada)
                resultados_tool.append(resultado)
                mensajes.append(ToolMessage(content=resultado, tool_call_id=llamada["id"]))

        # Sin texto final del modelo, lo que importa es el dato crudo de la
        # herramienta: sintesis lo usa tal cual, sin transformarlo.
        if resultados_tool:
            return "\n".join(resultados_tool)
        return f"ERROR: {self.nombre} no produjo resultado"
