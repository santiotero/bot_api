"""Modelos Pydantic de entrada y salida de la API y de los nodos del grafo."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

NEXT_NODES = Literal["investigador", "calculador", "buscador", "aprobacion", "sintesis"]


class MessageRequest(BaseModel):
    """Mensaje entrante reenviado por watbot_v2."""

    phone: str = Field(min_length=6, max_length=20, pattern=r"^\d+$")
    message: str = Field(min_length=1, max_length=4096)
    message_id: Optional[str] = Field(default=None, max_length=200)


class TaskResponse(BaseModel):
    """Estado de una tarea."""

    job_id: str
    status: Literal["PENDING", "RUNNING", "WAITING_HUMAN", "APPROVING", "DONE", "FAILED"]
    response: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Decision humana sobre una tarea en WAITING_HUMAN."""

    approved: bool
    reason: str = Field(default="", max_length=500)


class Documento(BaseModel):
    """Documento a ingestar. El servidor hace el chunking y los embeddings."""

    content: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Documento] = Field(min_length=1, max_length=500)


class IngestResponse(BaseModel):
    status: str
    count: int


class CheckDBResponse(BaseModel):
    status: Literal["ok", "empty", "inconsistent"]
    fts5_count: int
    vec_count: int
    consistent: bool


class SupervisorOutput(BaseModel):
    """Salida estructurada del supervisor. Solo decide, nunca responde.

    Las descripciones piden brevedad explicitamente: el modelo local tiende a
    divagar en campos de texto libre y agota el presupuesto de tokens.
    """

    next_node: NEXT_NODES = Field(description="El unico agente que debe actuar ahora.")
    instruccion: str = Field(
        default="",
        max_length=200,
        description="Instruccion concreta para ese agente, en una sola frase breve.",
    )
    razon: str = Field(
        default="",
        max_length=150,
        description="Motivo de la eleccion, maximo diez palabras.",
    )


class Contribucion(BaseModel):
    """Aporte de un agente al estado del grafo."""

    model_config = ConfigDict(extra="forbid")

    agente: str
    turno_id: str
    contenido: str
