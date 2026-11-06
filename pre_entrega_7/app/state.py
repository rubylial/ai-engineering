"""Estado compartido del grafo: el del M6 más los campos del flujo HITL.

Campos nuevos respecto del Módulo 6:
- `original_query`: la pregunta cruda del usuario (la usa el gate para mostrarla
  en el pedido de aprobación).
- `is_critical`: lo calcula la API al crear el job (keywords o flag explícito).
- `approved`: lo setea el nodo gate tras la aprobación/rechazo humano.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

AgentName = Literal["researcher", "analyst", "FINISH"]


class Contribution(TypedDict):
    """Qué agregó un agente en un paso."""

    agent: str
    step: int
    summary: str


class SupervisorState(MessagesState):
    """Estado del grafo: MessagesState + supervisor + HITL."""

    next_agent: AgentName
    task: str
    step: int
    task_completed: bool
    contributions: Annotated[list[Contribution], add]
    # HITL
    original_query: str
    is_critical: bool
    approved: bool


class SupervisorDecision(BaseModel):
    """Salida estructurada del supervisor (igual que el M6)."""

    next: AgentName = Field(..., description="Próximo nodo: researcher, analyst o FINISH.")
    reason: str = Field(..., min_length=3, description="Por qué esa decisión.")
    task: str = Field(
        ...,
        min_length=3,
        description="Instrucción para el worker, o la respuesta final si next=FINISH.",
    )
