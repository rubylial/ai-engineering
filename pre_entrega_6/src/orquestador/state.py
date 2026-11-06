"""Estado compartido del orquestador y contrato de decisión del supervisor.

`SupervisorState` hereda de `MessagesState` (el TypedDict estándar de LangGraph con
`messages: Annotated[list[BaseMessage], add_messages]`) y agrega los campos que
hacen falta para la topología jerárquica:

- `next_agent`: Literal con el próximo nodo. Los conditional edges del grafo lo
  leen para enrutar al worker o a END.
- `task`: la instrucción concreta que el supervisor le pasa al worker (así el
  worker no recibe todo el historial y se evita la contaminación de contexto).
- `step`: contador de pasos del supervisor. Es el presupuesto duro contra el
  "supervisor infinito".
- `task_completed`: queda en True cuando el supervisor decide FINISH o se agota
  el presupuesto.
- `contributions`: lista acumulada de qué aportó cada agente, con un reducer
  `operator.add` para que LangGraph mergee los deltas que escribe cada worker.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

AgentName = Literal["researcher", "analyst", "FINISH"]


class Contribution(TypedDict):
    """Qué agregó un agente en un paso. Se acumula en `SupervisorState.contributions`."""

    agent: str
    step: int
    summary: str


class SupervisorState(MessagesState):
    """Estado del grafo, extiende `MessagesState` con los campos del supervisor."""

    next_agent: AgentName
    task: str
    step: int
    task_completed: bool
    contributions: Annotated[list[Contribution], add]


class SupervisorDecision(BaseModel):
    """Salida estructurada del supervisor, devuelta por `model.with_structured_output()`.

    Si `next == "FINISH"`, `task` es la respuesta final para el usuario (una o dos
    oraciones). Si `next != "FINISH"`, `task` es la instrucción exacta para ese agente.
    """

    next: AgentName = Field(..., description="Próximo nodo: researcher, analyst o FINISH.")
    reason: str = Field(
        ...,
        min_length=3,
        description="Por qué esa decisión (lo loguea el supervisor).",
    )
    task: str = Field(
        ...,
        min_length=3,
        description=(
            "Si next != FINISH, la instrucción concreta para el worker. "
            "Si next == FINISH, la respuesta final para el usuario."
        ),
    )
