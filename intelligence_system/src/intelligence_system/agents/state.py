"""Estado tipado del grafo y su contrato de escritura por nodo.

El estado es un `TypedDict` de valores JSON-serializables: así el checkpointer
(Redis o memoria) lo persiste sin depender de clases propias, y la validación
fuerte ocurre en los bordes (modelos Pydantic de tools, salidas del LLM y API).

Aislamiento de escritura (ejercicio de estado compartido del módulo 7): cada
nodo solo puede escribir su "parcela". `NODE_OWNERSHIP` es la fuente de verdad
y `runtime.py` la hace cumplir en cada paso.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

WorkerName = Literal["researcher", "analyst", "writer"]
StepStatus = Literal["ok", "fallback", "timeout", "error"]


class EvidenceItem(TypedDict):
    id: str
    kind: Literal["document", "tool"]
    title: str
    source: str
    content: str
    score: float | None
    worker: str


class PlanStep(TypedDict):
    step: int
    next: str
    reason: str
    task: str
    decided_by: Literal["llm", "rule"]


class StepRecord(TypedDict):
    step: int
    node: str
    status: StepStatus
    detail: str
    duration_ms: float


class CitationRecord(TypedDict):
    marker: int
    id: str
    kind: str
    title: str
    source: str
    snippet: str
    score: float | None


class AgentState(TypedDict):
    # Memoria conversacional: el único canal que acumula entre turnos.
    messages: Annotated[list[AnyMessage], add_messages]
    request_id: str
    filters: dict[str, Any] | None
    # Canales del turno actual (se reinician en cada mensaje del usuario).
    plan: list[PlanStep]
    next: str | None
    current_task: str | None
    evidence: list[EvidenceItem]
    citations: list[CitationRecord]
    steps: list[StepRecord]
    step_count: int
    error: str | None
    answer: str | None
    grounded: bool | None


NODE_OWNERSHIP: dict[str, frozenset[str]] = {
    "supervisor": frozenset({"plan", "next", "current_task"}),
    "researcher": frozenset({"evidence", "step_count", "error"}),
    "analyst": frozenset({"evidence", "step_count", "error"}),
    "writer": frozenset({"messages", "answer", "citations", "grounded"}),
}
# `steps` lo escribe el runtime (bitácora común), nunca el nodo directamente.
RUNTIME_KEYS = frozenset({"steps"})


def new_turn_input(*, message: str, request_id: str, filters: dict[str, Any] | None) -> dict[str, Any]:
    """Input de un turno nuevo: agrega el mensaje y reinicia los canales del turno."""
    return {
        "messages": [HumanMessage(content=message, id=f"{request_id}-user")],
        "request_id": request_id,
        "filters": filters,
        "plan": [],
        "next": None,
        "current_task": None,
        "evidence": [],
        "citations": [],
        "steps": [],
        "step_count": 0,
        "error": None,
        "answer": None,
        "grounded": None,
    }


def last_user_message(state: AgentState) -> str:
    for message in reversed(state.get("messages") or []):
        if message.type == "human":
            return str(message.content)
    return ""


def merge_evidence(current: list[EvidenceItem], new: list[EvidenceItem]) -> list[EvidenceItem]:
    """Agrega evidencia nueva sin duplicar ids (orden estable)."""
    seen = {item["id"] for item in current}
    merged = list(current)
    for item in new:
        if item["id"] not in seen:
            merged.append(item)
            seen.add(item["id"])
    return merged
