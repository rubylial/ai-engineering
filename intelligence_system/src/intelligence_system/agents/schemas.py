"""Salidas estructuradas del LLM y modelos de resultado del turno (Pydantic v2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    """Decisión de ruteo del supervisor: quién trabaja a continuación y con qué tarea."""

    next: Literal["researcher", "analyst", "writer"] = Field(
        description="Worker que ejecuta el próximo paso."
    )
    reason: str = Field(min_length=3, max_length=400, description="Justificación breve de la decisión.")
    task: str = Field(
        min_length=3,
        max_length=500,
        description="Instrucción autocontenida para el worker (sin pronombres ambiguos).",
    )


class EvidenceGrade(BaseModel):
    """Qué fragmentos recuperados sirven de verdad para responder la consulta."""

    relevant_indices: list[int] = Field(
        default_factory=list,
        description="Números de los fragmentos útiles (por ejemplo [1, 3]). Vacío si ninguno sirve.",
    )
    rationale: str = Field(default="", max_length=400)


class Citation(BaseModel):
    """Cita verificable de la respuesta: apunta a un fragmento o a un cálculo."""

    marker: int = Field(ge=1, description="Número usado en el texto, p. ej. [1].")
    id: str
    kind: Literal["document", "tool"]
    title: str
    source: str
    snippet: str
    score: float | None = None


class StepInfo(BaseModel):
    """Un paso ejecutado por el grafo en este turno."""

    step: int
    node: str
    status: Literal["ok", "fallback", "timeout", "error"]
    detail: str
    duration_ms: float


class ThreadMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    id: str | None = None


class ThreadHistory(BaseModel):
    """Historial de un thread leído desde el checkpointer."""

    thread_id: str
    messages: list[ThreadMessage]
    last_answer: str | None = None
    last_grounded: bool | None = None
    last_citations: list[Citation] = Field(default_factory=list)
    last_steps: list[StepInfo] = Field(default_factory=list)
    checkpoint_id: str | None = None
    updated_at: str | None = None


class StreamEvent(BaseModel):
    """Evento SSE: token | step | citation | done | error."""

    event: Literal["token", "step", "citation", "done", "error"]
    data: dict[str, Any]


class TurnResult(BaseModel):
    """Resultado de un turno de conversación, independiente de la capa HTTP."""

    thread_id: str
    request_id: str
    answer: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)
    steps: list[StepInfo] = Field(default_factory=list)
    trace_id: str | None = None

    @classmethod
    def from_state(
        cls, state: dict[str, Any], *, thread_id: str, request_id: str, trace_id: str | None
    ) -> TurnResult:
        return cls(
            thread_id=thread_id,
            request_id=request_id,
            answer=str(state.get("answer") or ""),
            grounded=bool(state.get("grounded")),
            citations=[Citation.model_validate(c) for c in state.get("citations") or []],
            steps=[StepInfo.model_validate(s) for s in state.get("steps") or []],
            trace_id=trace_id,
        )
