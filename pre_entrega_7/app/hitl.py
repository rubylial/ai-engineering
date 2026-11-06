"""Human-in-the-loop: detección de tareas críticas y nodo de aprobación.

El nodo `gate` corre al principio del grafo. Si la tarea es crítica, llama a
`interrupt(...)`: LangGraph persiste el estado en el checkpointer (Redis) y
`ainvoke` devuelve con la clave `__interrupt__`. El worker detecta eso y marca
el job como WAITING_APPROVAL. Cuando llega la aprobación (POST .../approve), el
worker reanuda con `Command(resume={"approved": bool})`: el nodo gate se
re-ejecuta, `interrupt(...)` ahora devuelve ese valor en vez de pausar, y el
flujo sigue (o se corta si fue rechazada).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.state import SupervisorState


def is_critical(text: str, keywords: set[str]) -> bool:
    """True si el texto contiene alguna palabra marcada como crítica."""
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def gate_node(state: SupervisorState) -> dict[str, Any]:
    """Nodo de aprobación humana. No crítico → pasa; crítico → interrupt() hasta resume."""
    if not state.get("is_critical"):
        return {"approved": True}

    decision = interrupt(
        {
            "type": "approval_required",
            "query": state.get("original_query", ""),
            "reason": (
                "La tarea fue marcada como crítica (posibles efectos secundarios o "
                "costo alto). Requiere aprobación humana antes de ejecutarse."
            ),
        }
    )

    approved = _coerce_approved(decision)
    if not approved:
        return {
            "approved": False,
            "next_agent": "FINISH",
            "task_completed": True,
            "task": "Tarea rechazada por el revisor humano. No se ejecutó ninguna acción.",
            "messages": [AIMessage(content="[gate] Rechazada por el revisor humano.", name="gate")],
        }
    return {
        "approved": True,
        "messages": [AIMessage(content="[gate] Aprobada por el revisor humano.", name="gate")],
    }


def _coerce_approved(decision: Any) -> bool:
    """El resume puede llegar como {'approved': bool} o como bool directo."""
    if isinstance(decision, dict):
        return bool(decision.get("approved"))
    return bool(decision)
