"""Supervisor: decide el ruteo con salida estructurada `{next, reason, task}`.

Topología jerárquica (estrella): todos los workers vuelven al supervisor y
solo el writer termina el turno. Además de la decisión del LLM hay reglas
deterministas que la acotan:
- presupuesto de pasos (`AGENT_MAX_STEPS`) → writer;
- anti-bucle: el mismo worker con la misma tarea no se repite → writer;
- si el LLM falla (timeout, error, salida inválida) → ruteo por reglas.
"""

from __future__ import annotations

import re
import unicodedata

from intelligence_system.agents.prompts import SUPERVISOR_SYSTEM, SUPERVISOR_USER
from intelligence_system.agents.runtime import AgentDeps, NodeOutcome
from intelligence_system.agents.schemas import SupervisorDecision
from intelligence_system.agents.state import AgentState, PlanStep, StepStatus, last_user_message
from intelligence_system.llm.exceptions import LLMError
from intelligence_system.llm.schemas import ChatMessage
from intelligence_system.observability.logging import get_logger

log = get_logger(__name__)
WORKERS = ("researcher", "analyst")


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"\W+", " ", folded).strip()


def format_history(state: AgentState, window: int) -> str:
    """Últimos `window` mensajes previos a la pregunta actual (contexto mínimo)."""
    messages = list(state.get("messages") or [])[:-1]
    if window <= 0 or not messages:
        return "(sin historial)"
    lines = []
    for message in messages[-window:]:
        who = "Usuario" if message.type == "human" else "Asistente"
        lines.append(f"{who}: {str(message.content)[:600]}")
    return "\n".join(lines)


def executed_workers(state: AgentState) -> list[str]:
    return [s["node"] for s in state.get("steps") or [] if s["node"] in WORKERS]


def _progress(state: AgentState) -> str:
    lines = [
        f"- Paso {s['step']}: {s['node']} ({s['status']}) → {s['detail'][:200]}"
        for s in state.get("steps") or []
        if s["node"] in WORKERS
    ]
    return "\n".join(lines) if lines else "- (todavía no trabajó ningún worker)"


def build_supervisor_messages(state: AgentState, deps: AgentDeps) -> list[ChatMessage]:
    evidence = state.get("evidence") or []
    executed = executed_workers(state)
    error = state.get("error")
    content = SUPERVISOR_USER.format(
        history=format_history(state, deps.settings.agent_history_window),
        question=last_user_message(state),
        executed=", ".join(executed) if executed else "ninguno",
        used=state.get("step_count", 0),
        budget=deps.settings.agent_max_steps,
        progress=_progress(state),
        doc_count=sum(1 for e in evidence if e["kind"] == "document"),
        tool_count=sum(1 for e in evidence if e["kind"] == "tool"),
        error=f"Último error de un worker: {error}\n" if error else "",
    )
    return [ChatMessage.user(content)]


def rule_based_decision(state: AgentState) -> SupervisorDecision:
    """Ruteo determinista de respaldo: investigar una vez y luego redactar."""
    question = last_user_message(state) or "consulta del usuario"
    if not state.get("evidence") and "researcher" not in executed_workers(state):
        return SupervisorDecision(
            next="researcher", reason="regla: todavía no hay evidencia", task=question[:500]
        )
    return SupervisorDecision(next="writer", reason="regla: redactar con lo reunido", task=question[:500])


def apply_guards(decision: SupervisorDecision, state: AgentState) -> tuple[SupervisorDecision, bool]:
    """Anti-bucle: no repetir el mismo worker con la misma tarea."""
    if decision.next not in WORKERS:
        return decision, False
    task = _normalize(decision.task)
    for previous in state.get("plan") or []:
        if previous["next"] == decision.next and _normalize(previous["task"]) == task:
            guarded = SupervisorDecision(
                next="writer",
                reason=f"anti-bucle: {decision.next} ya ejecutó esta tarea",
                task=decision.task,
            )
            return guarded, True
    return decision, False


def _route(
    state: AgentState, decision: SupervisorDecision, decided_by: str, status: StepStatus
) -> NodeOutcome:
    plan = list(state.get("plan") or [])
    step = PlanStep(
        step=len(plan) + 1,
        next=decision.next,
        reason=decision.reason,
        task=decision.task,
        decided_by="llm" if decided_by == "llm" else "rule",
    )
    return NodeOutcome(
        goto=decision.next,
        update={"plan": [*plan, step], "next": decision.next, "current_task": decision.task},
        detail=f"→ {decision.next} ({decided_by}): {decision.reason}",
        status=status,
    )


async def supervisor_logic(state: AgentState, deps: AgentDeps) -> NodeOutcome:
    if state.get("step_count", 0) >= deps.settings.agent_max_steps:
        decision = SupervisorDecision(
            next="writer",
            reason="regla: presupuesto de pasos agotado",
            task=last_user_message(state)[:500] or "redactar",
        )
        return _route(state, decision, "rule", "ok")

    try:
        decision = await deps.llm.generate_structured(
            build_supervisor_messages(state, deps),
            deps.supervisor_config(SUPERVISOR_SYSTEM),
            SupervisorDecision,
            max_attempts=deps.settings.llm_structured_max_attempts,
        )
        decided_by, status = "llm", "ok"
    except LLMError as exc:
        log.warning("supervisor.llm_failed", error=type(exc).__name__)
        decision, decided_by, status = rule_based_decision(state), "rule", "fallback"

    decision, guarded = apply_guards(decision, state)
    if guarded:
        decided_by = "rule"
    log.info("supervisor.decision", next=decision.next, decided_by=decided_by, reason=decision.reason)
    return _route(state, decision, decided_by, status)


def supervisor_fallback(state: AgentState, deps: AgentDeps, reason: str, status: StepStatus) -> NodeOutcome:
    del deps
    decision = rule_based_decision(state)
    decision = decision.model_copy(update={"reason": f"{decision.reason} ({reason})"})
    return _route(state, decision, "rule", status)
