"""Nodo supervisor: decide a quién mandarle la tarea o cerrarla.

Diseño para evitar el "supervisor infinito":
1. **Presupuesto duro de pasos** (`max_steps`): si el contador supera el límite, el
   nodo responde FINISH sin consultar al LLM. Es el freno de emergencia.
2. **Rúbrica en el prompt**: el prompt explicita cuándo DEBE devolver FINISH
   (ambos agentes ya contribuyeron, o la pregunta no requiere ambos dominios).
3. **Anti-repetición**: si el supervisor intenta mandarle la misma instrucción al
   mismo agente dos veces seguidas, se fuerza FINISH. Evita loops de "corregir y
   volver a corregir" sin progreso.

El nodo recibe un `Runnable[Any, SupervisorDecision]` (típicamente
`llm.with_structured_output(SupervisorDecision)`) para que los tests puedan
inyectar un stub.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from orquestador.state import SupervisorDecision, SupervisorState

log = logging.getLogger(__name__)

SUPERVISOR_SYSTEM = (
    "Sos el supervisor de un sistema multi-agente. Rutear la tarea del usuario entre "
    "dos especialistas:\n"
    "- `researcher`: busca información en el knowledge base del curso. Usalo cuando "
    "falta información fáctica.\n"
    "- `analyst`: hace cálculos, cuenta menciones y analiza sentimiento de texto. "
    "Usalo cuando hay que procesar datos ya recolectados.\n\n"
    "Reglas duras:\n"
    "1. Si ya hay contribuciones de researcher Y analyst, DEBÉS devolver next='FINISH'.\n"
    "2. Si la pregunta es solo de búsqueda, con researcher alcanza.\n"
    "3. Si la pregunta es solo de cálculo (ej: 'calculá la media de [1,2,3]'), con "
    "analyst alcanza.\n"
    "4. Nunca pidas lo MISMO dos veces al mismo agente: revisá `contributions` antes "
    "de decidir.\n"
    "5. Cuando next='FINISH', poné en `task` la respuesta final para el usuario, en "
    "una o dos oraciones, citando los hallazgos.\n"
    "6. Cuando next != 'FINISH', poné en `task` la instrucción ESPECÍFICA para ese "
    "agente (qué buscar o qué calcular).\n"
    "7. Vas en el paso {step} de {max_steps}. Priorizá cerrar si ya tenés lo necesario."
)

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUPERVISOR_SYSTEM),
        MessagesPlaceholder("messages"),
        (
            "human",
            "Contribuciones hasta ahora: {contributions}. ¿Qué nodo activo ahora?",
        ),
    ]
)


def supervisor_node_factory(
    llm: Runnable[Any, SupervisorDecision],
    *,
    max_steps: int = 6,
) -> Callable[[SupervisorState], Awaitable[dict[str, Any]]]:
    """Devuelve el callable del nodo supervisor con el LLM inyectado.

    `llm` normalmente es `chat_model.with_structured_output(SupervisorDecision)`.
    Se inyecta así para que los tests pasen un stub sin red.
    """
    if max_steps < 1:
        raise ValueError("max_steps debe ser >= 1")

    async def _supervisor(state: SupervisorState) -> dict[str, Any]:
        step = state.get("step", 0) + 1
        contributions = state.get("contributions", [])

        # Presupuesto duro: freno de emergencia.
        if step > max_steps:
            log.warning("supervisor.budget_exceeded step=%d max=%d", step, max_steps)
            return _finish(
                step=step,
                reason=f"límite de {max_steps} pasos alcanzado",
                answer=_summarize_with_budget(contributions),
            )

        decision = await llm.ainvoke(
            SUPERVISOR_PROMPT.format_messages(
                step=step,
                max_steps=max_steps,
                messages=state.get("messages", []),
                contributions=_format_contributions(contributions),
            )
        )
        log.info(
            "supervisor.decision step=%d next=%s reason=%s",
            step,
            decision.next,
            decision.reason[:80],
        )

        # Anti-repetición: si pide lo mismo al mismo agente dos veces seguidas, FINISH.
        if _is_repeat(decision, contributions):
            log.warning("supervisor.repeat_detected forzando FINISH")
            return _finish(
                step=step,
                reason="el supervisor pidió lo mismo dos veces seguidas",
                answer=_summarize_with_budget(contributions),
            )

        if decision.next == "FINISH":
            return _finish(step=step, reason=decision.reason, answer=decision.task)

        return {
            "next_agent": decision.next,
            "task": decision.task,
            "step": step,
            "messages": [
                AIMessage(
                    content=f"[supervisor] → {decision.next}: {decision.task}",
                    name="supervisor",
                )
            ],
        }

    return _supervisor


def _finish(*, step: int, reason: str, answer: str) -> dict[str, Any]:
    return {
        "next_agent": "FINISH",
        "step": step,
        "task_completed": True,
        "task": answer,
        "messages": [
            AIMessage(
                content=f"[supervisor] FINISH ({reason}): {answer}",
                name="supervisor",
            )
        ],
    }


def _format_contributions(contributions: list[dict[str, Any]]) -> str:
    if not contributions:
        return "(ninguna todavía)"
    return "; ".join(f"{c['agent']} (paso {c['step']}): {c['summary'][:120]}" for c in contributions)


def _summarize_with_budget(contributions: list[dict[str, Any]]) -> str:
    if not contributions:
        return "No alcancé a reunir evidencia; corté la tarea por límite de pasos."
    pieces = [f"{c['agent']} aportó: {c['summary'][:200]}" for c in contributions]
    return "Cierre por presupuesto. " + " | ".join(pieces)


def _is_repeat(decision: SupervisorDecision, contributions: list[dict[str, Any]]) -> bool:
    if decision.next == "FINISH" or not contributions:
        return False
    last = contributions[-1]
    if last["agent"] != decision.next:
        return False
    # Mismo agente dos veces seguidas con una instrucción casi idéntica (prefijo).
    return decision.task.strip()[:80].casefold() == last["summary"].strip()[:80].casefold()
