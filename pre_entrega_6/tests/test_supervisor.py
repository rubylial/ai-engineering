"""Tests del nodo supervisor con un LLM stub (sin red)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableLambda

from orquestador.state import SupervisorDecision, SupervisorState
from orquestador.supervisor import supervisor_node_factory


def _scripted_llm(decisions: list[SupervisorDecision]) -> Runnable[Any, SupervisorDecision]:
    """Devuelve un Runnable que entrega decisiones una a una en cada ainvoke."""
    it = iter(decisions)

    async def _run(_: object) -> SupervisorDecision:
        return next(it)

    return RunnableLambda(_run)


def _empty_state(**overrides: Any) -> SupervisorState:
    base: dict[str, Any] = {
        "messages": [],
        "next_agent": "FINISH",
        "task": "",
        "step": 0,
        "task_completed": False,
        "contributions": [],
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


async def test_supervisor_routes_to_researcher() -> None:
    llm = _scripted_llm([SupervisorDecision(next="researcher", reason="falta info", task="Buscá")])
    node = supervisor_node_factory(llm, max_steps=4)
    out = await node(_empty_state())
    assert out["next_agent"] == "researcher"
    assert out["task"] == "Buscá"
    assert out["step"] == 1


async def test_supervisor_finishes_on_decision() -> None:
    llm = _scripted_llm(
        [SupervisorDecision(next="FINISH", reason="ya tengo todo", task="La respuesta final.")]
    )
    node = supervisor_node_factory(llm)
    out = await node(_empty_state())
    assert out["next_agent"] == "FINISH"
    assert out["task_completed"] is True
    assert "La respuesta final." in out["task"]


async def test_supervisor_forces_finish_when_budget_exceeded() -> None:
    # El LLM NO debe ser invocado: el presupuesto frena antes.
    llm = _scripted_llm([])
    node = supervisor_node_factory(llm, max_steps=2)
    out = await node(_empty_state(step=2))
    assert out["next_agent"] == "FINISH"
    assert out["task_completed"] is True
    assert out["step"] == 3


async def test_supervisor_rejects_invalid_max_steps() -> None:
    llm = _scripted_llm([])
    with pytest.raises(ValueError, match="max_steps"):
        supervisor_node_factory(llm, max_steps=0)


async def test_supervisor_blocks_repeat_same_agent_same_task() -> None:
    # Ya corrió researcher con "Buscá X"; el supervisor pide lo mismo otra vez → FINISH.
    llm = _scripted_llm([SupervisorDecision(next="researcher", reason="probar", task="Buscá X")])
    node = supervisor_node_factory(llm)
    state = _empty_state(contributions=[{"agent": "researcher", "step": 1, "summary": "Buscá X"}])
    out = await node(state)
    assert out["next_agent"] == "FINISH"
    assert out["task_completed"] is True
