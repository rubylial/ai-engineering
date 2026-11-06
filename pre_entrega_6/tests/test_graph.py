"""Tests end-to-end del grafo con nodos fake (sin LLM)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from orquestador.graph import build_graph
from orquestador.state import SupervisorDecision, SupervisorState
from orquestador.supervisor import supervisor_node_factory


def _scripted_llm(decisions: list[SupervisorDecision]) -> Runnable[Any, SupervisorDecision]:
    it = iter(decisions)

    async def _run(_: object) -> SupervisorDecision:
        return next(it)

    return RunnableLambda(_run)


async def _fake_research(state: SupervisorState) -> dict[str, Any]:
    return {
        "messages": [AIMessage(content="[researcher] Encontré el doc [doc:x].", name="researcher")],
        "contributions": [{"agent": "researcher", "step": state.get("step", 0), "summary": "KB hits"}],
    }


async def _fake_analyst(state: SupervisorState) -> dict[str, Any]:
    return {
        "messages": [AIMessage(content="[analyst] Media=30, n=3.", name="analyst")],
        "contributions": [{"agent": "analyst", "step": state.get("step", 0), "summary": "media=30"}],
    }


async def test_graph_full_flow_researcher_then_analyst_then_finish() -> None:
    llm = _scripted_llm(
        [
            SupervisorDecision(next="researcher", reason="primero buscar", task="Buscá X"),
            SupervisorDecision(next="analyst", reason="ahora procesar", task="Contá menciones"),
            SupervisorDecision(
                next="FINISH",
                reason="ambos aportaron",
                task="X aparece 3 veces según el KB.",
            ),
        ]
    )
    supervisor = supervisor_node_factory(llm, max_steps=6)
    graph = build_graph(supervisor, _fake_research, _fake_analyst)

    final = await graph.ainvoke({"messages": [HumanMessage(content="Buscá X y contá cuántas veces aparece")]})
    assert final["task_completed"] is True
    agents = [c["agent"] for c in final["contributions"]]
    assert agents == ["researcher", "analyst"]
    assert "X aparece 3 veces" in final["task"]


async def test_graph_single_domain_only_analyst() -> None:
    llm = _scripted_llm(
        [
            SupervisorDecision(next="analyst", reason="pregunta de cálculo", task="Calculá la media"),
            SupervisorDecision(next="FINISH", reason="listo", task="La media es 30."),
        ]
    )
    supervisor = supervisor_node_factory(llm, max_steps=6)
    graph = build_graph(supervisor, _fake_research, _fake_analyst)

    final = await graph.ainvoke({"messages": [HumanMessage(content="Media de [20,30,40]")]})
    assert [c["agent"] for c in final["contributions"]] == ["analyst"]
    assert "La media es 30." in final["task"]


async def test_graph_budget_guard_breaks_infinite_loop() -> None:
    # Simulamos un supervisor terco: siempre quiere researcher (nunca FINISH).
    # El grafo tiene que cortar por max_steps sin colgarse.
    llm = _scripted_llm(
        [SupervisorDecision(next="researcher", reason=f"intento {i}", task=f"Buscá t{i}") for i in range(10)]
    )
    supervisor = supervisor_node_factory(llm, max_steps=3)
    graph = build_graph(supervisor, _fake_research, _fake_analyst)
    final = await graph.ainvoke(
        {"messages": [HumanMessage(content="prueba")]},
        config={"recursion_limit": 25},
    )
    assert final["task_completed"] is True
    assert final["step"] >= 4  # pasó el límite antes de cortar


def test_graph_mermaid_contains_all_nodes() -> None:
    llm = _scripted_llm([])
    supervisor = supervisor_node_factory(llm)
    graph = build_graph(supervisor, _fake_research, _fake_analyst)
    mermaid = graph.get_graph().draw_mermaid()
    assert "supervisor" in mermaid
    assert "researcher" in mermaid
    assert "analyst" in mermaid
    assert "__start__" in mermaid.lower()
    assert "__end__" in mermaid.lower()
