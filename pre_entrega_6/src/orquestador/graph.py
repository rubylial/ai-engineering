"""Construcción del grafo: supervisor + researcher + analyst.

Topología (jerárquica, "star"):

    START → supervisor ──► researcher ──► supervisor
                       │                      │
                       ├─► analyst ──────────►│
                       │                      │
                       └─► END

El supervisor corre primero y en cada vuelta. Los conditional edges leen
`state["next_agent"]` (actualizado por el nodo supervisor) y rutean. Los workers
siempre vuelven al supervisor.

Dos builders:
- `build_graph(...)`: recibe los nodos ya armados. Lo usan los tests.
- `build_graph_from_llm(llm)`: arma researcher + analyst + supervisor con el LLM dado.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orquestador.agents import (
    build_analyst_agent,
    build_analyst_node,
    build_research_agent,
    build_research_node,
)
from orquestador.state import SupervisorDecision, SupervisorState
from orquestador.supervisor import supervisor_node_factory

NodeFn = Callable[[SupervisorState], Awaitable[dict[str, Any]]]


def build_graph(
    supervisor_node: NodeFn,
    research_node: NodeFn,
    analyst_node: NodeFn,
) -> CompiledStateGraph:
    """Compone el StateGraph con los tres nodos. No hace red ni llama al LLM."""
    builder: StateGraph = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("analyst", analyst_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "FINISH": END},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")

    return builder.compile()


def build_graph_from_llm(
    llm: BaseChatModel,
    *,
    max_steps: int = 6,
) -> CompiledStateGraph:
    """Builder "de producción": arma los react agents y el supervisor con el LLM."""
    structured = llm.with_structured_output(SupervisorDecision)
    supervisor = supervisor_node_factory(structured, max_steps=max_steps)
    researcher = build_research_node(build_research_agent(llm))
    analyst = build_analyst_node(build_analyst_agent(llm))
    return build_graph(supervisor, researcher, analyst)


def _route_from_supervisor(
    state: SupervisorState,
) -> Literal["researcher", "analyst", "FINISH"]:
    """Lee la decisión que el supervisor dejó en el state."""
    return state.get("next_agent", "FINISH")
