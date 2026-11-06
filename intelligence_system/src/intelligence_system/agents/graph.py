"""Construcción del StateGraph: supervisor + researcher + analyst + writer.

Handoffs con `Command(goto=...)`: cada nodo decide a dónde sigue el flujo y
lo declara en `destinations`, lo que además permite dibujar el grafo real con
`graph.get_graph().draw_mermaid()` (ver scripts/export_graph.py).

El checkpointer se inyecta desde afuera (lección de la pre-entrega 5: los
checkpointers async son context managers y su ciclo de vida lo maneja el
lifespan de la app, no el grafo).
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from intelligence_system.agents.runtime import AgentDeps, make_node
from intelligence_system.agents.state import AgentState
from intelligence_system.agents.supervisor import supervisor_fallback, supervisor_logic
from intelligence_system.agents.workers.analyst import analyst_fallback, analyst_logic
from intelligence_system.agents.workers.researcher import researcher_fallback, researcher_logic
from intelligence_system.agents.workers.writer import writer_logic

GRAPH_NAME = "intelligence_system"


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any] | None) -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node(
        "supervisor",
        make_node("supervisor", supervisor_logic, deps, fallback=supervisor_fallback),
        destinations=("researcher", "analyst", "writer"),
    )
    graph.add_node(
        "researcher",
        make_node("researcher", researcher_logic, deps, fallback=researcher_fallback),
        destinations=("supervisor",),
    )
    graph.add_node(
        "analyst",
        make_node("analyst", analyst_logic, deps, fallback=analyst_fallback),
        destinations=("supervisor",),
    )
    # Sin fallback: si el writer falla, el request falla (502/504) en vez de
    # devolver un 200 con una respuesta degradada.
    graph.add_node("writer", make_node("writer", writer_logic, deps, fallback=None))
    graph.add_edge(START, "supervisor")
    graph.add_edge("writer", END)
    return graph.compile(checkpointer=checkpointer, name=GRAPH_NAME)
