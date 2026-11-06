"""Grafo del M6 + nodo HITL + checkpointer inyectable.

Topología:

    START → gate ──(aprobada / no crítica)──► supervisor ──► researcher/analyst ──► supervisor
              │                                     │
              └──(rechazada)──► END                 └──► END

El `gate` es el único cambio estructural respecto del Módulo 6: intercepta antes
del supervisor y, si la tarea es crítica, pausa con interrupt() hasta la aprobación.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import (
    build_analyst_agent,
    build_analyst_node,
    build_research_agent,
    build_research_node,
)
from app.hitl import gate_node
from app.state import SupervisorDecision, SupervisorState
from app.supervisor import supervisor_node_factory

NodeFn = Callable[[SupervisorState], Awaitable[dict[str, Any]]]
SyncNodeFn = Callable[[SupervisorState], dict[str, Any]]


def build_graph(
    gate: SyncNodeFn,
    supervisor_node: NodeFn,
    research_node: NodeFn,
    analyst_node: NodeFn,
    *,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph:
    """Compone el StateGraph con el gate HITL y compila con el checkpointer dado."""
    builder: StateGraph = StateGraph(SupervisorState)
    builder.add_node("gate", gate)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("analyst", analyst_node)

    builder.add_edge(START, "gate")
    builder.add_conditional_edges("gate", _route_after_gate, {"supervisor": "supervisor", "FINISH": END})
    builder.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "FINISH": END},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")

    return builder.compile(checkpointer=checkpointer)


def build_graph_from_llm(
    llm: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver[Any],
    max_steps: int = 6,
) -> CompiledStateGraph:
    """Builder de producción: arma gate + supervisor + workers con el LLM."""
    structured = llm.with_structured_output(SupervisorDecision)
    supervisor = supervisor_node_factory(structured, max_steps=max_steps)
    researcher = build_research_node(build_research_agent(llm))
    analyst = build_analyst_node(build_analyst_agent(llm))
    return build_graph(gate_node, supervisor, researcher, analyst, checkpointer=checkpointer)


def _route_after_gate(state: SupervisorState) -> Literal["supervisor", "FINISH"]:
    """Si el gate rechazó (approved=False), va a END; si no, al supervisor."""
    return "supervisor" if state.get("approved", True) else "FINISH"


def _route_from_supervisor(state: SupervisorState) -> Literal["researcher", "analyst", "FINISH"]:
    return state.get("next_agent", "FINISH")
