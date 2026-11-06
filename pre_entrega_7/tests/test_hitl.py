"""Tests del gate HITL y la detección de criticidad (sin red)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.hitl import gate_node, is_critical
from tests.conftest import _build_test_graph

KEYWORDS = {"borrar", "producción", "ejecutar"}


def test_is_critical_detects_keyword() -> None:
    assert is_critical("por favor borrar la tabla", KEYWORDS) is True
    assert is_critical("en PRODUCCIÓN", KEYWORDS) is True


def test_is_critical_false_for_safe_text() -> None:
    assert is_critical("resumí la documentación de redis", KEYWORDS) is False


def test_gate_passes_through_when_not_critical() -> None:
    out = gate_node({"is_critical": False})  # type: ignore[arg-type]
    assert out == {"approved": True}


async def test_graph_pauses_on_critical_then_resumes_approved() -> None:
    graph = _build_test_graph()
    cfg = {"configurable": {"thread_id": "t-approve"}}
    first = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="borrar prod")],
            "original_query": "borrar prod",
            "is_critical": True,
        },
        config=cfg,
    )
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["type"] == "approval_required"

    resumed = await graph.ainvoke(Command(resume={"approved": True}), config=cfg)
    assert "__interrupt__" not in resumed
    assert resumed["task_completed"] is True
    assert resumed["approved"] is True


async def test_graph_rejected_does_not_execute() -> None:
    graph = _build_test_graph()
    cfg = {"configurable": {"thread_id": "t-reject"}}
    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="borrar prod")],
            "original_query": "borrar prod",
            "is_critical": True,
        },
        config=cfg,
    )
    resumed = await graph.ainvoke(Command(resume={"approved": False}), config=cfg)
    assert resumed["approved"] is False
    assert resumed["task_completed"] is True
    assert "rechazada" in resumed["task"].lower()
    # No corrió ningún worker: no hay contribuciones.
    assert resumed.get("contributions", []) == []


async def test_graph_non_critical_runs_without_pause() -> None:
    graph = _build_test_graph()
    cfg = {"configurable": {"thread_id": "t-normal"}}
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content="hola")], "original_query": "hola", "is_critical": False},
        config=cfg,
    )
    assert "__interrupt__" not in out
    assert out["task_completed"] is True
    assert [c["agent"] for c in out["contributions"]] == ["analyst"]
