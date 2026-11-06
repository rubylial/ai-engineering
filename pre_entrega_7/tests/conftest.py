"""Fixtures compartidos: grafo de prueba sin LLM, JobStore sobre fakeredis, Worker.

El grafo de prueba usa el gate_node REAL (con interrupt) y un InMemorySaver, pero
reemplaza supervisor/researcher/analyst por nodos fake deterministas. Así los tests
ejercitan el flujo completo (incluido HITL y FAILED) sin red ni Redis real.
"""

from __future__ import annotations

from typing import Any

import pytest
from fakeredis.aioredis import FakeRedis
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from app.graph import build_graph
from app.hitl import gate_node
from app.jobs import JobStore
from app.state import SupervisorState
from app.worker import Worker

CRITICAL_KEYWORDS = {"borrar", "producción", "ejecutar"}


async def _fake_supervisor(state: SupervisorState) -> dict[str, Any]:
    """Si no corrió ningún worker, manda al analyst; si ya corrió, FINISH."""
    contributions = state.get("contributions", [])
    step = state.get("step", 0) + 1
    if contributions:
        return {
            "next_agent": "FINISH",
            "step": step,
            "task_completed": True,
            "task": "Listo: la tarea se resolvió.",
            "messages": [AIMessage(content="[supervisor] FINISH", name="supervisor")],
        }
    return {
        "next_agent": "analyst",
        "step": step,
        "task": "procesá los datos",
        "messages": [AIMessage(content="[supervisor] → analyst", name="supervisor")],
    }


async def _fake_analyst(state: SupervisorState) -> dict[str, Any]:
    return {
        "messages": [AIMessage(content="[analyst] media=30", name="analyst")],
        "contributions": [{"agent": "analyst", "step": state.get("step", 0), "summary": "media=30"}],
    }


async def _fake_research(state: SupervisorState) -> dict[str, Any]:
    return {
        "messages": [AIMessage(content="[researcher] ok", name="researcher")],
        "contributions": [{"agent": "researcher", "step": state.get("step", 0), "summary": "hits"}],
    }


async def _boom_analyst(_state: SupervisorState) -> dict[str, Any]:
    raise RuntimeError("fallo simulado del analyst")


def _build_test_graph(*, failing: bool = False) -> CompiledStateGraph:
    analyst = _boom_analyst if failing else _fake_analyst
    return build_graph(gate_node, _fake_supervisor, _fake_research, analyst, checkpointer=InMemorySaver())


@pytest.fixture
def test_graph() -> CompiledStateGraph:
    return _build_test_graph()


@pytest.fixture
def failing_graph() -> CompiledStateGraph:
    return _build_test_graph(failing=True)


@pytest.fixture
async def redis_client() -> Any:
    client = FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def store(redis_client: Any) -> JobStore:
    return JobStore(redis_client, ttl_seconds=3600)


@pytest.fixture
async def worker(test_graph: CompiledStateGraph, store: JobStore) -> Any:
    w = Worker(test_graph, store, critical_keywords=CRITICAL_KEYWORDS, concurrency=2)
    await w.start()
    try:
        yield w
    finally:
        await w.stop()
