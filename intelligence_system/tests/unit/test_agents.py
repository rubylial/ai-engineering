"""Supervisor, workers y runtime de nodos con un LLM guionado (sin red)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from helpers import ScriptedLLM, build_container, make_settings, response, tool_call
from intelligence_system.agents.prompts import ABSTENTION_ANSWER
from intelligence_system.agents.runtime import (
    AgentDeps,
    NodeOutcome,
    OwnershipViolationError,
    check_ownership,
    make_node,
)
from intelligence_system.agents.schemas import SupervisorDecision
from intelligence_system.agents.state import AgentState, EvidenceItem, new_turn_input
from intelligence_system.agents.supervisor import apply_guards, rule_based_decision, supervisor_logic
from intelligence_system.agents.workers.analyst import analyst_logic, execute_tool
from intelligence_system.agents.workers.researcher import researcher_logic
from intelligence_system.agents.workers.writer import finalize_answer, writer_logic
from intelligence_system.container import AppContainer
from intelligence_system.llm.exceptions import LLMServerError
from intelligence_system.llm.providers.fake_client import FakeLLMClient


def _state(question: str = "¿Qué es BM25?", **extra: Any) -> AgentState:
    state = new_turn_input(message=question, request_id="req-test", filters=None)
    state.update(extra)
    return state  # type: ignore[return-value]


def _deps(container: AppContainer) -> AgentDeps:
    assert container.chat is not None and container.llm is not None and container.retriever is not None
    from intelligence_system.agents.tools import make_domain_tools, make_search_tool

    assert container.embeddings is not None
    return AgentDeps(
        settings=container.settings,
        llm=container.llm,
        search_tool=make_search_tool(container.retriever),
        domain_tools=make_domain_tools(container.embeddings),
    )


def _doc(i: int) -> EvidenceItem:
    return EvidenceItem(
        id=f"doc#{i}",
        kind="document",
        title=f"Doc {i}",
        source="d.md",
        content=f"contenido {i}",
        score=0.8,
        worker="researcher",
    )


# --------------------------------------------------------------------------- supervisor


async def test_supervisor_uses_structured_decision(ingested: AppContainer, scripted: ScriptedLLM) -> None:
    scripted.routes = [{"next": "analyst", "reason": "hay que calcular", "task": "calcular chunks"}]
    outcome = await supervisor_logic(_state(), _deps(ingested))
    assert outcome.goto == "analyst"
    assert outcome.update["current_task"] == "calcular chunks"
    assert outcome.update["plan"][0]["decided_by"] == "llm"
    assert "Workers ya ejecutados en este turno: ninguno" in scripted.prompts["supervisor"][0]


async def test_supervisor_enforces_step_budget(ingested: AppContainer, scripted: ScriptedLLM) -> None:
    outcome = await supervisor_logic(_state(step_count=ingested.settings.agent_max_steps), _deps(ingested))
    assert outcome.goto == "writer"
    assert scripted.prompts["supervisor"] == []  # ni siquiera consulta al LLM


async def test_supervisor_falls_back_to_rules_when_llm_fails(ingested: AppContainer) -> None:
    def broken(*_: Any) -> Any:
        raise LLMServerError("caído")

    deps = _deps(ingested)
    deps.llm = FakeLLMClient(broken)
    outcome = await supervisor_logic(_state(), deps)
    assert (outcome.goto, outcome.status) == ("researcher", "fallback")
    with_evidence = await supervisor_logic(_state(evidence=[_doc(1)]), deps)
    assert with_evidence.goto == "writer"


def test_anti_loop_guard_blocks_repeated_task() -> None:
    plan = [{"step": 1, "next": "researcher", "reason": "razón", "task": "Qué es BM25", "decided_by": "llm"}]
    state = _state(plan=plan)
    decision = SupervisorDecision(next="researcher", reason="otra vez", task="¿qué es bm25?")
    guarded, was_guarded = apply_guards(decision, state)
    assert was_guarded and guarded.next == "writer"
    fresh, _ = apply_guards(SupervisorDecision(next="researcher", reason="razón", task="otra cosa"), state)
    assert fresh.next == "researcher"


def test_rule_based_decision() -> None:
    assert rule_based_decision(_state()).next == "researcher"
    assert rule_based_decision(_state(evidence=[_doc(1)])).next == "writer"


# --------------------------------------------------------------------------- researcher


async def test_researcher_keeps_only_graded_evidence(ingested: AppContainer, scripted: ScriptedLLM) -> None:
    scripted.grade = [2, 99]  # 99 no existe: se descarta en código
    outcome = await researcher_logic(_state(current_task="BM25 y búsqueda léxica"), _deps(ingested))
    assert outcome.goto == "supervisor"
    assert len(outcome.update["evidence"]) == 1
    assert outcome.update["step_count"] == 1
    assert "1/5" in outcome.detail


async def test_researcher_uses_request_filters(ingested: AppContainer, scripted: ScriptedLLM) -> None:
    scripted.grade = [1, 2, 3, 4, 5]
    state = _state(current_task="namespaces y métricas", filters={"categories": ["vector_db"]})
    outcome = await researcher_logic(state, _deps(ingested))
    assert outcome.update["evidence"]
    assert all("pinecone" in e["id"] or "chromadb" in e["id"] for e in outcome.update["evidence"])


# --------------------------------------------------------------------------- analyst


async def test_analyst_react_loop_recovers_from_invalid_arguments(
    ingested: AppContainer, scripted: ScriptedLLM
) -> None:
    scripted.analyst = [
        response(
            tool_calls=[tool_call("estimar_chunks", total_tokens=12000, chunk_size=500, chunk_overlap=500)]
        ),
        response(
            tool_calls=[tool_call("estimar_chunks", total_tokens=12000, chunk_size=500, chunk_overlap=50)]
        ),
        response("Salen 27 chunks."),
    ]
    outcome = await analyst_logic(_state(current_task="chunks de 12000 tokens"), _deps(ingested))
    [item] = outcome.update["evidence"]
    assert item["kind"] == "tool" and json.loads(item["content"])["num_chunks"] == 27
    assert "1 cálculo(s) válidos" in outcome.detail


async def test_analyst_respects_tool_call_budget(ingested: AppContainer, scripted: ScriptedLLM) -> None:
    budget = ingested.settings.agent_analyst_max_tool_calls
    scripted.analyst = [
        response(tool_calls=[tool_call("estimar_memoria_vectores", num_vectores=10 + i, dimension=8)])
        for i in range(budget + 3)
    ]
    outcome = await analyst_logic(_state(), _deps(ingested))
    assert len(outcome.update["evidence"]) == budget


async def test_execute_tool_returns_errors_as_data() -> None:
    payload, ok = await execute_tool({}, tool_call("no_existe"))
    assert not ok and "desconocida" in json.loads(payload)["error"]


# --------------------------------------------------------------------------- writer


async def test_writer_abstains_without_calling_llm(ingested: AppContainer, scripted: ScriptedLLM) -> None:
    outcome = await writer_logic(_state(), _deps(ingested))
    assert outcome.update["answer"] == ABSTENTION_ANSWER
    assert outcome.update["grounded"] is False
    assert scripted.prompts["writer"] == []


async def test_writer_builds_citations(ingested: AppContainer, scripted: ScriptedLLM) -> None:
    scripted.writer = "BM25 es léxico [1] y RRF fusiona [2]."
    outcome = await writer_logic(_state(evidence=[_doc(1), _doc(2)]), _deps(ingested))
    assert outcome.update["grounded"] is True
    assert [c["marker"] for c in outcome.update["citations"]] == [1, 2]
    assert isinstance(outcome.update["messages"][0], AIMessage)


def test_finalize_answer_rules() -> None:
    evidence = [_doc(1)]
    answer, cites, grounded = finalize_answer("Dato [1] y otro inventado [7].", evidence, snippet_chars=50)
    assert grounded and "[7]" not in answer and [c["marker"] for c in cites] == [1]
    assert finalize_answer("Respuesta sin citas.", evidence, snippet_chars=50)[2] is False
    assert finalize_answer("SIN_EVIDENCIA", evidence, snippet_chars=50)[0] == ABSTENTION_ANSWER


# --------------------------------------------------------------------------- runtime


def test_ownership_contract() -> None:
    check_ownership("researcher", {"evidence": [], "step_count": 1, "steps": []})
    with pytest.raises(OwnershipViolationError, match="answer"):
        check_ownership("researcher", {"answer": "no me corresponde"})
    with pytest.raises(OwnershipViolationError, match="evidence"):
        check_ownership("writer", {"evidence": []})


async def test_node_timeout_applies_fallback(ingested: AppContainer) -> None:
    settings = make_settings(agent_node_timeout_s=0.05)
    deps = _deps(ingested)
    deps.settings = settings

    async def slow(state: AgentState, _: AgentDeps) -> NodeOutcome:
        await asyncio.sleep(1)
        return NodeOutcome(update={}, detail="nunca")

    def fallback(state: AgentState, _: AgentDeps, reason: str, status: Any) -> NodeOutcome:
        return NodeOutcome(goto="supervisor", update={"error": reason}, detail="fallback", status=status)

    node = make_node("researcher", slow, deps, fallback=fallback)
    command = await node(_state(), {"configurable": {"thread_id": "t"}})
    assert command.goto == "supervisor"  # type: ignore[union-attr]
    assert command.update["steps"][0]["status"] == "timeout"  # type: ignore[union-attr]
    assert "timeout" in command.update["error"]  # type: ignore[union-attr]


async def test_node_without_fallback_propagates_errors(ingested: AppContainer) -> None:
    async def boom(state: AgentState, _: AgentDeps) -> NodeOutcome:
        raise RuntimeError("falla")

    node = make_node("writer", boom, _deps(ingested), fallback=None)
    with pytest.raises(RuntimeError, match="falla"):
        await node(_state(), {"configurable": {"thread_id": "t"}})


def test_new_turn_input_resets_turn_channels() -> None:
    state = new_turn_input(message="hola", request_id="r1", filters={"categories": ["x"]})
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["evidence"] == [] and state["step_count"] == 0 and state["answer"] is None


def test_container_without_llm_has_no_chat() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from intelligence_system.container import assemble
    from intelligence_system.rag.embeddings import HashEmbeddings
    from intelligence_system.rag.retriever import NoopReranker
    from intelligence_system.rag.sparse import InMemoryBM25StatsStore
    from intelligence_system.rag.vector_store import InMemoryVectorStore

    settings = make_settings()
    container = assemble(
        AppContainer(settings=settings),
        llm=None,
        embeddings=HashEmbeddings(dimension=settings.embeddings_dimension),
        store=InMemoryVectorStore(dimension=settings.embeddings_dimension),
        stats=InMemoryBM25StatsStore(),
        reranker=NoopReranker(),
        checkpointer=InMemorySaver(),
    )
    assert container.chat is None and container.ingest is not None  # la ingesta no necesita LLM
    assert build_container(settings, responder=ScriptedLLM()).chat is not None
