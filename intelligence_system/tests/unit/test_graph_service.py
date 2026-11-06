"""Grafo completo + ChatService: turnos, persistencia por thread, deadlines, streaming y trazas."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from helpers import ScriptedLLM, build_container, make_settings, response, tool_call
from intelligence_system.agents.prompts import ABSTENTION_ANSWER
from intelligence_system.container import AppContainer
from intelligence_system.errors import AgentLoopError, GraphTimeoutError

MULTI_STEP = [
    {"next": "researcher", "reason": "buscar el chunk size recomendado", "task": "chunk size recomendado"},
    {"next": "analyst", "reason": "calcular", "task": "chunks para 12000 tokens con 500/50"},
    {"next": "writer", "reason": "hay evidencia", "task": "redactar"},
]


async def _ingested(scripted: ScriptedLLM, **overrides: Any) -> AppContainer:
    container = build_container(make_settings(**overrides), responder=scripted)
    assert container.ingest is not None
    await container.ingest.ingest_corpus(container.settings.rag_corpus_dir)
    return container


async def test_multi_worker_turn_with_citations(scripted: ScriptedLLM) -> None:
    scripted.routes = list(MULTI_STEP)
    scripted.grade = [1]
    scripted.analyst = [
        response(
            tool_calls=[tool_call("estimar_chunks", total_tokens=12000, chunk_size=500, chunk_overlap=50)]
        ),
        response("27 chunks"),
    ]
    scripted.writer = "Conviene empezar con 500 tokens [1]; salen 27 chunks [2]."
    container = await _ingested(scripted)
    assert container.chat is not None
    result = await container.chat.run_turn(message="¿Cuántos chunks?", thread_id="t-multi", request_id="r1")
    assert [s.node for s in result.steps] == [
        "supervisor",
        "researcher",
        "supervisor",
        "analyst",
        "supervisor",
        "writer",
    ]
    assert result.grounded and {c.kind for c in result.citations} == {"document", "tool"}
    assert result.request_id == "r1" and result.thread_id == "t-multi"


async def test_conversation_persists_by_thread(scripted: ScriptedLLM) -> None:
    scripted.routes = [MULTI_STEP[0], MULTI_STEP[2], MULTI_STEP[0], MULTI_STEP[2]]
    container = await _ingested(scripted)
    assert container.chat is not None
    await container.chat.run_turn(message="Primera pregunta sobre BM25", thread_id="t-mem", request_id="r1")
    await container.chat.run_turn(message="¿Y eso cómo se combina?", thread_id="t-mem", request_id="r2")
    second_turn_prompt = scripted.prompts["supervisor"][2]
    assert "Primera pregunta sobre BM25" in second_turn_prompt  # el historial llegó al supervisor
    history = await container.chat.thread_history("t-mem")
    assert history is not None
    assert [m.role for m in history.messages] == ["user", "assistant", "user", "assistant"]
    assert await container.chat.thread_history("otro-thread") is None


async def test_out_of_domain_question_abstains(scripted: ScriptedLLM) -> None:
    scripted.routes = [{"next": "writer", "reason": "fuera de dominio", "task": "capital de Australia"}]
    container = await _ingested(scripted)
    assert container.chat is not None
    result = await container.chat.run_turn(
        message="¿Capital de Australia?", thread_id="t-ood", request_id="r"
    )
    assert result.answer == ABSTENTION_ANSWER and not result.grounded and result.citations == []


async def test_request_deadline_raises_graph_timeout() -> None:
    async def slow(*_: Any) -> Any:
        await asyncio.sleep(2)

    container = await _ingested(ScriptedLLM())
    container.llm.responder = slow  # type: ignore[union-attr]
    assert container.chat is not None
    with pytest.raises(GraphTimeoutError, match="deadline"):
        await container.chat.run_turn(message="hola", thread_id="t-slow", request_id="r", deadline_s=0.1)


async def test_recursion_limit_maps_to_agent_loop_error(scripted: ScriptedLLM) -> None:
    scripted.routes = [
        {"next": "researcher", "reason": "seguir buscando", "task": f"tarea distinta número {i}"}
        for i in range(50)
    ]
    container = await _ingested(scripted, agent_max_steps=20, agent_recursion_limit=5)
    assert container.chat is not None
    with pytest.raises(AgentLoopError):
        await container.chat.run_turn(message="loop", thread_id="t-loop", request_id="r")


async def test_stream_emits_steps_tokens_citations_and_done(scripted: ScriptedLLM) -> None:
    scripted.routes = [MULTI_STEP[0], MULTI_STEP[2]]
    scripted.grade = [1]
    scripted.writer = "Respuesta larga que se transmite en varios tokens y cita [1]."
    container = await _ingested(scripted)
    assert container.chat is not None
    events = [e async for e in container.chat.stream_turn(message="¿BM25?", thread_id="t-s", request_id="r")]
    kinds = [e.event for e in events]
    assert kinds[-1] == "done"
    assert kinds.count("step") == 4 and "token" in kinds and "citation" in kinds
    assert kinds.index("token") < kinds.index("citation") < kinds.index("done")
    done = events[-1].data
    streamed = "".join(e.data["text"] for e in events if e.event == "token")
    assert streamed == done["answer"] and done["grounded"] is True


async def test_stream_holds_back_sentinel(scripted: ScriptedLLM) -> None:
    scripted.routes = [MULTI_STEP[0], MULTI_STEP[2]]
    scripted.writer = "SIN_EVIDENCIA"
    container = await _ingested(scripted)
    assert container.chat is not None
    events = [e async for e in container.chat.stream_turn(message="¿BM25?", thread_id="t-x", request_id="r")]
    assert not [e for e in events if e.event == "token"]  # el centinela nunca llega al cliente
    assert events[-1].data["answer"] == ABSTENTION_ANSWER


async def test_stream_reports_errors_as_events() -> None:
    async def slow(*_: Any) -> Any:
        await asyncio.sleep(2)

    container = await _ingested(ScriptedLLM())
    container.llm.responder = slow  # type: ignore[union-attr]
    assert container.chat is not None
    events = [
        e
        async for e in container.chat.stream_turn(
            message="x", thread_id="t-e", request_id="r", deadline_s=0.1
        )
    ]
    assert events[-1].event == "error"
    assert events[-1].data["status"] == 504 and events[-1].data["request_id"] == "r"


def test_graph_mermaid_has_supervisor_topology() -> None:
    container = build_container(make_settings())
    assert container.graph is not None
    mermaid = container.graph.get_graph().draw_mermaid()
    for edge in (
        "__start__ --> supervisor",
        "supervisor -.-> researcher",
        "supervisor -.-> analyst",
        "supervisor -.-> writer",
        "researcher -.-> supervisor",
        "analyst -.-> supervisor",
        "writer --> __end__",
    ):
        assert edge in mermaid


async def test_traces_carry_request_thread_and_worker(scripted: ScriptedLLM, span_exporter: Any) -> None:
    scripted.routes = [MULTI_STEP[0], MULTI_STEP[2]]
    container = await _ingested(scripted)
    assert container.chat is not None
    result = await container.chat.run_turn(message="¿BM25?", thread_id="t-trace", request_id="req-trace")
    spans = span_exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "chat.turn")
    assert root.attributes["request_id"] == "req-trace"
    assert root.attributes["thread_id"] == "t-trace"
    assert root.attributes["session.id"] == "t-trace"
    assert format(root.context.trace_id, "032x") == result.trace_id
    agents = [s for s in spans if s.name.startswith("agent.")]
    assert {s.attributes["worker"] for s in agents} == {"supervisor", "researcher", "writer"}
    assert all(s.context.trace_id == root.context.trace_id for s in agents)
    assert all(s.attributes["request_id"] == "req-trace" for s in agents)


async def test_agent_spans_nest_under_langgraph_nodes(
    scripted: ScriptedLLM, span_exporter: Any, tracer_provider: Any
) -> None:
    from openinference.instrumentation.langchain import LangChainInstrumentor

    scripted.routes = [MULTI_STEP[2]]
    container = await _ingested(scripted)
    assert container.chat is not None
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        await container.chat.run_turn(message="hola", thread_id="t-nest", request_id="r")
    finally:
        LangChainInstrumentor().uninstrument()
    spans = {s.context.span_id: s for s in span_exporter.get_finished_spans()}
    supervisor_agent = next(s for s in spans.values() if s.name == "agent.supervisor")
    parent = spans[supervisor_agent.parent.span_id]
    assert parent.name == "supervisor"  # span del nodo creado por OpenInference
    assert parent.attributes["session.id"] == "t-nest"


async def test_errors_carry_the_trace_id() -> None:
    async def slow(*_: Any) -> Any:
        await asyncio.sleep(2)

    container = await _ingested(ScriptedLLM())
    container.llm.responder = slow  # type: ignore[union-attr]
    assert container.chat is not None
    with pytest.raises(GraphTimeoutError) as info:
        await container.chat.run_turn(message="hola", thread_id="t-err", request_id="r", deadline_s=0.05)
    assert info.value.trace_id is not None and len(info.value.trace_id) == 32
