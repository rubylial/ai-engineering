"""Servicio de chat: ejecuta turnos del grafo con deadline, lock por thread y trazas.

- `run_turn`: respuesta completa (POST /v1/chat).
- `stream_turn`: eventos SSE (POST /v1/chat/stream). El grafo corre en una
  tarea productora con su propio contexto de traza y publica en una cola; el
  generador solo consume. Si el cliente se desconecta, la tarea se cancela.
- `thread_history`: lee el historial persistido en el checkpointer.

Un lock por `thread_id` serializa turnos concurrentes del mismo thread (dos
requests simultáneos no pisan el mismo checkpoint). Es por proceso: con
varias réplicas haría falta un lock distribuido (ver limitaciones).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pinecone import PineconeError

from intelligence_system.agents.runtime import OwnershipViolationError
from intelligence_system.agents.schemas import (
    Citation,
    StepInfo,
    StreamEvent,
    ThreadHistory,
    ThreadMessage,
    TurnResult,
)
from intelligence_system.agents.state import new_turn_input
from intelligence_system.config import Settings
from intelligence_system.errors import AgentLoopError, AppError, GraphTimeoutError, UpstreamError
from intelligence_system.llm.exceptions import LLMError
from intelligence_system.observability.context import bind_thread_id
from intelligence_system.observability.logging import get_logger
from intelligence_system.observability.tracing import (
    format_trace_id,
    mark_error,
    request_span,
    set_io,
)

log = get_logger(__name__)


class ThreadLocks:
    """Un `asyncio.Lock` por thread, liberado cuando nadie lo usa."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._refs: Counter[str] = Counter()

    @asynccontextmanager
    async def hold(self, thread_id: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        self._refs[thread_id] += 1
        try:
            async with lock:
                yield
        finally:
            self._refs[thread_id] -= 1
            if self._refs[thread_id] <= 0:
                self._locks.pop(thread_id, None)
                del self._refs[thread_id]


def to_app_error(exc: BaseException, *, deadline_s: float, deadline_hit: bool) -> AppError:
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, TimeoutError):
        if deadline_hit:
            return GraphTimeoutError(f"El turno superó el deadline de {deadline_s:g}s.")
        return GraphTimeoutError("Un paso del agente superó AGENT_NODE_TIMEOUT_S.")
    if isinstance(exc, GraphRecursionError):
        return AgentLoopError("Se alcanzó AGENT_RECURSION_LIMIT sin terminar el turno.")
    if isinstance(exc, LLMError):
        return UpstreamError(f"Proveedor LLM: {type(exc).__name__}: {exc}"[:500])
    if isinstance(exc, PineconeError):
        return UpstreamError(f"Pinecone: {type(exc).__name__}: {exc}"[:500])
    if isinstance(exc, OwnershipViolationError):
        return AppError(str(exc))
    return AppError(f"{type(exc).__name__}: {exc}"[:500])


class ChatService:
    def __init__(self, graph: CompiledStateGraph, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings
        self._locks = ThreadLocks()

    def _config(self, thread_id: str) -> RunnableConfig:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._settings.agent_recursion_limit,
        }

    def _deadline(self, deadline_s: float | None) -> float:
        return deadline_s or self._settings.agent_request_timeout_s

    async def run_turn(
        self,
        *,
        message: str,
        thread_id: str,
        request_id: str,
        filters: dict[str, Any] | None = None,
        deadline_s: float | None = None,
    ) -> TurnResult:
        bind_thread_id(thread_id)
        deadline = self._deadline(deadline_s)
        with request_span(
            "chat.turn", request_id=request_id, thread_id=thread_id, input_value=message
        ) as span:
            timer = asyncio.timeout(deadline)
            try:
                async with timer, self._locks.hold(thread_id):
                    state = await self._graph.ainvoke(
                        new_turn_input(message=message, request_id=request_id, filters=filters),
                        self._config(thread_id),
                    )
            except Exception as exc:
                error = to_app_error(exc, deadline_s=deadline, deadline_hit=timer.expired())
                error.trace_id = format_trace_id(span)
                mark_error(span, exc, description=error.title)
                log.warning("chat.turn_failed", error=error.slug, detail=error.detail)
                raise error from exc
            result = TurnResult.from_state(
                state, thread_id=thread_id, request_id=request_id, trace_id=format_trace_id(span)
            )
            set_io(span, output_value=result.answer)
            span.set_attribute("grounded", result.grounded)
            span.set_attribute("citations.count", len(result.citations))
            span.set_attribute("steps.count", len(result.steps))
        log.info(
            "chat.turn_completed",
            grounded=result.grounded,
            citations=len(result.citations),
            steps=[s.node for s in result.steps],
            trace_id=result.trace_id,
        )
        return result

    @staticmethod
    def _translate(mode: str, chunk: Any) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if mode == "custom" and isinstance(chunk, dict) and chunk.get("type") == "token":
            events.append(StreamEvent(event="token", data={"text": chunk["text"]}))
        elif mode == "updates" and isinstance(chunk, dict):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                if update.get("steps"):
                    step = StepInfo.model_validate(update["steps"][-1])
                    events.append(StreamEvent(event="step", data=step.model_dump()))
                if node == "writer":
                    events.extend(
                        StreamEvent(event="citation", data=Citation.model_validate(c).model_dump())
                        for c in update.get("citations") or []
                    )
        return events

    async def stream_turn(
        self,
        *,
        message: str,
        thread_id: str,
        request_id: str,
        filters: dict[str, Any] | None = None,
        deadline_s: float | None = None,
    ) -> AsyncIterator[StreamEvent]:
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        deadline = self._deadline(deadline_s)
        config = self._config(thread_id)

        async def produce() -> None:
            bind_thread_id(thread_id)
            try:
                with request_span(
                    "chat.stream", request_id=request_id, thread_id=thread_id, input_value=message
                ) as span:
                    trace_id = format_trace_id(span)
                    timer = asyncio.timeout(deadline)
                    try:
                        async with timer, self._locks.hold(thread_id):
                            async for mode, chunk in self._graph.astream(
                                new_turn_input(message=message, request_id=request_id, filters=filters),
                                config,
                                stream_mode=["updates", "custom"],
                            ):
                                for event in self._translate(mode, chunk):
                                    await queue.put(event)
                            snapshot = await self._graph.aget_state(config)
                        result = TurnResult.from_state(
                            snapshot.values, thread_id=thread_id, request_id=request_id, trace_id=trace_id
                        )
                        set_io(span, output_value=result.answer)
                        span.set_attribute("grounded", result.grounded)
                        await queue.put(StreamEvent(event="done", data=result.model_dump(mode="json")))
                    except Exception as exc:
                        error = to_app_error(exc, deadline_s=deadline, deadline_hit=timer.expired())
                        mark_error(span, exc, description=error.title)
                        log.warning("chat.stream_failed", error=error.slug, detail=error.detail)
                        await queue.put(
                            StreamEvent(
                                event="error",
                                data={
                                    "type": error.type_uri,
                                    "title": error.title,
                                    "status": error.status_code,
                                    "detail": error.detail,
                                    "request_id": request_id,
                                    "trace_id": trace_id,
                                },
                            )
                        )
            finally:
                await queue.put(None)

        task = asyncio.create_task(produce(), name=f"stream-{request_id}")
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def thread_history(self, thread_id: str) -> ThreadHistory | None:
        snapshot = await self._graph.aget_state(self._config(thread_id))
        values = snapshot.values or {}
        messages = [m for m in values.get("messages") or [] if m.type in ("human", "ai")]
        if not messages:
            return None
        configurable = (snapshot.config or {}).get("configurable") or {}
        return ThreadHistory(
            thread_id=thread_id,
            messages=[
                ThreadMessage(
                    role="user" if m.type == "human" else "assistant",
                    content=str(m.content),
                    id=m.id,
                )
                for m in messages
            ],
            last_answer=values.get("answer"),
            last_grounded=values.get("grounded"),
            last_citations=[Citation.model_validate(c) for c in values.get("citations") or []],
            last_steps=[StepInfo.model_validate(s) for s in values.get("steps") or []],
            checkpoint_id=configurable.get("checkpoint_id"),
            updated_at=snapshot.created_at,
        )
