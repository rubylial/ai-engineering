"""Worker pool: consume jobs de una asyncio.Queue y corre el grafo en segundo plano.

Patrón worker (no bloquea el event loop de la API):
- La API encola `JobCommand(job_id, resume=...)` y responde al toque.
- Un pool de N corrutinas consume la cola y ejecuta el grafo.
- Cada ejecución actualiza el estado del job en Redis.
- CUALQUIER excepción del agente se captura y deja el job en FAILED (así el cliente
  que hace polling no se queda en un loop infinito).

El mismo worker maneja el arranque de un job (resume=None) y la reanudación tras
aprobación (resume={"approved": bool}): LangGraph distingue por el tipo de input.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.hitl import is_critical
from app.jobs import JobStatus, JobStore
from app.observability import job_trace_context

log = logging.getLogger(__name__)


@dataclass(slots=True)
class JobCommand:
    """Unidad de trabajo en la cola: arrancar un job o reanudarlo con una aprobación."""

    job_id: str
    resume: dict[str, Any] | None = None


class Worker:
    """Pool de corrutinas que corren el grafo contra Redis."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        store: JobStore,
        *,
        critical_keywords: set[str],
        concurrency: int = 4,
        recursion_limit: int = 25,
    ) -> None:
        self._graph = graph
        self._store = store
        self._keywords = critical_keywords
        self._concurrency = concurrency
        self._recursion_limit = recursion_limit
        self._queue: asyncio.Queue[JobCommand] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._consume(i)) for i in range(self._concurrency)]
        log.info("worker.started concurrency=%d", self._concurrency)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        log.info("worker.stopped")

    async def submit(self, command: JobCommand) -> None:
        await self._queue.put(command)

    async def _consume(self, worker_id: int) -> None:
        while True:
            command = await self._queue.get()
            try:
                await self._run(command)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("worker.loop_error job=%s", command.job_id)
            finally:
                self._queue.task_done()

    async def _run(self, command: JobCommand) -> None:
        job_id = command.job_id
        config = {"configurable": {"thread_id": job_id}, "recursion_limit": self._recursion_limit}

        record = await self._store.get(job_id)
        if record is None:
            log.warning("worker.job_missing job=%s", job_id)
            return

        await self._store.update(job_id, status=JobStatus.RUNNING, interrupt=None, error=None)

        if command.resume is None:
            graph_input: Any = {
                "messages": [HumanMessage(content=record.query)],
                "original_query": record.query,
                "is_critical": record.is_critical,
            }
        else:
            graph_input = Command(resume=command.resume)

        try:
            with job_trace_context(job_id):
                result = await self._graph.ainvoke(graph_input, config=config)
        except Exception as exc:
            log.exception("worker.job_failed job=%s", job_id)
            await self._store.update(job_id, status=JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            return

        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = _interrupt_payload(interrupts[0])
            await self._store.update(job_id, status=JobStatus.WAITING_APPROVAL, interrupt=payload)
            log.info("worker.waiting_approval job=%s", job_id)
            return

        await self._store.update(job_id, status=JobStatus.DONE, result=_summarize(result), interrupt=None)
        log.info("worker.done job=%s", job_id)

    def classify(self, query: str, *, force: bool) -> bool:
        """True si el job necesita aprobación (flag explícito o keyword crítica)."""
        return force or is_critical(query, self._keywords)


def _interrupt_payload(interrupt: Any) -> dict[str, Any]:
    value = getattr(interrupt, "value", interrupt)
    if isinstance(value, dict):
        return value
    return {"reason": str(value)}


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    contributions = result.get("contributions", [])
    agents = [c["agent"] for c in contributions] if isinstance(contributions, list) else []
    return {
        "answer": result.get("task", ""),
        "task_completed": bool(result.get("task_completed", False)),
        "steps": result.get("step", 0),
        "agents": agents,
        "approved": result.get("approved"),
    }
