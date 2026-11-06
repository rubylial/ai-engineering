"""Tests del Worker end-to-end: grafo real (nodos fake) + fakeredis + InMemorySaver."""

from __future__ import annotations

import asyncio
from typing import Any

from app.jobs import JobStatus, JobStore
from app.worker import JobCommand, Worker


async def _wait_until(store: JobStore, job_id: str, statuses: set[JobStatus], timeout: float = 5.0) -> Any:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        rec = await store.get(job_id)
        if rec is not None and rec.status in statuses:
            return rec
        await asyncio.sleep(0.02)
    raise AssertionError(f"job {job_id} no llegó a {statuses}")


async def test_worker_runs_non_critical_job_to_done(worker: Worker, store: JobStore) -> None:
    await store.create("w1", "resumí redis", is_critical=False)
    await worker.submit(JobCommand(job_id="w1"))
    rec = await _wait_until(store, "w1", {JobStatus.DONE})
    assert rec.result is not None
    assert rec.result["task_completed"] is True
    assert "analyst" in rec.result["agents"]


async def test_worker_critical_job_waits_then_approves(worker: Worker, store: JobStore) -> None:
    await store.create("w2", "borrar prod", is_critical=True)
    await worker.submit(JobCommand(job_id="w2"))
    rec = await _wait_until(store, "w2", {JobStatus.WAITING_APPROVAL})
    assert rec.interrupt is not None
    assert rec.interrupt["type"] == "approval_required"

    # Aprobar → reanuda → DONE
    await worker.submit(JobCommand(job_id="w2", resume={"approved": True}))
    done = await _wait_until(store, "w2", {JobStatus.DONE})
    assert done.result is not None
    assert done.result["approved"] is True


async def test_worker_critical_job_rejected(worker: Worker, store: JobStore) -> None:
    await store.create("w3", "ejecutar deploy", is_critical=True)
    await worker.submit(JobCommand(job_id="w3"))
    await _wait_until(store, "w3", {JobStatus.WAITING_APPROVAL})
    await worker.submit(JobCommand(job_id="w3", resume={"approved": False}))
    done = await _wait_until(store, "w3", {JobStatus.DONE})
    assert done.result is not None
    assert done.result["approved"] is False


async def test_worker_sets_failed_on_exception(failing_graph: Any, store: JobStore) -> None:
    w = Worker(failing_graph, store, critical_keywords=set(), concurrency=1)
    await w.start()
    try:
        await store.create("w4", "algo que explota", is_critical=False)
        await w.submit(JobCommand(job_id="w4"))
        rec = await _wait_until(store, "w4", {JobStatus.FAILED})
        assert rec.error is not None
        assert "RuntimeError" in rec.error
    finally:
        await w.stop()


def test_worker_classify_detects_keyword_and_force(worker: Worker) -> None:
    assert worker.classify("borrar la tabla", force=False) is True
    assert worker.classify("consulta inocente", force=False) is False
    assert worker.classify("consulta inocente", force=True) is True
