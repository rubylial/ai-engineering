"""Tests de la API: endpoints async con el router montado sobre fakes (sin lifespan real)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.jobs import JobStore
from app.routes import router
from app.worker import Worker


@pytest.fixture
def api_app(worker: Worker, store: JobStore, redis_client: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.store = store
    app.state.worker = worker
    app.state.redis = redis_client
    return app


@pytest.fixture
async def client(api_app: FastAPI) -> Any:
    transport = ASGITransport(app=api_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _poll(client: AsyncClient, job_id: str, statuses: set[str], timeout: float = 5.0) -> dict:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        r = await client.get(f"/tasks/{job_id}")
        body = r.json()
        if body.get("status") in statuses:
            return body
        await asyncio.sleep(0.03)
    raise AssertionError(f"job {job_id} no llegó a {statuses}")


async def test_health_and_ready(client: AsyncClient) -> None:
    assert (await client.get("/health")).json() == {"status": "ok"}
    assert (await client.get("/ready")).json() == {"status": "ready"}


async def test_create_task_returns_job_id_immediately(client: AsyncClient) -> None:
    resp = await client.post("/tasks", json={"query": "resumí redis"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "PENDING"
    assert len(body["job_id"]) == 32
    # Y termina en DONE sin bloquear el POST.
    final = await _poll(client, body["job_id"], {"DONE"})
    assert final["result"]["task_completed"] is True


async def test_get_unknown_task_is_404(client: AsyncClient) -> None:
    resp = await client.get("/tasks/does-not-exist")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")


async def test_hitl_flow_over_api(client: AsyncClient) -> None:
    # Query crítica → el job queda esperando aprobación.
    resp = await client.post("/tasks", json={"query": "borrar la base de producción"})
    job_id = resp.json()["job_id"]
    waiting = await _poll(client, job_id, {"WAITING_APPROVAL"})
    assert waiting["interrupt"]["type"] == "approval_required"

    # Aprobar → reanuda → DONE.
    approve = await client.post(f"/tasks/{job_id}/approve", json={"approved": True})
    assert approve.status_code == 202
    done = await _poll(client, job_id, {"DONE"})
    assert done["result"]["approved"] is True


async def test_force_approval_flag_triggers_hitl(client: AsyncClient) -> None:
    resp = await client.post("/tasks", json={"query": "consulta inocente", "require_approval": True})
    job_id = resp.json()["job_id"]
    waiting = await _poll(client, job_id, {"WAITING_APPROVAL"})
    assert waiting["is_critical"] is True


async def test_approve_on_non_waiting_job_is_409(client: AsyncClient) -> None:
    resp = await client.post("/tasks", json={"query": "resumí redis"})
    job_id = resp.json()["job_id"]
    await _poll(client, job_id, {"DONE"})
    conflict = await client.post(f"/tasks/{job_id}/approve", json={"approved": True})
    assert conflict.status_code == 409
