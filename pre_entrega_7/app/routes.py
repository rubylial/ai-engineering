"""Rutas de la API, en un APIRouter para poder montarlas con o sin el lifespan real.

Los handlers solo tocan Redis (JobStore) y la cola del Worker: nunca corren el grafo
en el request, así el event loop no se bloquea. El trabajo pesado queda en el Worker.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.jobs import JobStatus, JobStore
from app.schemas import (
    ApproveRequest,
    CreateTaskRequest,
    CreateTaskResponse,
    ProblemDetail,
    TaskStatusResponse,
)
from app.worker import JobCommand, Worker

log = logging.getLogger(__name__)
router = APIRouter()


def _store(request: Request) -> JobStore:
    return request.app.state.store


def _worker(request: Request) -> Worker:
    return request.app.state.worker


def _problem(status: int, title: str, detail: str) -> JSONResponse:
    body = ProblemDetail(
        type=f"/errors/{title.lower().replace(' ', '-')}",
        title=title,
        status=status,
        detail=detail,
    )
    return JSONResponse(status_code=status, content=body.model_dump(), media_type="application/problem+json")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    try:
        await request.app.state.redis.ping()
    except Exception as exc:
        return _problem(503, "not ready", f"redis no responde: {exc}")
    return JSONResponse({"status": "ready"})


@router.post("/tasks", response_model=CreateTaskResponse, status_code=202)
async def create_task(body: CreateTaskRequest, request: Request) -> CreateTaskResponse:
    """Encola la tarea y devuelve el job_id sin esperar a que el agente termine."""
    worker = _worker(request)
    store = _store(request)

    job_id = uuid.uuid4().hex
    critical = worker.classify(body.query, force=body.require_approval)
    await store.create(job_id, body.query, is_critical=critical)
    await worker.submit(JobCommand(job_id=job_id))
    log.info("api.task_created job=%s critical=%s", job_id, critical)
    return CreateTaskResponse(job_id=job_id, status=JobStatus.PENDING)


@router.get("/tasks/{job_id}")
async def get_task(job_id: str, request: Request) -> JSONResponse:
    """Estado del job. No bloquea: lee Redis y responde."""
    record = await _store(request).get(job_id)
    if record is None:
        return _problem(404, "not found", f"no existe el job {job_id}")
    return JSONResponse(TaskStatusResponse.from_record(record).model_dump())


@router.post("/tasks/{job_id}/approve")
async def approve_task(job_id: str, body: ApproveRequest, request: Request) -> JSONResponse:
    """Aprueba o rechaza un job en WAITING_APPROVAL y lo reanuda en segundo plano."""
    store = _store(request)
    record = await store.get(job_id)
    if record is None:
        return _problem(404, "not found", f"no existe el job {job_id}")
    if record.status != JobStatus.WAITING_APPROVAL:
        return _problem(
            409,
            "conflict",
            f"el job está en {record.status}, no en WAITING_APPROVAL; no se puede aprobar",
        )

    await store.update(job_id, status=JobStatus.PENDING)
    await _worker(request).submit(JobCommand(job_id=job_id, resume={"approved": body.approved}))
    log.info("api.task_%s job=%s", "approved" if body.approved else "rejected", job_id)
    return JSONResponse(
        {"job_id": job_id, "status": JobStatus.PENDING, "approved": body.approved},
        status_code=202,
    )
