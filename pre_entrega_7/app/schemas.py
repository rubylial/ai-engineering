"""Contratos Pydantic v2 de la API (bordes de entrada/salida)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.jobs import JobRecord, JobStatus


class CreateTaskRequest(BaseModel):
    """Body de POST /tasks."""

    query: str = Field(..., min_length=1, max_length=4000, description="La tarea para el agente.")
    require_approval: bool = Field(
        default=False,
        description="Fuerza el flujo HITL aunque la query no tenga keywords críticas.",
    )


class CreateTaskResponse(BaseModel):
    """Respuesta inmediata de POST /tasks: el job quedó encolado."""

    job_id: str
    status: JobStatus


class ApproveRequest(BaseModel):
    """Body de POST /tasks/{id}/approve."""

    approved: bool = Field(..., description="True aprueba y reanuda; False rechaza y cierra.")
    note: str | None = Field(default=None, max_length=500)


class TaskStatusResponse(BaseModel):
    """Estado completo de un job (GET /tasks/{id})."""

    job_id: str
    status: JobStatus
    query: str
    is_critical: bool
    created_at: str
    updated_at: str
    result: dict | None = None
    error: str | None = None
    interrupt: dict | None = None

    @classmethod
    def from_record(cls, record: JobRecord) -> TaskStatusResponse:
        return cls(**record.model_dump())


class ProblemDetail(BaseModel):
    """Error uniforme estilo problem+json."""

    type: str
    title: str
    status: int
    detail: str
