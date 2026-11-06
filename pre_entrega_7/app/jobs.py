"""Estado de los jobs persistido en Redis (independiente del checkpointer del grafo).

Cada job es un JSON en la key `job:{id}` con TTL. El ciclo de vida es:

    PENDING → RUNNING → DONE
                     ├─► WAITING_APPROVAL → RUNNING → DONE
                     └─► FAILED

El JobStore solo usa comandos Redis básicos (GET/SET/EXPIRE), así que funciona
igual contra el Redis real del compose o contra fakeredis en los tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from redis.asyncio import Redis


class JobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobRecord(BaseModel):
    """Lo que se guarda y se devuelve por la API para un job."""

    job_id: str
    status: JobStatus
    query: str
    is_critical: bool = False
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    result: dict[str, Any] | None = None
    error: str | None = None
    interrupt: dict[str, Any] | None = None  # payload de aprobación pendiente


class JobStore:
    """CRUD de jobs sobre Redis."""

    def __init__(self, redis: Redis, *, ttl_seconds: int = 86400) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @staticmethod
    def _key(job_id: str) -> str:
        return f"job:{job_id}"

    async def create(self, job_id: str, query: str, *, is_critical: bool) -> JobRecord:
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.PENDING,
            query=query,
            is_critical=is_critical,
        )
        await self._save(record)
        return record

    async def get(self, job_id: str) -> JobRecord | None:
        raw = await self._redis.get(self._key(job_id))
        if raw is None:
            return None
        return JobRecord.model_validate_json(raw)

    async def update(self, job_id: str, **changes: Any) -> JobRecord:
        record = await self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        updated = record.model_copy(update={**changes, "updated_at": _now()})
        await self._save(updated)
        return updated

    async def _save(self, record: JobRecord) -> None:
        await self._redis.set(self._key(record.job_id), record.model_dump_json(), ex=self._ttl)
