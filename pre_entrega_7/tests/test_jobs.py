"""Tests del JobStore sobre fakeredis."""

from __future__ import annotations

import pytest

from app.jobs import JobStatus, JobStore


async def test_create_and_get(store: JobStore) -> None:
    rec = await store.create("j1", "hacé algo", is_critical=False)
    assert rec.status == JobStatus.PENDING
    fetched = await store.get("j1")
    assert fetched is not None
    assert fetched.query == "hacé algo"


async def test_get_missing_returns_none(store: JobStore) -> None:
    assert await store.get("nope") is None


async def test_update_changes_status_and_timestamp(store: JobStore) -> None:
    await store.create("j2", "x", is_critical=False)
    updated = await store.update("j2", status=JobStatus.DONE, result={"answer": "ok"})
    assert updated.status == JobStatus.DONE
    assert updated.result == {"answer": "ok"}


async def test_update_missing_raises(store: JobStore) -> None:
    with pytest.raises(KeyError):
        await store.update("ghost", status=JobStatus.FAILED)


async def test_round_trip_preserves_interrupt_payload(store: JobStore) -> None:
    await store.create("j3", "borrar", is_critical=True)
    await store.update("j3", status=JobStatus.WAITING_APPROVAL, interrupt={"type": "approval_required"})
    rec = await store.get("j3")
    assert rec is not None
    assert rec.interrupt == {"type": "approval_required"}
