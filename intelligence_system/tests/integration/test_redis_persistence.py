"""Persistencia real con AsyncRedisSaver (requiere Redis 8+ con JSON y Search).

Ejecutar con:  RUN_INTEGRATION=1 REDIS_URL=redis://localhost:6379/0 uv run pytest -m integration
(`make test-integration` lo hace contra el Redis de docker compose).
"""

from __future__ import annotations

import os
import uuid

import pytest
from redis.asyncio import Redis

from helpers import ScriptedLLM, build_container, make_settings
from intelligence_system.agents.checkpointer import open_checkpointer
from intelligence_system.rag.sparse import ChunkTerms, RedisBM25StatsStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="RUN_INTEGRATION=1 para correr"),
]
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ROUTES = [
    {"next": "researcher", "reason": "buscar evidencia", "task": "qué es BM25"},
    {"next": "writer", "reason": "redactar respuesta", "task": "redactar"},
]


async def _run_turn(thread_id: str, message: str, request_id: str) -> list[str]:
    settings = make_settings(checkpointer_backend="redis", redis_url=REDIS_URL)
    scripted = ScriptedLLM(routes=list(ROUTES), grade=[1], writer="BM25 es léxico [1].")
    async with open_checkpointer(settings) as saver:
        container = build_container(settings, responder=scripted, checkpointer=saver)
        assert container.ingest is not None and container.chat is not None
        await container.ingest.ingest_corpus(settings.rag_corpus_dir)
        await container.chat.run_turn(message=message, thread_id=thread_id, request_id=request_id)
        history = await container.chat.thread_history(thread_id)
        assert history is not None
        return [m.content for m in history.messages if m.role == "user"]


async def test_conversation_survives_api_restart() -> None:
    thread_id = f"it-{uuid.uuid4().hex[:8]}"
    first = await _run_turn(thread_id, "¿Qué es BM25?", "r1")
    assert first == ["¿Qué es BM25?"]
    # Nuevo checkpointer + nuevo grafo = proceso reiniciado. El historial sigue en Redis.
    second = await _run_turn(thread_id, "¿Y cómo se combina con embeddings?", "r2")
    assert second == ["¿Qué es BM25?", "¿Y cómo se combina con embeddings?"]


async def test_history_is_readable_from_a_fresh_saver() -> None:
    thread_id = f"it-{uuid.uuid4().hex[:8]}"
    await _run_turn(thread_id, "Pregunta persistida", "r1")
    settings = make_settings(checkpointer_backend="redis", redis_url=REDIS_URL)
    async with open_checkpointer(settings) as saver:
        container = build_container(settings, checkpointer=saver)
        assert container.chat is not None
        history = await container.chat.thread_history(thread_id)
    assert history is not None
    assert history.messages[0].content == "Pregunta persistida"
    assert history.checkpoint_id


async def test_bm25_stats_in_real_redis_are_idempotent() -> None:
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    store = RedisBM25StatsStore(redis, prefix="bm25-it", namespace=uuid.uuid4().hex[:8])
    try:
        items = [ChunkTerms("a", ["bm25", "lexico"]), ChunkTerms("b", ["bm25"])]
        assert await store.register(items) == 2
        assert await store.register(items) == 0
        snapshot = await store.snapshot(["bm25", "lexico"])
        assert (snapshot.n_docs, snapshot.doc_freq) == (2, {"bm25": 2, "lexico": 1})
    finally:
        await redis.aclose()
