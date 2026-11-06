from __future__ import annotations

import math

import pytest
from fakeredis import FakeAsyncRedis

from intelligence_system.rag.sparse import (
    BM25Encoder,
    ChunkTerms,
    InMemoryBM25StatsStore,
    RedisBM25StatsStore,
    token_index,
    tokenize,
)


def test_tokenize_normalizes_accents_and_stopwords() -> None:
    tokens = tokenize("¿Qué es la Búsqueda Híbrida con BM25?")
    assert tokens == ["busqueda", "hibrida", "bm25"]


def test_tokenize_keeps_compound_terms_and_parts() -> None:
    tokens = tokenize("Usá asyncio.Semaphore y gpt-4o-mini")
    assert "asyncio.semaphore" in tokens and "asyncio" in tokens and "semaphore" in tokens
    assert "gpt-4o-mini" in tokens and "mini" in tokens


def test_token_index_is_stable_uint32() -> None:
    assert token_index("bm25") == token_index("bm25")
    assert 0 <= token_index("bm25") < 2**32


@pytest.fixture(params=["memory", "redis"])
def stats_store(request: pytest.FixtureRequest) -> object:
    if request.param == "memory":
        return InMemoryBM25StatsStore()
    return RedisBM25StatsStore(FakeAsyncRedis(decode_responses=True), prefix="bm25", namespace="test")


async def test_stats_registration_is_idempotent(stats_store: InMemoryBM25StatsStore) -> None:
    items = [ChunkTerms("c1", ["bm25", "lexico", "bm25"]), ChunkTerms("c2", ["semantico"])]
    assert await stats_store.register(items) == 2
    assert await stats_store.register(items) == 0  # mismo chunk_id: no infla contadores
    snap = await stats_store.snapshot(["bm25", "semantico", "nada"])
    assert snap.n_docs == 2
    assert snap.total_length == 4
    assert snap.doc_freq == {"bm25": 1, "semantico": 1, "nada": 0}
    assert await stats_store.unregister([items[0]]) == 1
    assert await stats_store.unregister([items[0]]) == 0
    snap = await stats_store.snapshot(["bm25"])
    assert (snap.n_docs, snap.doc_freq["bm25"]) == (1, 0)


async def test_query_vector_weights_rare_terms_higher() -> None:
    store = InMemoryBM25StatsStore()
    corpus = ["bm25 lexico exacto", "bm25 semantico", "bm25 vectores densos", "rrf fusion"]
    await store.register([ChunkTerms(f"c{i}", tokenize(t)) for i, t in enumerate(corpus)])
    encoder = BM25Encoder(store, k1=1.2, b=0.75)
    query = await encoder.encode_query("bm25 rrf")
    weights = dict(zip(query.indices, query.values, strict=True))
    assert math.isclose(sum(query.values), 1.0)
    assert weights[token_index("rrf")] > weights[token_index("bm25")]  # rrf es más raro


async def test_unknown_terms_produce_empty_query() -> None:
    store = InMemoryBM25StatsStore()
    await store.register([ChunkTerms("c1", ["asyncio"])])
    encoder = BM25Encoder(store, k1=1.2, b=0.75)
    assert (await encoder.encode_query("capital de australia")).is_empty
    assert (await BM25Encoder(InMemoryBM25StatsStore(), k1=1.2, b=0.75).encode_query("bm25")).is_empty


async def test_document_vector_values_are_bounded() -> None:
    store = InMemoryBM25StatsStore()
    await store.register([ChunkTerms("c1", tokenize("bm25 bm25 bm25 lexico"))])
    encoder = BM25Encoder(store, k1=1.2, b=0.75)
    [vector] = await encoder.encode_documents(["bm25 bm25 bm25 lexico"])
    assert all(0 < v < 1 for v in vector.values)
    assert vector.indices == sorted(vector.indices)
