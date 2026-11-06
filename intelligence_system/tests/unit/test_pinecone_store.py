"""PineconeVectorStore contra un doble del SDK 10.x (mismas firmas que AsyncPinecone)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pinecone import RateLimitError

from intelligence_system.llm.base import RetryPolicy
from intelligence_system.rag.schemas import SparseVector
from intelligence_system.rag.vector_store import (
    PineconeVectorStore,
    VectorRecord,
    VectorStoreConfigError,
    hybrid_schema,
)

FAST = RetryPolicy(max_attempts=3, min_wait_s=0.0, max_wait_s=0.0)


class FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.fail_query_times = 0
        self.closed = False

    async def upsert(self, **kwargs: Any) -> Any:
        self.upserts.append(kwargs)
        return SimpleNamespace(failed_item_count=0)

    async def query(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        if self.fail_query_times:
            self.fail_query_times -= 1
            raise RateLimitError.__new__(RateLimitError)
        match = SimpleNamespace(id="d#1", score=0.9, metadata={"doc_id": "d", "text": "t"})
        return SimpleNamespace(matches=[match])

    def list(self, *, prefix: str, namespace: str) -> AsyncIterator[Any]:
        async def pages() -> AsyncIterator[Any]:
            yield SimpleNamespace(
                vectors=[SimpleNamespace(id=f"{prefix}a"), SimpleNamespace(id=f"{prefix}b")]
            )
            yield SimpleNamespace(vectors=[SimpleNamespace(id=f"{prefix}c")])

        return pages()

    async def describe_index_stats(self) -> Any:
        return SimpleNamespace(dimension=8, namespaces={"ns": SimpleNamespace(vector_count=3)})

    async def close(self) -> None:
        self.closed = True


class FakeIndexes:
    def __init__(self, existing: dict[str, Any] | None = None) -> None:
        self.existing = existing or {}
        self.created: list[dict[str, Any]] = []

    async def exists(self, name: str) -> bool:
        return name in self.existing

    async def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        self.existing[kwargs["name"]] = _description(kwargs["schema"]["fields"])

    async def describe(self, name: str) -> Any:
        return self.existing[name]


def _description(fields: dict[str, Any]) -> Any:
    typed = {
        name: SimpleNamespace(**{k: v for k, v in spec.items() if k != "type"})
        for name, spec in fields.items()
    }
    return SimpleNamespace(
        schema=SimpleNamespace(fields=typed), status=SimpleNamespace(ready=True), host="idx.svc.pinecone.io"
    )


class FakePinecone:
    def __init__(self, existing: dict[str, Any] | None = None) -> None:
        self.indexes = FakeIndexes(existing)
        self.index = FakeIndex()

    def IndexAsyncio(self, host: str) -> FakeIndex:
        assert host == "idx.svc.pinecone.io"
        return self.index


def _store(client: FakePinecone) -> PineconeVectorStore:
    return PineconeVectorStore(
        client,
        index_name="intel",
        namespace="ns",
        dimension=8,
        cloud="aws",
        region="us-east-1",
        create_timeout_s=60,
        upsert_batch_size=50,
        retry=FAST,
    )


async def test_creates_hybrid_index_with_reserved_fields() -> None:
    client = FakePinecone()
    store = _store(client)
    await store.ensure_ready()
    await store.ensure_ready()  # idempotente
    [created] = client.indexes.created
    assert created["schema"] == hybrid_schema(8)
    assert set(created["schema"]["fields"]) == {"_values", "_sparse_values"}
    assert created["schema"]["fields"]["_values"]["metric"] == "dotproduct"
    assert created["deployment"] == {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"_values": {"type": "dense_vector", "dimension": 8, "metric": "dotproduct"}}, "no es híbrido"),
        ({**hybrid_schema(16)["fields"]}, "Dimensión"),
        (
            {"_values": {"type": "dense_vector", "dimension": 8, "metric": "cosine"}, "_sparse_values": {}},
            "dotproduct",
        ),
    ],
)
async def test_rejects_incompatible_existing_index(fields: dict[str, Any], message: str) -> None:
    store = _store(FakePinecone(existing={"intel": _description(fields)}))
    with pytest.raises(VectorStoreConfigError, match=message):
        await store.ensure_ready()


async def test_upsert_sends_dense_sparse_and_metadata() -> None:
    client = FakePinecone()
    store = _store(client)
    await store.ensure_ready()
    records = [
        VectorRecord(
            id="a", values=[0.1] * 8, sparse=SparseVector(indices=[3], values=[0.5]), metadata={"x": 1}
        ),
        VectorRecord(id="b", values=[0.2] * 8, sparse=SparseVector(), metadata={}),
    ]
    assert await store.upsert(records) == 2
    call = client.index.upserts[0]
    assert call["namespace"] == "ns" and call["show_progress"] is False
    assert call["vectors"][0]["sparse_values"] == {"indices": [3], "values": [0.5]}
    assert "sparse_values" not in call["vectors"][1]  # vector disperso vacío: no se envía


async def test_query_retries_transient_errors_and_maps_matches() -> None:
    client = FakePinecone()
    store = _store(client)
    await store.ensure_ready()
    client.index.fail_query_times = 2
    matches = await store.query(
        dense=[0.1] * 8,
        sparse=SparseVector(indices=[1], values=[1.0]),
        top_k=5,
        metadata_filter={"category": {"$in": ["x"]}},
    )
    assert matches[0].id == "d#1" and matches[0].metadata["doc_id"] == "d"
    last = client.index.queries[-1]
    assert last["sparse_vector"] == {"indices": [1], "values": [1.0]}
    assert last["include_metadata"] is True and last["filter"] == {"category": {"$in": ["x"]}}
    assert len(client.index.queries) == 3


async def test_list_ids_paginates_and_health_reports_namespace() -> None:
    client = FakePinecone()
    store = _store(client)
    await store.ensure_ready()
    assert await store.list_ids("doc#") == ["doc#a", "doc#b", "doc#c"]
    health = await store.health()
    assert (health.ok, health.vector_count, health.dimension) == (True, 3, 8)
    await store.aclose()
    assert client.index.closed
