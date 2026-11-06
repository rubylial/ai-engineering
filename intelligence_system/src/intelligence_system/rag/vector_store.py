"""Vector store: Pinecone serverless híbrido (async) + implementación en memoria.

Índice híbrido en Pinecone (SDK 10.x, API 2026-07), verificado en el código
del SDK: un índice cuyo schema declara EXACTAMENTE los campos reservados
`_values` (dense_vector, métrica dotproduct) y `_sparse_values` (sparse_vector)
es servido de forma nativa por la API de vectores (`upsert`/`query` con
`values` + `sparse_values`). Un índice denso creado solo con `dimension` NO
acepta vectores dispersos: por eso el campo sparse se declara al crearlo.

Toda la comunicación usa `PineconeAsyncio` / `IndexAsyncio`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from intelligence_system.llm.base import RetryPolicy
from intelligence_system.observability.logging import get_logger
from intelligence_system.rag.retry import pinecone_retrying
from intelligence_system.rag.schemas import SparseVector

DENSE_FIELD = "_values"
SPARSE_FIELD = "_sparse_values"
_FETCH_BATCH = 100
_DELETE_BATCH = 1000
log = get_logger(__name__)


class VectorStoreError(RuntimeError):
    """Error del vector store."""


class VectorStoreConfigError(VectorStoreError):
    """El índice existe pero no es compatible (dimensión, métrica o falta el campo sparse)."""


class VectorRecord(BaseModel):
    id: str = Field(min_length=1, max_length=512)
    values: list[float]
    sparse: SparseVector | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorMatch(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoreHealth(BaseModel):
    ok: bool
    detail: str
    vector_count: int | None = None
    dimension: int | None = None


class VectorStore(Protocol):
    async def ensure_ready(self) -> None: ...

    async def upsert(self, records: Sequence[VectorRecord]) -> int: ...

    async def query(
        self,
        *,
        dense: list[float] | None,
        sparse: SparseVector | None,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[VectorMatch]: ...

    async def list_ids(self, prefix: str) -> list[str]: ...

    async def fetch_metadata(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]: ...

    async def delete(self, ids: Sequence[str]) -> None: ...

    async def health(self) -> StoreHealth: ...

    async def aclose(self) -> None: ...


def hybrid_schema(dimension: int) -> dict[str, Any]:
    return {
        "fields": {
            DENSE_FIELD: {"type": "dense_vector", "dimension": dimension, "metric": "dotproduct"},
            SPARSE_FIELD: {"type": "sparse_vector"},
        }
    }


class PineconeVectorStore:
    """Índice serverless híbrido en Pinecone."""

    def __init__(
        self,
        client: Any,
        *,
        index_name: str,
        namespace: str,
        dimension: int,
        cloud: str,
        region: str,
        create_timeout_s: int,
        upsert_batch_size: int,
        retry: RetryPolicy,
    ) -> None:
        self._client = client
        self._index_name = index_name
        self._namespace = namespace
        self._dimension = dimension
        self._cloud = cloud
        self._region = region
        self._create_timeout_s = create_timeout_s
        self._batch_size = upsert_batch_size
        self._retry = retry
        self._index: Any = None
        self._ready_lock = asyncio.Lock()

    @property
    def index(self) -> Any:
        if self._index is None:
            raise VectorStoreError("el índice no está inicializado (ensure_ready no se ejecutó)")
        return self._index

    def _validate_schema(self, description: Any) -> None:
        fields = getattr(getattr(description, "schema", None), "fields", None) or {}
        if not fields:
            log.warning("pinecone.schema_unknown", index=self._index_name)
            return
        dense = fields.get(DENSE_FIELD)
        if dense is None or SPARSE_FIELD not in fields:
            raise VectorStoreConfigError(
                f"El índice '{self._index_name}' existe pero no es híbrido (faltan los campos "
                f"{DENSE_FIELD}/{SPARSE_FIELD}). Borralo o usá otro PINECONE_INDEX_NAME."
            )
        dim = getattr(dense, "dimension", None)
        metric = str(getattr(dense, "metric", "")).lower()
        if dim != self._dimension:
            raise VectorStoreConfigError(
                f"Dimensión del índice {dim} != EMBEDDINGS_DIMENSION={self._dimension}."
            )
        if metric != "dotproduct":
            raise VectorStoreConfigError(
                f"La métrica del índice es {metric}; el híbrido requiere dotproduct."
            )

    async def ensure_ready(self) -> None:
        """Crea el índice híbrido si no existe (idempotente) y abre el cliente de datos."""
        async with self._ready_lock:
            if self._index is not None:
                return
            if not await self._client.indexes.exists(self._index_name):
                log.info(
                    "pinecone.create_index",
                    index=self._index_name,
                    dimension=self._dimension,
                    cloud=self._cloud,
                    region=self._region,
                )
                await self._client.indexes.create(
                    name=self._index_name,
                    schema=hybrid_schema(self._dimension),
                    deployment={
                        "deployment_type": "managed",
                        "cloud": self._cloud,
                        "region": self._region,
                    },
                    timeout=self._create_timeout_s,
                )
            description = await self._client.indexes.describe(self._index_name)
            self._validate_schema(description)
            if not description.status.ready:
                raise VectorStoreError(f"el índice '{self._index_name}' todavía no está listo")
            self._index = self._client.IndexAsyncio(host=description.host)
            log.info("pinecone.ready", index=self._index_name, namespace=self._namespace)

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        payload: list[dict[str, Any]] = []
        for record in records:
            item: dict[str, Any] = {"id": record.id, "values": record.values, "metadata": record.metadata}
            if record.sparse is not None and not record.sparse.is_empty:
                item["sparse_values"] = record.sparse.as_dict()
            payload.append(item)
        response: Any = None
        async for attempt in pinecone_retrying(self._retry):
            with attempt:
                response = await self.index.upsert(
                    vectors=payload,
                    namespace=self._namespace,
                    batch_size=self._batch_size,
                    show_progress=False,
                )
        failed = int(getattr(response, "failed_item_count", 0) or 0)
        if failed:
            raise VectorStoreError(f"Pinecone rechazó {failed} vectores en el upsert")
        return len(payload)

    async def query(
        self,
        *,
        dense: list[float] | None,
        sparse: SparseVector | None,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[VectorMatch]:
        kwargs: dict[str, Any] = {
            "top_k": top_k,
            "namespace": self._namespace,
            "include_metadata": True,
        }
        if dense is not None:
            kwargs["vector"] = dense
        if sparse is not None and not sparse.is_empty:
            kwargs["sparse_vector"] = sparse.as_dict()
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        response: Any = None
        async for attempt in pinecone_retrying(self._retry):
            with attempt:
                response = await self.index.query(**kwargs)
        return [
            VectorMatch(id=m.id, score=float(m.score), metadata=dict(m.metadata or {}))
            for m in response.matches
        ]

    async def list_ids(self, prefix: str) -> list[str]:
        ids: list[str] = []
        async for page in self.index.list(prefix=prefix, namespace=self._namespace):
            ids.extend(item.id for item in page.vectors)
        return ids

    async def fetch_metadata(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        for start in range(0, len(ids), _FETCH_BATCH):
            batch = list(ids[start : start + _FETCH_BATCH])
            response = await self.index.fetch(ids=batch, namespace=self._namespace)
            for vid, vector in response.vectors.items():
                found[vid] = dict(vector.metadata or {})
        return found

    async def delete(self, ids: Sequence[str]) -> None:
        for start in range(0, len(ids), _DELETE_BATCH):
            await self.index.delete(ids=list(ids[start : start + _DELETE_BATCH]), namespace=self._namespace)

    async def health(self) -> StoreHealth:
        if self._index is None:
            return StoreHealth(ok=False, detail="índice no inicializado")
        stats = await self.index.describe_index_stats()
        summary = (stats.namespaces or {}).get(self._namespace)
        count = int(summary.vector_count) if summary is not None else 0
        return StoreHealth(
            ok=True,
            detail=f"{self._index_name}/{self._namespace}",
            vector_count=count,
            dimension=stats.dimension,
        )

    async def aclose(self) -> None:
        if self._index is not None:
            await self._index.close()
            self._index = None


def _matches_filter(metadata: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    """Subconjunto de la sintaxis de filtros de Pinecone: $eq, $in, $and."""
    if not flt:
        return True
    for key, condition in flt.items():
        if key == "$and":
            if not all(_matches_filter(metadata, sub) for sub in condition):
                return False
            continue
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition and value not in condition["$in"]:
                return False
            if "$eq" in condition and value != condition["$eq"]:
                return False
        elif value != condition:
            return False
    return True


class InMemoryVectorStore:
    """Mismo contrato y misma semántica de score (dotproduct denso + disperso), sin red."""

    def __init__(self, *, dimension: int) -> None:
        self._dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    async def ensure_ready(self) -> None:
        return None

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        for record in records:
            if len(record.values) != self._dimension:
                raise VectorStoreConfigError(f"dimensión {len(record.values)} != {self._dimension}")
            self._records[record.id] = record
        return len(records)

    async def query(
        self,
        *,
        dense: list[float] | None,
        sparse: SparseVector | None,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[VectorMatch]:
        query_sparse = dict(zip(sparse.indices, sparse.values, strict=True)) if sparse else {}
        scored: list[VectorMatch] = []
        for record in self._records.values():
            if not _matches_filter(record.metadata, metadata_filter):
                continue
            score = 0.0
            if dense is not None:
                score += sum(a * b for a, b in zip(dense, record.values, strict=True))
            if query_sparse and record.sparse is not None:
                score += sum(
                    v * query_sparse.get(i, 0.0)
                    for i, v in zip(record.sparse.indices, record.sparse.values, strict=True)
                )
            scored.append(VectorMatch(id=record.id, score=score, metadata=dict(record.metadata)))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    async def list_ids(self, prefix: str) -> list[str]:
        return sorted(rid for rid in self._records if rid.startswith(prefix))

    async def fetch_metadata(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        return {rid: dict(self._records[rid].metadata) for rid in ids if rid in self._records}

    async def delete(self, ids: Sequence[str]) -> None:
        for rid in ids:
            self._records.pop(rid, None)

    async def health(self) -> StoreHealth:
        return StoreHealth(
            ok=True, detail="memoria", vector_count=len(self._records), dimension=self._dimension
        )

    async def aclose(self) -> None:
        return None
