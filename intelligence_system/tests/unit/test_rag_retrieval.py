from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from helpers import CORPUS_DIR, DIM
from intelligence_system.container import AppContainer
from intelligence_system.rag.chunking import MarkdownChunker
from intelligence_system.rag.embeddings import HashEmbeddings
from intelligence_system.rag.ingest import IngestService
from intelligence_system.rag.loaders import load_corpus
from intelligence_system.rag.retriever import HybridRetriever, NoopReranker, hybrid_scale
from intelligence_system.rag.schemas import Document, MetadataFilter, SparseVector
from intelligence_system.rag.sparse import BM25Encoder, InMemoryBM25StatsStore
from intelligence_system.rag.vector_store import InMemoryVectorStore

# --------------------------------------------------------------------------- hybrid_scale


def test_hybrid_scale_convex_combination() -> None:
    dense, sparse = hybrid_scale([1.0, 2.0], SparseVector(indices=[1], values=[1.0]), 0.75)
    assert dense == [0.75, 1.5]
    assert sparse is not None and sparse.values == [0.25]


def test_hybrid_scale_edges_never_produce_empty_query() -> None:
    dense_only, none = hybrid_scale([1.0], SparseVector(indices=[1], values=[1.0]), 1.0)
    assert dense_only == [1.0] and none is None
    # alpha=0.0: el denso va escalado a cero (no None), porque el índice denso de
    # Pinecone exige un vector denso en toda query; el score queda puro BM25.
    zero_dense, sparse_only = hybrid_scale([1.0], SparseVector(indices=[1], values=[1.0]), 0.0)
    assert zero_dense == [0.0] and sparse_only is not None
    fallback, empty = hybrid_scale([1.0], SparseVector(), 0.0)
    assert fallback == [1.0] and empty is None
    with pytest.raises(ValueError, match="alpha"):
        hybrid_scale([1.0], None, 1.5)


# --------------------------------------------------------------------------- ingest + retrieve


class CountingEmbeddings(HashEmbeddings):
    def __init__(self) -> None:
        super().__init__(dimension=DIM)
        self.embedded: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return await super().embed_documents(texts)


def _service(alpha: float = 0.7) -> tuple[IngestService, HybridRetriever, CountingEmbeddings, Any]:
    embeddings = CountingEmbeddings()
    store = InMemoryVectorStore(dimension=DIM)
    encoder = BM25Encoder(InMemoryBM25StatsStore(), k1=1.2, b=0.75)
    ingest = IngestService(
        chunker=MarkdownChunker(chunk_size=1500, chunk_overlap=150, min_chunk_chars=40),
        embeddings=embeddings,
        sparse_encoder=encoder,
        store=store,
    )
    retriever = HybridRetriever(
        embeddings=embeddings,
        sparse_encoder=encoder,
        store=store,
        reranker=NoopReranker(),
        alpha=alpha,
        fetch_k=20,
        default_top_k=5,
    )
    return ingest, retriever, embeddings, store


async def test_ingest_is_idempotent_and_skips_embeddings() -> None:
    ingest, _, embeddings, _ = _service()
    first = await ingest.ingest_corpus(CORPUS_DIR)
    assert first.chunks_upserted == first.chunks_total > 0
    embedded = len(embeddings.embedded)
    second = await ingest.ingest_corpus(CORPUS_DIR)
    assert (second.chunks_upserted, second.chunks_unchanged) == (0, first.chunks_total)
    assert len(embeddings.embedded) == embedded  # cero costo en la reingesta


async def test_reingest_rebuilds_bm25_stats_after_redis_wipe_without_embedding() -> None:
    ingest, retriever, embeddings, _ = _service()
    first = await ingest.ingest_corpus(CORPUS_DIR)
    before = await retriever._sparse.store.snapshot(["bm25"])
    embedded = len(embeddings.embedded)
    retriever._sparse.store.__init__()  # simula `make clean`: Redis vacío, Pinecone intacto
    assert (await retriever._sparse.store.snapshot()).n_docs == 0
    second = await ingest.ingest_corpus(CORPUS_DIR)
    assert (second.chunks_upserted, second.chunks_unchanged) == (0, first.chunks_total)
    assert await retriever._sparse.store.snapshot(["bm25"]) == before
    assert len(embeddings.embedded) == embedded


async def test_ingest_updates_changed_document_and_deletes_stale_chunks() -> None:
    ingest, retriever, _, store = _service()
    doc = Document(
        doc_id="nota", title="Nota", text="# Nota\n\n## A\n\nPrimera versión del texto A.", source="n.md"
    )
    await ingest.ingest([doc])
    old_ids = await store.list_ids("nota#")
    changed = doc.model_copy(update={"text": "# Nota\n\n## A\n\nSegunda versión distinta del texto A."})
    report = await ingest.ingest([changed])
    assert report.chunks_upserted == 1 and report.chunks_deleted == len(old_ids)
    assert set(await store.list_ids("nota#")).isdisjoint(old_ids)
    snap = await retriever._sparse.store.snapshot()
    assert snap.n_docs == 1  # el chunk viejo se descontó de las estadísticas BM25


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("¿Qué es asyncio.Semaphore y cuándo lo usarías?", "01_asyncio_python"),
        ("¿Para qué sirven los namespaces en Pinecone?", "05_pinecone_serverless"),
        ("¿Qué ventaja tiene BM25 sobre la búsqueda puramente semántica?", "06_hybrid_retrieval"),
    ],
)
async def test_hybrid_retrieval_finds_expected_document(query: str, expected_doc: str) -> None:
    ingest, retriever, _, _ = _service()
    await ingest.ingest_corpus(CORPUS_DIR)
    results = await retriever.retrieve(query, top_k=3)
    assert expected_doc in {r.doc_id for r in results}
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


async def test_retrieval_respects_metadata_filters() -> None:
    ingest, retriever, _, _ = _service()
    await ingest.ingest_corpus(CORPUS_DIR)
    results = await retriever.retrieve(
        "métricas de evaluación", top_k=5, metadata_filter=MetadataFilter(categories=["vector_db"])
    )
    assert results and {r.category for r in results} == {"vector_db"}


async def test_retrieval_records_retriever_span(span_exporter: Any) -> None:
    ingest, retriever, _, _ = _service()
    await ingest.ingest_corpus(CORPUS_DIR)
    await retriever.retrieve("¿Qué es RRF?", top_k=2)
    span = next(s for s in span_exporter.get_finished_spans() if s.name == "rag.hybrid_search")
    assert span.attributes["openinference.span.kind"] == "RETRIEVER"
    assert span.attributes["rag.results"] == 2
    assert "retrieval.documents.0.document.id" in span.attributes


async def test_container_ingests_full_corpus(ingested: AppContainer) -> None:
    assert ingested.vector_store is not None
    health = await ingested.vector_store.health()
    docs = await load_corpus(CORPUS_DIR)
    assert health.vector_count and health.vector_count >= len(docs)
