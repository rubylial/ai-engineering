"""Retriever híbrido: denso (e5) + disperso (BM25) en UNA consulta a Pinecone.

Combinación convexa estándar de Pinecone: se escala el vector denso por
`alpha` y el disperso por `(1 - alpha)`; como el índice usa `dotproduct`, el
score final es `alpha·sim_densa + (1 − alpha)·bm25`.
- alpha = 1.0 → solo semántico   - alpha = 0.0 → solo léxico
Reemplaza el EnsembleRetriever + RRF en memoria de la pre-entrega 4 (que
necesitaba un pickle local con todos los chunks) por un híbrido nativo del
índice, que escala sin estado local.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from openinference.semconv.trace import OpenInferenceSpanKindValues

from intelligence_system.llm.base import RetryPolicy
from intelligence_system.observability.logging import get_logger
from intelligence_system.observability.tracing import child_span, record_documents, set_io
from intelligence_system.rag.embeddings import EmbeddingsProvider
from intelligence_system.rag.retry import pinecone_retrying
from intelligence_system.rag.schemas import MetadataFilter, RetrievalResult, SparseVector
from intelligence_system.rag.sparse import BM25Encoder
from intelligence_system.rag.vector_store import VectorStore

log = get_logger(__name__)


def hybrid_scale(
    dense: list[float], sparse: SparseVector | None, alpha: float
) -> tuple[list[float], SparseVector | None]:
    """Aplica la combinación convexa alpha·denso + (1−alpha)·BM25.

    El índice de Pinecone es de tipo denso (con un campo sparse adjunto), así que
    TODA consulta debe incluir un vector denso. Por eso en alpha=0.0 se manda el
    vector denso escalado a cero (aporta 0 al dotproduct → el score queda puro
    BM25) en vez de omitirlo: omitirlo hace que Pinecone rechace la consulta con
    "Cannot query index with dense 'vector_type' with only sparse vector".
    Nunca devuelve una consulta vacía.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha debe estar en [0, 1]")
    has_sparse = sparse is not None and not sparse.is_empty
    scaled_sparse = sparse.scaled(1.0 - alpha) if has_sparse and alpha < 1.0 else None
    # alpha=0.0 sin señal léxica: el vector cero no rankea nada, así que caemos al
    # denso sin escalar para no mandar una consulta de puro ruido.
    if alpha == 0.0 and not has_sparse:
        return list(dense), None
    return [v * alpha for v in dense], scaled_sparse


class Reranker(Protocol):
    async def rerank(
        self, query: str, results: Sequence[RetrievalResult], top_n: int
    ) -> list[RetrievalResult]: ...


class NoopReranker:
    async def rerank(
        self, query: str, results: Sequence[RetrievalResult], top_n: int
    ) -> list[RetrievalResult]:
        del query
        return list(results[:top_n])


class PineconeReranker:
    """Rerank opcional con un cross-encoder hosteado (bge-reranker-v2-m3, multilingüe)."""

    def __init__(self, inference: Any, *, model: str, retry: RetryPolicy) -> None:
        self._inference = inference
        self._model = model
        self._retry = retry

    async def rerank(
        self, query: str, results: Sequence[RetrievalResult], top_n: int
    ) -> list[RetrievalResult]:
        if not results:
            return []
        with child_span(
            "rag.rerank", OpenInferenceSpanKindValues.RERANKER, **{"reranker.model_name": self._model}
        ):
            response: Any = None
            async for attempt in pinecone_retrying(self._retry):
                with attempt:
                    response = await self._inference.rerank(
                        model=self._model,
                        query=query,
                        documents=[{"id": r.chunk_id, "text": r.text} for r in results],
                        rank_fields=["text"],
                        return_documents=False,
                        top_n=min(top_n, len(results)),
                    )
        return [
            results[item.index].model_copy(update={"rerank_score": float(item.score)})
            for item in response.data
        ]


def _dedupe(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Elimina fragmentos con el mismo contenido (mismo hash) conservando el mejor score."""
    best: dict[str, RetrievalResult] = {}
    for result in results:
        key = result.content_hash or result.chunk_id
        if key not in best or result.score > best[key].score:
            best[key] = result
    return sorted(best.values(), key=lambda r: r.score, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        *,
        embeddings: EmbeddingsProvider,
        sparse_encoder: BM25Encoder,
        store: VectorStore,
        reranker: Reranker,
        alpha: float,
        fetch_k: int,
        default_top_k: int,
    ) -> None:
        self._embeddings = embeddings
        self._sparse = sparse_encoder
        self._store = store
        self._reranker = reranker
        self._alpha = alpha
        self._fetch_k = fetch_k
        self._default_top_k = default_top_k

    @property
    def alpha(self) -> float:
        return self._alpha

    def with_alpha(self, alpha: float) -> HybridRetriever:
        """Misma configuración con otro peso denso/disperso (evaluación de variantes)."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha debe estar en [0, 1]")
        return HybridRetriever(
            embeddings=self._embeddings,
            sparse_encoder=self._sparse,
            store=self._store,
            reranker=self._reranker,
            alpha=alpha,
            fetch_k=self._fetch_k,
            default_top_k=self._default_top_k,
        )

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RetrievalResult]:
        k = top_k or self._default_top_k
        pinecone_filter = metadata_filter.to_pinecone() if metadata_filter else None
        with child_span(
            "rag.hybrid_search",
            OpenInferenceSpanKindValues.RETRIEVER,
            **{"rag.alpha": self._alpha, "rag.top_k": k, "rag.fetch_k": self._fetch_k},
        ) as span:
            set_io(span, input_value=query)
            dense, sparse = await asyncio.gather(
                self._embeddings.embed_query(query), self._sparse.encode_query(query)
            )
            scaled_dense, scaled_sparse = hybrid_scale(dense, sparse, self._alpha)
            matches = await self._store.query(
                dense=scaled_dense,
                sparse=scaled_sparse,
                top_k=max(self._fetch_k, k),
                metadata_filter=pinecone_filter,
            )
            results = _dedupe([RetrievalResult.from_metadata(m.id, m.score, m.metadata) for m in matches])
            results = await self._reranker.rerank(query, results, k)
            span.set_attribute("rag.sparse_terms", len(sparse.indices))
            span.set_attribute("rag.results", len(results))
            record_documents(
                span,
                [
                    {
                        "id": r.chunk_id,
                        "content": r.text,
                        "score": r.rerank_score if r.rerank_score is not None else r.score,
                        "metadata": {"doc_id": r.doc_id, "section": r.section},
                    }
                    for r in results
                ],
            )
        log.info(
            "rag.retrieve",
            top_k=k,
            alpha=self._alpha,
            sparse_terms=len(sparse.indices),
            matches=len(matches),
            results=len(results),
            filtered=pinecone_filter is not None,
        )
        return results
