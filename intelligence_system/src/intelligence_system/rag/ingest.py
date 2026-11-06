"""Ingesta idempotente: upsert por doc_id + hash de contenido.

Para cada documento:
1. Se chunkea (en `asyncio.to_thread`: es CPU puro).
2. Se listan en Pinecone los ids existentes con prefijo `<doc_id>#`.
3. Chunks nuevos (id no existe) → se embeben y se suben; chunks iguales
   (mismo hash → mismo id) → se saltean SIN volver a embeber (cero costo);
   chunks que ya no existen en el documento → se borran.
Las estadísticas BM25 se actualizan en el mismo paso, de forma idempotente.
Correr la ingesta dos veces seguidas sube 0 vectores la segunda vez.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from intelligence_system.observability.logging import get_logger
from intelligence_system.rag.chunking import MarkdownChunker, chunk_prefix
from intelligence_system.rag.embeddings import EmbeddingsProvider
from intelligence_system.rag.loaders import load_corpus
from intelligence_system.rag.schemas import Chunk, Document, IngestReport
from intelligence_system.rag.sparse import BM25Encoder, ChunkTerms, tokenize
from intelligence_system.rag.vector_store import VectorRecord, VectorStore

log = get_logger(__name__)


class IngestService:
    def __init__(
        self,
        *,
        chunker: MarkdownChunker,
        embeddings: EmbeddingsProvider,
        sparse_encoder: BM25Encoder,
        store: VectorStore,
    ) -> None:
        self._chunker = chunker
        self._embeddings = embeddings
        self._sparse = sparse_encoder
        self._store = store
        # Una ingesta a la vez por proceso: evita carreras en las estadísticas.
        self._lock = asyncio.Lock()

    async def ingest_corpus(self, directory: Path) -> IngestReport:
        documents = await load_corpus(directory)
        return await self.ingest(documents)

    async def ingest(self, documents: Sequence[Document]) -> IngestReport:
        started = time.perf_counter()
        async with self._lock:
            chunks = await asyncio.to_thread(self._chunker.split_many, documents)
            by_doc: dict[str, list[Chunk]] = defaultdict(list)
            for chunk in chunks:
                by_doc[chunk.doc_id].append(chunk)

            existing_lists = await asyncio.gather(
                *(self._store.list_ids(chunk_prefix(doc.doc_id)) for doc in documents)
            )
            to_add: list[Chunk] = []
            kept: list[Chunk] = []
            stale: list[str] = []
            for doc, existing_ids in zip(documents, existing_lists, strict=True):
                existing = set(existing_ids)
                current = {c.chunk_id: c for c in by_doc.get(doc.doc_id, [])}
                kept.extend(c for cid, c in current.items() if cid in existing)
                to_add.extend(c for cid, c in current.items() if cid not in existing)
                stale.extend(sorted(existing - current.keys()))

            if stale:
                await self._remove(stale)
            if kept:
                await self._restore_stats(kept)
            if to_add:
                await self._add(to_add)

        report = IngestReport(
            documents=len(documents),
            chunks_total=len(chunks),
            chunks_upserted=len(to_add),
            chunks_unchanged=len(kept),
            chunks_deleted=len(stale),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        log.info("rag.ingest", **report.model_dump())
        return report

    async def _remove(self, ids: list[str]) -> None:
        metadata = await self._store.fetch_metadata(ids)
        terms = [
            ChunkTerms(chunk_id=cid, tokens=tokenize(str(meta.get("text", ""))))
            for cid, meta in metadata.items()
        ]
        await self._sparse.store.unregister(terms)
        await self._store.delete(ids)

    async def _restore_stats(self, chunks: list[Chunk]) -> None:
        """Re-registra los chunks que ya están en el índice (sin embeber nada).

        `register` es idempotente por chunk_id, así que normalmente no cambia nada. Si Redis se
        vació (`make clean`) pero los vectores siguen en Pinecone, reconstruye el IDF de BM25.
        """
        restored = await self._sparse.store.register(
            [ChunkTerms(chunk_id=c.chunk_id, tokens=tokenize(c.text)) for c in chunks]
        )
        if restored:
            log.warning("rag.bm25_stats_restored", chunks=restored)

    async def _add(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        # 1) estadísticas primero: avgdl debe incluir los chunks nuevos
        await self._sparse.store.register(
            [ChunkTerms(chunk_id=c.chunk_id, tokens=tokenize(c.text)) for c in chunks]
        )
        dense, sparse = await asyncio.gather(
            self._embeddings.embed_documents(texts), self._sparse.encode_documents(texts)
        )
        records = [
            VectorRecord(id=c.chunk_id, values=d, sparse=s, metadata=c.metadata())
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        await self._store.upsert(records)
