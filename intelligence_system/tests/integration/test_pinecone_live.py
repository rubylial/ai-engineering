"""Smoke test contra Pinecone real (crea/usa el índice y un namespace temporal).

COSTO: 1 índice serverless (si no existe) + ~3 embeddings de Pinecone Inference.
Ejecutar solo a propósito:
  RUN_INTEGRATION=1 PINECONE_API_KEY=... uv run pytest tests/integration/test_pinecone_live.py
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from intelligence_system.llm.base import RetryPolicy
from intelligence_system.rag.embeddings import PineconeInferenceEmbeddings
from intelligence_system.rag.retriever import HybridRetriever, NoopReranker
from intelligence_system.rag.schemas import SparseVector
from intelligence_system.rag.sparse import BM25Encoder, ChunkTerms, InMemoryBM25StatsStore, tokenize
from intelligence_system.rag.vector_store import PineconeVectorStore, VectorRecord

# Se leen al importar: el fixture hermético de conftest limpia el entorno durante cada test.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "intelligence-system")
DIMENSION = int(os.getenv("EMBEDDINGS_DIMENSION", "1024"))
MODEL = os.getenv("EMBEDDINGS_MODEL", "multilingual-e5-large")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1" or not PINECONE_API_KEY,
        reason="requiere RUN_INTEGRATION=1 y PINECONE_API_KEY",
    ),
]


async def test_hybrid_upsert_and_query_roundtrip() -> None:
    from pinecone import PineconeAsyncio

    retry = RetryPolicy(max_attempts=3, min_wait_s=1, max_wait_s=8)
    namespace = f"it-{uuid.uuid4().hex[:8]}"
    async with PineconeAsyncio(api_key=PINECONE_API_KEY) as pc:
        store = PineconeVectorStore(
            pc,
            index_name=INDEX_NAME,
            namespace=namespace,
            dimension=DIMENSION,
            cloud=CLOUD,
            region=REGION,
            create_timeout_s=300,
            upsert_batch_size=50,
            retry=retry,
        )
        embeddings = PineconeInferenceEmbeddings(
            pc.inference,
            model=MODEL,
            dimension=DIMENSION,
            batch_size=16,
            max_concurrency=2,
            retry=retry,
        )
        try:
            await store.ensure_ready()
            texts = ["BM25 es un algoritmo de ranking léxico.", "Los embeddings capturan significado."]
            stats = InMemoryBM25StatsStore()
            await stats.register([ChunkTerms(f"t#{i}", tokenize(t)) for i, t in enumerate(texts)])
            encoder = BM25Encoder(stats, k1=1.2, b=0.75)
            dense = await embeddings.embed_documents(texts)
            sparse: list[SparseVector] = await encoder.encode_documents(texts)
            await store.upsert(
                [
                    VectorRecord(id=f"t#{i}", values=d, sparse=s, metadata={"doc_id": "t", "text": t})
                    for i, (t, d, s) in enumerate(zip(texts, dense, sparse, strict=True))
                ]
            )
            retriever = HybridRetriever(
                embeddings=embeddings,
                sparse_encoder=encoder,
                store=store,
                reranker=NoopReranker(),
                alpha=0.5,
                fetch_k=5,
                default_top_k=2,
            )
            results = []
            for _ in range(10):  # los upserts de Pinecone son eventualmente consistentes
                results = await retriever.retrieve("¿Qué es BM25?")
                if results:
                    break
                await asyncio.sleep(2)
            assert results and results[0].chunk_id == "t#0"
        finally:
            await store.delete(["t#0", "t#1"])
            await store.aclose()
