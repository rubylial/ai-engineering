"""Embeddings densos async por lotes.

Proveedor por defecto: Pinecone Inference (`multilingual-e5-large`, 1024 dims).
Es async nativo (`PineconeAsyncio.inference.embed`), multilingüe y usa la misma
API key que el vector store: no hace falta OpenAI ni cargar torch en la imagen
(la pre-entrega 4 usaba sentence-transformers local; ver ADR-0003).

Regla de oro (pre-entrega 4): el mismo modelo para indexar y para consultar.
e5 distingue `input_type="passage"` (documentos) de `"query"` (preguntas).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Sequence
from typing import Any, Protocol

from openinference.semconv.trace import OpenInferenceSpanKindValues

from intelligence_system.llm.base import RetryPolicy
from intelligence_system.observability.logging import get_logger
from intelligence_system.observability.tracing import child_span
from intelligence_system.rag.retry import pinecone_retrying
from intelligence_system.rag.sparse import tokenize

log = get_logger(__name__)


class EmbeddingsError(RuntimeError):
    """El proveedor devolvió un número o dimensión de vectores inesperado."""


class EmbeddingsProvider(Protocol):
    model: str
    dimension: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _batches(items: Sequence[str], size: int) -> list[list[str]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


class PineconeInferenceEmbeddings:
    """Embeddings vía Pinecone Inference, en lotes concurrentes acotados por semáforo."""

    def __init__(
        self,
        inference: Any,
        *,
        model: str,
        dimension: int,
        batch_size: int,
        max_concurrency: int,
        retry: RetryPolicy,
    ) -> None:
        self._inference = inference
        self.model = model
        self.dimension = dimension
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._retry = retry

    async def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        async with self._semaphore:
            result: Any = None
            async for attempt in pinecone_retrying(self._retry):
                with attempt:
                    result = await self._inference.embed(
                        model=self.model,
                        inputs=texts,
                        parameters={"input_type": input_type, "truncate": "END"},
                    )
        vectors = [list(item.values) for item in result.data]
        if len(vectors) != len(texts):
            raise EmbeddingsError(f"se pidieron {len(texts)} embeddings y llegaron {len(vectors)}")
        bad = [len(v) for v in vectors if len(v) != self.dimension]
        if bad:
            raise EmbeddingsError(
                f"dimensión {bad[0]} != EMBEDDINGS_DIMENSION={self.dimension} ({self.model})"
            )
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with child_span(
            "rag.embed_documents",
            OpenInferenceSpanKindValues.EMBEDDING,
            **{"embedding.model_name": self.model, "embedding.count": len(texts)},
        ):
            batches = _batches(texts, self._batch_size)
            results = await asyncio.gather(*(self._embed_batch(b, "passage") for b in batches))
        log.info("rag.embed_documents", model=self.model, texts=len(texts), batches=len(batches))
        return [vector for batch in results for vector in batch]

    async def embed_query(self, text: str) -> list[float]:
        with child_span(
            "rag.embed_query",
            OpenInferenceSpanKindValues.EMBEDDING,
            **{"embedding.model_name": self.model},
        ):
            return (await self._embed_batch([text], "query"))[0]


class HashEmbeddings:
    """Embeddings deterministas por hashing de términos (offline/tests, sin red).

    Proyecta cada término a una dimensión con signo y normaliza L2: textos que
    comparten términos quedan cerca. Suficiente para probar el pipeline; no
    captura sinónimos como un modelo real.
    """

    def __init__(self, *, dimension: int, model: str = "hash-embeddings") -> None:
        self.dimension = dimension
        self.model = model

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
