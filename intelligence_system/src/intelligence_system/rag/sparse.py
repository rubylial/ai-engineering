"""BM25 como vector disperso (la parte léxica del retrieval híbrido).

Mismo esquema que el BM25Encoder de pinecone-text, pero sin dependencias:
- Documento: cada término pesa su componente TF de BM25 normalizado a [0, 1):
  tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl)) / (k1+1).
- Consulta: cada término pesa su IDF, normalizado para que sumen 1.
El producto punto documento·consulta reproduce el score BM25 (escalado), por
eso Pinecone puede calcularlo dentro del índice con métrica `dotproduct`.

Las estadísticas del corpus (N, longitud total, document frequency) viven en
Redis para que la API y el script de ingesta compartan el mismo vocabulario.
El registro es idempotente por chunk_id (SET en Redis): reingestar el mismo
chunk no infla los contadores.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from redis.asyncio import Redis

from intelligence_system.rag.schemas import SparseVector

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_STOPWORDS_TEXT = """
    a al algo algun alguna algunas alguno algunos ante antes aqui asi cada como con contra
    cual cuales cuando de del desde donde dos e el ella ellas ellos en entre era es esa esas
    ese eso esos esta estan estas este esto estos fue fueron ha hace hay la las le les lo los
    mas me mi mientras muy nada ni no nos o otra otras otro otros para pero poco por porque
    puede que quien se segun ser si sin sobre son su sus tambien tan te tiene tienen todo
    todos tu un una unas uno unos usa usar y ya yo
    the of and or to in is are for on with as by be this that it an at from
"""
_STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    """Minúsculas, sin acentos, sin stopwords. Conserva términos compuestos
    (`asyncio.semaphore`, `gpt-4o-mini`) y además sus partes."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalized):
        if any(sep in raw for sep in "._-"):
            tokens.append(raw)
            tokens.extend(p for p in re.split(r"[._-]", raw) if len(p) >= 2 and p not in _STOPWORDS)
        elif len(raw) >= 2 and raw not in _STOPWORDS:
            tokens.append(raw)
    return tokens


def token_index(token: str) -> int:
    """Índice disperso estable (uint32) para un término."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


@dataclass(frozen=True)
class BM25Snapshot:
    n_docs: int
    total_length: int
    doc_freq: dict[str, int] = field(default_factory=dict)

    @property
    def avgdl(self) -> float:
        return self.total_length / self.n_docs if self.n_docs else 0.0


@dataclass(frozen=True)
class ChunkTerms:
    chunk_id: str
    tokens: list[str]


class BM25StatsStore(Protocol):
    async def register(self, items: Sequence[ChunkTerms]) -> int: ...

    async def unregister(self, items: Sequence[ChunkTerms]) -> int: ...

    async def snapshot(self, tokens: Collection[str] = ()) -> BM25Snapshot: ...


class InMemoryBM25StatsStore:
    """Estadísticas en memoria (tests y modo offline)."""

    def __init__(self) -> None:
        self._chunks: set[str] = set()
        self._df: Counter[str] = Counter()
        self._n_docs = 0
        self._total_len = 0
        self._lock = asyncio.Lock()

    async def register(self, items: Sequence[ChunkTerms]) -> int:
        async with self._lock:
            added = 0
            for item in items:
                if item.chunk_id in self._chunks:
                    continue
                self._chunks.add(item.chunk_id)
                self._df.update(set(item.tokens))
                self._n_docs += 1
                self._total_len += len(item.tokens)
                added += 1
            return added

    async def unregister(self, items: Sequence[ChunkTerms]) -> int:
        async with self._lock:
            removed = 0
            for item in items:
                if item.chunk_id not in self._chunks:
                    continue
                self._chunks.discard(item.chunk_id)
                self._df.subtract(set(item.tokens))
                self._n_docs -= 1
                self._total_len -= len(item.tokens)
                removed += 1
            return removed

    async def snapshot(self, tokens: Collection[str] = ()) -> BM25Snapshot:
        async with self._lock:
            return BM25Snapshot(
                n_docs=self._n_docs,
                total_length=self._total_len,
                doc_freq={t: max(self._df.get(t, 0), 0) for t in tokens},
            )


class RedisBM25StatsStore:
    """Estadísticas BM25 en Redis, idempotentes por chunk_id y atómicas (MULTI/EXEC)."""

    def __init__(self, redis: Redis, *, prefix: str, namespace: str) -> None:
        self._redis = redis
        base = f"{prefix}:{namespace}"
        self._chunks_key = f"{base}:chunks"
        self._df_key = f"{base}:df"
        self._n_docs_key = f"{base}:n_docs"
        self._total_len_key = f"{base}:total_len"

    async def _membership(self, items: Sequence[ChunkTerms], *, add: bool) -> list[ChunkTerms]:
        async with self._redis.pipeline(transaction=False) as pipe:
            for item in items:
                if add:
                    pipe.sadd(self._chunks_key, item.chunk_id)
                else:
                    pipe.srem(self._chunks_key, item.chunk_id)
            flags = await pipe.execute()
        return [item for item, flag in zip(items, flags, strict=True) if flag]

    async def _apply(self, items: Sequence[ChunkTerms], sign: int) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            for item in items:
                for token in set(item.tokens):
                    pipe.hincrby(self._df_key, token, sign)
            pipe.incrby(self._n_docs_key, sign * len(items))
            pipe.incrby(self._total_len_key, sign * sum(len(i.tokens) for i in items))
            await pipe.execute()

    async def register(self, items: Sequence[ChunkTerms]) -> int:
        new_items = await self._membership(items, add=True)
        if new_items:
            await self._apply(new_items, +1)
        return len(new_items)

    async def unregister(self, items: Sequence[ChunkTerms]) -> int:
        gone = await self._membership(items, add=False)
        if gone:
            await self._apply(gone, -1)
        return len(gone)

    async def snapshot(self, tokens: Collection[str] = ()) -> BM25Snapshot:
        token_list = list(tokens)
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.get(self._n_docs_key)
            pipe.get(self._total_len_key)
            if token_list:
                pipe.hmget(self._df_key, token_list)
            results = await pipe.execute()
        n_docs = int(results[0] or 0)
        total = int(results[1] or 0)
        df_values = results[2] if token_list else []
        doc_freq = {t: max(int(v or 0), 0) for t, v in zip(token_list, df_values, strict=True)}
        return BM25Snapshot(n_docs=max(n_docs, 0), total_length=max(total, 0), doc_freq=doc_freq)


class BM25Encoder:
    """Codifica documentos y consultas como `SparseVector` BM25."""

    def __init__(self, store: BM25StatsStore, *, k1: float, b: float) -> None:
        self._store = store
        self._k1 = k1
        self._b = b

    @property
    def store(self) -> BM25StatsStore:
        return self._store

    def _document_vector(self, tokens: list[str], avgdl: float) -> SparseVector:
        if not tokens:
            return SparseVector()
        counts = Counter(tokens)
        length = len(tokens)
        avgdl = avgdl or float(length)
        weights: dict[int, float] = {}
        for token, tf in counts.items():
            norm = self._k1 * (1 - self._b + self._b * length / avgdl)
            weight = tf * (self._k1 + 1) / (tf + norm) / (self._k1 + 1)
            idx = token_index(token)
            weights[idx] = weights.get(idx, 0.0) + weight  # colisión de hash: se suman
        indices = sorted(weights)
        return SparseVector(indices=indices, values=[weights[i] for i in indices])

    async def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        snapshot = await self._store.snapshot()
        return [self._document_vector(tokenize(t), snapshot.avgdl) for t in texts]

    async def encode_query(self, text: str) -> SparseVector:
        tokens = list(dict.fromkeys(tokenize(text)))
        if not tokens:
            return SparseVector()
        snapshot = await self._store.snapshot(tokens)
        if snapshot.n_docs == 0:
            return SparseVector()
        weights: dict[int, float] = {}
        for token in tokens:
            df = snapshot.doc_freq.get(token, 0)
            if df <= 0:
                continue  # término que no existe en el corpus: no puede sumar score
            idf = math.log(1 + (snapshot.n_docs - df + 0.5) / (df + 0.5))
            idx = token_index(token)
            weights[idx] = weights.get(idx, 0.0) + idf
        total = sum(weights.values())
        if total <= 0:
            return SparseVector()
        indices = sorted(weights)
        return SparseVector(indices=indices, values=[weights[i] / total for i in indices])
