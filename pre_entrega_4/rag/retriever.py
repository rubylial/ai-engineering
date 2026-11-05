"""
RAGSystem con fusión manual RRF (Reciprocal Rank Fusion).

Combina Pinecone vectorial + BM25 léxico SIN depender de EnsembleRetriever,
que cambió de ubicación entre LangChain 0.3 y 1.x. La fusión RRF es
exactamente lo que LangChain usa por dentro, solo que acá lo hacemos
explícito en vez de delegarlo.
"""

import pickle
from collections import defaultdict
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from .config import get_settings
from .pinecone_setup import get_pinecone_client


# ============================================================
# Reciprocal Rank Fusion
# ============================================================
# Fórmula: score(d) = Σᵢ [ wᵢ / (k_rrf + rankᵢ(d)) ]
# donde rankᵢ(d) es la posición (1-indexed) del doc d en el ranking del
# retriever i, y k_rrf es un hiperparámetro (típicamente 60).
#
# Ventaja de RRF sobre normalizar y sumar scores: es robusto a rankings
# con escalas distintas (semántico [0,1] vs BM25 [0, +∞]).

RRF_K = 60


def _reciprocal_rank_fusion(
    rankings: list[tuple[list[Document], float]],
    top_k: int,
) -> list[Document]:
    """
    Fusiona varios rankings en uno solo.

    Args:
        rankings: lista de (docs_del_retriever, peso).
        top_k: cuántos documentos devolver.
    """
    scores: dict[str, float] = defaultdict(float)
    docs_by_key: dict[str, Document] = {}

    for docs, weight in rankings:
        for rank, doc in enumerate(docs, start=1):
            # Usamos page_content como clave: es lo único estable entre
            # los dos retrievers (Pinecone y BM25 pueden asignar IDs
            # internos distintos al mismo chunk).
            key = doc.page_content
            scores[key] += weight / (RRF_K + rank)
            if key not in docs_by_key:
                docs_by_key[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [docs_by_key[k] for k in sorted_keys[:top_k]]


# ============================================================
# RAGSystem
# ============================================================

class RAGSystem:
    """
    Retrieval híbrido: Pinecone semántico + BM25 léxico fusionados con RRF.
    """

    def __init__(self, top_k: int | None = None):
        self.settings = get_settings()
        self.top_k = top_k or self.settings.top_k

        # --- Embedder (mismo que en ingesta) ---
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # --- Rama 1: retriever vectorial (Pinecone) ---
        pc = get_pinecone_client()
        index = pc.Index(self.settings.pinecone_index_name)
        self._vectorstore = PineconeVectorStore(
            index=index,
            embedding=self._embeddings,
            namespace=self.settings.pinecone_namespace,
            text_key="text",
        )
        self._semantic_retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        # --- Rama 2: retriever léxico (BM25 in-memory) ---
        chunks = self._load_chunks_from_cache()
        self._bm25_retriever = BM25Retriever.from_documents(chunks)
        self._bm25_retriever.k = self.top_k

    def _load_chunks_from_cache(self) -> List[Document]:
        path = self.settings.chunks_cache_path
        if not path.exists():
            raise RuntimeError(
                f"No se encontró {path}. Corré `python ingest.py` primero."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------
    # API pública de retrieval
    # ------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        """Retrieval híbrido: fusión RRF del ranking de ambos retrievers."""
        semantic_docs = self._semantic_retriever.invoke(query)
        bm25_docs = self._bm25_retriever.invoke(query)
        return _reciprocal_rank_fusion(
            [
                (semantic_docs, self.settings.semantic_weight),
                (bm25_docs, self.settings.bm25_weight),
            ],
            top_k=self.top_k,
        )

    async def aretrieve(self, query: str) -> List[Document]:
        """Versión async para integrar en cadenas LCEL."""
        semantic_docs = await self._semantic_retriever.ainvoke(query)
        bm25_docs = await self._bm25_retriever.ainvoke(query)
        return _reciprocal_rank_fusion(
            [
                (semantic_docs, self.settings.semantic_weight),
                (bm25_docs, self.settings.bm25_weight),
            ],
            top_k=self.top_k,
        )

    # Acceso directo a cada rama — útil para ablation studies y debug.
    def retrieve_semantic_only(self, query: str) -> List[Document]:
        return self._semantic_retriever.invoke(query)

    def retrieve_bm25_only(self, query: str) -> List[Document]:
        return self._bm25_retriever.invoke(query)