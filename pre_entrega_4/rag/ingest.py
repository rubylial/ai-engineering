"""
Pipeline de ingesta: lee docs de data/, chunkea, embebe, sube a Pinecone.

Patrón de persistencia: además de subir a Pinecone, guarda los chunks en un
pickle local (chunks_cache.pkl). Esto es NECESARIO para el BM25Retriever,
que necesita tener los chunks en RAM al construirse. Sin el pickle,
tendríamos que re-chunkear en cada arranque (lento y propenso a diferencias).
"""

import pickle
from pathlib import Path
from typing import List

import tiktoken
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings
from .pinecone_setup import ensure_index


# ============================================================
# Helpers
# ============================================================

# Mapa source → categoría. Sirve para enriquecer la metadata y permitir
# filtros avanzados al estilo "buscá solo en infra/python/eval".
CATEGORY_MAP = {
    "01_asyncio_python.md":       "python",
    "02_embeddings_similitud.md": "embeddings",
    "03_langchain_lcel.md":       "langchain",
    "04_chromadb_rag.md":         "vector_db",
    "05_pinecone_serverless.md":  "vector_db",
    "06_hybrid_retrieval.md":     "retrieval",
    "07_rag_metricas.md":         "evaluation",
}


def _load_documents(data_dir: Path) -> List[Document]:
    """Lee .md/.txt de data/ y los envuelve en Documents con metadata rica."""
    docs: list[Document] = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=text,
            metadata={
                "source": path.name,
                "category": CATEGORY_MAP.get(path.name, "general"),
            },
        ))
    return docs


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Splitter que mide en TOKENS reales (no caracteres)."""
    s = get_settings()
    enc = tiktoken.get_encoding("cl100k_base")
    return RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        length_function=lambda t: len(enc.encode(t)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Embedder local. Mismo en ingesta y en query (crítico)."""
    s = get_settings()
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _enrich_chunks(chunks: list[Document]) -> list[Document]:
    """
    Agrega chunk_index y text al metadata.

    ¿Por qué guardar text en metadata? Para evitar consultas adicionales a
    otra base de datos para recuperar el contenido original. Pinecone
    devuelve metadata junto con el vector en cada query — así todo el
    contexto está en una sola llamada.
    """
    enriched = []
    # Contador por source para numerar chunks dentro de cada documento.
    per_source: dict[str, int] = {}
    for c in chunks:
        src = c.metadata["source"]
        idx = per_source.get(src, 0)
        per_source[src] = idx + 1
        new_meta = {
            **c.metadata,
            "chunk_index": idx,
            "text": c.page_content,  # redundante para PVS pero útil para debug
        }
        enriched.append(Document(page_content=c.page_content, metadata=new_meta))
    return enriched


# ============================================================
# API pública
# ============================================================

def ingest_data_directory(force_reindex: bool = False) -> PineconeVectorStore:
    """
    Pipeline completo. Idempotente por defecto.

    Args:
        force_reindex: si True, borra el namespace entero y re-indexa.
    """
    settings = get_settings()

    # 1. Asegurar que el índice exista y esté ready
    pc = ensure_index()
    index = pc.Index(settings.pinecone_index_name)

    # 2. Estado actual: ¿ya hay vectores?
    stats = index.describe_index_stats()
    existing = stats.get("namespaces", {}).get(settings.pinecone_namespace, {}).get("vector_count", 0)

    if existing > 0 and not force_reindex and settings.chunks_cache_path.exists():
        print(
            f"✓ Namespace '{settings.pinecone_namespace}' ya tiene {existing} vectores "
            f"y existe {settings.chunks_cache_path.name}. Salteando re-indexación."
        )
        return PineconeVectorStore(
            index=index,
            embedding=_get_embeddings(),
            namespace=settings.pinecone_namespace,
            text_key="text",
        )

    if force_reindex and existing > 0:
        print(f"⚠ force_reindex=True → borrando {existing} vectores de '{settings.pinecone_namespace}'...")
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)

    # 3. Cargar + chunkear
    docs = _load_documents(settings.data_dir)
    if not docs:
        raise RuntimeError(f"No se encontraron docs en {settings.data_dir}/")
    print(f"→ {len(docs)} documentos cargados.")

    splitter = _build_splitter()
    chunks = splitter.split_documents(docs)
    chunks = _enrich_chunks(chunks)
    print(f"→ {len(chunks)} chunks generados (size={settings.chunk_size}t, overlap={settings.chunk_overlap}t).")

    # 4. Persistir chunks para el BM25Retriever (en memoria, no cabe en Pinecone)
    with open(settings.chunks_cache_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"→ Chunks serializados en {settings.chunks_cache_path} (para BM25).")

    # 5. Upsert a Pinecone (por lotes: PineconeVectorStore ya lo hace adentro)
    embeddings = _get_embeddings()
    # IDs deterministas: re-ingesta en el mismo corpus no duplica.
    ids = [f"{c.metadata['source']}::chunk_{c.metadata['chunk_index']}" for c in chunks]

    vs = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        ids=ids,
        text_key="text",
    )
    print(f"✓ Ingesta completa: {len(chunks)} vectores en Pinecone.")
    return vs


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    ingest_data_directory(force_reindex=force)
