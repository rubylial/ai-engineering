"""
Módulo de ingesta.

Pipeline:
  1. Leer archivos .md / .txt de la carpeta data/
  2. Chunking recursivo midiendo en tokens (no en chars)
  3. Persistir en ChromaDB con metadata (source) para citación posterior

Optimización clave: si la vectorstore ya existe y tiene documentos, NO
re-indexa. Evita gastar tiempo cada vez que se ejecuta el script.
"""

from pathlib import Path
from typing import List

import tiktoken
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .config import get_settings


# ============================================================
# Helpers
# ============================================================

def _load_documents(data_dir: Path) -> List[Document]:
    """
    Lee todos los archivos .md y .txt de data/ y los envuelve en Documents
    de LangChain (texto + metadata con el nombre del archivo).
    """
    documents: list[Document] = []
    for file_path in sorted(data_dir.glob("*")):
        if file_path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = file_path.read_text(encoding="utf-8")
        documents.append(Document(
            page_content=text,
            metadata={"source": file_path.name},
        ))
    return documents


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Splitter que mide en TOKENS reales (no caracteres), usando el tokenizador
    cl100k_base — es una aproximación razonable incluso cuando el LLM final
    es Claude, porque sirve como presupuesto consistente entre modelos.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda t: len(enc.encode(t)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Modelo de embeddings local (HuggingFace sentence-transformers).
    Se descarga la primera vez (~120 MB), después corre offline.

    CRÍTICO: debe ser el mismo en ingesta y en query. Si cambiás esto,
    re-indexás todo (los vectores viejos quedan inútiles).
    """
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        # CPU es suficiente para corpus chicos; cambiá a 'cuda' si tenés GPU.
        model_kwargs={"device": "cpu"},
        # normalize=True convierte en vectores unitarios (norma 1).
        # Chroma con L2 sobre vectores normalizados ≈ coseno.
        encode_kwargs={"normalize_embeddings": True},
    )


# ============================================================
# API pública
# ============================================================

def get_vectorstore() -> Chroma:
    """
    Abre (o crea) la vectorstore persistente.
    Esta función es segura de llamar muchas veces: no re-crea nada, solo abre.
    """
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=str(settings.vectorstore_dir),
    )


def ingest_data_directory(force_reindex: bool = False) -> Chroma:
    """
    Pipeline completo de ingesta.

    Args:
        force_reindex: si True, borra la colección existente y re-indexa.
                       Útil cuando cambiás los documentos o el modelo de embeddings.

    Returns:
        La Chroma vectorstore lista para usar como retriever.
    """
    settings = get_settings()
    vs = get_vectorstore()

    existing_count = vs._collection.count()

    if existing_count > 0 and not force_reindex:
        print(
            f"✓ Vectorstore ya contiene {existing_count} chunks en "
            f"'{settings.vectorstore_dir}'. Salteando re-indexación."
        )
        return vs

    if force_reindex and existing_count > 0:
        print(f"⚠ force_reindex=True → borrando {existing_count} chunks anteriores...")
        all_ids = vs._collection.get()["ids"]
        if all_ids:
            vs._collection.delete(ids=all_ids)

    # --- 1. Cargar documentos ---
    docs = _load_documents(settings.data_dir)
    if not docs:
        raise RuntimeError(
            f"No se encontraron archivos .md/.txt en {settings.data_dir}. "
            f"Revisá que el directorio exista y contenga documentos."
        )
    print(f"→ {len(docs)} documentos cargados de {settings.data_dir}/")

    # --- 2. Chunking ---
    splitter = _build_splitter(settings.chunk_size, settings.chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(
        f"→ {len(chunks)} chunks generados "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}, en tokens)"
    )

    # --- 3. Ingesta a ChromaDB ---
    # IDs deterministas por posición: permiten que un re-run con el mismo
    # corpus no duplique entradas (chroma hace upsert por ID).
    ids = [f"{c.metadata['source']}::chunk_{i}" for i, c in enumerate(chunks)]
    vs.add_documents(documents=chunks, ids=ids)
    print(f"✓ Indexados {len(chunks)} chunks en Chroma ({settings.vectorstore_dir}/)")

    return vs


if __name__ == "__main__":
    # Permite ejecutar la ingesta como script standalone:
    #   python -m rag.ingest
    import sys
    force = "--force" in sys.argv
    ingest_data_directory(force_reindex=force)
