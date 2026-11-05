"""Configuración centralizada. Lee de .env."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    """Único lugar donde viven las constantes del sistema."""

    # --- Rutas ---
    data_dir: Path = Path("./data")
    # Dónde se persiste el corpus chunkeado (para que BM25 lo lea sin
    # re-chunkear cada vez). Se regenera con el pipeline de ingesta.
    chunks_cache_path: Path = Path("./chunks_cache.pkl")

    # --- Chunking (dentro del rango 500-800 que recomienda el enunciado) ---
    chunk_size: int = 600
    chunk_overlap: int = 60

    # --- Retrieval ---
    top_k: int = 5

    # --- Pesos del ensemble (semántico + léxico deben sumar 1.0) ---
    # Más peso al semántico porque captura sinónimos y paráfrasis, pero
    # BM25 aporta el 40% para no perder matches léxicos exactos
    # (nombres propios, siglas, códigos).
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4

    # --- Embeddings ---
    # HuggingFace multilingual: 384 dims, gratis, offline tras descarga.
    # Elegido para no requerir OPENAI_API_KEY (el usuario solo tiene Anthropic).
    # Si querés 1536 dims con OpenAI, cambiá también la dimensión del índice
    # y re-ingestá todo.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384

    # --- Pinecone ---
    pinecone_index_name: str = Field(
        default_factory=lambda: os.getenv("INDEX_NAME", "ai-eng-rag")
    )
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "docs"
    pinecone_metric: str = "cosine"

    # --- LLM (Anthropic) ---
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 1024

    # --- API keys ---
    pinecone_api_key: str | None = Field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY")
    )
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    if not s.pinecone_api_key:
        raise RuntimeError(
            "Falta PINECONE_API_KEY. Copiá .env.example a .env y cargá tu clave."
        )
    return s
