"""
Configuración centralizada.
Lee de .env y expone un objeto Settings que el resto del código consume.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    """
    Un único lugar para todas las constantes del sistema.
    Si cambiás embeddings o LLM, lo hacés acá y el resto se adapta.
    """

    # --- Rutas ---
    data_dir: Path = Path("./data")
    vectorstore_dir: Path = Path("./vectorstore")
    collection_name: str = "ai_engineering_kb"

    # --- Chunking (mínimos exigidos por el enunciado) ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    # top_k entre 3 y 5: más documentos no mejoran la respuesta y aumentan el
    # riesgo de "lost in the middle" (el modelo pierde atención sobre contexto
    # en posiciones intermedias de un prompt largo).
    top_k: int = 4

    # --- Modelos ---
    # Anthropic NO tiene API de embeddings propia, así que usamos un modelo
    # local de HuggingFace (sentence-transformers). Se descarga una sola vez
    # (~120 MB) y después corre offline, gratis.
    #
    # paraphrase-multilingual-MiniLM-L12-v2:
    #   - 384 dimensiones
    #   - Multilingual (soporta español sin prefijos especiales)
    #   - Rápido en CPU
    #
    # CRÍTICO: debe ser el MISMO modelo para indexar y para consultar.
    # Si cambia, hay que re-indexar todo.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM: Claude vía Anthropic. Usamos haiku (barato y rápido) por defecto;
    # cambiá a sonnet/opus si querés más calidad a más costo.
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 1024

    # --- API key ---
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve una instancia singleton de Settings."""
    s = Settings()
    if not s.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY no está definida. Copiá .env.example a .env y "
            "cargá tu clave."
        )
    return s
