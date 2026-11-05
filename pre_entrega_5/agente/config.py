"""Configuración centralizada del agente."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


load_dotenv()


class Settings(BaseModel):
    """Único lugar donde viven las constantes del sistema."""

    # --- LLM ---
    # El enunciado permite OpenAI o Anthropic. Usamos Claude porque
    # es el único modelo al que tenemos acceso en el entorno.
    llm_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 1024

    # --- Persistencia ---
    # SqliteSaver escribe acá. Sobrevive al reinicio del proceso.
    checkpoint_db: Path = Path("./checkpoints.sqlite")

    # --- Traza ReAct ---
    trace_dir: Path = Path("./traces")

    # --- Guardarraíl contra loops infinitos ---
    # Cada iteración del agente consume tokens. Un tope duro evita
    # costos imprevistos si el modelo entra en bucle.
    recursion_limit: int = 15

    # --- API key ---
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    if not s.anthropic_api_key:
        raise RuntimeError(
            "Falta ANTHROPIC_API_KEY. Copiá .env.example a .env y cargá tu clave."
        )
    s.trace_dir.mkdir(parents=True, exist_ok=True)
    return s
