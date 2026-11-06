"""Configuración centralizada (pydantic-settings). Todo sale de variables de entorno."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings de la API. Se leen de `.env` o del entorno."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    # LLM
    anthropic_api_key: str = Field(default="", description="Clave de Anthropic.")
    llm_model: str = Field(default="claude-haiku-5-5")

    # Redis: estado de jobs + checkpoints de LangGraph
    redis_url: str = Field(default="redis://localhost:6379/0")
    job_ttl_seconds: int = Field(default=86400, description="TTL de los jobs en Redis (1 día).")
    checkpoint_ttl_minutes: int | None = Field(default=1440)

    # Worker
    worker_concurrency: int = Field(default=4, ge=1, le=32)

    # HITL: palabras que marcan una tarea como crítica (requiere aprobación)
    critical_keywords: str = Field(
        default="ejecutar,borrar,eliminar,drop,delete,producción,produccion,pago,transferir,deploy,desplegar",
        description="Lista separada por comas. Si la query contiene alguna, se pide aprobación.",
    )

    # Observabilidad (Arize Phoenix por OpenInference)
    phoenix_enabled: bool = Field(default=True)
    phoenix_endpoint: str = Field(default="http://localhost:6006")
    phoenix_project_name: str = Field(default="api-agente")
    phoenix_batch_export: bool = Field(default=True)

    # Límite de recursión del grafo (defensa extra contra loops)
    recursion_limit: int = Field(default=25, ge=5, le=100)

    @property
    def phoenix_traces_endpoint(self) -> str:
        """Phoenix recibe OTLP/HTTP en /v1/traces."""
        return f"{self.phoenix_endpoint.rstrip('/')}/v1/traces"

    @property
    def critical_keyword_set(self) -> set[str]:
        return {kw.strip().lower() for kw in self.critical_keywords.split(",") if kw.strip()}


_settings: Settings | None = None


def get_settings() -> Settings:
    """Singleton perezoso de Settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
