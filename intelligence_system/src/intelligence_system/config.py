"""Configuración central del sistema (pydantic-settings).

Regla ZERO HARD-CODING: modelos, URLs, keys, puertos, índices, namespaces,
tamaños de chunk, límites del agente y endpoints de observabilidad viven acá
y se sobreescriben desde variables de entorno o `.env`. El resto del código
recibe un `Settings` ya validado y nunca lee `os.environ` directamente.

Las API keys son opcionales a propósito: la app arranca igual sin ellas y
`GET /ready` informa exactamente qué falta, en lugar de morir en el arranque.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["anthropic", "openai", "fake"]
EmbeddingsProviderName = Literal["pinecone", "fake"]
VectorStoreName = Literal["pinecone", "memory"]
CheckpointerName = Literal["redis", "memory"]


class Settings(BaseSettings):
    """Toda la configuración del sistema, validada al arrancar."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # `LLM_TEMPERATURE=` vacío en .env = no definido (usa el default).
        env_ignore_empty=True,
    )

    # --- Aplicación -------------------------------------------------------
    app_name: str = "intelligence-system"
    app_env: Literal["local", "test", "production"] = "local"
    app_version: str = "1.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # --- API ----------------------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104 - dentro del contenedor escucha en todas las interfaces
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: SecretStr | None = None

    # --- LLM (cliente unificado de la pre-entrega 1) -------------------------
    llm_provider: LLMProviderName = "anthropic"
    llm_model_supervisor: str = "claude-haiku-5-5"
    llm_model_worker: str = "claude-haiku-5-5"
    llm_max_tokens_supervisor: int = Field(default=400, ge=64, le=8192)
    llm_max_tokens_worker: int = Field(default=800, ge=64, le=8192)
    llm_max_tokens_writer: int = Field(default=1024, ge=64, le=16384)
    # Opcional: el SDK anthropic 2.5.0 ya no acepta `temperature`; solo se
    # envía a proveedores que lo soportan (OpenAI).
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=1, le=10)
    llm_retry_min_wait_s: float = Field(default=1.0, ge=0)
    llm_retry_max_wait_s: float = Field(default=20.0, ge=0)
    llm_max_concurrency: int = Field(default=8, ge=1, le=64)
    llm_structured_max_attempts: int = Field(default=2, ge=1, le=5)
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # --- Embeddings ---------------------------------------------------------
    embeddings_provider: EmbeddingsProviderName = "pinecone"
    embeddings_model: str = "multilingual-e5-large"
    embeddings_dimension: int = Field(default=1024, ge=8, le=20000)
    embeddings_batch_size: int = Field(default=64, ge=1, le=96)
    embeddings_max_concurrency: int = Field(default=4, ge=1, le=32)

    # --- Vector store (Pinecone serverless) -------------------------------
    vector_store: VectorStoreName = "pinecone"
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = Field(default="intelligence-system", pattern=r"^[a-z0-9][a-z0-9-]{0,44}$")
    pinecone_namespace: str = "ai-engineering-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_upsert_batch_size: int = Field(default=100, ge=1, le=1000)
    pinecone_timeout_s: float = Field(default=30.0, gt=0)
    pinecone_create_timeout_s: int = Field(default=180, ge=10)

    # --- RAG ------------------------------------------------------------------
    rag_corpus_dir: Path = Path("data/corpus")
    rag_chunk_size: int = Field(default=1500, ge=200, le=8000)
    rag_chunk_overlap: int = Field(default=150, ge=0, le=2000)
    rag_min_chunk_chars: int = Field(default=40, ge=1, le=1000)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_fetch_k: int = Field(default=20, ge=1, le=100)
    rag_hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    rag_rerank_enabled: bool = False
    rag_rerank_model: str = "bge-reranker-v2-m3"
    rag_grade_evidence: bool = True
    rag_snippet_chars: int = Field(default=300, ge=50, le=2000)
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    bm25_redis_prefix: str = "bm25"

    # --- Redis / persistencia -----------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    checkpointer_backend: CheckpointerName = "redis"
    checkpoint_ttl_minutes: int | None = Field(default=None, ge=1)

    # --- Agentes ------------------------------------------------------------
    agent_max_steps: int = Field(default=4, ge=1, le=20)
    agent_recursion_limit: int = Field(default=25, ge=5, le=200)
    agent_node_timeout_s: float = Field(default=45.0, gt=0)
    agent_request_timeout_s: float = Field(default=120.0, gt=0)
    agent_history_window: int = Field(default=8, ge=0, le=50)
    agent_analyst_max_tool_calls: int = Field(default=4, ge=1, le=12)

    # --- Observabilidad (Arize Phoenix) -------------------------------------
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "intelligence-system"
    phoenix_ui_url: str = "http://localhost:6006"
    phoenix_batch_export: bool = True
    # Spans HTTP nativos de FastAPI (POST /v1/chat, fastapi.*). Apagados por defecto
    # para que la raíz de cada traza sea `chat.turn`, con pregunta y respuesta.
    phoenix_http_spans: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        """Reglas que cruzan varios campos: fallan en el arranque, no en runtime."""
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP debe ser menor que RAG_CHUNK_SIZE")
        if self.rag_fetch_k < self.rag_top_k:
            raise ValueError("RAG_FETCH_K debe ser mayor o igual que RAG_TOP_K")
        if self.llm_retry_max_wait_s < self.llm_retry_min_wait_s:
            raise ValueError("LLM_RETRY_MAX_WAIT_S debe ser >= LLM_RETRY_MIN_WAIT_S")
        if self.agent_request_timeout_s < self.agent_node_timeout_s:
            raise ValueError("AGENT_REQUEST_TIMEOUT_S debe ser >= AGENT_NODE_TIMEOUT_S")
        return self

    @property
    def phoenix_traces_endpoint(self) -> str:
        """Endpoint OTLP/HTTP completo (Phoenix recibe trazas en /v1/traces)."""
        return f"{self.phoenix_collector_endpoint.rstrip('/')}/v1/traces"

    def missing_credentials(self) -> list[str]:
        """Variables obligatorias que faltan según los proveedores elegidos."""
        missing: list[str] = []
        if self.api_key is None or not self.api_key.get_secret_value():
            missing.append("API_KEY")
        if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
            missing.append("ANTHROPIC_API_KEY")
        if self.llm_provider == "openai" and self.openai_api_key is None:
            missing.append("OPENAI_API_KEY")
        uses_pinecone = self.vector_store == "pinecone" or self.embeddings_provider == "pinecone"
        if uses_pinecone and self.pinecone_api_key is None:
            missing.append("PINECONE_API_KEY")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings cacheado para el proceso (los tests construyen el suyo)."""
    return Settings()
