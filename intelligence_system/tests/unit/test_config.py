from __future__ import annotations

import pytest
from pydantic import ValidationError

from helpers import make_settings
from intelligence_system.config import Settings


def test_defaults_are_production_oriented() -> None:
    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
    assert s.vector_store == "pinecone"
    assert s.checkpointer_backend == "redis"
    assert s.embeddings_dimension == 1024
    assert s.llm_temperature is None  # anthropic 2.5.0 no acepta temperature
    assert s.phoenix_traces_endpoint == "http://localhost:6006/v1/traces"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_HYBRID_ALPHA", "0.3")
    monkeypatch.setenv("PINECONE_NAMESPACE", "otro")
    monkeypatch.setenv("API_PORT", "9000")
    s = Settings(_env_file=None)
    assert s.rag_hybrid_alpha == 0.3
    assert s.pinecone_namespace == "otro"
    assert s.api_port == 9000


def test_secrets_are_masked() -> None:
    s = make_settings(anthropic_api_key="sk-ant-secret")
    assert "sk-ant-secret" not in repr(s)
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-secret"


@pytest.mark.parametrize(
    ("field", "value"),
    [("rag_hybrid_alpha", 1.5), ("rag_hybrid_alpha", -0.1), ("api_port", 0), ("embeddings_batch_size", 500)],
)
def test_invalid_values_fail_fast(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_cross_field_rules() -> None:
    with pytest.raises(ValidationError, match="RAG_CHUNK_OVERLAP"):
        make_settings(rag_chunk_size=500, rag_chunk_overlap=500)
    with pytest.raises(ValidationError, match="RAG_FETCH_K"):
        make_settings(rag_top_k=10, rag_fetch_k=5)


def test_missing_credentials_reports_what_is_needed() -> None:
    s = make_settings(
        api_key=None, llm_provider="anthropic", vector_store="pinecone", embeddings_provider="pinecone"
    )
    assert set(s.missing_credentials()) == {"API_KEY", "ANTHROPIC_API_KEY", "PINECONE_API_KEY"}
    assert make_settings().missing_credentials() == []


def test_empty_env_values_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_TEMPERATURE", "CHECKPOINT_TTL_MINUTES", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "")
    s = Settings(_env_file=None)
    assert s.llm_temperature is None and s.checkpoint_ttl_minutes is None
    assert s.anthropic_api_key is None and "ANTHROPIC_API_KEY" in s.missing_credentials()


def test_env_example_parses_and_covers_every_setting() -> None:
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / ".env.example"
    s = Settings(_env_file=example)
    assert s.pinecone_index_name == "intelligence-system"
    declared = {
        line.split("=", 1)[0].strip().lower()
        for line in example.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    host_only = {"redis_port", "phoenix_port"}  # solo los usa docker compose
    assert set(Settings.model_fields) <= declared | {"app_name", "app_version", "api_host"}
    assert host_only <= declared
