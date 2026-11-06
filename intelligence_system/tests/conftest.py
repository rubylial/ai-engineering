"""Fixtures comunes. Los tests unitarios nunca tocan red: LLM, Pinecone y Redis son dobles."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import ScriptedLLM, build_container, make_settings
from intelligence_system.api.app import create_app
from intelligence_system.config import Settings
from intelligence_system.container import AppContainer

# Un único TracerProvider global con exportador en memoria: los tests pueden
# inspeccionar los spans que produce el código real.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ningún test lee variables reales del entorno (claves, URLs) por accidente."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


@pytest.fixture
def tracer_provider() -> TracerProvider:
    return _PROVIDER


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def scripted() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
async def ingested(settings: Settings, scripted: ScriptedLLM) -> AppContainer:
    """Contenedor en memoria con el corpus ya ingestado (en el loop del test)."""
    built = build_container(settings, responder=scripted)
    assert built.ingest is not None
    await built.ingest.ingest_corpus(settings.rag_corpus_dir)
    return built


@pytest.fixture
def client(settings: Settings, scripted: ScriptedLLM) -> Iterator[TestClient]:
    """API completa con dobles; la ingesta corre dentro del lifespan (mismo loop)."""
    built = build_container(settings, responder=scripted)

    @asynccontextmanager
    async def factory(_: Settings) -> AsyncIterator[AppContainer]:
        assert built.ingest is not None
        await built.ingest.ingest_corpus(settings.rag_corpus_dir)
        yield built

    app = create_app(settings, container_factory=factory)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": "test-key"}
