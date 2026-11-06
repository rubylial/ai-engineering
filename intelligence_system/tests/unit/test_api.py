"""API HTTP de punta a punta con TestClient (lifespan real, dependencias en memoria)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from helpers import ScriptedLLM, make_settings
from intelligence_system.api.app import create_app
from intelligence_system.config import Settings
from intelligence_system.container import AppContainer

ROUTES = [
    {"next": "researcher", "reason": "buscar", "task": "qué es BM25"},
    {"next": "writer", "reason": "listo", "task": "redactar"},
]


def test_health_and_request_id_header(client: TestClient) -> None:
    resp = client.get("/health", headers={"X-Request-ID": "req-from-client"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.headers["X-Request-ID"] == "req-from-client"
    generated = client.get("/health").headers["X-Request-ID"]
    assert generated.startswith("req_")


def test_ready_reports_all_components(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ready"
    assert {c["name"] for c in body["components"]} == {"redis", "vector_store", "llm", "agent"}


@pytest.fixture
def broken_client() -> Iterator[TestClient]:
    settings = make_settings(api_key=None, llm_provider="anthropic")

    @asynccontextmanager
    async def factory(s: Settings) -> AsyncIterator[AppContainer]:
        yield AppContainer(settings=s, problems={"llm": "falta ANTHROPIC_API_KEY"})

    with TestClient(create_app(settings, container_factory=factory)) as test_client:
        yield test_client


def test_ready_explains_what_is_missing(broken_client: TestClient) -> None:
    resp = broken_client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert set(body["missing"]) >= {"API_KEY", "ANTHROPIC_API_KEY"}
    assert broken_client.get("/health").status_code == 200  # liveness no depende de credenciales


def test_endpoints_require_api_key(client: TestClient) -> None:
    for headers in ({}, {"X-API-Key": "incorrecta"}):
        resp = client.post("/v1/chat", json={"message": "hola"}, headers=headers)
        assert resp.status_code == 401
        body = resp.json()
        assert resp.headers["content-type"].startswith("application/problem+json")
        assert body["type"] == "/errors/unauthorized" and body["request_id"]


def test_validation_errors_are_problem_json(client: TestClient, auth: dict[str, str]) -> None:
    resp = client.post("/v1/chat", json={"message": "", "extra": 1}, headers=auth)
    assert resp.status_code == 422
    body = resp.json()
    assert set(body) >= {"type", "title", "detail", "request_id"}
    assert {e["loc"] for e in body["errors"]} >= {"body.message", "body.extra"}


def test_chat_returns_answer_citations_steps_and_ids(
    client: TestClient, auth: dict[str, str], scripted: ScriptedLLM
) -> None:
    scripted.routes = list(ROUTES)
    scripted.grade = [1]
    scripted.writer = "BM25 puntúa por frecuencia de términos [1]."
    resp = client.post("/v1/chat", json={"message": "¿Qué es BM25?"}, headers=auth)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grounded"] is True and body["citations"][0]["marker"] == 1
    assert [s["node"] for s in body["steps"]] == ["supervisor", "researcher", "supervisor", "writer"]
    assert body["thread_id"].startswith("thr_")
    assert body["request_id"] == resp.headers["X-Request-ID"]
    assert len(body["trace_id"]) == 32


def test_thread_history_roundtrip(client: TestClient, auth: dict[str, str], scripted: ScriptedLLM) -> None:
    assert client.get("/v1/threads/hilo-1", headers=auth).status_code == 404
    scripted.routes = list(ROUTES)
    client.post("/v1/chat", json={"message": "¿Qué es BM25?", "thread_id": "hilo-1"}, headers=auth)
    resp = client.get("/v1/threads/hilo-1", headers=auth)
    assert resp.status_code == 200
    assert [m["role"] for m in resp.json()["messages"]] == ["user", "assistant"]
    assert client.get("/v1/threads/inválido!", headers=auth).status_code == 422


def test_chat_stream_sse(client: TestClient, auth: dict[str, str], scripted: ScriptedLLM) -> None:
    scripted.routes = list(ROUTES)
    scripted.grade = [1]
    scripted.writer = "BM25 es léxico [1]."
    events: list[tuple[str, dict[str, object]]] = []
    with client.stream("POST", "/v1/chat/stream", json={"message": "¿BM25?"}, headers=auth) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        event = ""
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((event, json.loads(line.split(":", 1)[1])))
    kinds = [k for k, _ in events]
    assert kinds[-1] == "done" and "step" in kinds and "token" in kinds and "citation" in kinds
    assert events[-1][1]["grounded"] is True


def test_ingest_is_idempotent_via_api(client: TestClient, auth: dict[str, str]) -> None:
    resp = client.post("/v1/ingest", json={"from_corpus": True}, headers=auth)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["chunks_upserted"] == 0 and body["chunks_unchanged"] == body["chunks_total"] > 0
    new_doc = {
        "doc_id": "nota-01",
        "title": "Nota",
        "text": "# Nota\n\nTexto nuevo sobre chunking y overlap.",
    }
    added = client.post("/v1/ingest", json={"documents": [new_doc]}, headers=auth).json()
    assert added["chunks_upserted"] == 1
    assert client.post("/v1/ingest", json={}, headers=auth).status_code == 422


def test_chat_unavailable_returns_503(broken_client: TestClient) -> None:
    broken_client.app.state.container.settings = make_settings(api_key="k")  # type: ignore[attr-defined]
    resp = broken_client.post("/v1/chat", json={"message": "hola"}, headers={"X-API-Key": "k"})
    assert resp.status_code == 503
    assert "falta ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_openapi_documents_examples(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert {
        "/health",
        "/ready",
        "/v1/chat",
        "/v1/chat/stream",
        "/v1/ingest",
        "/v1/threads/{thread_id}",
    } <= set(spec["paths"])
    assert spec["components"]["schemas"]["ChatRequest"]["examples"]
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]


def test_chat_trace_root_is_chat_turn_with_io(
    client: TestClient, auth: dict[str, str], scripted: ScriptedLLM, span_exporter: object
) -> None:
    scripted.routes = list(ROUTES)
    scripted.grade = [1]
    scripted.writer = "BM25 puntúa términos [1]."
    body = client.post("/v1/chat", json={"message": "¿Qué es BM25?"}, headers=auth).json()
    spans = [
        s
        for s in span_exporter.get_finished_spans()  # type: ignore[attr-defined]
        if format(s.context.trace_id, "032x") == body["trace_id"]
    ]
    roots = [s for s in spans if s.parent is None]
    assert [r.name for r in roots] == ["chat.turn"]  # sin spans HTTP nativos por defecto
    root = roots[0]
    assert root.attributes["request_id"] == body["request_id"]
    assert root.attributes["input.value"] == "¿Qué es BM25?"
    assert root.attributes["output.value"] == body["answer"]


def test_timeout_returns_504_with_trace_id(client: TestClient, auth: dict[str, str]) -> None:
    resp = client.post("/v1/chat", json={"message": "Explicá RRF", "deadline_s": 0.000001}, headers=auth)
    assert resp.status_code == 504
    body = resp.json()
    assert body["type"] == "/errors/graph-timeout"
    assert body["request_id"] == resp.headers["X-Request-ID"]
    assert body["trace_id"] and len(body["trace_id"]) == 32
