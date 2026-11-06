from __future__ import annotations

import json

import pytest
from opentelemetry import trace

from intelligence_system.observability.context import (
    bind_request_id,
    bind_thread_id,
    clear_context,
    sanitize_request_id,
)
from intelligence_system.observability.logging import configure_logging, get_logger


def test_json_logs_carry_correlation_ids(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True)
    bind_request_id("req-123")
    bind_thread_id("thr-9")
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("unit"):
        get_logger("test").info("evento.prueba", detalle="x")
    clear_context()
    line = [ln for ln in capsys.readouterr().out.splitlines() if "evento.prueba" in ln][-1]
    record = json.loads(line)
    assert record["request_id"] == "req-123"
    assert record["thread_id"] == "thr-9"
    assert record["level"] == "info"
    assert len(record["trace_id"]) == 32 and len(record["span_id"]) == 16


def test_sanitize_request_id() -> None:
    assert sanitize_request_id("abc-123") == "abc-123"
    assert sanitize_request_id("con espacios y $$$").startswith("req_")
    assert sanitize_request_id("x" * 200).startswith("req_")
    assert sanitize_request_id(None).startswith("req_")
