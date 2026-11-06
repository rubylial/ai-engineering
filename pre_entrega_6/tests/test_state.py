"""Tests del esquema de state y de la decisión del supervisor."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orquestador.state import SupervisorDecision


def test_decision_accepts_valid_payload() -> None:
    d = SupervisorDecision(next="researcher", reason="falta info del KB", task="Buscá LangGraph")
    assert d.next == "researcher"


def test_decision_rejects_invalid_agent_name() -> None:
    with pytest.raises(ValidationError):
        SupervisorDecision(next="writer", reason="test", task="algo")  # type: ignore[arg-type]


def test_decision_rejects_empty_task() -> None:
    with pytest.raises(ValidationError):
        SupervisorDecision(next="FINISH", reason="ok", task="")


def test_decision_rejects_short_reason() -> None:
    with pytest.raises(ValidationError):
        SupervisorDecision(next="analyst", reason="", task="procesar los datos")
