"""Tests de las herramientas: búsqueda KB y análisis. Sin LLM."""

from __future__ import annotations

import pytest

from orquestador.agents.analyst_agent import (
    analizar_sentimiento,
    calcular_estadisticas,
    contar_menciones,
)
from orquestador.agents.knowledge_base import search
from orquestador.agents.research_agent import search_knowledge_base


def test_kb_search_finds_matching_doc() -> None:
    hits = search("supervisor langgraph", k=3)
    ids = [doc["id"] for doc in hits]
    assert "doc:langgraph-supervisor" in ids


def test_kb_search_returns_empty_for_unknown_terms() -> None:
    assert search("zxqwvutsrp", k=3) == []


def test_kb_search_ignores_short_terms() -> None:
    assert search("a b c", k=3) == []


def test_research_tool_formats_hits() -> None:
    result = search_knowledge_base.invoke({"query": "Redis módulos", "k": 2})
    assert "doc:redis" in result
    assert "RedisJSON" in result or "RediSearch" in result


def test_research_tool_handles_empty() -> None:
    assert search_knowledge_base.invoke({"query": "zxqwvutsrp"}) == "No hay documentos relevantes."


def test_contar_menciones_case_insensitive() -> None:
    out = contar_menciones.invoke(
        {"texto": "Supervisor supervisor SUPERVISOR analyst", "terminos": ["supervisor", "analyst"]}
    )
    assert out == {"supervisor": 3, "analyst": 1}


def test_calcular_estadisticas_simple() -> None:
    out = calcular_estadisticas.invoke({"numeros": [10.0, 20.0, 30.0, 40.0, 50.0]})
    assert out["n"] == 5
    assert out["media"] == pytest.approx(30.0)
    assert out["mediana"] == pytest.approx(30.0)
    assert out["min"] == pytest.approx(10.0)
    assert out["max"] == pytest.approx(50.0)


def test_calcular_estadisticas_empty() -> None:
    out = calcular_estadisticas.invoke({"numeros": []})
    assert out == {"error": -1, "n": 0}


def test_sentimiento_positivo() -> None:
    assert (
        analizar_sentimiento.invoke({"texto": "la API es rápida, estable y eficiente"})["sentimiento"]
        == "positivo"
    )


def test_sentimiento_negativo() -> None:
    assert (
        analizar_sentimiento.invoke({"texto": "la API cae y satura todo bloqueado roto"})["sentimiento"]
        == "negativo"
    )


def test_sentimiento_neutro_sin_keywords() -> None:
    assert analizar_sentimiento.invoke({"texto": "texto sin palabras clave"})["sentimiento"] == "neutro"
