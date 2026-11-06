"""Unit tests del contrato Pydantic. Sin red."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from entity_pipeline.schemas import NivelCriticidad, TechExtraction


def test_schema_rejects_empty_tech_list() -> None:
    with pytest.raises(ValidationError):
        TechExtraction(
            tecnologias=[],
            nivel_de_criticidad="baja",
            resumen_tecnico="un resumen largo suficiente",
        )


def test_schema_rejects_list_of_only_whitespace() -> None:
    with pytest.raises(ValidationError):
        TechExtraction(
            tecnologias=["", "   "],
            nivel_de_criticidad="baja",
            resumen_tecnico="un resumen largo suficiente",
        )


def test_schema_dedupes_case_insensitive_preserves_first() -> None:
    result = TechExtraction(
        tecnologias=["Redis", "redis", " FastAPI ", "FASTAPI"],
        nivel_de_criticidad="media",
        resumen_tecnico="API con cache y framework asíncrono funcionando bien",
    )
    assert result.tecnologias == ["Redis", "FastAPI"]


def test_schema_rejects_invalid_criticality() -> None:
    with pytest.raises(ValidationError):
        TechExtraction(
            tecnologias=["x"],
            nivel_de_criticidad="urgente",
            resumen_tecnico="un resumen suficientemente largo",
        )


def test_schema_accepts_enum_or_string() -> None:
    assert (
        TechExtraction(
            tecnologias=["Redis"],
            nivel_de_criticidad=NivelCriticidad.ALTA,
            resumen_tecnico="caida completa del servicio en produccion",
        ).nivel_de_criticidad
        is NivelCriticidad.ALTA
    )


def test_resumen_tecnico_collapses_whitespace() -> None:
    result = TechExtraction(
        tecnologias=["x"],
        nivel_de_criticidad="baja",
        resumen_tecnico="varios   espacios   seguidos entre palabras",
    )
    assert "   " not in result.resumen_tecnico


def test_resumen_tecnico_max_length_enforced() -> None:
    with pytest.raises(ValidationError):
        TechExtraction(
            tecnologias=["x"],
            nivel_de_criticidad="baja",
            resumen_tecnico="a" * 241,
        )


def test_json_output_shape_matches_readme_example() -> None:
    result = TechExtraction(
        tecnologias=["FastAPI", "Redis", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico=(
            "API con cache en Redis y persistencia en PostgreSQL; cuello de botella "
            "en conexiones concurrentes."
        ),
    )
    payload = result.model_dump()
    assert set(payload) == {"tecnologias", "nivel_de_criticidad", "resumen_tecnico"}
    assert payload["nivel_de_criticidad"] == "alta"
