"""Contrato de salida del pipeline: entidades técnicas extraídas del texto.

Este módulo es el "contrato" entre el LLM y el resto del sistema. Pydantic v2 valida
que la lista de tecnologías no esté vacía, que la criticidad sea uno de los tres valores
permitidos y que el resumen tenga largo razonable. Si el LLM devuelve algo que no cumple,
`with_structured_output()` lanza `ValidationError` y la cadena reintenta.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class NivelCriticidad(StrEnum):
    """Nivel de criticidad inferido del texto."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class TechExtraction(BaseModel):
    """Entidades técnicas extraídas de un párrafo de texto.

    Devuelta por la cadena LCEL del módulo `chain`. Los nombres de los campos van en
    español porque el prompt los pide así (y así salen al JSON final).
    """

    tecnologias: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Tecnologías, frameworks, lenguajes o servicios mencionados en el texto. "
            "Al menos uno; sin duplicados (case-insensitive)."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description=(
            "Impacto operacional inferido del texto: 'baja' (info o cambio menor), "
            "'media' (degradación parcial) o 'alta' (caída o riesgo de datos)."
        ),
    )
    resumen_tecnico: str = Field(
        ...,
        min_length=10,
        max_length=240,
        description="Resumen técnico en una sola oración, máximo 240 caracteres.",
    )

    @field_validator("tecnologias")
    @classmethod
    def _unique_non_empty(cls, raw: list[str]) -> list[str]:
        """Limpia, descarta vacíos y dedupe preservando orden (case-insensitive)."""
        cleaned = [t.strip() for t in raw if t and t.strip()]
        if not cleaned:
            raise ValueError("tecnologias no puede quedar vacía después de limpiar")
        seen: set[str] = set()
        out: list[str] = []
        for tech in cleaned:
            key = tech.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(tech)
        return out

    @field_validator("resumen_tecnico")
    @classmethod
    def _collapse_whitespace(cls, raw: str) -> str:
        """Colapsa espacios en blanco repetidos: así `max_length` mide lo que se lee."""
        return " ".join(raw.split())
