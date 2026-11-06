"""Tests de `build_chain` con un chat model falso (sin red).

En langchain-core 1.x, los chat models son Pydantic v2 con validación estricta de
sus campos, así que no heredamos de ninguno: un objeto duck-typed con el método
`with_structured_output` alcanza porque `build_chain` solo llama a ese método y
luego compone con `|` y `.with_retry()`.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from entity_pipeline.chain import build_chain
from entity_pipeline.schemas import NivelCriticidad, TechExtraction


class _FakeStructuredModel:
    """Stand-in minimal: solo expone `with_structured_output`, que es lo que necesita `build_chain`."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)

    def with_structured_output(self, schema: Any, **_: Any) -> Runnable[Any, TechExtraction]:
        iterator = iter(self._responses)

        async def _emit(_messages: object) -> TechExtraction:
            value = next(iterator)
            if isinstance(value, BaseException):
                raise value
            if isinstance(value, schema):
                return value
            raise TypeError(f"respuesta inesperada: {type(value).__name__}")

        return RunnableLambda(_emit)


def _as_model(fake: _FakeStructuredModel) -> BaseChatModel:
    """Duck-type cast: `build_chain` solo llama a `.with_structured_output()`."""
    return cast("BaseChatModel", fake)


@pytest.fixture
def ok_extraction() -> TechExtraction:
    return TechExtraction(
        tecnologias=["FastAPI", "PostgreSQL"],
        nivel_de_criticidad=NivelCriticidad.MEDIA,
        resumen_tecnico="API FastAPI con Postgres; latencia elevada bajo carga.",
    )


async def test_build_chain_happy_path(ok_extraction: TechExtraction) -> None:
    model = _as_model(_FakeStructuredModel(responses=[ok_extraction]))
    chain = build_chain(model, max_attempts=1)
    result = await chain.ainvoke({"text": "FastAPI con Postgres"})
    assert result is ok_extraction


async def test_build_chain_retries_until_success(ok_extraction: TechExtraction) -> None:
    model = _as_model(
        _FakeStructuredModel(responses=[OutputParserException("JSON mal formado"), ok_extraction])
    )
    chain = build_chain(model, max_attempts=3, wait_exponential_jitter=False)
    result = await chain.ainvoke({"text": "Postgres"})
    assert result is ok_extraction


async def test_build_chain_rejects_invalid_max_attempts() -> None:
    model = _as_model(_FakeStructuredModel(responses=[]))
    with pytest.raises(ValueError, match="max_attempts"):
        build_chain(model, max_attempts=0)
