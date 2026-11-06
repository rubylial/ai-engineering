"""Unit tests de la cadena LCEL. Sin red: usamos un Runnable fake en lugar del LLM."""

from __future__ import annotations

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable, RunnableLambda

from entity_pipeline.chain import PROMPT, process_text
from entity_pipeline.schemas import NivelCriticidad, TechExtraction


def _prompt_bound_chain(responses: list[TechExtraction | Exception]) -> Runnable:
    """Arma `PROMPT | fake_structured_output` donde cada ainvoke consume una respuesta."""
    iterator = iter(responses)

    async def _emit(_messages: object) -> TechExtraction:
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return value

    return PROMPT | RunnableLambda(_emit)


async def test_process_text_happy_path_returns_schema() -> None:
    expected = TechExtraction(
        tecnologias=["FastAPI", "Redis"],
        nivel_de_criticidad=NivelCriticidad.ALTA,
        resumen_tecnico="API con Redis y Postgres; cuello de botella bajo carga.",
    )
    chain = _prompt_bound_chain([expected])
    result = await process_text(chain, "descripción técnica de prueba")
    assert result is expected


async def test_process_text_rejects_blank_input() -> None:
    chain = _prompt_bound_chain([])
    with pytest.raises(ValueError, match="texto vacío"):
        await process_text(chain, "   ")


async def test_chain_retries_on_output_parser_exception() -> None:
    expected = TechExtraction(
        tecnologias=["Redis"],
        nivel_de_criticidad=NivelCriticidad.BAJA,
        resumen_tecnico="cache simple con redis para endpoints de lectura",
    )
    chain = _prompt_bound_chain([OutputParserException("JSON incompleto"), expected]).with_retry(
        retry_if_exception_type=(OutputParserException,),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )
    result = await process_text(chain, "Redis cache")
    assert result is expected


async def test_chain_gives_up_after_max_attempts() -> None:
    chain = _prompt_bound_chain([OutputParserException("1"), OutputParserException("2")]).with_retry(
        retry_if_exception_type=(OutputParserException,),
        stop_after_attempt=2,
        wait_exponential_jitter=False,
    )
    with pytest.raises(OutputParserException):
        await process_text(chain, "algo")


async def test_chain_does_not_retry_on_unrelated_error() -> None:
    chain = _prompt_bound_chain([RuntimeError("401 unauthorized")]).with_retry(
        retry_if_exception_type=(OutputParserException,),
        stop_after_attempt=5,
        wait_exponential_jitter=False,
    )
    with pytest.raises(RuntimeError, match="401"):
        await process_text(chain, "algo")
