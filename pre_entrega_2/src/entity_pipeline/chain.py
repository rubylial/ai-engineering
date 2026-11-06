"""Cadena LCEL: `prompt | model.with_structured_output(Schema)` con reintentos.

Diseño:
- El prompt es un `ChatPromptTemplate` (no f-strings): LangChain gestiona las variables.
- `model.with_structured_output(TechExtraction)` fuerza al LLM a devolver el schema
  vía tool calling (recomendado para Anthropic/OpenAI). Si el JSON está incompleto o
  no cumple el schema, LangChain lanza `OutputParserException` o `ValidationError`.
- `.with_retry(...)` envuelve la cadena con backoff exponencial y reintenta hasta
  `max_attempts` veces ante esos dos errores. Otros errores (red, auth, 429) suben
  tal cual: no tiene sentido reintentar un 401.
- `process_text` corre la cadena con `.ainvoke()` y loguea entrada/salida para que
  un reintento sea visible en los logs sin tener que prender DEBUG de LangChain.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from entity_pipeline.schemas import TechExtraction

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sos un asistente técnico. Dado un párrafo (descripción de arquitectura, "
    "log de error o incidente), extraés estrictamente:\n"
    "- `tecnologias`: la lista de tecnologías, frameworks, lenguajes o servicios "
    "mencionados. No inventes nombres: usá únicamente los que aparecen en el texto.\n"
    "- `nivel_de_criticidad`: 'baja' si es informativo o menor; 'media' si hay "
    "degradación parcial; 'alta' si hay caída, pérdida de datos o riesgo de seguridad.\n"
    "- `resumen_tecnico`: una sola oración, máximo 240 caracteres, en español.\n"
    "Si el texto no menciona ninguna tecnología, devolvé igual al menos un término "
    "general extraído del propio texto (por ejemplo 'servicio' o 'base de datos')."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{text}"),
    ]
)


def build_chain(
    model: BaseChatModel,
    *,
    max_attempts: int = 3,
    wait_exponential_jitter: bool = True,
) -> Runnable[dict[str, Any], TechExtraction]:
    """Compone la cadena `prompt | model.with_structured_output(Schema)` + reintentos.

    Parameters
    ----------
    model:
        Un chat model de LangChain que implemente `.with_structured_output()`
        (típicamente `ChatAnthropic` o `ChatOpenAI`).
    max_attempts:
        Intentos totales, incluyendo el primero. 1 = sin reintentos.
    wait_exponential_jitter:
        Si True, backoff exponencial con jitter entre reintentos (por defecto).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts debe ser >= 1")

    # method="function_calling" es el default para Anthropic/OpenAI en langchain>=0.3.
    # Fuerza una tool call con el schema; si llega incompleta o inválida, el parser
    # levanta OutputParserException o ValidationError (se reintenta abajo).
    structured = model.with_structured_output(TechExtraction)
    chain: Runnable[dict[str, Any], TechExtraction] = PROMPT | structured

    if max_attempts == 1:
        return chain

    return chain.with_retry(
        retry_if_exception_type=(ValidationError, OutputParserException),
        stop_after_attempt=max_attempts,
        wait_exponential_jitter=wait_exponential_jitter,
    )


async def process_text(
    chain: Runnable[dict[str, Any], TechExtraction],
    text: str,
) -> TechExtraction:
    """Ejecuta la cadena sobre `text` y loguea validación y reintentos.

    Lanza `ValidationError` u `OutputParserException` si la cadena agotó sus reintentos,
    y las excepciones del proveedor (auth, cuota, red) sin reintentar.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("process_text recibió un texto vacío")

    log.info("pipeline.start chars=%d preview=%r", len(text), text[:80])
    try:
        result = await chain.ainvoke({"text": text})
    except ValidationError as err:
        log.warning("pipeline.validation_error errors=%d", len(err.errors()))
        raise
    except OutputParserException as err:
        log.warning("pipeline.parser_error detail=%s", str(err)[:200])
        raise

    log.info(
        "pipeline.ok tecnologias=%s criticidad=%s resumen_chars=%d",
        result.tecnologias,
        result.nivel_de_criticidad.value,
        len(result.resumen_tecnico),
    )
    return result
