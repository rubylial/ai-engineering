"""Interfaz común (ABC) de todo proveedor LLM — patrón de la pre-entrega 1.

El resto del sistema programa contra `BaseLLMClient`, nunca contra un SDK.
La clase base resuelve una sola vez lo transversal para todos los proveedores:

- límite de concurrencia (`asyncio.Semaphore`) para no golpear rate limits;
- reintentos con backoff exponencial + jitter (tenacity) solo ante errores
  transitorios;
- salida estructurada con bucle de reparación: si el JSON no valida contra el
  esquema Pydantic, se reenvía el output fallido junto con el error exacto
  del validador para que el modelo corrija solo lo afectado.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from intelligence_system.llm.exceptions import LLMStructuredOutputError, LLMTransientError
from intelligence_system.llm.schemas import (
    ChatMessage,
    ModelConfig,
    ModelResponse,
    Provider,
    ToolSpec,
)
from intelligence_system.observability.logging import get_logger

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    min_wait_s: float = 1.0
    max_wait_s: float = 20.0

    def wait_strategy(self) -> wait_exponential_jitter:
        """Backoff exponencial con jitter: min_wait · 2^(n-1) + U(0, min_wait), tope max_wait."""
        return wait_exponential_jitter(
            multiplier=self.min_wait_s, max=self.max_wait_s, jitter=self.min_wait_s
        )

    def backoff(self, attempt: int) -> float:
        base = min(self.max_wait_s, self.min_wait_s * (2 ** (attempt - 1)))
        return base + random.uniform(0, self.min_wait_s)  # noqa: S311 - jitter, no criptografía


def structured_tool_name(schema: type[BaseModel]) -> str:
    """`SupervisorDecision` -> `supervisor_decision` (nombre de tool válido)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", schema.__name__).lower()


def format_validation_error(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in err['loc']) or '(raíz)'}: {err['msg']}" for err in exc.errors()]
    return "; ".join(parts)


class BaseLLMClient(ABC):
    """Contrato + comportamiento transversal de todos los clientes LLM."""

    provider: ClassVar[Provider]

    def __init__(self, *, retry: RetryPolicy, max_concurrency: int) -> None:
        self._retry = retry
        self._semaphore = asyncio.Semaphore(max_concurrency)

    # ------------------------------------------------------------------
    # Lo que implementa cada proveedor
    # ------------------------------------------------------------------
    @abstractmethod
    async def _generate(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> ModelResponse: ...

    @abstractmethod
    def _stream(self, messages: Sequence[ChatMessage], config: ModelConfig) -> AsyncIterator[str]: ...

    @abstractmethod
    async def check_model(self, model: str) -> None:
        """Verificación barata (sin tokens) de credenciales + nombre de modelo."""

    async def aclose(self) -> None:  # noqa: B027 - hook opcional
        """Libera conexiones HTTP del SDK (opcional)."""

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self._retry.max_attempts),
            wait=self._retry.wait_strategy(),
            retry=retry_if_exception_type(LLMTransientError),
            reraise=True,
            before_sleep=self._log_retry,
        )

    def _log_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log.warning(
            "llm.retry",
            provider=self.provider.value,
            attempt=state.attempt_number,
            error=type(exc).__name__ if exc else None,
            sleep_s=round(state.next_action.sleep, 2) if state.next_action else None,
        )

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        *,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | None = None,
    ) -> ModelResponse:
        """Respuesta completa (con tools opcionales). Reintenta solo errores transitorios."""
        started = time.perf_counter()
        response: ModelResponse | None = None
        async for attempt in self._retrying():
            with attempt:
                async with self._semaphore:
                    response = await self._generate(messages, config, tools, tool_choice)
        if response is None:  # pragma: no cover - reraise=True ya propaga la excepción
            raise RuntimeError("el proveedor no devolvió respuesta")
        log.info(
            "llm.generate",
            provider=self.provider.value,
            model=config.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            tool_calls=[c.name for c in response.tool_calls],
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    async def stream(self, messages: Sequence[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        """Tokens a medida que llegan. Solo reintenta si el fallo ocurre antes del 1er token."""
        attempt = 0
        while True:
            attempt += 1
            emitted = False
            try:
                async with self._semaphore:
                    async for token in self._stream(messages, config):
                        emitted = True
                        yield token
                return
            except LLMTransientError as exc:
                if emitted or attempt >= self._retry.max_attempts:
                    raise
                delay = self._retry.backoff(attempt)
                log.warning(
                    "llm.stream.retry",
                    provider=self.provider.value,
                    attempt=attempt,
                    error=type(exc).__name__,
                    sleep_s=round(delay, 2),
                )
                await asyncio.sleep(delay)

    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        schema: type[T],
        *,
        max_attempts: int = 2,
        description: str | None = None,
    ) -> T:
        """Fuerza una salida que valide contra `schema` vía tool calling.

        Bucle de reparación: ante un `ValidationError` se reenvía la llamada
        fallida + el mensaje exacto del validador (no se reinicia a ciegas).
        """
        name = structured_tool_name(schema)
        spec = ToolSpec.from_model(name, description or (schema.__doc__ or name).strip(), schema)
        history = list(messages)
        last_error = "sin respuesta"
        for attempt in range(1, max_attempts + 1):
            response = await self.generate(history, config, tools=[spec], tool_choice=name)
            call = next((c for c in response.tool_calls if c.name == name), None)
            if call is None:
                last_error = "el modelo no llamó a la herramienta de salida estructurada"
                history = [
                    *history,
                    ChatMessage.assistant(response.content or "(sin contenido)"),
                    ChatMessage.user(f"Respondé únicamente llamando a la herramienta `{name}`."),
                ]
                continue
            try:
                return schema.model_validate(call.arguments)
            except ValidationError as exc:
                last_error = format_validation_error(exc)
                log.warning(
                    "llm.structured.invalid",
                    schema=schema.__name__,
                    attempt=attempt,
                    errors=last_error,
                )
                history = [
                    *history,
                    ChatMessage.assistant(response.content, [call]),
                    ChatMessage.tool(
                        call.id,
                        f"ERROR DE VALIDACIÓN: {last_error}. Corregí solo los campos "
                        f"inválidos y volvé a llamar a `{name}`.",
                    ),
                ]
        raise LLMStructuredOutputError(
            f"{schema.__name__}: sin salida válida tras {max_attempts} intentos ({last_error})"
        )
