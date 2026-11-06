"""Proveedor Anthropic (AsyncAnthropic) — migrado de la pre-entrega 1.

Notas verificadas contra anthropic==2.5.0:
- `messages.create` ya no acepta `temperature` ni `top_p` (no existen en la
  firma), por eso este cliente nunca los envía.
- `system` va aparte, no como un mensaje más.
- Tool calling: `tools=[{name, description, input_schema}]` y
  `tool_choice={"type": "tool", "name": ...}` para forzar salida estructurada.
- `max_retries=0` en el SDK: los reintentos los hace tenacity en la clase base
  (con logs y backoff controlado), así no se duplican.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic
from pydantic import SecretStr

from intelligence_system.llm.base import BaseLLMClient, RetryPolicy
from intelligence_system.llm.exceptions import (
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
)
from intelligence_system.llm.schemas import (
    ChatMessage,
    ModelConfig,
    ModelResponse,
    Provider,
    Role,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


def map_anthropic_error(exc: BaseException) -> LLMError:
    """Traduce excepciones del SDK a la jerarquía propia (transitorias vs. no)."""
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return LLMAuthError(f"Anthropic auth: {exc}")
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError(f"Anthropic rate limit: {exc}")
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.RetryableError)):
        return LLMConnectionError(f"Anthropic conexión/timeout: {exc}")
    if isinstance(exc, anthropic.InternalServerError):
        return LLMServerError(f"Anthropic 5xx/sobrecarga: {exc}")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMConfigError(f"Anthropic: modelo o recurso inexistente: {exc}")
    return LLMProviderError(f"Anthropic: {exc}")


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(
        self,
        api_key: SecretStr,
        *,
        timeout_s: float,
        retry: RetryPolicy,
        max_concurrency: int,
        client: AsyncAnthropic | None = None,
    ) -> None:
        super().__init__(retry=retry, max_concurrency=max_concurrency)
        # open_telemetry=False: el SDK 2.x emite un span CLIENT propio por llamada. OpenInference ya
        # lo reemplaza en las llamadas que traza (messages.*), así que lo único que quedaría son
        # trazas sueltas `anthropic.models.retrieve` del probe de /ready, sin request_id/thread_id.
        self._client = client or AsyncAnthropic(
            api_key=api_key.get_secret_value(), max_retries=0, timeout=timeout_s, open_telemetry=False
        )

    # ------------------------------------------------------------------
    # Traducción al formato de Anthropic
    # ------------------------------------------------------------------
    @staticmethod
    def to_wire(
        messages: Sequence[ChatMessage], config: ModelConfig
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Separa el system prompt y agrupa turnos consecutivos del mismo rol."""
        system_parts: list[str] = [config.system_prompt] if config.system_prompt else []
        wire: list[dict[str, Any]] = []

        def append(role: str, blocks: list[dict[str, Any]]) -> None:
            if wire and wire[-1]["role"] == role:
                wire[-1]["content"].extend(blocks)
            else:
                wire.append({"role": role, "content": blocks})

        for msg in messages:
            if msg.role is Role.SYSTEM:
                system_parts.append(msg.content)
            elif msg.role is Role.USER:
                append("user", [{"type": "text", "text": msg.content}])
            elif msg.role is Role.ASSISTANT:
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                blocks.extend(
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in msg.tool_calls
                )
                append("assistant", blocks)
            else:  # Role.TOOL
                append(
                    "user",
                    [{"type": "tool_result", "tool_use_id": msg.tool_call_id, "content": msg.content}],
                )
        system = "\n\n".join(system_parts) if system_parts else None
        return system, wire

    def _build_kwargs(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> dict[str, Any]:
        system, wire = self.to_wire(messages, config)
        kwargs: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": wire,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools
            ]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        return kwargs

    # ------------------------------------------------------------------
    # Implementación del contrato
    # ------------------------------------------------------------------
    async def _generate(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> ModelResponse:
        kwargs = self._build_kwargs(messages, config, tools, tool_choice)
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.AnthropicError as exc:
            raise map_anthropic_error(exc) from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {"__raw__": block.input}
                calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
        return ModelResponse(
            provider=self.provider,
            model=response.model,
            content="".join(text_parts),
            tool_calls=calls,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason,
        )

    async def _stream(self, messages: Sequence[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(messages, config, None, None)
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except anthropic.AnthropicError as exc:
            raise map_anthropic_error(exc) from exc

    async def check_model(self, model: str) -> None:
        try:
            await self._client.models.retrieve(model)
        except anthropic.AnthropicError as exc:
            raise map_anthropic_error(exc) from exc

    async def aclose(self) -> None:
        await self._client.close()
