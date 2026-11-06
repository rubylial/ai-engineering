"""Proveedor OpenAI (AsyncOpenAI, Chat Completions) — migrado de la pre-entrega 1.

Verificado contra openai==3.46.0: `chat.completions.create` acepta `tools`,
`tool_choice`, `max_completion_tokens` y `temperature`. Se mantiene como
proveedor alternativo de la Factory (LLM_PROVIDER=openai).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI
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


def map_openai_error(exc: BaseException) -> LLMError:
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMAuthError(f"OpenAI auth: {exc}")
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(f"OpenAI rate limit: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        return LLMConnectionError(f"OpenAI conexión/timeout: {exc}")
    if isinstance(exc, openai.InternalServerError):
        return LLMServerError(f"OpenAI 5xx: {exc}")
    if isinstance(exc, openai.NotFoundError):
        return LLMConfigError(f"OpenAI: modelo o recurso inexistente: {exc}")
    return LLMProviderError(f"OpenAI: {exc}")


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(
        self,
        api_key: SecretStr,
        *,
        timeout_s: float,
        retry: RetryPolicy,
        max_concurrency: int,
        client: AsyncOpenAI | None = None,
    ) -> None:
        super().__init__(retry=retry, max_concurrency=max_concurrency)
        self._client = client or AsyncOpenAI(
            api_key=api_key.get_secret_value(), max_retries=0, timeout=timeout_s
        )

    @staticmethod
    def to_wire(messages: Sequence[ChatMessage], config: ModelConfig) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        if config.system_prompt:
            wire.append({"role": "system", "content": config.system_prompt})
        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.USER):
                wire.append({"role": msg.role.value, "content": msg.content})
            elif msg.role is Role.ASSISTANT:
                item: dict[str, Any] = {"role": "assistant", "content": msg.content or None}
                if msg.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in msg.tool_calls
                    ]
                wire.append(item)
            else:  # Role.TOOL
                wire.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
        return wire

    def _build_kwargs(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": self.to_wire(messages, config),
            "max_completion_tokens": config.max_tokens,
        }
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        return kwargs

    async def _generate(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                **self._build_kwargs(messages, config, tools, tool_choice)
            )
        except openai.OpenAIError as exc:
            raise map_openai_error(exc) from exc

        choice = response.choices[0]
        calls: list[ToolCall] = []
        for call in choice.message.tool_calls or []:
            if getattr(call, "type", None) != "function":
                continue
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"__raw__": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))
        usage = response.usage
        return ModelResponse(
            provider=self.provider,
            model=response.model,
            content=choice.message.content or "",
            tool_calls=calls,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason,
        )

    async def _stream(self, messages: Sequence[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                **self._build_kwargs(messages, config, None, None), stream=True
            )
            async for chunk in stream:
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    yield delta
        except openai.OpenAIError as exc:
            raise map_openai_error(exc) from exc

    async def check_model(self, model: str) -> None:
        try:
            await self._client.models.retrieve(model)
        except openai.OpenAIError as exc:
            raise map_openai_error(exc) from exc

    async def aclose(self) -> None:
        await self._client.close()
