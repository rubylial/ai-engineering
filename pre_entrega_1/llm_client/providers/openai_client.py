"""Implementación concreta para OpenAI (modo normal + streaming)."""

from typing import AsyncIterator

import openai
from openai import AsyncOpenAI
from pydantic import SecretStr

from ..base import BaseLLMClient
from ..exceptions import (
    LLMAuthError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
)
from ..schemas import ChatMessage, ModelConfig, ModelResponse, Provider, TokenUsage


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: SecretStr):
        # AsyncOpenAI usa httpx por debajo — ya es async-native.
        # NUNCA uses el cliente síncrono dentro de async def: bloquea el loop.
        self._client = AsyncOpenAI(api_key=api_key.get_secret_value())

    # ------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------
    def _build_messages(self, messages: list[ChatMessage], config: ModelConfig) -> list[dict]:
        """Convierte nuestros ChatMessage al formato esperado por el SDK."""
        out: list[dict] = []
        if config.system_prompt:
            out.append({"role": "system", "content": config.system_prompt})
        out.extend(m.to_dict() for m in messages)
        return out

    # ------------------------------------------------------------
    # Modo normal (respuesta completa)
    # ------------------------------------------------------------
    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=config.model,
                messages=self._build_messages(messages, config),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
            )
        except openai.AuthenticationError as e:
            raise LLMAuthError(f"OpenAI auth failed: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit: {e}") from e
        except openai.APIConnectionError as e:
            raise LLMConnectionError(f"OpenAI connection error: {e}") from e
        except openai.APIError as e:
            raise LLMProviderError(f"OpenAI API error: {e}") from e

        choice = response.choices[0]
        usage = response.usage

        return ModelResponse(
            provider=Provider.OPENAI,
            model=response.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )

    # ------------------------------------------------------------
    # Streaming (generador asíncrono de tokens)
    # ------------------------------------------------------------
    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                model=config.model,
                messages=self._build_messages(messages, config),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stream=True,
            )
        except openai.AuthenticationError as e:
            raise LLMAuthError(f"OpenAI auth failed: {e}") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit: {e}") from e
        except openai.APIConnectionError as e:
            raise LLMConnectionError(f"OpenAI connection error: {e}") from e

        try:
            async for chunk in stream:
                # Un chunk de OpenAI trae un delta; puede venir vacío al inicio/fin.
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except openai.APIError as e:
            # Error en medio del stream (ej. cortado por el servidor).
            raise LLMProviderError(f"OpenAI stream error: {e}") from e
