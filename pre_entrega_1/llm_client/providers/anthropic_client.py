"""Implementación concreta para Anthropic (modo normal + streaming).

Compatible con anthropic SDK >= 2.5.0, donde los parámetros
`temperature` y `top_p` fueron removidos de messages.create().
En su lugar se expone `speed` ('standard' | 'fast').
"""

from typing import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic
from pydantic import SecretStr

from ..base import BaseLLMClient
from ..exceptions import (
    LLMAuthError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
)
from ..schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role, TokenUsage


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: SecretStr):
        self._client = AsyncAnthropic(api_key=api_key.get_secret_value())

    # ------------------------------------------------------------
    # Helper: Anthropic trata 'system' aparte, NO como un mensaje más
    # ------------------------------------------------------------
    def _split_system_and_messages(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> tuple[str | None, list[dict]]:
        system = config.system_prompt
        chat_messages: list[dict] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system = f"{system}\n\n{m.content}" if system else m.content
            else:
                chat_messages.append(m.to_dict())
        return system, chat_messages

    def _build_kwargs(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> dict:
        """
        Construye kwargs compatibles con el SDK moderno.
        En SDK >= 2.5.0 ya NO se envían temperature ni top_p.
        """
        system, chat_messages = self._split_system_and_messages(messages, config)
        kwargs: dict = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": chat_messages,
        }
        if system:
            kwargs["system"] = system
        return kwargs

    # ------------------------------------------------------------
    # Modo normal
    # ------------------------------------------------------------
    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> ModelResponse:
        kwargs = self._build_kwargs(messages, config)

        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMAuthError(f"Anthropic auth failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(f"Anthropic connection error: {e}") from e
        except anthropic.APIError as e:
            raise LLMProviderError(f"Anthropic API error: {e}") from e

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return ModelResponse(
            provider=Provider.ANTHROPIC,
            model=response.model,
            content=text,
            finish_reason=response.stop_reason,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )

    # ------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------
    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(messages, config)

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AuthenticationError as e:
            raise LLMAuthError(f"Anthropic auth failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMConnectionError(f"Anthropic connection error: {e}") from e
        except anthropic.APIError as e:
            raise LLMProviderError(f"Anthropic stream error: {e}") from e