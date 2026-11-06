"""Proveedor falso y determinista (sin red, sin costo).

Se usa en tests unitarios y en el modo offline (`LLM_PROVIDER=fake`) para
validar el cableado completo (API, grafo, checkpointer, trazas) sin gastar
tokens. NO es un modelo: responde lo que le dicta su `responder`.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from intelligence_system.llm.base import BaseLLMClient, RetryPolicy
from intelligence_system.llm.schemas import (
    ChatMessage,
    ModelConfig,
    ModelResponse,
    Provider,
    ToolSpec,
)

Responder = Callable[
    [Sequence[ChatMessage], ModelConfig, Sequence[ToolSpec] | None, str | None],
    ModelResponse | Awaitable[ModelResponse],
]


def echo_responder(
    messages: Sequence[ChatMessage],
    config: ModelConfig,
    tools: Sequence[ToolSpec] | None,
    tool_choice: str | None,
) -> ModelResponse:
    """Responder por defecto: repite el último mensaje del usuario."""
    del tools, tool_choice
    last_user = next((m.content for m in reversed(messages) if m.role.value == "user"), "")
    return ModelResponse(provider=Provider.FAKE, model=config.model, content=f"eco: {last_user}")


class FakeLLMClient(BaseLLMClient):
    provider = Provider.FAKE

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        stream_chunk_chars: int = 16,
        max_concurrency: int = 8,
    ) -> None:
        super().__init__(
            retry=RetryPolicy(max_attempts=1, min_wait_s=0.0, max_wait_s=0.0),
            max_concurrency=max_concurrency,
        )
        self.responder: Responder = responder or echo_responder
        self.stream_chunk_chars = stream_chunk_chars
        self.calls: list[dict[str, Any]] = []

    async def _generate(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "model": config.model,
                "tools": [t.name for t in tools or []],
                "tool_choice": tool_choice,
            }
        )
        result = self.responder(messages, config, tools, tool_choice)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _stream(self, messages: Sequence[ChatMessage], config: ModelConfig) -> AsyncIterator[str]:
        response = await self._generate(messages, config, None, None)
        text = response.content
        for start in range(0, len(text), self.stream_chunk_chars):
            yield text[start : start + self.stream_chunk_chars]

    async def check_model(self, model: str) -> None:
        return None
