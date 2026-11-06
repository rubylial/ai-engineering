from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import anthropic
import openai
import pytest
from pydantic import BaseModel, Field

from helpers import make_settings, response, tool_call
from intelligence_system.llm import (
    ChatMessage,
    LLMAuthError,
    LLMClientFactory,
    LLMConfigError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
    LLMStructuredOutputError,
    ModelConfig,
    ModelResponse,
    ToolSpec,
)
from intelligence_system.llm.base import BaseLLMClient, RetryPolicy, structured_tool_name
from intelligence_system.llm.providers.anthropic_client import AnthropicClient, map_anthropic_error
from intelligence_system.llm.providers.fake_client import FakeLLMClient
from intelligence_system.llm.providers.openai_client import OpenAIClient, map_openai_error

CFG = ModelConfig(model="m", max_tokens=100, system_prompt="sistema")
FAST = RetryPolicy(max_attempts=3, min_wait_s=0.0, max_wait_s=0.0)


class Decision(BaseModel):
    """Decisión de prueba."""

    next: str = Field(pattern="^(a|b)$")
    score: int = Field(ge=0, le=10)


class FlakyClient(BaseLLMClient):
    """Falla con los errores indicados antes de responder."""

    provider = FakeLLMClient.provider

    def __init__(self, errors: list[Exception], stream_tokens: list[str] | None = None) -> None:
        super().__init__(retry=FAST, max_concurrency=2)
        self.errors = list(errors)
        self.calls = 0
        self.stream_tokens = stream_tokens or ["ho", "la"]
        self.fail_after_first_token = False

    async def _generate(self, messages: Any, config: Any, tools: Any, tool_choice: Any) -> ModelResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return response("ok")

    async def _stream(self, messages: Any, config: Any) -> AsyncIterator[str]:
        self.calls += 1
        if self.errors and not self.fail_after_first_token:
            raise self.errors.pop(0)
        for i, token in enumerate(self.stream_tokens):
            yield token
            if i == 0 and self.fail_after_first_token:
                raise LLMConnectionError("cortado")

    async def check_model(self, model: str) -> None:
        return None


# --------------------------------------------------------------------------- base


async def test_generate_retries_transient_errors() -> None:
    client = FlakyClient([LLMRateLimitError("429"), LLMServerError("503")])
    result = await client.generate([ChatMessage.user("hola")], CFG)
    assert result.content == "ok"
    assert client.calls == 3


async def test_generate_does_not_retry_logic_errors() -> None:
    client = FlakyClient([LLMAuthError("401")])
    with pytest.raises(LLMAuthError):
        await client.generate([ChatMessage.user("hola")], CFG)
    assert client.calls == 1


async def test_generate_gives_up_after_max_attempts() -> None:
    client = FlakyClient([LLMConnectionError("x")] * 5)
    with pytest.raises(LLMConnectionError):
        await client.generate([ChatMessage.user("hola")], CFG)
    assert client.calls == FAST.max_attempts


async def test_stream_retries_only_before_first_token() -> None:
    client = FlakyClient([LLMConnectionError("x")])
    tokens = [t async for t in client.stream([ChatMessage.user("hola")], CFG)]
    assert tokens == ["ho", "la"]
    assert client.calls == 2

    broken = FlakyClient([])
    broken.fail_after_first_token = True
    received: list[str] = []

    async def consume() -> None:
        async for token in broken.stream([ChatMessage.user("hola")], CFG):
            received.append(token)

    with pytest.raises(LLMConnectionError):
        await consume()
    assert received == ["ho"]
    assert broken.calls == 1  # no se reintenta: el cliente ya vio tokens


async def test_structured_output_repair_loop_sends_validator_error() -> None:
    seen: list[list[ChatMessage]] = []
    answers = [
        response(tool_calls=[tool_call("decision", next="c", score=99)]),
        response(tool_calls=[tool_call("decision", next="a", score=7)]),
    ]

    def responder(messages: Sequence[ChatMessage], config: Any, tools: Any, choice: Any) -> ModelResponse:
        seen.append(list(messages))
        assert choice == "decision"
        return answers.pop(0)

    client = FakeLLMClient(responder)
    result = await client.generate_structured([ChatMessage.user("decidí")], CFG, Decision, max_attempts=2)
    assert result == Decision(next="a", score=7)
    repair = seen[1][-1]
    assert repair.role.value == "tool"
    assert "ERROR DE VALIDACIÓN" in repair.content
    assert "next" in repair.content and "score" in repair.content


async def test_structured_output_fails_after_attempts() -> None:
    client = FakeLLMClient(lambda *_: response("no uso tools"))
    with pytest.raises(LLMStructuredOutputError):
        await client.generate_structured([ChatMessage.user("x")], CFG, Decision, max_attempts=2)
    assert len(client.calls) == 2


def test_structured_tool_name() -> None:
    assert structured_tool_name(Decision) == "decision"

    class SupervisorDecision(BaseModel):
        x: int

    assert structured_tool_name(SupervisorDecision) == "supervisor_decision"


def test_chat_message_shape_validation() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        ChatMessage(role="tool", content="x")
    with pytest.raises(ValueError, match="vacío"):
        ChatMessage.user("   ")
    with pytest.raises(ValueError, match="assistant"):
        ChatMessage.assistant("")


# --------------------------------------------------------------------------- anthropic


class _FakeAnthropicStream:
    def __init__(self, tokens: list[str]) -> None:
        self.text_stream = self._gen(tokens)

    @staticmethod
    async def _gen(tokens: list[str]) -> AsyncIterator[str]:
        for token in tokens:
            yield token

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeAnthropicSDK:
    def __init__(self, message: Any = None, tokens: list[str] | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self._message = message
        self._tokens = tokens or []
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)
        self.models = SimpleNamespace(retrieve=self._retrieve)

    async def _create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self._message

    def _stream(self, **kwargs: Any) -> _FakeAnthropicStream:
        self.kwargs = kwargs
        return _FakeAnthropicStream(self._tokens)

    async def _retrieve(self, model: str) -> Any:
        return SimpleNamespace(id=model)

    async def close(self) -> None:
        return None


def _anthropic_message() -> Any:
    return SimpleNamespace(
        model="claude-haiku-5-5",
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=12, output_tokens=5),
        content=[
            SimpleNamespace(type="text", text="pienso "),
            SimpleNamespace(type="tool_use", id="tu_1", name="estimar_chunks", input={"total_tokens": 10}),
        ],
    )


def test_anthropic_wire_format_splits_system_and_merges_tool_results() -> None:
    call = tool_call("calc", x=1)
    messages = [
        ChatMessage.system("extra"),
        ChatMessage.user("hola"),
        ChatMessage.assistant("uso tool", [call]),
        ChatMessage.tool(call.id, '{"y": 2}'),
        ChatMessage.user("seguí"),
    ]
    system, wire = AnthropicClient.to_wire(messages, CFG)
    assert system == "sistema\n\nextra"
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert wire[1]["content"][1] == {"type": "tool_use", "id": call.id, "name": "calc", "input": {"x": 1}}
    assert wire[2]["content"][0]["type"] == "tool_result"  # tool_result primero, luego el texto
    assert wire[2]["content"][1] == {"type": "text", "text": "seguí"}


async def test_anthropic_generate_parses_tools_and_never_sends_temperature() -> None:
    sdk = _FakeAnthropicSDK(message=_anthropic_message())
    client = AnthropicClient(
        make_settings(anthropic_api_key="x").anthropic_api_key,  # type: ignore[arg-type]
        timeout_s=5,
        retry=FAST,
        max_concurrency=1,
        client=sdk,  # type: ignore[arg-type]
    )
    cfg = ModelConfig(model="claude-haiku-5-5", temperature=0.2)
    spec = ToolSpec(name="estimar_chunks", description="d", input_schema={"type": "object"})
    result = await client.generate(
        [ChatMessage.user("hola")], cfg, tools=[spec], tool_choice="estimar_chunks"
    )
    assert result.content == "pienso "
    assert result.tool_calls[0].arguments == {"total_tokens": 10}
    assert result.usage.total == 17
    assert "temperature" not in sdk.kwargs and "top_p" not in sdk.kwargs
    assert sdk.kwargs["tool_choice"] == {"type": "tool", "name": "estimar_chunks"}


async def test_anthropic_stream_and_check_model() -> None:
    sdk = _FakeAnthropicSDK(tokens=["a", "", "b"])
    client = AnthropicClient(
        make_settings(anthropic_api_key="x").anthropic_api_key,  # type: ignore[arg-type]
        timeout_s=5,
        retry=FAST,
        max_concurrency=1,
        client=sdk,  # type: ignore[arg-type]
    )
    assert [t async for t in client.stream([ChatMessage.user("hola")], CFG)] == ["a", "b"]
    await client.check_model("claude-haiku-5-5")


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (anthropic.AuthenticationError, LLMAuthError),
        (anthropic.RateLimitError, LLMRateLimitError),
        (anthropic.APIConnectionError, LLMConnectionError),
        (anthropic.APITimeoutError, LLMConnectionError),
        (anthropic.OverloadedError, LLMServerError),
        (anthropic.InternalServerError, LLMServerError),
        (anthropic.NotFoundError, LLMConfigError),
        (anthropic.BadRequestError, LLMProviderError),
    ],
)
def test_anthropic_error_mapping(sdk_error: type[Exception], expected: type[Exception]) -> None:
    assert isinstance(map_anthropic_error(sdk_error.__new__(sdk_error)), expected)


# --------------------------------------------------------------------------- openai


def test_openai_wire_format() -> None:
    call = tool_call("calc", x=1)
    wire = OpenAIClient.to_wire(
        [ChatMessage.user("hola"), ChatMessage.assistant("", [call]), ChatMessage.tool(call.id, "{}")], CFG
    )
    assert wire[0] == {"role": "system", "content": "sistema"}
    assert wire[2]["tool_calls"][0]["function"] == {"name": "calc", "arguments": json.dumps({"x": 1})}
    assert wire[3] == {"role": "tool", "tool_call_id": call.id, "content": "{}"}


async def test_openai_generate_parses_tool_calls() -> None:
    captured: dict[str, Any] = {}

    async def create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        fn = SimpleNamespace(name="calc", arguments='{"x": 3}')
        message = SimpleNamespace(
            content=None, tool_calls=[SimpleNamespace(id="c1", type="function", function=fn)]
        )
        return SimpleNamespace(
            model="gpt",
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = OpenAIClient(
        make_settings(openai_api_key="x").openai_api_key,  # type: ignore[arg-type]
        timeout_s=5,
        retry=FAST,
        max_concurrency=1,
        client=sdk,  # type: ignore[arg-type]
    )
    cfg = ModelConfig(model="gpt", temperature=0.1)
    result = await client.generate([ChatMessage.user("hola")], cfg)
    assert result.tool_calls[0].arguments == {"x": 3}
    assert captured["temperature"] == 0.1
    assert captured["max_completion_tokens"] == cfg.max_tokens


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (openai.AuthenticationError, LLMAuthError),
        (openai.RateLimitError, LLMRateLimitError),
        (openai.APIConnectionError, LLMConnectionError),
        (openai.InternalServerError, LLMServerError),
    ],
)
def test_openai_error_mapping(sdk_error: type[Exception], expected: type[Exception]) -> None:
    assert isinstance(map_openai_error(sdk_error.__new__(sdk_error)), expected)


# --------------------------------------------------------------------------- factory


def test_factory_builds_each_provider() -> None:
    assert isinstance(LLMClientFactory.create(make_settings(llm_provider="fake")), FakeLLMClient)
    anthropic_client = LLMClientFactory.create(make_settings(llm_provider="anthropic", anthropic_api_key="k"))
    assert isinstance(anthropic_client, AnthropicClient)
    openai_client = LLMClientFactory.create(make_settings(llm_provider="openai", openai_api_key="k"))
    assert isinstance(openai_client, OpenAIClient)


def test_factory_fails_fast_without_key() -> None:
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        LLMClientFactory.create(make_settings(llm_provider="anthropic"))
