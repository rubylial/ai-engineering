"""Factory de clientes LLM (patrón Factory de la pre-entrega 1).

Cambiar de proveedor = cambiar `LLM_PROVIDER` en `.env`; ningún otro módulo
importa un SDK de LLM. Agregar un proveedor nuevo = registrar un builder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from intelligence_system.config import Settings
from intelligence_system.llm.base import BaseLLMClient, RetryPolicy
from intelligence_system.llm.exceptions import LLMConfigError
from intelligence_system.llm.providers.fake_client import FakeLLMClient
from intelligence_system.llm.schemas import Provider
from intelligence_system.observability.logging import get_logger

log = get_logger(__name__)
Builder = Callable[[Settings, RetryPolicy], BaseLLMClient]


def _build_anthropic(settings: Settings, retry: RetryPolicy) -> BaseLLMClient:
    from intelligence_system.llm.providers.anthropic_client import AnthropicClient

    if settings.anthropic_api_key is None:
        raise LLMConfigError("LLM_PROVIDER=anthropic pero falta ANTHROPIC_API_KEY")
    if settings.llm_temperature is not None:
        log.warning("llm.temperature_ignored", reason="anthropic>=2.5.0 no acepta temperature")
    return AnthropicClient(
        settings.anthropic_api_key,
        timeout_s=settings.llm_timeout_s,
        retry=retry,
        max_concurrency=settings.llm_max_concurrency,
    )


def _build_openai(settings: Settings, retry: RetryPolicy) -> BaseLLMClient:
    from intelligence_system.llm.providers.openai_client import OpenAIClient

    if settings.openai_api_key is None:
        raise LLMConfigError("LLM_PROVIDER=openai pero falta OPENAI_API_KEY")
    return OpenAIClient(
        settings.openai_api_key,
        timeout_s=settings.llm_timeout_s,
        retry=retry,
        max_concurrency=settings.llm_max_concurrency,
    )


def _build_fake(settings: Settings, retry: RetryPolicy) -> BaseLLMClient:
    del retry
    return FakeLLMClient(max_concurrency=settings.llm_max_concurrency)


class LLMClientFactory:
    """Registro proveedor -> builder."""

    _builders: ClassVar[dict[Provider, Builder]] = {
        Provider.ANTHROPIC: _build_anthropic,
        Provider.OPENAI: _build_openai,
        Provider.FAKE: _build_fake,
    }

    @classmethod
    def register(cls, provider: Provider, builder: Builder) -> None:
        cls._builders[provider] = builder

    @classmethod
    def create(cls, settings: Settings) -> BaseLLMClient:
        try:
            provider = Provider(settings.llm_provider)
        except ValueError as exc:
            raise LLMConfigError(f"LLM_PROVIDER inválido: {settings.llm_provider!r}") from exc
        retry = RetryPolicy(
            max_attempts=settings.llm_max_retries,
            min_wait_s=settings.llm_retry_min_wait_s,
            max_wait_s=settings.llm_retry_max_wait_s,
        )
        client = cls._builders[provider](settings, retry)
        log.info("llm.client_created", provider=provider.value)
        return client
