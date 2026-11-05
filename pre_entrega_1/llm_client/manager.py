"""
AsyncLLMManager — fachada única que la aplicación usa.

Responsabilidades:
- Leer configuración (incluyendo variables de entorno vía .env)
- Instanciar el proveedor correcto según LLM_PROVIDER
- Exponer generate() y stream() con la misma firma independientemente del proveedor
- Validar que exista la API key del proveedor seleccionado
"""

import os
from typing import AsyncIterator

from pydantic import BaseModel, SecretStr, model_validator

from .base import BaseLLMClient
from .exceptions import LLMConfigError
from .providers import AnthropicClient, OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider


# ============================================================
# Configuración del manager (leída de env o pasada explícita)
# ============================================================

class ManagerConfig(BaseModel):
    provider: Provider
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _check_key_for_selected_provider(self) -> "ManagerConfig":
        """Falla rápido si la key del proveedor elegido no está."""
        if self.provider == Provider.OPENAI and self.openai_api_key is None:
            raise LLMConfigError(
                "LLM_PROVIDER=openai pero falta OPENAI_API_KEY en el entorno."
            )
        if self.provider == Provider.ANTHROPIC and self.anthropic_api_key is None:
            raise LLMConfigError(
                "LLM_PROVIDER=anthropic pero falta ANTHROPIC_API_KEY en el entorno."
            )
        return self

    @classmethod
    def from_env(cls) -> "ManagerConfig":
        """
        Lee la configuración de variables de entorno.
        Espera: LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY.
        """
        raw_provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
        try:
            provider = Provider(raw_provider)
        except ValueError as e:
            raise LLMConfigError(
                f"LLM_PROVIDER inválido: '{raw_provider}'. "
                f"Valores válidos: {[p.value for p in Provider]}"
            ) from e

        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        return cls(
            provider=provider,
            openai_api_key=SecretStr(openai_key) if openai_key else None,
            anthropic_api_key=SecretStr(anthropic_key) if anthropic_key else None,
        )


# ============================================================
# Fachada principal
# ============================================================

class AsyncLLMManager:
    """
    Punto único de acceso a los LLMs. Toda la app debería interactuar
    con esta clase, nunca con OpenAIClient/AnthropicClient directamente.
    """

    def __init__(self, config: ManagerConfig):
        self._config = config
        self._client: BaseLLMClient = self._build_client(config)

    @staticmethod
    def _build_client(config: ManagerConfig) -> BaseLLMClient:
        if config.provider == Provider.OPENAI:
            # El validator ya garantizó que la key existe
            assert config.openai_api_key is not None
            return OpenAIClient(api_key=config.openai_api_key)
        if config.provider == Provider.ANTHROPIC:
            assert config.anthropic_api_key is not None
            return AnthropicClient(api_key=config.anthropic_api_key)
        raise LLMConfigError(f"Provider no soportado: {config.provider}")

    @classmethod
    def from_env(cls) -> "AsyncLLMManager":
        """Atajo: construir el manager leyendo todo del entorno."""
        return cls(ManagerConfig.from_env())

    @property
    def provider(self) -> Provider:
        return self._config.provider

    # ------------------------------------------------------------
    # API pública — espejo de BaseLLMClient
    # ------------------------------------------------------------
    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> ModelResponse:
        return await self._client.generate(messages, config)

    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> AsyncIterator[str]:
        """
        Devuelve un generador asíncrono. Uso:
            async for token in manager.stream(msgs, cfg):
                print(token, end="", flush=True)
        """
        async for token in self._client.stream(messages, config):
            yield token
