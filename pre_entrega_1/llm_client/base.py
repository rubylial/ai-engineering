"""
Interfaz común (ABC) que todo proveedor LLM debe implementar.

Esta es la clave del patrón: el resto de la app programa contra BaseLLMClient,
no contra OpenAI o Anthropic directamente. Cambiar de proveedor = cambiar
una línea de configuración, no reescribir el 60% del código.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from .schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """Contrato que todo cliente concreto debe cumplir."""

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> ModelResponse:
        """
        Genera una respuesta completa (modo no-stream).
        Devuelve un ModelResponse normalizado — el llamador NO necesita
        saber qué proveedor atendió la llamada.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig,
    ) -> AsyncIterator[str]:
        """
        Generador asíncrono de tokens (modo streaming).

        Uso:
            async for token in client.stream(messages, config):
                print(token, end="", flush=True)

        Esto es lo que habilita la experiencia 'ChatGPT' — ver palabras
        apareciendo una a una en lugar de esperar 10s por el bloque entero.
        Reduce drásticamente el Time-To-First-Token (TTFT).
        """
        ...
        # NOTA: la implementación concreta usa `yield` dentro de un
        # `async for` sobre el stream del SDK. En la clase abstracta
        # el `...` alcanza porque el tipo de retorno declara AsyncIterator.
