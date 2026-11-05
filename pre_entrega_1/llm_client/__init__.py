"""Unified Async LLM Client — OpenAI + Anthropic bajo una interfaz común."""

from .manager import AsyncLLMManager, ManagerConfig
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role, TokenUsage
from .exceptions import (
    LLMError,
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
)

__all__ = [
    "AsyncLLMManager",
    "ManagerConfig",
    "ChatMessage",
    "ModelConfig",
    "ModelResponse",
    "Provider",
    "Role",
    "TokenUsage",
    "LLMError",
    "LLMAuthError",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMProviderError",
    "LLMRateLimitError",
]
