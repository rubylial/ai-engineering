"""Cliente LLM async unificado (evolución de la pre-entrega 1)."""

from intelligence_system.llm.base import BaseLLMClient, RetryPolicy
from intelligence_system.llm.exceptions import (
    LLMAuthError,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServerError,
    LLMStructuredOutputError,
    LLMTransientError,
)
from intelligence_system.llm.factory import LLMClientFactory
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

__all__ = [
    "BaseLLMClient",
    "ChatMessage",
    "LLMAuthError",
    "LLMClientFactory",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMServerError",
    "LLMStructuredOutputError",
    "LLMTransientError",
    "ModelConfig",
    "ModelResponse",
    "Provider",
    "RetryPolicy",
    "Role",
    "TokenUsage",
    "ToolCall",
    "ToolSpec",
]
