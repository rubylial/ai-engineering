"""
Jerarquía de excepciones del cliente unificado.

El resto de la app solo necesita capturar LLMError para atrapar cualquier
fallo del proveedor. Para lógica de retry/fallback, se capturan las
subclases específicas (ej: solo reintentar en LLMRateLimitError).
"""


class LLMError(Exception):
    """Clase base — capturala para atrapar cualquier error del cliente."""


class LLMConfigError(LLMError):
    """Configuración inválida (falta API key, provider inexistente, etc.)."""


class LLMAuthError(LLMError):
    """API key inválida o sin permisos (401/403)."""


class LLMRateLimitError(LLMError):
    """Rate limit golpeado (429). Candidato natural para retry con backoff."""


class LLMConnectionError(LLMError):
    """Error de red: timeout, DNS, conexión caída, etc."""


class LLMProviderError(LLMError):
    """Error genérico del proveedor (5xx, respuesta malformada, etc.)."""
