"""Jerarquía de excepciones del cliente unificado (pre-entrega 1, extendida).

`LLMTransientError` agrupa los fallos que tiene sentido reintentar (429, red,
5xx). Los errores lógicos (auth, request inválido, salida estructurada que no
valida) NO se reintentan a ciegas: reintentarlos solo gasta créditos.
"""


class LLMError(Exception):
    """Base: capturala para atrapar cualquier fallo del cliente LLM."""


class LLMConfigError(LLMError):
    """Configuración inválida (falta API key, proveedor o modelo inexistente)."""


class LLMAuthError(LLMError):
    """API key inválida o sin permisos (401/403)."""


class LLMProviderError(LLMError):
    """Error no transitorio del proveedor (request inválido, 4xx, respuesta rara)."""


class LLMTransientError(LLMError):
    """Base de los errores reintentables."""


class LLMRateLimitError(LLMTransientError):
    """Rate limit (429)."""


class LLMConnectionError(LLMTransientError):
    """Timeout o red caída."""


class LLMServerError(LLMTransientError):
    """5xx / sobrecarga del proveedor."""


class LLMStructuredOutputError(LLMError):
    """El modelo no produjo una salida que valide contra el esquema pedido."""
