"""
Esquemas Pydantic — estructuras de datos validadas para el cliente LLM.

Tener esto centralizado evita el 'error de diccionarios anidados' típico
de principiantes y te da autocompletado + validación en tiempo de ejecución.
"""

from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr


# ============================================================
# ENUMS
# ============================================================

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# ============================================================
# MENSAJES
# ============================================================

class ChatMessage(BaseModel):
    """Un turno de conversación. Compatible con ambos SDKs tras adaptación."""
    role: Role
    content: str = Field(..., min_length=1)

    def to_dict(self) -> dict:
        """Serialización al formato esperado por los SDKs (role/content planos)."""
        return {"role": self.role.value, "content": self.content}


# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

class ModelConfig(BaseModel):
    """
    Parámetros de inferencia validados.
    - temperature: 0.0 (determinista) a 2.0 (muy creativo)
    - max_tokens: tope duro de tokens de salida
    - top_p: nucleus sampling
    """
    model_config = ConfigDict(extra="forbid")  # rechaza campos desconocidos

    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=200_000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    system_prompt: str | None = None


# ============================================================
# RESPUESTAS
# ============================================================

class TokenUsage(BaseModel):
    """Contabilidad de tokens — útil para costos y observabilidad."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    """Respuesta normalizada entre proveedores."""
    provider: Provider
    model: str
    content: str
    usage: TokenUsage
    finish_reason: str | None = None
