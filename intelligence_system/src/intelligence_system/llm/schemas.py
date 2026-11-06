"""Esquemas Pydantic del cliente LLM unificado (migrados de la pre-entrega 1).

Cambios respecto a la pre-entrega 1:
- `ChatMessage` ahora soporta tool calling (rol `tool`, `tool_calls`).
- `ModelConfig.temperature` es opcional y se eliminó `top_p`: el SDK
  anthropic 2.5.0 ya no acepta esos parámetros.
- `ToolSpec` describe una herramienta en formato neutral; cada proveedor la
  traduce a su formato de wire.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    FAKE = "fake"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Una invocación de herramienta pedida por el modelo."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """Un turno de conversación, neutral respecto del proveedor."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> ChatMessage:
        if self.role is Role.TOOL and not self.tool_call_id:
            raise ValueError("un mensaje con rol 'tool' necesita tool_call_id")
        if self.role in (Role.SYSTEM, Role.USER) and not self.content.strip():
            raise ValueError(f"un mensaje '{self.role}' no puede estar vacío")
        if self.role is Role.ASSISTANT and not (self.content or self.tool_calls):
            raise ValueError("un mensaje 'assistant' necesita contenido o tool_calls")
        return self

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> ChatMessage:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> ChatMessage:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)


class ToolSpec(BaseModel):
    """Definición neutral de una herramienta (nombre + descripción + JSON schema)."""

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str
    input_schema: dict[str, Any]

    @classmethod
    def from_model(cls, name: str, description: str, model: type[BaseModel]) -> ToolSpec:
        return cls(name=name, description=description, input_schema=model.model_json_schema())


class ModelConfig(BaseModel):
    """Parámetros de inferencia validados (extra=forbid: un typo explota en dev)."""

    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = Field(default=1024, gt=0, le=200_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    system_prompt: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    """Respuesta normalizada entre proveedores."""

    provider: Provider
    model: str
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
