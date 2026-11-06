"""Factory del cliente LLM. Aislado para poder mockearlo en tests."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from app.config import get_settings

DEFAULT_MODEL = "claude-haiku-5-5"


def build_chat_model(model: str | None = None) -> BaseChatModel:
    """Devuelve un ChatAnthropic listo para usar.

    Los modelos claude-5-* NO aceptan `temperature` (devuelven HTTP 400), así que no
    se pasa. `max_retries=0`: los reintentos que valen están en el grafo.
    """
    settings = get_settings()
    return ChatAnthropic(
        model=model or settings.llm_model,
        api_key=settings.anthropic_api_key or None,  # type: ignore[arg-type]
        max_retries=0,
    )
