"""Factory del cliente LLM. Aislado para que los tests puedan mockearlo."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "claude-haiku-5-5"


def build_chat_model(model: str | None = None) -> BaseChatModel:
    """Devuelve un `ChatAnthropic` listo para usar.

    Nota: los modelos claude-5-* no aceptan `temperature` (devuelven HTTP 400), así
    que NO se pasa. `max_retries=0` evita reintentos del SDK; los que valen están en
    los nodos del grafo.
    """
    return ChatAnthropic(
        model=model or os.getenv("LLM_MODEL", DEFAULT_MODEL),
        max_retries=0,
    )
