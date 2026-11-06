"""Writer: sintetiza la respuesta final con citas y la valida en código.

Política "sin cita no hay afirmación":
- Sin evidencia → responde que no sabe, SIN llamar al LLM.
- El LLM redacta con marcadores [n] (streaming de tokens vía `get_stream_writer`).
- Validación determinista: los marcadores deben apuntar a evidencia real;
  los inventados se eliminan. Si no queda ninguna cita válida, o el modelo
  declaró `SIN_EVIDENCIA`, la respuesta se reemplaza por la abstención.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer

from intelligence_system.agents.prompts import (
    ABSTENTION_ANSWER,
    WRITER_SENTINEL,
    WRITER_SYSTEM,
    WRITER_USER,
)
from intelligence_system.agents.runtime import AgentDeps, NodeOutcome
from intelligence_system.agents.state import (
    AgentState,
    CitationRecord,
    EvidenceItem,
    last_user_message,
)
from intelligence_system.agents.supervisor import format_history
from intelligence_system.llm.schemas import ChatMessage
from intelligence_system.observability.logging import get_logger

log = get_logger(__name__)
_MARKER = re.compile(r"\[(\d{1,3})\]")
_EVIDENCE_CHARS = 1800


def token_sink() -> Callable[[Any], None]:
    """Writer de streaming de LangGraph; no-op si se ejecuta fuera de un run del grafo."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _chunk: None


def format_evidence(evidence: list[EvidenceItem]) -> str:
    blocks = []
    for i, item in enumerate(evidence, start=1):
        kind = "documento" if item["kind"] == "document" else "herramienta"
        blocks.append(
            f"[{i}] ({kind}) {item['title']} — fuente: {item['source']}\n{item['content'][:_EVIDENCE_CHARS]}"
        )
    return "\n\n".join(blocks)


def finalize_answer(
    text: str, evidence: list[EvidenceItem], *, snippet_chars: int
) -> tuple[str, list[CitationRecord], bool]:
    """Valida marcadores contra la evidencia. Devuelve (respuesta, citas, grounded)."""
    cleaned = text.strip()
    if not cleaned or cleaned.startswith(WRITER_SENTINEL) or WRITER_SENTINEL in cleaned:
        return ABSTENTION_ANSWER, [], False
    markers = [int(m) for m in _MARKER.findall(cleaned)]
    valid = sorted({m for m in markers if 1 <= m <= len(evidence)})
    invalid = sorted({m for m in markers if not 1 <= m <= len(evidence)})
    if invalid:
        log.warning("writer.invalid_markers", markers=invalid)
        cleaned = _MARKER.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", cleaned).strip()
    if not valid:
        log.warning("writer.no_valid_citations")
        return ABSTENTION_ANSWER, [], False
    citations = [
        CitationRecord(
            marker=m,
            id=evidence[m - 1]["id"],
            kind=evidence[m - 1]["kind"],
            title=evidence[m - 1]["title"],
            source=evidence[m - 1]["source"],
            snippet=evidence[m - 1]["content"][:snippet_chars],
            score=evidence[m - 1]["score"],
        )
        for m in valid
    ]
    return cleaned, citations, True


def _answer_update(state: AgentState, answer: str, citations: list[CitationRecord], grounded: bool) -> dict:
    message_id = f"{state.get('request_id', 'turn')}-assistant"
    return {
        "answer": answer,
        "citations": citations,
        "grounded": grounded,
        "messages": [AIMessage(content=answer, name="writer", id=message_id)],
    }


async def writer_logic(state: AgentState, deps: AgentDeps) -> NodeOutcome:
    evidence = list(state.get("evidence") or [])
    if not evidence:
        return NodeOutcome(
            update=_answer_update(state, ABSTENTION_ANSWER, [], False),
            detail="sin evidencia: abstención sin llamar al LLM",
        )

    messages = [
        ChatMessage.user(
            WRITER_USER.format(
                history=format_history(state, deps.settings.agent_history_window),
                question=last_user_message(state),
                evidence=format_evidence(evidence),
            )
        )
    ]
    config = deps.worker_config(WRITER_SYSTEM, max_tokens=deps.settings.llm_max_tokens_writer)
    emit = token_sink()
    parts: list[str] = []
    pending = ""
    streaming = False
    async for token in deps.llm.stream(messages, config):
        parts.append(token)
        if streaming:
            emit({"type": "token", "text": token})
            continue
        # Se retiene el arranque hasta descartar que sea el centinela SIN_EVIDENCIA.
        pending += token
        head = pending.lstrip()
        if WRITER_SENTINEL.startswith(head) or head.startswith(WRITER_SENTINEL):
            continue
        streaming = True
        emit({"type": "token", "text": pending})

    answer, citations, grounded = finalize_answer(
        "".join(parts), evidence, snippet_chars=deps.settings.rag_snippet_chars
    )
    detail = f"{len(citations)} cita(s) válidas" if grounded else "abstención: sin citas válidas"
    return NodeOutcome(update=_answer_update(state, answer, citations, grounded), detail=detail)
