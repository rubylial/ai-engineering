"""
Serialización de la traza ReAct a JSON.

Convierte los BaseMessage de LangChain (que no son JSON-serializables
directamente) en dicts limpios que dejan clara la secuencia:
    HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage(final)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def serialize_message(msg: BaseMessage) -> dict:
    """Convierte un BaseMessage en un dict JSON-serializable."""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)

    out: dict = {
        "type": msg.__class__.__name__,
        "content": content,
    }

    if isinstance(msg, AIMessage) and msg.tool_calls:
        # El AIMessage que invoca tools: guardamos los tool_calls
        out["tool_calls"] = [
            {
                "name": tc["name"],
                "args": tc["args"],
                "id": tc.get("id", ""),
            }
            for tc in msg.tool_calls
        ]

    if isinstance(msg, ToolMessage):
        # La "observación" que vuelve del ToolNode al agent
        out["tool_name"] = msg.name
        out["tool_call_id"] = msg.tool_call_id

    if isinstance(msg, HumanMessage):
        out["source"] = "user"

    return out


def dump_trace(
    messages: list[BaseMessage],
    out_path: Path,
    metadata: dict | None = None,
) -> Path:
    """
    Serializa la traza completa a un JSON legible y versionable.

    Args:
        messages: lista de mensajes del estado final del grafo.
        out_path: dónde escribir el JSON.
        metadata: info del run (thread_id, modelo, queries ejecutadas).

    Returns:
        El path donde se escribió el archivo.
    """
    trace = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_messages": len(messages),
        "metadata": metadata or {},
        "messages": [serialize_message(m) for m in messages],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    return out_path
