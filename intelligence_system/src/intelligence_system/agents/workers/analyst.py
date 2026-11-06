"""Analyst: ciclo ReAct (razonar → llamar tool → observar) sobre el cliente unificado.

Es el patrón de la pre-entrega 5 implementado sobre `BaseLLMClient` (tool
calling neutral entre proveedores):
- el LLM decide qué tool de cálculo usar y con qué argumentos;
- los argumentos se validan con el `args_schema` Pydantic de cada tool;
- un error de validación vuelve al LLM como observación para que corrija
  (recovery loop), no como excepción;
- tope duro de llamadas (`AGENT_ANALYST_MAX_TOOL_CALLS`).
Cada resultado exitoso se guarda como evidencia citable de tipo `tool`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from intelligence_system.agents.prompts import ANALYST_SYSTEM, ANALYST_USER
from intelligence_system.agents.runtime import AgentDeps, NodeOutcome
from intelligence_system.agents.state import (
    AgentState,
    EvidenceItem,
    StepStatus,
    last_user_message,
    merge_evidence,
)
from intelligence_system.llm.base import format_validation_error
from intelligence_system.llm.schemas import ChatMessage, ToolCall, ToolSpec
from intelligence_system.observability.logging import get_logger

log = get_logger(__name__)
_CONTEXT_DOCS = 6
_CONTEXT_CHARS = 500


def tool_specs(tools: list[BaseTool]) -> list[ToolSpec]:
    specs = []
    for tool in tools:
        schema = tool.args_schema
        input_schema = (
            schema.model_json_schema()
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else {"type": "object", "properties": {}}
        )
        specs.append(ToolSpec(name=tool.name, description=tool.description, input_schema=input_schema))
    return specs


async def execute_tool(tools: dict[str, BaseTool], call: ToolCall) -> tuple[str, bool]:
    """Ejecuta una tool y devuelve (payload JSON, ok). Los errores se devuelven como datos."""
    tool = tools.get(call.name)
    if tool is None:
        return json.dumps({"error": f"herramienta desconocida: {call.name}"}), False
    try:
        output: Any = await tool.ainvoke(call.arguments)
    except ValidationError as exc:
        detail = format_validation_error(exc)
        log.info("analyst.tool_invalid_args", tool=call.name, detail=detail)
        return json.dumps({"error": "argumentos inválidos", "detalle": detail}, ensure_ascii=False), False
    except Exception as exc:
        log.warning("analyst.tool_failed", tool=call.name, error=type(exc).__name__)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), False
    if isinstance(output, BaseModel):
        return output.model_dump_json(), True
    return json.dumps(output, ensure_ascii=False, default=str), True


def _evidence_for(call: ToolCall, payload: str) -> EvidenceItem:
    args = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(f"{call.name}:{args}".encode()).hexdigest()[:12]
    return EvidenceItem(
        id=f"tool:{call.name}:{digest}",
        kind="tool",
        title=f"cálculo · {call.name}",
        source=args,
        content=payload,
        score=None,
        worker="analyst",
    )


def _document_context(state: AgentState) -> str:
    docs = [e for e in state.get("evidence") or [] if e["kind"] == "document"][:_CONTEXT_DOCS]
    if not docs:
        return "(no hay evidencia documental todavía)"
    return "\n".join(f"- {e['title']}: {e['content'][:_CONTEXT_CHARS]}" for e in docs)


async def analyst_logic(state: AgentState, deps: AgentDeps) -> NodeOutcome:
    task = (state.get("current_task") or last_user_message(state))[:500]
    tools = {t.name: t for t in deps.domain_tools}
    specs = tool_specs(deps.domain_tools)
    config = deps.worker_config(ANALYST_SYSTEM)
    messages = [ChatMessage.user(ANALYST_USER.format(task=task, context=_document_context(state)))]
    max_calls = deps.settings.agent_analyst_max_tool_calls

    new_items: list[EvidenceItem] = []
    calls_made = 0
    summary = ""
    while True:
        response = await deps.llm.generate(messages, config, tools=specs)
        if not response.tool_calls or calls_made >= max_calls:
            summary = response.content or "sin resumen"
            break
        messages.append(ChatMessage.assistant(response.content, response.tool_calls))
        for call in response.tool_calls:
            if calls_made >= max_calls:
                payload, ok = json.dumps({"error": "límite de llamadas alcanzado"}), False
            else:
                calls_made += 1
                payload, ok = await execute_tool(tools, call)
            messages.append(ChatMessage.tool(call.id, payload))
            if ok:
                new_items.append(_evidence_for(call, payload))

    evidence = merge_evidence(list(state.get("evidence") or []), new_items)
    names = ", ".join(sorted({item["title"].removeprefix("cálculo · ") for item in new_items})) or "ninguno"
    return NodeOutcome(
        goto="supervisor",
        update={"evidence": evidence, "step_count": state.get("step_count", 0) + 1, "error": None},
        detail=f"{len(new_items)} cálculo(s) válidos [{names}] en {calls_made} llamada(s). {summary[:160]}",
    )


def analyst_fallback(state: AgentState, deps: AgentDeps, reason: str, status: StepStatus) -> NodeOutcome:
    del deps
    return NodeOutcome(
        goto="supervisor",
        update={"step_count": state.get("step_count", 0) + 1, "error": f"analyst: {reason}"},
        detail=f"sin cálculos ({reason})",
        status=status,
    )
