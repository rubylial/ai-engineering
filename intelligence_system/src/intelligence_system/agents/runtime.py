"""Runtime común de los nodos: timeout, trazas, logs, bitácora y contrato de escritura.

Cada nodo implementa solo su lógica y devuelve un `NodeOutcome`. Este módulo:
- corta el nodo con `asyncio.timeout(AGENT_NODE_TIMEOUT_S)`;
- abre el span `agent.<worker>` (con request_id, thread_id y worker);
- ante timeout o error aplica el fallback del nodo (degradación controlada);
- agrega el `StepRecord` a la bitácora `steps`;
- verifica que el nodo solo escriba su parcela del estado (`NODE_OWNERSHIP`);
- traduce el resultado a `Command(goto=...)` (handoff) o a un update simple.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.types import Command

from intelligence_system.agents.state import (
    NODE_OWNERSHIP,
    RUNTIME_KEYS,
    AgentState,
    StepRecord,
    StepStatus,
)
from intelligence_system.config import Settings
from intelligence_system.llm.base import BaseLLMClient
from intelligence_system.llm.schemas import ModelConfig
from intelligence_system.observability.logging import get_logger
from intelligence_system.observability.tracing import agent_span, mark_error

log = get_logger(__name__)


@dataclass
class AgentDeps:
    """Dependencias inyectadas al grafo (nada se crea adentro de los nodos)."""

    settings: Settings
    llm: BaseLLMClient
    search_tool: BaseTool
    domain_tools: list[BaseTool] = field(default_factory=list)

    def supervisor_config(self, system_prompt: str) -> ModelConfig:
        s = self.settings
        return ModelConfig(
            model=s.llm_model_supervisor,
            max_tokens=s.llm_max_tokens_supervisor,
            temperature=s.llm_temperature,
            system_prompt=system_prompt,
        )

    def worker_config(self, system_prompt: str, *, max_tokens: int | None = None) -> ModelConfig:
        s = self.settings
        return ModelConfig(
            model=s.llm_model_worker,
            max_tokens=max_tokens or s.llm_max_tokens_worker,
            temperature=s.llm_temperature,
            system_prompt=system_prompt,
        )


@dataclass
class NodeOutcome:
    update: dict[str, Any]
    detail: str
    goto: str | None = None
    status: StepStatus = "ok"


class OwnershipViolationError(RuntimeError):
    """Un nodo intentó escribir un campo del estado que no le pertenece."""


NodeLogic = Callable[[AgentState, AgentDeps], Awaitable[NodeOutcome]]
FallbackLogic = Callable[[AgentState, AgentDeps, str, StepStatus], NodeOutcome]


def thread_id_from(config: RunnableConfig | None) -> str | None:
    if not config:
        return None
    return (config.get("configurable") or {}).get("thread_id")


def check_ownership(node: str, update: dict[str, Any]) -> None:
    allowed = NODE_OWNERSHIP[node] | RUNTIME_KEYS
    illegal = set(update) - allowed
    if illegal:
        raise OwnershipViolationError(f"el nodo '{node}' no puede escribir {sorted(illegal)}")


def make_node(
    name: str,
    logic: NodeLogic,
    deps: AgentDeps,
    *,
    fallback: FallbackLogic | None,
) -> Callable[[AgentState, RunnableConfig], Awaitable[Command[Any] | dict[str, Any]]]:
    timeout_s = deps.settings.agent_node_timeout_s

    async def node(state: AgentState, config: RunnableConfig) -> Command[Any] | dict[str, Any]:
        step = len(state.get("steps") or []) + 1
        request_id = state.get("request_id")
        thread_id = thread_id_from(config)
        started = time.perf_counter()
        with (
            structlog.contextvars.bound_contextvars(worker=name, agent_step=step),
            agent_span(name, request_id=request_id, thread_id=thread_id, step=step) as span,
        ):
            try:
                async with asyncio.timeout(timeout_s):
                    outcome = await logic(state, deps)
            except TimeoutError as exc:
                mark_error(span, exc, description=f"timeout del nodo ({timeout_s}s)")
                if fallback is None:
                    raise
                log.warning("agent.node_timeout", timeout_s=timeout_s)
                outcome = fallback(state, deps, f"timeout tras {timeout_s:.0f}s", "timeout")
            except Exception as exc:
                mark_error(span, exc)
                if fallback is None:
                    raise
                log.warning("agent.node_error", error=type(exc).__name__, detail=str(exc)[:300])
                outcome = fallback(state, deps, f"{type(exc).__name__}: {exc}"[:300], "error")

            check_ownership(name, outcome.update)
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            record = StepRecord(
                step=step,
                node=name,
                status=outcome.status,
                detail=outcome.detail[:500],
                duration_ms=duration_ms,
            )
            span.set_attribute("agent.status", outcome.status)
            span.set_attribute("agent.detail", record["detail"])
            if outcome.goto:
                span.set_attribute("agent.goto", outcome.goto)
            log.info(
                "agent.node",
                status=outcome.status,
                goto=outcome.goto,
                duration_ms=duration_ms,
                detail=record["detail"][:200],
            )
        update = {**outcome.update, "steps": [*(state.get("steps") or []), record]}
        if outcome.goto:
            return Command(goto=outcome.goto, update=update)
        return update

    node.__name__ = name
    return node
