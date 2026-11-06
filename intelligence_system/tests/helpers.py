"""Dobles de prueba: LLM guionado por rol y constructores de contenedores en memoria."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from intelligence_system.config import Settings
from intelligence_system.container import AppContainer, assemble
from intelligence_system.llm.providers.fake_client import FakeLLMClient
from intelligence_system.llm.schemas import (
    ChatMessage,
    ModelConfig,
    ModelResponse,
    Provider,
    ToolCall,
    ToolSpec,
)
from intelligence_system.rag.embeddings import HashEmbeddings
from intelligence_system.rag.retriever import NoopReranker
from intelligence_system.rag.sparse import InMemoryBM25StatsStore
from intelligence_system.rag.vector_store import InMemoryVectorStore

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"
DIM = 256


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "_env_file": None,
        "app_env": "test",
        "api_key": "test-key",
        "llm_provider": "fake",
        "embeddings_provider": "fake",
        "vector_store": "memory",
        "checkpointer_backend": "memory",
        "phoenix_enabled": False,
        "embeddings_dimension": DIM,
        "rag_corpus_dir": CORPUS_DIR,
        "log_json": True,
        "log_level": "WARNING",
        "anthropic_api_key": None,
        "openai_api_key": None,
        "pinecone_api_key": None,
    }
    values.update(overrides)
    return Settings(**values)


def response(content: str = "", tool_calls: list[ToolCall] | None = None) -> ModelResponse:
    return ModelResponse(provider=Provider.FAKE, model="fake", content=content, tool_calls=tool_calls or [])


def tool_call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)


Handler = Callable[[Sequence[ChatMessage], ModelConfig, Sequence[ToolSpec] | None], ModelResponse]


class ScriptedLLM:
    """Responder por rol: supervisor (lista de decisiones), grader, analyst y writer."""

    def __init__(
        self,
        *,
        routes: list[dict[str, str]] | None = None,
        grade: list[int] | Callable[[str], list[int]] | None = None,
        analyst: list[ModelResponse] | None = None,
        writer: str | Callable[[str], str] = "Respuesta con cita [1].",
    ) -> None:
        self.routes = list(routes or [])
        self.grade = grade if grade is not None else [1, 2]
        self.analyst = list(analyst or [])
        self.writer = writer
        self.prompts: dict[str, list[str]] = {"supervisor": [], "grader": [], "analyst": [], "writer": []}

    def __call__(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> ModelResponse:
        last = messages[-1].content
        if tool_choice == "supervisor_decision":
            self.prompts["supervisor"].append(last)
            decision = self.routes.pop(0) if self.routes else {"next": "writer", "reason": "fin", "task": "x"}
            return response(tool_calls=[tool_call("supervisor_decision", **decision)])
        if tool_choice == "evidence_grade":
            self.prompts["grader"].append(last)
            indices = self.grade(last) if callable(self.grade) else self.grade
            return response(
                tool_calls=[tool_call("evidence_grade", relevant_indices=indices, rationale="ok")]
            )
        if tools:
            self.prompts["analyst"].append(last)
            return self.analyst.pop(0) if self.analyst else response("análisis terminado")
        self.prompts["writer"].append(last)
        return response(self.writer(last) if callable(self.writer) else self.writer)


def build_container(
    settings: Settings | None = None,
    *,
    responder: Any = None,
    checkpointer: Any = None,
) -> AppContainer:
    settings = settings or make_settings()
    llm = FakeLLMClient(responder or ScriptedLLM())
    return assemble(
        AppContainer(settings=settings),
        llm=llm,
        embeddings=HashEmbeddings(dimension=settings.embeddings_dimension),
        store=InMemoryVectorStore(dimension=settings.embeddings_dimension),
        stats=InMemoryBM25StatsStore(),
        reranker=NoopReranker(),
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
