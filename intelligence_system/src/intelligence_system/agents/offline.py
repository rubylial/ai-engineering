"""Responder heurístico para `LLM_PROVIDER=fake` (demo y CI sin API keys).

Permite levantar el sistema completo (API, grafo, checkpointer, Phoenix) sin
gastar tokens para validar el cableado. Imita el contrato de cada rol con
reglas simples sobre los prompts de `prompts.py`. NO es un modelo: las
respuestas son extractivas y la calidad no representa al sistema real.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from intelligence_system.agents.prompts import WRITER_SENTINEL
from intelligence_system.llm.schemas import (
    ChatMessage,
    ModelConfig,
    ModelResponse,
    Provider,
    Role,
    ToolCall,
    ToolSpec,
)
from intelligence_system.rag.sparse import tokenize

_DOMAIN = (
    "asyncio",
    "semaforo",
    "semaphore",
    "gather",
    "embedding",
    "coseno",
    "similitud",
    "lcel",
    "langchain",
    "runnable",
    "chroma",
    "pinecone",
    "namespace",
    "indice",
    "vector",
    "bm25",
    "hibrid",
    "hybrid",
    "rrf",
    "rag",
    "retriev",
    "chunk",
    "precision",
    "recall",
    "metrica",
    "golden",
    "dimension",
    "overlap",
    "token",
    "fusion",
    "rank",
    "semantic",
    "lexic",
    "busqueda",
)
_CALC = ("calcul", "cuant", "precision", "recall", "memoria", "ocupa", "almacen", "rrf", "fusion")
_FOLLOW_UP = ("y ", "eso", "esa", "ese", "lo anterior", "tambien", "entonces")


def _fold(text: str) -> str:
    return " ".join(tokenize(text))


def _section(text: str, header: str) -> str:
    match = re.search(rf"## {re.escape(header)}\n(.*?)(?:\n## |\Z)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _call(name: str, arguments: dict[str, Any], config: ModelConfig) -> ModelResponse:
    call = ToolCall(id=f"call_{uuid.uuid4().hex[:10]}", name=name, arguments=arguments)
    return ModelResponse(provider=Provider.FAKE, model=config.model, tool_calls=[call])


def _numbers(text: str) -> list[int]:
    cleaned = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)
    values = [int(n) for n in re.findall(r"\d+", cleaned)]
    if re.search(r"\bmill[oó]n(es)?\b", text.lower()) and values:
        values = [v * 1_000_000 if v < 1000 else v for v in values]
    return values


class OfflineResponder:
    def __call__(
        self,
        messages: Sequence[ChatMessage],
        config: ModelConfig,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> ModelResponse:
        last = messages[-1]
        if tool_choice == "supervisor_decision":
            return self._route(last.content, config)
        if tool_choice == "evidence_grade":
            return self._grade(last.content, config)
        if tools:
            return self._analyst(messages, config)
        return self._write(last.content, config)

    def _route(self, prompt: str, config: ModelConfig) -> ModelResponse:
        question = _section(prompt, "Pregunta actual")
        previous = re.findall(r"^Usuario: (.*)$", _section(prompt, "Historial reciente"), re.MULTILINE)
        lowered = question.lower().lstrip("¿¡ ")
        if previous and lowered.startswith(_FOLLOW_UP):
            # Imita la "reescritura autocontenida" que hace el LLM real con el historial.
            question = f"{previous[-1]} {question}"
        executed = re.search(r"Workers ya ejecutados en este turno: (.*)", prompt)
        done = executed.group(1) if executed else ""
        folded = _fold(question)
        in_domain = any(k in folded for k in _DOMAIN)
        needs_calc = any(k in folded for k in _CALC) and bool(re.search(r"\d", question))
        if in_domain and "researcher" not in done:
            nxt, reason = "researcher", "offline: falta evidencia documental"
        elif needs_calc and "analyst" not in done:
            nxt, reason = "analyst", "offline: la pregunta pide un cálculo"
        else:
            nxt, reason = "writer", "offline: redactar con lo reunido"
        args = {"next": nxt, "reason": reason, "task": question[:500] or "consulta"}
        return _call("supervisor_decision", args, config)

    def _grade(self, prompt: str, config: ModelConfig) -> ModelResponse:
        head, _, fragments = prompt.partition("Fragmentos:")
        query_terms = set(tokenize(head.replace("Consulta:", "")))
        needed = 1 if len(query_terms) <= 2 else 2
        relevant = []
        for match in re.finditer(r"^\[(\d+)\] (.*?)(?=^\[\d+\] |\Z)", fragments, re.MULTILINE | re.DOTALL):
            overlap = query_terms & set(tokenize(match.group(2)))
            if len(overlap) >= needed:
                relevant.append(int(match.group(1)))
        return _call("evidence_grade", {"relevant_indices": relevant, "rationale": "offline"}, config)

    def _analyst(self, messages: Sequence[ChatMessage], config: ModelConfig) -> ModelResponse:
        if messages[-1].role is Role.TOOL:
            return ModelResponse(
                provider=Provider.FAKE, model=config.model, content=f"Resultado: {messages[-1].content}"
            )
        prompt = messages[-1].content
        task = prompt.split("Evidencia documental")[0].lower()
        numbers = _numbers(task)
        lists = [
            [x.strip(" '\"") for x in group.split(",") if x.strip()]
            for group in re.findall(r"\[([^\]]+)\]", task)
        ]
        if "chunk" in task and numbers:
            size = re.search(r"(\d+)\s*tokens\s*con\s*(\d+)\s*de\s*overlap", prompt.lower())
            chunk_size, overlap = (int(size.group(1)), int(size.group(2))) if size else (500, 50)
            args = {"total_tokens": max(numbers), "chunk_size": chunk_size, "chunk_overlap": overlap}
            return _call("estimar_chunks", args, config)
        if ("precision" in task or "recall" in task) and len(lists) >= 2:
            k = next((n for n in numbers if 0 < n <= len(lists[0])), len(lists[0]))
            return _call(
                "calcular_metricas_retrieval",
                {"recuperados": lists[0], "relevantes": lists[1], "k": k},
                config,
            )
        if any(w in task for w in ("memoria", "ocupa", "almacen")) and len(numbers) >= 2:
            dimension = min(numbers)
            return _call(
                "estimar_memoria_vectores",
                {"num_vectores": max(numbers), "dimension": dimension},
                config,
            )
        return ModelResponse(
            provider=Provider.FAKE, model=config.model, content="No identifiqué un cálculo aplicable."
        )

    def _write(self, prompt: str, config: ModelConfig) -> ModelResponse:
        # La evidencia es la última sección del prompt y puede contener "## " propios.
        evidence = prompt.partition("## Evidencia\n")[2]
        blocks = re.findall(
            r"^\[(\d+)\] \((documento|herramienta)\) (.*?)\n(.*?)(?=^\[\d+\] |\Z)",
            evidence,
            re.MULTILINE | re.DOTALL,
        )
        if not blocks:
            return ModelResponse(provider=Provider.FAKE, model=config.model, content=WRITER_SENTINEL)
        lines = []
        for number, kind, _title, body in blocks[:3]:
            if kind == "herramienta":
                try:
                    payload = json.loads(body.strip())
                    summary = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:4])
                except json.JSONDecodeError:
                    summary = body.strip()[:200]
                lines.append(f"Resultado del cálculo: {summary} [{number}].")
            else:
                text = re.sub(r"^#+ .*$", "", body.strip(), flags=re.MULTILINE).strip()
                sentence = re.split(r"(?<=[.!?])\s", " ".join(text.split()))[0]
                lines.append(f"{sentence[:300]} [{number}]")
        return ModelResponse(provider=Provider.FAKE, model=config.model, content="\n\n".join(lines))
