"""Researcher: busca en la base de conocimiento y filtra la evidencia relevante.

1. Invoca la tool `search_knowledge_base` (args validados por Pydantic).
2. Un grader LLM (salida estructurada) marca qué fragmentos responden de
   verdad la consulta: los índices inventados se descartan en código.
3. Solo la evidencia aprobada pasa al writer. Si no queda nada, el writer
   responderá que no sabe (anti-alucinación).
"""

from __future__ import annotations

from intelligence_system.agents.prompts import GRADER_SYSTEM, GRADER_USER
from intelligence_system.agents.runtime import AgentDeps, NodeOutcome
from intelligence_system.agents.schemas import EvidenceGrade
from intelligence_system.agents.state import (
    AgentState,
    EvidenceItem,
    StepStatus,
    last_user_message,
    merge_evidence,
)
from intelligence_system.agents.tools.knowledge import SearchKnowledgeBaseOutput
from intelligence_system.llm.exceptions import LLMError
from intelligence_system.llm.schemas import ChatMessage
from intelligence_system.observability.logging import get_logger
from intelligence_system.rag.schemas import RetrievalResult

log = get_logger(__name__)
_GRADER_FRAGMENT_CHARS = 700


def to_evidence(result: RetrievalResult) -> EvidenceItem:
    return EvidenceItem(
        id=result.chunk_id,
        kind="document",
        title=f"{result.title} › {result.section}" if result.section != result.title else result.title,
        source=result.source,
        content=result.text,
        score=result.rerank_score if result.rerank_score is not None else round(result.score, 4),
        worker="researcher",
    )


async def grade_results(
    query: str, results: list[RetrievalResult], deps: AgentDeps
) -> tuple[list[RetrievalResult], str]:
    fragments = "\n\n".join(
        f"[{i}] {r.title} › {r.section}\n{r.text[:_GRADER_FRAGMENT_CHARS]}"
        for i, r in enumerate(results, start=1)
    )
    try:
        grade = await deps.llm.generate_structured(
            [ChatMessage.user(GRADER_USER.format(query=query, fragments=fragments))],
            deps.worker_config(GRADER_SYSTEM),
            EvidenceGrade,
            max_attempts=deps.settings.llm_structured_max_attempts,
        )
    except LLMError as exc:
        log.warning("researcher.grader_failed", error=type(exc).__name__)
        return results, "grader no disponible: se conservan todos los fragmentos"
    valid = sorted({i for i in grade.relevant_indices if 1 <= i <= len(results)})
    invented = sorted(set(grade.relevant_indices) - set(valid))
    if invented:
        log.warning("researcher.grader_invented_indices", indices=invented)
    return [results[i - 1] for i in valid], grade.rationale


async def researcher_logic(state: AgentState, deps: AgentDeps) -> NodeOutcome:
    task = (state.get("current_task") or last_user_message(state))[:500]
    filters = state.get("filters") or {}
    args = {
        "query": task,
        "top_k": deps.settings.rag_top_k,
        "doc_ids": filters.get("doc_ids"),
        "categories": filters.get("categories"),
    }
    output: SearchKnowledgeBaseOutput = await deps.search_tool.ainvoke(args)
    candidates = list(output.results)
    kept, note = candidates, "sin grading"
    if candidates and deps.settings.rag_grade_evidence:
        kept, note = await grade_results(task, candidates, deps)
    evidence = merge_evidence(list(state.get("evidence") or []), [to_evidence(r) for r in kept])
    detail = f"{len(kept)}/{len(candidates)} fragmentos relevantes para «{task[:90]}». {note[:150]}"
    return NodeOutcome(
        goto="supervisor",
        update={"evidence": evidence, "step_count": state.get("step_count", 0) + 1, "error": None},
        detail=detail,
    )


def researcher_fallback(state: AgentState, deps: AgentDeps, reason: str, status: StepStatus) -> NodeOutcome:
    del deps
    return NodeOutcome(
        goto="supervisor",
        update={"step_count": state.get("step_count", 0) + 1, "error": f"researcher: {reason}"},
        detail=f"sin evidencia nueva ({reason})",
        status=status,
    )
