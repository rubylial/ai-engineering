"""POST /v1/ingest: ingesta idempotente (upsert por doc_id + hash de contenido)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from openinference.semconv.trace import OpenInferenceSpanKindValues
from pinecone import PineconeError

from intelligence_system.api.deps import ContainerDep, IngestServiceDep, require_api_key
from intelligence_system.api.schemas import ERROR_RESPONSES, IngestRequest, IngestResponse
from intelligence_system.errors import InvalidRequestError, UpstreamError
from intelligence_system.observability.context import get_request_id
from intelligence_system.observability.tracing import child_span, format_trace_id, mark_error
from intelligence_system.rag.embeddings import EmbeddingsError
from intelligence_system.rag.loaders import CorpusError, load_corpus
from intelligence_system.rag.schemas import Document

router = APIRouter(prefix="/v1", tags=["ingest"], dependencies=[Depends(require_api_key)])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingesta idempotente del corpus y/o de documentos enviados",
    responses=ERROR_RESPONSES,
)
async def ingest(body: IngestRequest, service: IngestServiceDep, container: ContainerDep) -> IngestResponse:
    with child_span("ingest.run", OpenInferenceSpanKindValues.CHAIN, request_id=get_request_id()) as span:
        documents: list[Document] = [d.to_document() for d in body.documents]
        try:
            if body.from_corpus:
                corpus = await load_corpus(container.settings.rag_corpus_dir)
                overridden = {d.doc_id for d in documents}
                documents = [d for d in corpus if d.doc_id not in overridden] + documents
            report = await service.ingest(documents)
        except CorpusError as exc:
            mark_error(span, exc)
            raise InvalidRequestError(str(exc), trace_id=format_trace_id(span)) from exc
        except (PineconeError, EmbeddingsError) as exc:
            mark_error(span, exc)
            detail = f"{type(exc).__name__}: {exc}"[:500]
            raise UpstreamError(detail, trace_id=format_trace_id(span)) from exc
        span.set_attribute("ingest.upserted", report.chunks_upserted)
        span.set_attribute("ingest.unchanged", report.chunks_unchanged)
    return IngestResponse(**report.model_dump())
