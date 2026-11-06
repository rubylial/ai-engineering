"""Tool de búsqueda en la base de conocimiento (la usa el researcher).

`args_schema` y salida son modelos Pydantic: un argumento inválido se rechaza
antes de tocar Pinecone, y el resultado viaja tipado hasta el writer.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, model_validator

from intelligence_system.rag.retriever import HybridRetriever
from intelligence_system.rag.schemas import MetadataFilter, RetrievalResult


class SearchKnowledgeBaseInput(BaseModel):
    query: str = Field(min_length=3, max_length=500, description="Consulta en lenguaje natural.")
    top_k: int = Field(default=5, ge=1, le=20, description="Cantidad de fragmentos a devolver.")
    doc_ids: list[str] | None = Field(default=None, description="Restringe a estos documentos.")
    categories: list[str] | None = Field(default=None, description="Restringe a estas categorías.")

    @model_validator(mode="after")
    def _validate_filters(self) -> SearchKnowledgeBaseInput:
        MetadataFilter(doc_ids=self.doc_ids, categories=self.categories)
        return self

    def metadata_filter(self) -> MetadataFilter | None:
        if not self.doc_ids and not self.categories:
            return None
        return MetadataFilter(doc_ids=self.doc_ids, categories=self.categories)


class SearchKnowledgeBaseOutput(BaseModel):
    query: str
    results: list[RetrievalResult]
    total: int


def make_search_tool(retriever: HybridRetriever) -> BaseTool:
    async def search_knowledge_base(
        query: str,
        top_k: int = 5,
        doc_ids: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> SearchKnowledgeBaseOutput:
        args = SearchKnowledgeBaseInput(query=query, top_k=top_k, doc_ids=doc_ids, categories=categories)
        results = await retriever.retrieve(
            args.query, top_k=args.top_k, metadata_filter=args.metadata_filter()
        )
        return SearchKnowledgeBaseOutput(query=args.query, results=results, total=len(results))

    return StructuredTool.from_function(
        coroutine=search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Busca fragmentos relevantes en la documentación de AI Engineering con retrieval "
            "híbrido (semántico + BM25). Usá esta herramienta cuando necesites definiciones, "
            "recomendaciones o datos que deben salir de la documentación. No la uses para "
            "hacer cálculos."
        ),
        args_schema=SearchKnowledgeBaseInput,
    )
