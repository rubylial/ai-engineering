"""Composition root: construye y cierra todos los clientes compartidos.

Se ejecuta una vez en el lifespan de FastAPI. Si falta una credencial o un
servicio no responde, el componente queda marcado en `problems` y la app
arranca igual: `/health` responde y `/ready` explica qué falta. Los endpoints
que dependen de ese componente devuelven 503 con el detalle.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from intelligence_system.agents.graph import build_graph
from intelligence_system.agents.offline import OfflineResponder
from intelligence_system.agents.runtime import AgentDeps
from intelligence_system.agents.service import ChatService
from intelligence_system.agents.tools import make_domain_tools, make_search_tool
from intelligence_system.config import Settings
from intelligence_system.llm.base import BaseLLMClient, RetryPolicy
from intelligence_system.llm.exceptions import LLMConfigError
from intelligence_system.llm.factory import LLMClientFactory
from intelligence_system.llm.providers.fake_client import FakeLLMClient
from intelligence_system.observability.logging import get_logger
from intelligence_system.rag.chunking import MarkdownChunker
from intelligence_system.rag.embeddings import (
    EmbeddingsProvider,
    HashEmbeddings,
    PineconeInferenceEmbeddings,
)
from intelligence_system.rag.ingest import IngestService
from intelligence_system.rag.retriever import (
    HybridRetriever,
    NoopReranker,
    PineconeReranker,
    Reranker,
)
from intelligence_system.rag.sparse import (
    BM25Encoder,
    BM25StatsStore,
    InMemoryBM25StatsStore,
    RedisBM25StatsStore,
)
from intelligence_system.rag.vector_store import (
    InMemoryVectorStore,
    PineconeVectorStore,
    VectorStore,
)

log = get_logger(__name__)


@dataclass
class AppContainer:
    settings: Settings
    llm: BaseLLMClient | None = None
    redis: Redis | None = None
    vector_store: VectorStore | None = None
    embeddings: EmbeddingsProvider | None = None
    retriever: HybridRetriever | None = None
    ingest: IngestService | None = None
    checkpointer: BaseCheckpointSaver[Any] | None = None
    graph: CompiledStateGraph | None = None
    chat: ChatService | None = None
    problems: dict[str, str] = field(default_factory=dict)


def assemble(
    container: AppContainer,
    *,
    llm: BaseLLMClient | None,
    embeddings: EmbeddingsProvider | None,
    store: VectorStore | None,
    stats: BM25StatsStore,
    reranker: Reranker,
    checkpointer: BaseCheckpointSaver[Any] | None,
) -> AppContainer:
    """Conecta las piezas disponibles (también lo usan los tests con fakes)."""
    s = container.settings
    container.llm, container.embeddings = llm, embeddings
    container.vector_store, container.checkpointer = store, checkpointer
    if embeddings is not None and store is not None:
        encoder = BM25Encoder(stats, k1=s.bm25_k1, b=s.bm25_b)
        container.retriever = HybridRetriever(
            embeddings=embeddings,
            sparse_encoder=encoder,
            store=store,
            reranker=reranker,
            alpha=s.rag_hybrid_alpha,
            fetch_k=s.rag_fetch_k,
            default_top_k=s.rag_top_k,
        )
        container.ingest = IngestService(
            chunker=MarkdownChunker(
                chunk_size=s.rag_chunk_size,
                chunk_overlap=s.rag_chunk_overlap,
                min_chunk_chars=s.rag_min_chunk_chars,
            ),
            embeddings=embeddings,
            sparse_encoder=encoder,
            store=store,
        )
    if (
        llm is not None
        and container.retriever is not None
        and embeddings is not None
        and checkpointer is not None
    ):
        deps = AgentDeps(
            settings=s,
            llm=llm,
            search_tool=make_search_tool(container.retriever),
            domain_tools=make_domain_tools(embeddings),
        )
        container.graph = build_graph(deps, checkpointer)
        container.chat = ChatService(container.graph, s)
    return container


@asynccontextmanager
async def build_container(settings: Settings) -> AsyncIterator[AppContainer]:
    from intelligence_system.agents.checkpointer import open_checkpointer

    container = AppContainer(settings=settings)
    problems = container.problems
    for name in settings.missing_credentials():
        problems[f"config.{name}"] = f"falta la variable {name}"
    retry = RetryPolicy(
        max_attempts=settings.llm_max_retries,
        min_wait_s=settings.llm_retry_min_wait_s,
        max_wait_s=settings.llm_retry_max_wait_s,
    )

    async with AsyncExitStack() as stack:
        # --- Redis (checkpointer + estadísticas BM25) ---------------------
        stats: BM25StatsStore = InMemoryBM25StatsStore()
        if settings.checkpointer_backend == "redis":
            container.redis = Redis.from_url(settings.redis_url, decode_responses=True)
            stack.push_async_callback(container.redis.aclose)
            stats = RedisBM25StatsStore(
                container.redis, prefix=settings.bm25_redis_prefix, namespace=settings.pinecone_namespace
            )

        # --- Pinecone (vector store + inference) --------------------------
        pinecone_client: Any = None
        uses_pinecone = settings.vector_store == "pinecone" or settings.embeddings_provider == "pinecone"
        if uses_pinecone and settings.pinecone_api_key is not None:
            from pinecone import PineconeAsyncio

            pinecone_client = await stack.enter_async_context(
                PineconeAsyncio(
                    api_key=settings.pinecone_api_key.get_secret_value(),
                    timeout=settings.pinecone_timeout_s,
                )
            )

        embeddings: EmbeddingsProvider | None = None
        if settings.embeddings_provider == "fake":
            embeddings = HashEmbeddings(dimension=settings.embeddings_dimension)
        elif pinecone_client is not None:
            embeddings = PineconeInferenceEmbeddings(
                pinecone_client.inference,
                model=settings.embeddings_model,
                dimension=settings.embeddings_dimension,
                batch_size=settings.embeddings_batch_size,
                max_concurrency=settings.embeddings_max_concurrency,
                retry=retry,
            )

        store: VectorStore | None = None
        if settings.vector_store == "memory":
            store = InMemoryVectorStore(dimension=settings.embeddings_dimension)
        elif pinecone_client is not None:
            pinecone_store = PineconeVectorStore(
                pinecone_client,
                index_name=settings.pinecone_index_name,
                namespace=settings.pinecone_namespace,
                dimension=settings.embeddings_dimension,
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
                create_timeout_s=settings.pinecone_create_timeout_s,
                upsert_batch_size=settings.pinecone_upsert_batch_size,
                retry=retry,
            )
            try:
                await pinecone_store.ensure_ready()
                store = pinecone_store
                stack.push_async_callback(pinecone_store.aclose)
            except Exception as exc:
                problems["vector_store"] = f"{type(exc).__name__}: {exc}"[:500]
                log.error("container.vector_store_failed", error=problems["vector_store"])

        reranker: Reranker = NoopReranker()
        if settings.rag_rerank_enabled and pinecone_client is not None:
            reranker = PineconeReranker(
                pinecone_client.inference, model=settings.rag_rerank_model, retry=retry
            )

        # --- LLM -------------------------------------------------------------
        llm: BaseLLMClient | None = None
        try:
            if settings.llm_provider == "fake":
                llm = FakeLLMClient(OfflineResponder(), max_concurrency=settings.llm_max_concurrency)
            else:
                llm = LLMClientFactory.create(settings)
            stack.push_async_callback(llm.aclose)
        except LLMConfigError as exc:
            problems["llm"] = str(exc)

        # --- Checkpointer ------------------------------------------------------
        checkpointer: BaseCheckpointSaver[Any] | None = None
        try:
            checkpointer = await stack.enter_async_context(open_checkpointer(settings))
        except Exception as exc:
            problems["checkpointer"] = f"{type(exc).__name__}: {exc}"[:500]
            log.error("container.checkpointer_failed", error=problems["checkpointer"])

        assemble(
            container,
            llm=llm,
            embeddings=embeddings,
            store=store,
            stats=stats,
            reranker=reranker,
            checkpointer=checkpointer,
        )
        log.info(
            "container.ready",
            chat=container.chat is not None,
            ingest=container.ingest is not None,
            problems=sorted(problems),
        )
        yield container
