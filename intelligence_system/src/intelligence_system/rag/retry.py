"""Política de reintentos para Pinecone (solo errores transitorios)."""

from __future__ import annotations

from pinecone import PineconeConnectionError, PineconeTimeoutError, RateLimitError, ServiceError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
)

from intelligence_system.llm.base import RetryPolicy
from intelligence_system.observability.logging import get_logger

PINECONE_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    RateLimitError,
    ServiceError,
    PineconeTimeoutError,
    PineconeConnectionError,
)
log = get_logger(__name__)


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "pinecone.retry",
        attempt=state.attempt_number,
        error=type(exc).__name__ if exc else None,
    )


def pinecone_retrying(policy: RetryPolicy) -> AsyncRetrying:
    return AsyncRetrying(
        stop=stop_after_attempt(policy.max_attempts),
        wait=policy.wait_strategy(),
        retry=retry_if_exception_type(PINECONE_TRANSIENT_ERRORS),
        reraise=True,
        before_sleep=_log_retry,
    )
