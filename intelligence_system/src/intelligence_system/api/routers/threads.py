"""GET /v1/threads/{thread_id}: historial persistido en el checkpointer."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Path

from intelligence_system.agents.schemas import ThreadHistory
from intelligence_system.api.deps import ChatServiceDep, require_api_key
from intelligence_system.api.schemas import ERROR_RESPONSES, THREAD_ID_PATTERN, ProblemDetail
from intelligence_system.errors import NotFoundError

router = APIRouter(prefix="/v1", tags=["threads"], dependencies=[Depends(require_api_key)])


@router.get(
    "/threads/{thread_id}",
    response_model=ThreadHistory,
    summary="Historial de una conversación (sobrevive reinicios del servidor)",
    responses={404: {"model": ProblemDetail, "description": "El thread no existe"}, **ERROR_RESPONSES},
)
async def get_thread(
    thread_id: Annotated[str, Path(pattern=THREAD_ID_PATTERN)], service: ChatServiceDep
) -> ThreadHistory:
    history = await service.thread_history(thread_id)
    if history is None:
        raise NotFoundError(f"No existe historial para el thread '{thread_id}'.")
    return history
