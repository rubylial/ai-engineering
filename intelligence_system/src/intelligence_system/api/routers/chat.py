"""POST /v1/chat y POST /v1/chat/stream (SSE: token | step | citation | done | error)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from sse_starlette import EventSourceResponse

from intelligence_system.api.deps import ChatServiceDep, require_api_key
from intelligence_system.api.schemas import ERROR_RESPONSES, ChatRequest, ChatResponse
from intelligence_system.observability.context import get_request_id, new_request_id, new_thread_id

router = APIRouter(prefix="/v1", tags=["chat"], dependencies=[Depends(require_api_key)])
_SSE_PING_S = 15


def _filters(body: ChatRequest) -> dict[str, Any] | None:
    return body.filters.model_dump(exclude_none=True) if body.filters else None


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Pregunta al sistema multi-agente (respuesta completa con citas)",
    responses=ERROR_RESPONSES,
)
async def chat(body: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    result = await service.run_turn(
        message=body.message,
        thread_id=body.thread_id or new_thread_id(),
        request_id=get_request_id() or new_request_id(),
        filters=_filters(body),
        deadline_s=body.deadline_s,
    )
    return ChatResponse(**result.model_dump())


@router.post(
    "/chat/stream",
    summary="Igual que /v1/chat pero por Server-Sent Events",
    response_class=EventSourceResponse,
    responses={
        200: {
            "description": (
                "Stream SSE. Eventos: `step` (cada nodo terminado), `token` (texto del writer), "
                "`citation` (cada cita validada), `done` (resultado final validado: si "
                "`grounded=false`, reemplaza el texto acumulado) y `error` (problem+json)."
            ),
            "content": {"text/event-stream": {}},
        },
        **ERROR_RESPONSES,
    },
)
async def chat_stream(body: ChatRequest, service: ChatServiceDep) -> EventSourceResponse:
    thread_id = body.thread_id or new_thread_id()
    request_id = get_request_id() or new_request_id()

    async def events() -> AsyncIterator[dict[str, str]]:
        async for event in service.stream_turn(
            message=body.message,
            thread_id=thread_id,
            request_id=request_id,
            filters=_filters(body),
            deadline_s=body.deadline_s,
        ):
            yield {"event": event.event, "data": json.dumps(event.data, ensure_ascii=False)}

    return EventSourceResponse(
        events(), ping=_SSE_PING_S, headers={"X-Thread-ID": thread_id, "X-Request-ID": request_id}
    )
