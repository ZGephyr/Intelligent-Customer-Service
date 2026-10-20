"""C 端会话接口（问答走 WebSocket）。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request

from rag_service.api.schemas import ChatMessageOut, SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=dict)
def create_session(request: Request) -> dict:
    service = request.app.state.services.chat
    session_id = service.create_session()
    payload = SessionOut(session_id=session_id)
    return {"code": 0, "data": payload.model_dump(), "request_id": getattr(request.state, "request_id", uuid.uuid4().hex)}


@router.get("/{session_id}/messages", response_model=dict)
def list_messages(request: Request, session_id: str, limit: int = 50) -> dict:
    service = request.app.state.services.chat
    messages = service.list_messages(session_id, limit=limit)
    items = [ChatMessageOut.model_validate(message).model_dump(mode="json") for message in messages]
    return {
        "code": 0,
        "data": {"items": items},
        "request_id": getattr(request.state, "request_id", uuid.uuid4().hex),
    }
