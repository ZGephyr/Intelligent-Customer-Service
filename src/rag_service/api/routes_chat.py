"""会话接口：列表、新建、历史。全部要求登录，且只能操作自己的会话。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request

from rag_service.api.deps import require_user
from rag_service.api.schemas import ChatMessageOut, SessionOut
from rag_service.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


@router.get("")
def list_sessions(
    request: Request,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    user=Depends(require_user),
) -> dict:
    items, total = request.app.state.services.sessions.list_for_user(user.id, page=page, size=size)
    payload = {
        "items": [
            {
                "id": item["id"],
                "title": item["title"],
                "message_count": item["message_count"],
                "last_message_at": item["last_message_at"].isoformat()
                if item["last_message_at"]
                else None,
                "created_at": item["created_at"].isoformat() if item["created_at"] else None,
            }
            for item in items
        ],
        "total": total,
        "page": page,
        "size": size,
    }
    return {"code": 0, "data": payload, "request_id": _request_id(request)}


@router.post("")
def create_session(request: Request, user=Depends(require_user)) -> dict:
    session_id = request.app.state.services.sessions.create(user.id)
    payload = SessionOut(session_id=session_id)
    return {"code": 0, "data": payload.model_dump(), "request_id": _request_id(request)}


@router.delete("/{session_id}")
def delete_session(request: Request, session_id: str, user=Depends(require_user)) -> dict:
    services = request.app.state.services
    services.sessions.delete(session_id, user.id)
    chat = getattr(services, "chat", None)
    if chat is not None and hasattr(chat, "clear_history"):
        chat.clear_history(session_id)
    return {"code": 0, "data": {"deleted": session_id}, "request_id": _request_id(request)}


@router.get("/{session_id}/messages")
def list_messages(
    request: Request,
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    user=Depends(require_user),
) -> dict:
    messages = request.app.state.services.sessions.messages(session_id, user.id, limit=limit)
    items = [ChatMessageOut.model_validate(message).model_dump(mode="json") for message in messages]
    return {
        "code": 0,
        "data": {"items": items},
        "request_id": _request_id(request),
    }
