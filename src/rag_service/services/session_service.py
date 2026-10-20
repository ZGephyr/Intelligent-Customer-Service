"""会话列表与归属校验；越权一律 404，不暴露会话是否存在。"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import func

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.models.tables import ChatMessage, ChatSession

logger = get_logger(__name__)

TITLE_MAX_CHARS = 20


class SessionService:
    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def create(self, user_id: str | None = None) -> str:
        session_id = str(uuid4())
        with self._session_factory() as session:
            session.add(ChatSession(id=session_id, user_id=user_id))
            session.commit()
        logger.info("新建会话: session=%s user=%s", session_id, user_id)
        return session_id

    def list_for_user(self, user_id: str, page: int = 1, size: int = 20) -> tuple[list[dict], int]:
        with self._session_factory() as session:
            total = (
                session.query(func.count(ChatSession.id))
                .filter(ChatSession.user_id == user_id)
                .scalar()
                or 0
            )
            rows = (
                session.query(
                    ChatSession,
                    func.count(ChatMessage.id),
                    func.max(ChatMessage.created_at),
                )
                .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
                .filter(ChatSession.user_id == user_id)
                .group_by(ChatSession.id)
                .order_by(ChatSession.updated_at.desc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )
            items = [
                {
                    "id": chat_session.id,
                    "title": chat_session.title or "新会话",
                    "message_count": int(count),
                    "last_message_at": last_at,
                    "created_at": chat_session.created_at,
                }
                for chat_session, count, last_at in rows
            ]
            return items, int(total)

    def get_owned(self, session_id: str, user_id: str) -> ChatSession:
        with self._session_factory() as session:
            row = session.get(ChatSession, session_id)
            if row is None or row.user_id != user_id:
                raise AppError(
                    ErrorCode.CHAT_SESSION_NOT_FOUND, "会话不存在", http_status=404
                )
            return row

    def messages(self, session_id: str, user_id: str, limit: int = 50) -> list[ChatMessage]:
        self.get_owned(session_id, user_id)
        with self._session_factory() as session:
            rows = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(
                    ChatMessage.seq.is_(None),
                    ChatMessage.seq.asc(),
                    ChatMessage.created_at.asc(),
                    ChatMessage.id.asc(),
                )
                .limit(limit)
                .all()
            )
            return list(rows)

    def delete(self, session_id: str, user_id: str) -> None:
        """硬删除：先删消息再删会话；越权或不存在按 404 处理。"""
        self.get_owned(session_id, user_id)
        with self._session_factory() as session:
            session.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete(
                synchronize_session=False
            )
            row = session.get(ChatSession, session_id)
            if row is not None:
                session.delete(row)
            session.commit()
        logger.info("删除会话: session=%s user=%s", session_id, user_id)

    def touch(self, session_id: str) -> None:
        with self._session_factory() as session:
            row = session.get(ChatSession, session_id)
            if row is not None:
                session.commit()

    def set_title_if_empty(self, session_id: str, text: str) -> None:
        title = (text or "").strip()[:TITLE_MAX_CHARS]
        if not title:
            return
        with self._session_factory() as session:
            row = session.get(ChatSession, session_id)
            if row is None or row.title:
                return
            row.title = title
            session.commit()
