"""SQLAlchemy ORM 模型。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now()


class Base(DeclarativeBase):
    pass


class KbDocument(Base):
    __tablename__ = "kb_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_ext: Mapped[str] = mapped_column(String(16))
    file_size: Mapped[int] = mapped_column(Integer)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    storage_path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="processing", index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class KbChunk(Base):
    __tablename__ = "kb_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uk_doc_chunk"),
        Index("idx_document_id", "document_id"),
        Index("idx_parent_id", "parent_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(36))
    parent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    level: Mapped[str] = mapped_column(String(8), default="child")
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class User(Base):
    """登录用户；role 为 admin / user。"""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(16), default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class AuthCaptcha(Base):
    """注册验证码：答案只存 HMAC，5 分钟过期、用过即作废。"""

    __tablename__ = "auth_captchas"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    answer_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (Index("idx_sessions_user", "user_id", "updated_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    title: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (Index("idx_session_created", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    seq: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rewritten_query: Mapped[str | None] = mapped_column(String(512), nullable=True)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    intent: Mapped[str | None] = mapped_column(String(16), nullable=True)
    route: Mapped[str | None] = mapped_column(String(16), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class FaqPair(Base):
    """高频问答对：命中后原样返回 answer，不经过模型生成。"""

    __tablename__ = "faq_pairs"
    __table_args__ = (Index("idx_faq_enabled", "enabled"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    question: Mapped[str] = mapped_column(String(500), unique=True)
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(32), nullable=True)
    aliases: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_hit_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
