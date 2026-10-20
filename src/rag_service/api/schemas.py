"""Pydantic 请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class SessionOut(BaseModel):
    session_id: str


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    role: str
    content: str
    rewritten_query: str | None = None
    sources: list[dict[str, Any]] | None = None
    status: str
    latency_ms: int | None = None
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_ext: str
    file_size: int
    status: str
    chunk_count: int
    error_message: str | None = None
    created_at: datetime


class PageOut(BaseModel):
    items: list[DocumentOut]
    total: int
    page: int
    size: int
