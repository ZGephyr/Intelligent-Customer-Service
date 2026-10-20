"""跨层共享的领域数据类型，避免 infra 反向依赖 services。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedBlock:
    """解析器输出的一段文本。"""

    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedChunk:
    """切分后、尚未向量化的 chunk。"""

    content: str
    meta: dict[str, Any]
    index: int


@dataclass(frozen=True)
class ParsedParent:
    """父块：较大的语义段，children 为切分后的子块（用于向量化）。"""

    content: str
    meta: dict[str, Any]
    index: int
    children: tuple[ParsedChunk, ...]


@dataclass(frozen=True)
class IndexedChunk:
    """写入 Milvus 的子块：子块文本 + 向量 + 父块上下文。"""

    id: str
    document_id: str
    chunk_index: int
    content: str
    source: str
    doc_type: str
    created_at: int
    embedding: list[float]
    parent_id: str = ""
    parent_content: str = ""
    sparse: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """检索返回的片段：content 为父块上下文，snippet 为命中的子块。"""

    id: str
    document_id: str
    content: str
    source: str
    chunk_index: int
    score: float
    snippet: str = ""


@dataclass(frozen=True)
class ChatTurn:
    """一轮对话消息，供改写与提示词使用。"""

    role: str
    content: str


@dataclass
class ChatEvent:
    """问答流式事件，协议层原样转成 WebSocket 帧。"""

    type: str
    data: dict[str, Any]
