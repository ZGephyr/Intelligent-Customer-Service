"""Redis 会话历史缓存；故障时降级为空历史，不阻断问答。"""

from __future__ import annotations

import json
from typing import Any

from redis import Redis

from rag_service.core.logging import get_logger
from rag_service.core.types import ChatTurn

logger = get_logger(__name__)

KEY_PREFIX = "chat:history:"


def build_redis_client(url: str) -> Redis:
    # 服务端可能是旧版 Redis（不支持 RESP3 的 HELLO），固定协议 2 兼容
    return Redis.from_url(url, decode_responses=True, protocol=2)


class HistoryStore:
    def __init__(self, client: Any, ttl_seconds: int = 3600, max_messages: int = 20) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._max = max_messages

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{KEY_PREFIX}{session_id}"

    def get(self, session_id: str) -> list[ChatTurn]:
        try:
            raw = self._client.lrange(self._key(session_id), -self._max, -1)
        except Exception:  # noqa: BLE001 - 缓存故障必须降级
            logger.warning("读取 Redis 历史失败，降级为空历史: session=%s", session_id)
            return []

        turns: list[ChatTurn] = []
        for item in raw:
            try:
                payload = json.loads(item)
                turns.append(ChatTurn(role=payload["role"], content=payload["content"]))
            except (ValueError, KeyError, TypeError):
                logger.warning("跳过无法解析的历史记录: session=%s", session_id)
        return turns

    def append(self, session_id: str, turns: list[ChatTurn]) -> None:
        if not turns:
            return
        key = self._key(session_id)
        payloads = [
            json.dumps({"role": turn.role, "content": turn.content}, ensure_ascii=False)
            for turn in turns
        ]
        try:
            self._client.rpush(key, *payloads)
            self._client.ltrim(key, -self._max, -1)
            self._client.expire(key, self._ttl)
        except Exception:  # noqa: BLE001 - 缓存故障必须降级
            logger.warning("写入 Redis 历史失败: session=%s", session_id)

    def clear(self, session_id: str) -> None:
        """删除会话时清掉缓存；故障只记 warning，不影响删除结果。"""
        try:
            self._client.delete(self._key(session_id))
        except Exception:  # noqa: BLE001 - 缓存清理失败可忽略
            logger.warning("清理 Redis 历史失败: session=%s", session_id)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False
