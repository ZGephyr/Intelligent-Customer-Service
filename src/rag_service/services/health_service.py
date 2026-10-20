"""依赖健康检查：四项任一失败即 degraded。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from rag_service.core.logging import get_logger

logger = get_logger(__name__)


class HealthService:
    def __init__(
        self,
        db_engine: Any,
        redis_client: Any,
        milvus_store: Any,
        embedding: Any,
    ) -> None:
        self._engine = db_engine
        self._redis = redis_client
        self._milvus = milvus_store
        self._embedding = embedding

    def _check_db(self) -> str:
        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return "ok"

    def _check_redis(self) -> str:
        return "ok" if self._redis.ping() else "error: ping failed"

    def _check_milvus(self) -> str:
        return "ok" if self._milvus.ping() else "error: unreachable"

    def _check_embedding(self) -> str:
        return "ok" if getattr(self._embedding, "dimension", 0) > 0 else "error: not configured"

    def check(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for name, checker in (
            ("mysql", self._check_db),
            ("redis", self._check_redis),
            ("milvus", self._check_milvus),
            ("embedding", self._check_embedding),
        ):
            try:
                results[name] = checker()
            except Exception as exc:  # noqa: BLE001 - 健康检查不抛错
                logger.warning("健康检查失败: %s -> %s", name, exc)
                results[name] = f"error: {exc}"
        return results
