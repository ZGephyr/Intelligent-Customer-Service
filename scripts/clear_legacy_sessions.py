"""一次性清空旧会话（升级 1.1 用）。

升级后会话按用户隔离，旧会话没有 user_id，按用户决策直接清空。
默认只打印提示，必须显式加 --yes 才执行。
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import delete

from rag_service.core.config import get_settings
from rag_service.core.logging import get_logger, setup_logging
from rag_service.infra.db import build_engine, ensure_schema
from rag_service.models.tables import ChatMessage, ChatSession

logger = get_logger(__name__)


def clear_sessions(engine) -> tuple[int, int]:
    """删除全部消息与会话，返回 (消息行数, 会话行数)。"""
    ensure_schema(engine)
    with engine.begin() as conn:
        messages = conn.execute(delete(ChatMessage)).rowcount or 0
        sessions = conn.execute(delete(ChatSession)).rowcount or 0
    logger.info("清空旧会话完成: messages=%s sessions=%s", messages, sessions)
    return int(messages), int(sessions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清空 chat_messages 与 chat_sessions")
    parser.add_argument("--yes", action="store_true", help="确认执行删除")
    args = parser.parse_args(argv)

    if not args.yes:
        print("将删除全部 chat_messages 与 chat_sessions（不可恢复）。确认后加 --yes 再执行。")
        return 1

    settings = get_settings()
    setup_logging(settings)
    engine = build_engine(settings.mysql_url)
    messages, sessions = clear_sessions(engine)
    print(f"已删除 chat_messages {messages} 行、chat_sessions {sessions} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
