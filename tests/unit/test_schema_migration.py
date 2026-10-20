"""建表与幂等迁移测试。"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.models.tables import FaqPair

LEGACY_DDL = """
CREATE TABLE chat_messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    rewritten_query VARCHAR(512),
    sources JSON,
    status VARCHAR(16),
    latency_ms INTEGER,
    created_at DATETIME
)
"""


def test_ensure_schema_adds_missing_message_columns(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.execute(text(LEGACY_DDL))

    ensure_schema(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("chat_messages")}
    assert {"intent", "route"} <= columns


def test_ensure_schema_is_idempotent(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twice.db'}")
    ensure_schema(engine)
    ensure_schema(engine)
    assert inspect(engine).has_table("faq_pairs")


def test_faq_pair_question_is_unique(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'faq.db'}")
    ensure_schema(engine)
    factory = build_session_factory(engine)

    with factory() as session:
        session.add(FaqPair(id="f1", question="多久发货", answer="48 小时"))
        session.commit()

    with pytest.raises(IntegrityError):
        with factory() as session:
            session.add(FaqPair(id="f2", question="多久发货", answer="重复"))
            session.commit()


def test_ensure_schema_creates_user_and_captcha_tables(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'auth.db'}")
    ensure_schema(engine)

    tables = set(inspect(engine).get_table_names())
    assert {"users", "auth_captchas"} <= tables


def test_ensure_schema_adds_user_id_to_legacy_sessions(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'legacy_sessions.db'}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE chat_sessions ("
                "id VARCHAR(36) PRIMARY KEY, title VARCHAR(100), "
                "created_at DATETIME, updated_at DATETIME)"
            )
        )

    ensure_schema(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("chat_sessions")}
    assert "user_id" in columns


def test_clear_sessions_deletes_messages_and_sessions(tmp_path):
    from scripts.clear_legacy_sessions import clear_sessions
    from rag_service.models.tables import ChatMessage, ChatSession

    engine = build_engine(f"sqlite:///{tmp_path / 'clear.db'}")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    with factory() as session:
        session.add(ChatSession(id="s1"))
        session.add(ChatMessage(id="m1", session_id="s1", role="user", content="hi"))
        session.commit()

    removed = clear_sessions(engine)

    assert removed == (1, 1)
    with factory() as session:
        assert session.query(ChatMessage).count() == 0
        assert session.query(ChatSession).count() == 0
