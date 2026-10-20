"""数据模型与数据库工具测试。"""

from __future__ import annotations

from rag_service.infra.db import (
    build_engine,
    build_session_factory,
    create_all,
    database_name_from_url,
    server_url_without_database,
)
from rag_service.models.tables import ChatMessage, ChatSession, FaqPair, KbChunk, KbDocument


def test_create_all_and_insert_document_with_chunk():
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)
    with factory() as s:
        doc = KbDocument(
            id="d1",
            filename="a.md",
            file_ext="md",
            file_size=3,
            file_hash="h" * 64,
            storage_path="x",
            status="ready",
            chunk_count=1,
        )
        s.add(doc)
        s.add(KbChunk(id="c1", document_id="d1", chunk_index=0, content="hi", char_count=2, meta={}))
        s.commit()
        assert s.query(KbChunk).filter_by(document_id="d1").one().content == "hi"


def test_create_all_creates_chat_tables():
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)
    with factory() as s:
        s.add(ChatSession(id="s1", title="会话"))
        s.add(ChatMessage(id="m1", session_id="s1", role="user", content="你好"))
        s.commit()
        assert s.query(ChatMessage).filter_by(session_id="s1").one().content == "你好"


def test_faq_pair_defaults_enabled_and_zero_hits():
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)
    with factory() as s:
        s.add(FaqPair(id="f1", question="多久发货", answer="48 小时"))
        s.commit()
        row = s.get(FaqPair, "f1")
        assert row.enabled is True and row.hit_count == 0 and row.last_hit_at is None


def test_database_url_helpers_extract_name_and_server_url():
    url = "mysql+pymysql://root:pwd@localhost:3306/agentic_rag?charset=utf8mb4"
    assert database_name_from_url(url) == "agentic_rag"
    assert server_url_without_database(url).database is None
