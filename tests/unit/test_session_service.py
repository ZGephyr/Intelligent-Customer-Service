"""会话服务：归属、列表聚合、标题。"""

from __future__ import annotations

import pytest

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.models.tables import ChatMessage, ChatSession
from rag_service.services.session_service import SessionService


@pytest.fixture
def session_ctx():
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    return SessionService(factory), factory


def test_create_assigns_owner(session_ctx):
    service, factory = session_ctx

    session_id = service.create("u1")

    assert service.get_owned(session_id, "u1").user_id == "u1"


def test_list_for_user_only_returns_own_sessions(session_ctx):
    service, _ = session_ctx
    mine = service.create("u1")
    service.create("u2")

    items, total = service.list_for_user("u1")

    assert total == 1 and [item["id"] for item in items] == [mine]


def test_list_aggregates_message_count_and_last_time(session_ctx):
    service, factory = session_ctx
    session_id = service.create("u1")
    with factory() as session:
        session.add(ChatMessage(id="m1", session_id=session_id, role="user", content="hi"))
        session.add(ChatMessage(id="m2", session_id=session_id, role="assistant", content="hello"))
        session.commit()

    items, _ = service.list_for_user("u1")

    assert items[0]["message_count"] == 2
    assert items[0]["last_message_at"] is not None


def test_get_owned_raises_404_for_other_user(session_ctx):
    service, _ = session_ctx
    session_id = service.create("u2")

    with pytest.raises(AppError) as excinfo:
        service.get_owned(session_id, "u1")

    assert excinfo.value.code == ErrorCode.CHAT_SESSION_NOT_FOUND.value
    assert excinfo.value.http_status == 404


def test_get_owned_raises_404_for_missing_session(session_ctx):
    service, _ = session_ctx

    with pytest.raises(AppError) as excinfo:
        service.get_owned("missing", "u1")

    assert excinfo.value.http_status == 404


def test_first_message_sets_title_truncated(session_ctx):
    service, _ = session_ctx
    session_id = service.create("u1")
    question = "耳机保修多久，另外发票怎么开，还有运费谁承担"

    service.set_title_if_empty(session_id, question)
    service.set_title_if_empty(session_id, "第二条不应覆盖")

    assert service.get_owned(session_id, "u1").title == question[:20]


def test_messages_returns_chronological(session_ctx):
    service, factory = session_ctx
    session_id = service.create("u1")
    with factory() as session:
        session.add(ChatMessage(id="m1", session_id=session_id, role="user", content="first"))
        session.add(ChatMessage(id="m2", session_id=session_id, role="assistant", content="second"))
        session.commit()

    rows = service.messages(session_id, "u1")

    assert [row.content for row in rows] == ["first", "second"]


def test_messages_rejects_foreign_session(session_ctx):
    service, _ = session_ctx
    session_id = service.create("u2")

    with pytest.raises(AppError):
        service.messages(session_id, "u1")


def test_delete_removes_session_and_messages(session_ctx):
    service, factory = session_ctx
    session_id = service.create("u1")
    with factory() as session:
        session.add(ChatMessage(id="m1", session_id=session_id, role="user", content="hi"))
        session.commit()

    service.delete(session_id, "u1")

    with factory() as session:
        assert session.get(ChatSession, session_id) is None
        assert session.query(ChatMessage).filter_by(session_id=session_id).count() == 0


def test_delete_foreign_session_raises_404_and_keeps_data(session_ctx):
    service, factory = session_ctx
    session_id = service.create("u2")

    with pytest.raises(AppError) as excinfo:
        service.delete(session_id, "u1")

    assert excinfo.value.code == ErrorCode.CHAT_SESSION_NOT_FOUND.value
    with factory() as session:
        assert session.get(ChatSession, session_id) is not None


def test_delete_missing_session_raises_404(session_ctx):
    service, _ = session_ctx

    with pytest.raises(AppError) as excinfo:
        service.delete("missing", "u1")

    assert excinfo.value.http_status == 404

def test_messages_orders_by_seq_when_timestamps_tie(session_ctx):
    from datetime import datetime

    service, factory = session_ctx
    session_id = service.create("u1")
    same = datetime(2026, 1, 1, 12, 0, 0)
    with factory() as session:
        session.add(ChatMessage(id="m2", session_id=session_id, role="assistant", content="second", created_at=same, seq=2))
        session.add(ChatMessage(id="m1", session_id=session_id, role="user", content="first", created_at=same, seq=1))
        session.commit()

    rows = service.messages(session_id, "u1")

    assert [row.seq for row in rows] == [1, 2]
    assert [row.content for row in rows] == ["first", "second"]


def test_messages_put_rows_without_seq_last(session_ctx):
    from datetime import datetime

    service, factory = session_ctx
    session_id = service.create("u1")
    same = datetime(2026, 1, 1, 12, 0, 0)
    with factory() as session:
        session.add(ChatMessage(id="m0", session_id=session_id, role="user", content="legacy", created_at=same))
        session.add(ChatMessage(id="m1", session_id=session_id, role="user", content="first", created_at=same, seq=1))
        session.commit()

    rows = service.messages(session_id, "u1")

    assert [row.content for row in rows] == ["first", "legacy"]