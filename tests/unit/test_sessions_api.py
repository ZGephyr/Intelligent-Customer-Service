"""会话接口：登录要求、归属隔离、分页。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.main import AppServices, create_app
from rag_service.services.auth_service import AuthService
from rag_service.services.session_service import SessionService


@pytest.fixture
def session_app(settings_factory, register_user):
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    settings = settings_factory(bcrypt_rounds=4)
    app = create_app(
        settings,
        services=AppServices(
            chat=None,
            ingestion=None,
            health=None,
            auth=AuthService(factory, "s" * 32, bcrypt_rounds=4),
            sessions=SessionService(factory),
        ),
    )

    def make_client(username: str, role: str = "user") -> TestClient:
        client = TestClient(app)
        register_user(client, username=username, role=role)
        return client

    return make_client


def test_sessions_require_login(session_app):
    anonymous = TestClient(session_app("alice").app)
    assert anonymous.get("/api/sessions").status_code == 401
    assert anonymous.post("/api/sessions").status_code == 401


def test_create_then_list_own_sessions(session_app):
    client = session_app("alice")

    session_id = client.post("/api/sessions").json()["data"]["session_id"]
    items = client.get("/api/sessions").json()["data"]["items"]

    assert [item["id"] for item in items] == [session_id]
    assert items[0]["title"] == "新会话" and items[0]["message_count"] == 0


def test_other_users_session_returns_404(session_app):
    owner = session_app("alice")
    intruder = session_app("bob")
    session_id = owner.post("/api/sessions").json()["data"]["session_id"]

    response = intruder.get(f"/api/sessions/{session_id}/messages")

    assert response.status_code == 404
    assert response.json()["code"] == "CHAT_SESSION_NOT_FOUND"


def test_sessions_are_isolated_between_users(session_app):
    alice = session_app("alice")
    bob = session_app("bob")
    alice.post("/api/sessions")
    bob.post("/api/sessions")
    bob.post("/api/sessions")

    assert alice.get("/api/sessions").json()["data"]["total"] == 1
    assert bob.get("/api/sessions").json()["data"]["total"] == 2


def test_sessions_pagination_metadata(session_app):
    client = session_app("alice")
    for _ in range(3):
        client.post("/api/sessions")

    payload = client.get("/api/sessions?page=2&size=2").json()["data"]

    assert payload["total"] == 3 and payload["page"] == 2 and payload["size"] == 2
    assert len(payload["items"]) == 1


def test_delete_own_session_removes_it(session_app):
    client = session_app("alice")
    session_id = client.post("/api/sessions").json()["data"]["session_id"]

    response = client.delete(f"/api/sessions/{session_id}")

    assert response.status_code == 200
    assert client.get("/api/sessions").json()["data"]["total"] == 0
    assert client.get(f"/api/sessions/{session_id}/messages").status_code == 404


def test_delete_foreign_session_returns_404_and_keeps_it(session_app):
    owner = session_app("alice")
    intruder = session_app("bob")
    session_id = owner.post("/api/sessions").json()["data"]["session_id"]

    response = intruder.delete(f"/api/sessions/{session_id}")

    assert response.status_code == 404
    assert owner.get("/api/sessions").json()["data"]["total"] == 1


def test_delete_session_requires_login(session_app):
    client = session_app("alice")
    session_id = client.post("/api/sessions").json()["data"]["session_id"]
    anonymous = TestClient(client.app)

    assert anonymous.delete(f"/api/sessions/{session_id}").status_code == 401