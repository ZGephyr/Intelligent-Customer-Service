"""WebSocket 鉴权：未登录 4401、越权 4404、无 session 时新建并归属。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.main import AppServices, create_app
from rag_service.services.auth_service import AuthService
from rag_service.services.chat_service import ChatService
from rag_service.services.session_service import SessionService
from rag_service.services.rewrite_service import RewriteOutcome


class FakeHistory:
    def get(self, session_id):
        return []

    def append(self, session_id, turns):
        pass


class FakeRetrieval:
    def retrieve(self, query):
        return []


class FakeRewrite:
    def rewrite(self, question, history):
        return RewriteOutcome(question, False)


class FakeLLM:
    def stream(self, messages):
        yield "好的"

    def invoke(self, messages, max_tokens=None):
        return ""


@pytest.fixture
def ws_auth_app(settings_factory, register_user):
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    settings = settings_factory(bcrypt_rounds=4, ws_allowed_origins="http://testserver")
    sessions = SessionService(factory)
    chat = ChatService(
        factory, FakeHistory(), FakeRetrieval(), FakeRewrite(), FakeLLM(), settings, sessions=sessions
    )
    app = create_app(
        settings,
        services=AppServices(
            chat=chat,
            ingestion=None,
            health=None,
            auth=AuthService(factory, "s" * 32, bcrypt_rounds=4),
            sessions=sessions,
        ),
    )

    def make_client(username: str) -> TestClient:
        client = TestClient(app)
        register_user(client, username=username, role="user")
        return client

    return app, make_client


def test_ws_without_login_is_rejected(ws_auth_app):
    app, _ = ws_auth_app
    anonymous = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with anonymous.websocket_connect("/ws/chat") as websocket:
            websocket.receive_json()

    assert excinfo.value.code == 4401


def test_ws_without_session_id_creates_one_owned_by_user(ws_auth_app):
    _, make_client = ws_auth_app
    client = make_client("alice")

    with client.websocket_connect("/ws/chat") as websocket:
        ready = websocket.receive_json()

    assert ready["type"] == "ready" and ready["session_id"]
    items = client.get("/api/sessions").json()["data"]["items"]
    assert items[0]["id"] == ready["session_id"]


def test_ws_foreign_session_is_rejected(ws_auth_app):
    _, make_client = ws_auth_app
    alice = make_client("alice")
    bob = make_client("bob")
    foreign_id = alice.post("/api/sessions").json()["data"]["session_id"]

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with bob.websocket_connect(f"/ws/chat?session_id={foreign_id}") as websocket:
            websocket.receive_json()

    assert excinfo.value.code == 4404


def test_ws_with_own_session_streams_events(ws_auth_app):
    _, make_client = ws_auth_app
    client = make_client("alice")
    session_id = client.post("/api/sessions").json()["data"]["session_id"]

    with client.websocket_connect(f"/ws/chat?session_id={session_id}") as websocket:
        ready = websocket.receive_json()
        assert ready["session_id"] == session_id
        websocket.send_json({"type": "message", "content": "保修多久"})
        types = []
        while True:
            frame = websocket.receive_json()
            types.append(frame["type"])
            if frame["type"] in ("done", "error"):
                break

    assert types[0] == "meta" and types[-1] == "done"
