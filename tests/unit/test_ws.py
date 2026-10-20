"""WebSocket 接口测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from rag_service.infra.db import build_engine, build_session_factory, create_all
from rag_service.main import AppServices, create_app
from rag_service.services.chat_service import ChatService
from rag_service.services.health_service import HealthService
from rag_service.services.ingestion_service import IngestionService
from rag_service.services.rewrite_service import RewriteOutcome


class FakeEmbedding:
    dimension = 8

    def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


class FakeStore:
    def upsert(self, chunks):
        pass

    def delete_document(self, document_id):
        pass

    def drop_and_recreate(self):
        pass

    def ping(self):
        return True


class FakeRedis:
    def ping(self):
        return True


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
        yield "抱歉"

    def invoke(self, messages, max_tokens=None):
        return ""


def build_ws_client(tmp_path, settings_factory):
    settings = settings_factory(
        upload_dir=tmp_path / "uploads",
        log_dir=tmp_path / "logs",
        ws_allowed_origins="http://testserver",
    )
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)

    chat = ChatService(factory, FakeHistory(), FakeRetrieval(), FakeRewrite(), FakeLLM(), settings)
    ingestion = IngestionService(
        factory, FakeEmbedding(), FakeStore(), tmp_path / "uploads", max_upload_mb=1
    )
    health = HealthService(engine, FakeRedis(), FakeStore(), FakeEmbedding())
    app = create_app(settings, AppServices(chat=chat, ingestion=ingestion, health=health))
    return TestClient(app)


@pytest.fixture
def ws_client(tmp_path, settings_factory):
    return build_ws_client(tmp_path, settings_factory)


def test_ws_connection_sends_ready_with_session_id(ws_client):
    with ws_client.websocket_connect("/ws/chat") as websocket:
        frame = websocket.receive_json()
        assert frame["type"] == "ready"
        assert frame["session_id"]


def test_ws_message_streams_events_in_order(ws_client):
    with ws_client.websocket_connect("/ws/chat") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "message", "content": "保修多久"})

        types = []
        while True:
            frame = websocket.receive_json()
            types.append(frame["type"])
            if frame["type"] in ("done", "error"):
                break

        assert types[0] == "meta"
        assert "sources" in types
        assert types[-1] == "done"


def test_ws_invalid_frame_returns_error_without_closing(ws_client):
    with ws_client.websocket_connect("/ws/chat") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "message", "content": "  "})
        assert websocket.receive_json()["code"] == "CHAT_INVALID_MESSAGE"

        websocket.send_json({"type": "message", "content": "保修多久"})
        assert websocket.receive_json()["type"] == "meta"


def test_ws_rejects_unknown_origin(ws_client):
    with pytest.raises(WebSocketDisconnect):
        with ws_client.websocket_connect(
            "/ws/chat", headers={"origin": "http://evil.com"}
        ) as websocket:
            websocket.receive_json()
