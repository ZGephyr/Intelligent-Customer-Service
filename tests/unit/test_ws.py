"""WebSocket 接口测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from rag_service.infra.db import build_engine, build_session_factory, create_all
from rag_service.main import AppServices, create_app
from rag_service.services.auth_service import AuthService
from rag_service.services.chat_service import ChatService
from rag_service.services.health_service import HealthService
from rag_service.services.session_service import SessionService
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


class FakeIntent:
    def __init__(self, intent):
        self.intent = intent

    def classify(self, question, history):
        from rag_service.services.intent_service import IntentResult

        return IntentResult(self.intent)


class FakeFaq:
    def __init__(self, hit=None):
        self.hit = hit

    def match(self, question):
        return self.hit

    def record_hit(self, faq_id):
        pass


def build_ws_client(tmp_path, settings_factory, intent=None, faq=None, register_user=None):
    settings = settings_factory(
        upload_dir=tmp_path / "uploads",
        log_dir=tmp_path / "logs",
        ws_allowed_origins="http://testserver",
        bcrypt_rounds=4,
    )
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)

    sessions = SessionService(factory)
    chat = ChatService(
        factory,
        FakeHistory(),
        FakeRetrieval(),
        FakeRewrite(),
        FakeLLM(),
        settings,
        intent=intent,
        faq=faq,
        sessions=sessions,
    )
    ingestion = IngestionService(
        factory, FakeEmbedding(), FakeStore(), tmp_path / "uploads", max_upload_mb=1
    )
    health = HealthService(engine, FakeRedis(), FakeStore(), FakeEmbedding())
    app = create_app(
        settings,
        AppServices(
            chat=chat,
            ingestion=ingestion,
            health=health,
            auth=AuthService(factory, "s" * 32, bcrypt_rounds=4),
            sessions=sessions,
        ),
    )
    client = TestClient(app)
    if register_user is not None:
        register_user(client, username="ws_tester", role="user")
    return client


@pytest.fixture
def ws_client(tmp_path, settings_factory, register_user):
    return build_ws_client(tmp_path, settings_factory, register_user=register_user)


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


def test_ws_meta_carries_intent_and_sources_type(tmp_path, settings_factory, register_user):
    from rag_service.core.types import FaqHit
    from rag_service.services.intent_service import IN_SCOPE, OUT_OF_SCOPE

    client = build_ws_client(
        tmp_path,
        settings_factory,
        intent=FakeIntent(IN_SCOPE),
        faq=FakeFaq(FaqHit("f1", "多久发货", "48 小时", 0.9, "多久发货")),
        register_user=register_user,
    )
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "message", "content": "多久发货"})

        frames = []
        while True:
            frame = websocket.receive_json()
            frames.append(frame)
            if frame["type"] in ("done", "error"):
                break

    assert frames[0]["type"] == "meta" and frames[0]["intent"] == IN_SCOPE
    sources = [frame for frame in frames if frame["type"] == "sources"][0]
    assert sources["sources"][0]["type"] == "faq"
    assert frames[-1]["status"] == "faq"


def test_ws_out_of_scope_status_reaches_client(tmp_path, settings_factory, register_user):
    from rag_service.services.intent_service import OUT_OF_SCOPE

    client = build_ws_client(
        tmp_path, settings_factory, intent=FakeIntent(OUT_OF_SCOPE), register_user=register_user
    )
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "message", "content": "今天天气怎么样"})

        frames = []
        while True:
            frame = websocket.receive_json()
            frames.append(frame)
            if frame["type"] in ("done", "error"):
                break

    assert frames[0]["intent"] == OUT_OF_SCOPE
    assert frames[-1]["type"] == "done" and frames[-1]["status"] == "out_of_scope"


def test_ws_chitchat_status_reaches_client(tmp_path, settings_factory, register_user):
    from rag_service.services.intent_service import CHITCHAT

    client = build_ws_client(
        tmp_path, settings_factory, intent=FakeIntent(CHITCHAT), register_user=register_user
    )
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "message", "content": "你好呀"})

        frames = []
        while True:
            frame = websocket.receive_json()
            frames.append(frame)
            if frame["type"] in ("done", "error"):
                break

    assert frames[0]["intent"] == CHITCHAT
    assert frames[-1]["status"] == "chitchat"
