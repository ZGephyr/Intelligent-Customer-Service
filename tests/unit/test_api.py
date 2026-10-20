"""HTTP 接口测试：管理端鉴权、上传、会话、健康检查。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

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
    def __init__(self):
        self.upserted = []
        self.deleted = []

    def upsert(self, chunks):
        self.upserted.extend(chunks)

    def delete_document(self, document_id):
        self.deleted.append(document_id)

    def drop_and_recreate(self):
        pass

    def ping(self):
        return True


class FakeMilvusDown(FakeStore):
    def ping(self):
        return False


class FakeRedis:
    def ping(self):
        return True


class FakeHistory:
    def __init__(self):
        self.data = {}

    def get(self, session_id):
        return list(self.data.get(session_id, []))

    def append(self, session_id, turns):
        self.data.setdefault(session_id, []).extend(turns)


class FakeRetrieval:
    def retrieve(self, query):
        return []


class FakeRewrite:
    def rewrite(self, question, history):
        return RewriteOutcome(question, False)


class FakeLLM:
    def stream(self, messages):
        yield "你好"

    def invoke(self, messages, max_tokens=None):
        return ""


def build_client(tmp_path, settings_factory, milvus_down: bool = False, admin_auth_enabled: bool = False):
    settings = settings_factory(
        upload_dir=tmp_path / "uploads",
        log_dir=tmp_path / "logs",
        admin_auth_enabled=admin_auth_enabled,
    )
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)

    chat = ChatService(factory, FakeHistory(), FakeRetrieval(), FakeRewrite(), FakeLLM(), settings)
    ingestion = IngestionService(
        factory, FakeEmbedding(), FakeStore(), tmp_path / "uploads", max_upload_mb=1
    )
    store = FakeMilvusDown() if milvus_down else FakeStore()
    health = HealthService(engine, FakeRedis(), store, FakeEmbedding())

    app = create_app(settings, AppServices(chat=chat, ingestion=ingestion, health=health))
    return TestClient(app)


@pytest.fixture
def app_client(tmp_path, settings_factory):
    return build_client(tmp_path, settings_factory)


@pytest.fixture
def admin_headers():
    return {"X-Admin-Token": "admin-token"}


def test_admin_upload_without_token_allowed_when_auth_disabled(app_client):
    response = app_client.post(
        "/api/admin/documents", files={"file": ("a.md", b"x", "text/markdown")}
    )
    assert response.status_code == 202


def test_admin_requires_token_when_auth_enabled(tmp_path, settings_factory):
    client = build_client(tmp_path, settings_factory, admin_auth_enabled=True)
    response = client.post(
        "/api/admin/documents", files={"file": ("a.md", b"x", "text/markdown")}
    )
    assert response.status_code == 401
    assert response.json()["code"] == "UNAUTHORIZED"


def test_admin_config_returns_supported_extensions(app_client):
    response = app_client.get("/api/admin/config")
    assert response.status_code == 200
    data = response.json()["data"]
    assert {"md", "txt", "csv", "xlsx", "pdf", "docx"} <= set(data["supported_extensions"])
    assert data["max_upload_mb"] > 0
    assert data["admin_auth_enabled"] is False


def test_admin_upload_unsupported_type_returns_400(app_client, admin_headers):
    response = app_client.post(
        "/api/admin/documents",
        headers=admin_headers,
        files={"file": ("a.png", b"x", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "KB_FILE_TYPE_UNSUPPORTED"


def test_admin_upload_markdown_returns_202_and_appears_in_list(app_client, admin_headers):
    response = app_client.post(
        "/api/admin/documents",
        headers=admin_headers,
        files={"file": ("policy.md", "退货政策".encode(), "text/markdown")},
    )
    assert response.status_code == 202

    listing = app_client.get("/api/admin/documents?page=1&size=10", headers=admin_headers)
    assert listing.status_code == 200
    data = listing.json()["data"]
    assert data["total"] == 1
    assert {"items", "total", "page", "size"} <= data.keys()


def test_admin_delete_unknown_document_returns_404(app_client, admin_headers):
    response = app_client.delete("/api/admin/documents/not-exist", headers=admin_headers)
    assert response.status_code == 404
    assert response.json()["code"] == "KB_DOCUMENT_NOT_FOUND"


def test_create_session_then_list_messages_returns_empty_list(app_client):
    created = app_client.post("/api/sessions")
    session_id = created.json()["data"]["session_id"]
    response = app_client.get(f"/api/sessions/{session_id}/messages")
    assert response.status_code == 200
    assert response.json()["data"]["items"] == []


def test_health_returns_ok_when_all_dependencies_up(app_client):
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["checks"]["mysql"] == "ok"


def test_health_returns_503_when_milvus_down(tmp_path, settings_factory):
    client = build_client(tmp_path, settings_factory, milvus_down=True)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["checks"]["milvus"] != "ok"
