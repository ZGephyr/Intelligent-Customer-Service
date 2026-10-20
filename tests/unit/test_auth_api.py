"""登录 / 注册接口与权限依赖测试。"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.main import AppServices, create_app
from rag_service.services.auth_service import AuthService


def _answer(question: str) -> str:
    left, right = re.findall(r"\d+", question)[:2]
    return str(int(left) + int(right))


class StubIngestion:
    def list_documents(self, page, size):
        return [], 0


@pytest.fixture
def auth_app(settings_factory):
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    auth = AuthService(factory, "s" * 32, bcrypt_rounds=4)
    settings = settings_factory(bcrypt_rounds=4, admin_auth_enabled=False)
    app = create_app(
        settings,
        services=AppServices(chat=None, ingestion=StubIngestion(), health=None, auth=auth),
    )
    return TestClient(app)


def _register(client, username="alice", password="password123", role="user"):
    captcha = client.get("/api/auth/captcha").json()["data"]
    return client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": password,
            "captcha_id": captcha["captcha_id"],
            "captcha_answer": _answer(captcha["question"]),
            "role": role,
        },
    )


def test_captcha_endpoint_exposes_admin_signup_flag(auth_app):
    payload = auth_app.get("/api/auth/captcha").json()["data"]

    assert payload["allow_admin_signup"] is True
    assert payload["captcha_id"] and "=" in payload["question"]


def test_register_sets_httponly_cookie_and_returns_user(auth_app):
    response = _register(auth_app)

    assert response.status_code == 200
    assert response.json()["data"]["username"] == "alice"
    cookie = response.headers["set-cookie"]
    lowered = cookie.lower()
    assert "httponly" in lowered and "samesite=lax" in lowered


def test_me_returns_user_after_register(auth_app):
    _register(auth_app)

    payload = auth_app.get("/api/auth/me").json()["data"]

    assert payload["username"] == "alice" and payload["role"] == "user"


def test_me_without_cookie_returns_401(auth_app):
    response = auth_app.get("/api/auth/me")

    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_INVALID_CREDENTIALS"


def test_logout_clears_cookie_and_session(auth_app):
    _register(auth_app)

    response = auth_app.post("/api/auth/logout")

    assert response.status_code == 200
    assert auth_app.get("/api/auth/me").status_code == 401


def test_login_with_wrong_password_returns_401(auth_app):
    _register(auth_app)

    response = auth_app.post("/api/auth/login", json={"username": "alice", "password": "wrong"})

    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_INVALID_CREDENTIALS"


def test_login_with_correct_password_sets_cookie(auth_app):
    _register(auth_app)
    auth_app.post("/api/auth/logout")

    response = auth_app.post("/api/auth/login", json={"username": "alice", "password": "password123"})

    assert response.status_code == 200 and "set-cookie" in response.headers
    assert auth_app.get("/api/auth/me").json()["data"]["username"] == "alice"


def test_non_admin_cannot_reach_admin_api(auth_app):
    _register(auth_app, username="bob", role="user")

    assert auth_app.get("/api/admin/documents").status_code == 403


def test_admin_can_reach_admin_api(auth_app):
    _register(auth_app, username="boss", role="admin")

    assert auth_app.get("/api/admin/documents").status_code == 200
