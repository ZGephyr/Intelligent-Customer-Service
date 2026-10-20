"""注册、登录、验证码与失败限流测试。"""

from __future__ import annotations

import re

import pytest

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.services.auth_service import AuthService

SECRET = "s" * 32


def _answer(question: str) -> str:
    left, right = re.findall(r"\d+", question)[:2]
    return str(int(left) + int(right))


@pytest.fixture
def auth_ctx():
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    service = AuthService(factory, SECRET, bcrypt_rounds=4)
    return service, factory


def _register(service, username="alice_01", password="password123", role="user"):
    captcha_id, question = service.create_captcha("1.1.1.1")
    return service.register(username, password, captcha_id, _answer(question), role)


def test_register_creates_user_and_normalizes_username(auth_ctx):
    service, _ = auth_ctx

    user = _register(service, username="Alice_01")

    assert user.username == "alice_01" and user.role == "user"
    assert user.password_hash.startswith("$2")


def test_register_admin_role_when_allowed(auth_ctx):
    service, _ = auth_ctx

    user = _register(service, username="boss", role="admin")

    assert user.role == "admin"


def test_register_duplicate_username_raises(auth_ctx):
    service, _ = auth_ctx
    _register(service, username="alice_01")

    with pytest.raises(AppError) as excinfo:
        _register(service, username="alice_01")

    assert excinfo.value.code == ErrorCode.AUTH_USERNAME_TAKEN.value


def test_register_rejects_invalid_username(auth_ctx):
    service, _ = auth_ctx

    with pytest.raises(AppError) as excinfo:
        _register(service, username="ab")

    assert excinfo.value.code == ErrorCode.AUTH_INVALID_CREDENTIALS.value


def test_register_rejects_short_password(auth_ctx):
    service, _ = auth_ctx

    with pytest.raises(AppError) as excinfo:
        _register(service, password="short")

    assert excinfo.value.code == ErrorCode.AUTH_INVALID_CREDENTIALS.value


def test_register_with_wrong_captcha_answer_raises(auth_ctx):
    service, _ = auth_ctx
    captcha_id, _ = service.create_captcha("1.1.1.1")

    with pytest.raises(AppError) as excinfo:
        service.register("alice", "password123", captcha_id, "999", "user")

    assert excinfo.value.code == ErrorCode.AUTH_CAPTCHA_INVALID.value


def test_captcha_is_single_use(auth_ctx):
    service, _ = auth_ctx
    captcha_id, question = service.create_captcha("1.1.1.1")
    service.register("alice", "password123", captcha_id, _answer(question), "user")

    with pytest.raises(AppError) as excinfo:
        service.register("bob", "password123", captcha_id, _answer(question), "user")

    assert excinfo.value.code == ErrorCode.AUTH_CAPTCHA_INVALID.value


def test_captcha_unknown_id_raises(auth_ctx):
    service, _ = auth_ctx

    with pytest.raises(AppError) as excinfo:
        service.register("alice", "password123", "missing", "3", "user")

    assert excinfo.value.code == ErrorCode.AUTH_CAPTCHA_INVALID.value


def test_captcha_expired_raises(auth_ctx):
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    service = AuthService(factory, SECRET, bcrypt_rounds=4, captcha_ttl_seconds=-1)
    captcha_id, question = service.create_captcha("1.1.1.1")

    with pytest.raises(AppError) as excinfo:
        service.register("alice", "password123", captcha_id, _answer(question), "user")

    assert excinfo.value.code == ErrorCode.AUTH_CAPTCHA_EXPIRED.value


def test_register_admin_rejected_when_signup_disabled(auth_ctx):
    _, factory = auth_ctx
    service = AuthService(factory, SECRET, bcrypt_rounds=4, allow_admin_signup=False)
    captcha_id, question = service.create_captcha("1.1.1.1")

    with pytest.raises(AppError) as excinfo:
        service.register("boss", "password123", captcha_id, _answer(question), "admin")

    assert excinfo.value.code == ErrorCode.AUTH_FORBIDDEN.value


def test_login_returns_user_for_correct_password(auth_ctx):
    service, _ = auth_ctx
    _register(service)

    user = service.login("ALICE_01", "password123", "1.1.1.1")

    assert user.username == "alice_01"


def test_login_wrong_password_and_unknown_user_share_error(auth_ctx):
    service, _ = auth_ctx
    _register(service)

    with pytest.raises(AppError) as wrong:
        service.login("alice_01", "bad-password", "2.2.2.2")
    with pytest.raises(AppError) as missing:
        service.login("nobody", "password123", "2.2.2.2")

    assert wrong.value.code == ErrorCode.AUTH_INVALID_CREDENTIALS.value
    assert missing.value.code == ErrorCode.AUTH_INVALID_CREDENTIALS.value


def test_login_locked_after_too_many_failures(auth_ctx):
    service, _ = auth_ctx
    _register(service)

    for _ in range(10):
        with pytest.raises(AppError):
            service.login("alice_01", "bad-password", "3.3.3.3")

    with pytest.raises(AppError) as excinfo:
        service.login("alice_01", "password123", "3.3.3.3")

    assert excinfo.value.code == ErrorCode.AUTH_TOO_MANY_ATTEMPTS.value


def test_get_user_returns_none_for_unknown_id(auth_ctx):
    service, _ = auth_ctx

    assert service.get_user("missing") is None
