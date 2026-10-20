"""密码哈希、JWT 与验证码答案哈希测试。"""

from __future__ import annotations

from rag_service.infra.security import (
    create_token,
    decode_token,
    hash_captcha_answer,
    hash_password,
    load_or_create_secret,
    verify_password,
)

SECRET = "k" * 32


def test_password_hash_is_not_plaintext_and_verifies():
    hashed = hash_password("secret123", rounds=4)

    assert hashed != "secret123" and hashed.startswith("$2")
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_password_longer_than_72_bytes_still_verifies():
    long_password = "长" * 200
    hashed = hash_password(long_password, rounds=4)

    assert verify_password(long_password, hashed) is True


def test_verify_password_with_broken_hash_returns_false():
    assert verify_password("x", "not-a-bcrypt-hash") is False


def test_token_roundtrip_and_tamper_rejected():
    token = create_token("u1", "admin", SECRET, ttl_seconds=60)

    payload = decode_token(token, SECRET)
    assert payload["sub"] == "u1" and payload["role"] == "admin"
    assert decode_token(token + "x", SECRET) is None


def test_token_signed_with_other_secret_is_rejected():
    token = create_token("u1", "user", SECRET, ttl_seconds=60)

    assert decode_token(token, "other" * 8) is None


def test_expired_token_is_rejected():
    token = create_token("u1", "user", SECRET, ttl_seconds=-1)

    assert decode_token(token, SECRET) is None


def test_load_or_create_secret_is_stable(tmp_path):
    path = tmp_path / "secret"

    first = load_or_create_secret(path)

    assert len(first) == 64
    assert load_or_create_secret(path) == first


def test_captcha_answer_hash_is_deterministic_and_keyed():
    assert hash_captcha_answer("9", "s") == hash_captcha_answer("9", "s")
    assert hash_captcha_answer("9", "s") != hash_captcha_answer("9", "other")
