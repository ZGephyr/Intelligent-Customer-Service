"""配置模块测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_service.core.config import Settings

BASE = dict(
    llm_api_key="test-key",
    llm_base_url="http://127.0.0.1:9999/v1",
    llm_model="test-model",
    embedding_model_path="models/bge",
    rerank_model_path="models/rerank",
    milvus_uri="http://127.0.0.1:19530",
    mysql_url="sqlite://",
    redis_url="redis://127.0.0.1:6379/0",
    admin_token="admin-token",
)


def test_settings_missing_admin_token_raises_validation_error(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    env = {k: v for k, v in BASE.items() if k != "admin_token"}
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **env)


def test_settings_defaults_and_allowed_origins_parse_to_list():
    s = Settings(_env_file=None, ws_allowed_origins="http://a.com, http://b.com", **BASE)
    assert s.top_k == 3
    assert s.score_threshold == 0.35
    assert s.embedding_dim == 1024
    assert s.embedding_batch_size == 32
    assert s.rerank_enabled is True
    assert s.candidate_k == 20
    assert s.rerank_candidates == 10
    assert s.rerank_score_threshold == 0.3
    assert s.allowed_origins == ["http://a.com", "http://b.com"]


def test_settings_accepts_dashscope_env_alias(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "alias-key")
    env = {k: v for k, v in BASE.items() if k != "llm_api_key"}
    s = Settings(_env_file=None, **env)
    assert s.llm_api_key == "alias-key"


def test_settings_allowed_origins_ignores_blank_items():
    s = Settings(_env_file=None, ws_allowed_origins="http://a.com, ,", **BASE)
    assert s.allowed_origins == ["http://a.com"]


def test_settings_intent_and_faq_defaults(settings_factory):
    s = settings_factory()
    assert s.intent_enabled is True and s.intent_timeout_seconds == 60
    assert s.faq_score_threshold == 0.8 and s.faq_margin == 0.0
    assert s.faq_min_coverage == 0.30
    assert s.faq_enabled is True and s.hyde_enabled is True and s.faq_softmax_temperature == 1.0
    assert s.rewrite_mode == "full" and s.rewrite_timeout_seconds == 60


def test_settings_abbreviations_map_parses_pairs(settings_factory):
    s = settings_factory(abbreviations="降噪=主动降噪, hrv=心率变异性, 坏项, =, a=")
    assert s.abbreviations_map == {"降噪": "主动降噪", "hrv": "心率变异性"}


def test_settings_abbreviations_map_empty_when_unset(settings_factory):
    assert settings_factory().abbreviations_map == {}


def test_error_codes_include_faq_members():
    from rag_service.core.exceptions import ErrorCode

    assert ErrorCode.FAQ_DUPLICATE.value == "FAQ_DUPLICATE"
    assert ErrorCode.FAQ_NOT_FOUND.value == "FAQ_NOT_FOUND"
    assert ErrorCode.FAQ_IMPORT_INVALID.value == "FAQ_IMPORT_INVALID"


def test_settings_auth_defaults(settings_factory):
    s = settings_factory()

    assert s.auth_cookie_name == "rag_session"
    assert s.auth_token_ttl_days == 7 and s.cookie_secure is False
    assert s.allow_admin_signup is True
    assert s.captcha_ttl_seconds == 300 and s.bcrypt_rounds == 12
    assert s.auth_max_attempts == 10 and s.auth_lock_seconds == 60


def test_error_codes_include_auth_members():
    from rag_service.core.exceptions import ErrorCode

    for name in (
        "AUTH_INVALID_CREDENTIALS",
        "AUTH_USERNAME_TAKEN",
        "AUTH_CAPTCHA_INVALID",
        "AUTH_CAPTCHA_EXPIRED",
        "AUTH_FORBIDDEN",
        "AUTH_TOO_MANY_ATTEMPTS",
        "CHAT_SESSION_NOT_FOUND",
    ):
        assert getattr(ErrorCode, name).value == name
