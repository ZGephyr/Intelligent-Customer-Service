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
    assert s.top_k == 5
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
