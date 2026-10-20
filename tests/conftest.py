"""共享测试夹具。"""

from __future__ import annotations

import pytest

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


@pytest.fixture
def settings_factory():
    """构造 Settings，默认值对测试友好，可逐项覆盖。"""

    def make(**overrides):
        return Settings(_env_file=None, **{**BASE, **overrides})

    return make


@pytest.fixture
def settings(settings_factory):
    return settings_factory()
