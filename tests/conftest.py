"""共享测试夹具。"""

from __future__ import annotations

import re

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


@pytest.fixture
def register_user():
    """通过接口注册并登录，返回用户信息；测试里用它构造已登录客户端。"""

    def _register(client, username="tester", role="admin", password="password123"):
        captcha = client.get("/api/auth/captcha").json()["data"]
        numbers = re.findall(r"\d+", captcha["question"])[:2]
        response = client.post(
            "/api/auth/register",
            json={
                "username": username,
                "password": password,
                "captcha_id": captcha["captcha_id"],
                "captcha_answer": str(sum(int(value) for value in numbers)),
                "role": role,
            },
        )
        assert response.status_code == 200, response.text
        return response.json()["data"]

    return _register
