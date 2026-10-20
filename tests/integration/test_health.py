"""集成测试：依赖真实 MySQL / Redis / Milvus，默认跳过。"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"),
    reason="需设置 RUN_INTEGRATION=1 且外部依赖可用",
)


def test_health_endpoint_with_real_dependencies():
    from rag_service.core.config import get_settings
    from rag_service.main import create_app

    client = TestClient(create_app(get_settings()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
