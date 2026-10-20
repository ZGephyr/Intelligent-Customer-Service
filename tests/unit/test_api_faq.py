"""FAQ 管理接口测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.main import AppServices, create_app
from rag_service.services.auth_service import AuthService
from rag_service.services.faq_admin_service import FaqAdminService
from rag_service.services.faq_service import FaqService
from rag_service.services.session_service import SessionService


@pytest.fixture
def faq_client(settings_factory, register_user):
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    admin = FaqAdminService(factory, FaqService(factory))
    settings = settings_factory(admin_auth_enabled=False, bcrypt_rounds=4)
    app = create_app(
        settings,
        services=AppServices(
            chat=None,
            ingestion=None,
            health=None,
            faq_admin=admin,
            auth=AuthService(factory, "s" * 32, bcrypt_rounds=4),
            sessions=SessionService(factory),
        ),
    )
    client = TestClient(app)
    register_user(client, role="admin")
    return client


def test_create_then_list_faq(faq_client):
    created = faq_client.post("/api/admin/faqs", json={"question": "多久发货", "answer": "48 小时"})

    assert created.status_code == 200
    assert created.json()["data"]["question"] == "多久发货"
    listed = faq_client.get("/api/admin/faqs?page=1&size=20")
    assert listed.json()["data"]["total"] == 1


def test_create_missing_answer_returns_400(faq_client):
    response = faq_client.post("/api/admin/faqs", json={"question": "只有问题", "answer": ""})

    assert response.status_code == 400
    assert response.json()["code"] == "FAQ_IMPORT_INVALID"


def test_create_duplicate_returns_400_with_code(faq_client):
    faq_client.post("/api/admin/faqs", json={"question": "多久发货", "answer": "48 小时"})

    duplicate = faq_client.post("/api/admin/faqs", json={"question": "多久发货", "answer": "24 小时"})

    assert duplicate.status_code == 400 and duplicate.json()["code"] == "FAQ_DUPLICATE"


def test_update_and_delete_faq(faq_client):
    faq_id = faq_client.post(
        "/api/admin/faqs", json={"question": "多久发货", "answer": "48 小时"}
    ).json()["data"]["id"]

    updated = faq_client.put(f"/api/admin/faqs/{faq_id}", json={"answer": "24 小时"})
    assert updated.json()["data"]["answer"] == "24 小时"

    assert faq_client.delete(f"/api/admin/faqs/{faq_id}").status_code == 200
    assert faq_client.get(f"/api/admin/faqs/{faq_id}").status_code in (404, 405)


def test_delete_missing_returns_404(faq_client):
    response = faq_client.delete("/api/admin/faqs/missing")

    assert response.status_code == 404 and response.json()["code"] == "FAQ_NOT_FOUND"


def test_import_csv_returns_counts(faq_client):
    payload = "问题,答案\n多久发货,48 小时\n发票怎么开,备注即可\n"

    response = faq_client.post(
        "/api/admin/faqs/import",
        files={"file": ("faqs.csv", payload.encode("utf-8"), "text/csv")},
    )

    data = response.json()["data"]
    assert response.status_code == 200
    assert (data["created"], data["updated"], data["skipped"]) == (2, 0, 0)


def test_import_unsupported_file_returns_400(faq_client):
    response = faq_client.post(
        "/api/admin/faqs/import",
        files={"file": ("faqs.png", b"x", "image/png")},
    )

    assert response.status_code == 400 and response.json()["code"] == "FAQ_IMPORT_INVALID"


def test_import_broken_xlsx_returns_400(faq_client):
    response = faq_client.post(
        "/api/admin/faqs/import",
        files={"file": ("bad.xlsx", b"not a zip file", "application/vnd.ms-excel")},
    )

    assert response.status_code == 400 and response.json()["code"] == "FAQ_IMPORT_INVALID"


def test_import_empty_xlsx_returns_400(faq_client):
    response = faq_client.post(
        "/api/admin/faqs/import",
        files={"file": ("empty.xlsx", b"", "application/vnd.ms-excel")},
    )

    assert response.status_code == 400 and response.json()["code"] == "FAQ_IMPORT_INVALID"


def test_import_csv_without_required_columns_returns_400(faq_client):
    payload = "foo,bar\n1,2\n"

    response = faq_client.post(
        "/api/admin/faqs/import",
        files={"file": ("wrong.csv", payload.encode("utf-8"), "text/csv")},
    )

    assert response.status_code == 400 and response.json()["code"] == "FAQ_IMPORT_INVALID"


def test_update_blank_answer_returns_400(faq_client):
    faq_id = faq_client.post(
        "/api/admin/faqs", json={"question": "多久发货", "answer": "48 小时"}
    ).json()["data"]["id"]

    response = faq_client.put(f"/api/admin/faqs/{faq_id}", json={"answer": "   "})

    assert response.status_code == 400 and response.json()["code"] == "FAQ_IMPORT_INVALID"


def test_stats_returns_totals(faq_client):
    faq_client.post("/api/admin/faqs", json={"question": "多久发货", "answer": "48 小时"})

    stats = faq_client.get("/api/admin/faqs/stats").json()["data"]

    assert stats["total"] == 1 and stats["enabled"] == 1 and stats["hit_total"] == 0
