"""FAQ 管理服务测试：CRUD、导入与统计。"""

from __future__ import annotations

import pytest

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.services.faq_admin_service import FaqAdminService
from rag_service.services.faq_service import FaqService


@pytest.fixture
def admin_ctx():
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    faq = FaqService(factory)
    return FaqAdminService(factory, faq), factory


def test_create_duplicate_question_raises(admin_ctx):
    service, _ = admin_ctx
    service.create("多久发货", "48 小时")

    with pytest.raises(AppError) as excinfo:
        service.create("多久发货", "72 小时")
    assert excinfo.value.code == ErrorCode.FAQ_DUPLICATE.value


def test_list_filters_by_keyword_and_enabled(admin_ctx):
    service, _ = admin_ctx
    service.create("多久发货", "48 小时")
    disabled = service.create("发票怎么开", "备注即可")
    service.update(disabled.id, enabled=False)

    items, total = service.list(page=1, size=20, keyword="发货")

    assert total == 1 and items[0].question == "多久发货"
    assert service.list(page=1, size=20, enabled=False)[1] == 1


def test_update_missing_raises_not_found(admin_ctx):
    service, _ = admin_ctx

    with pytest.raises(AppError) as excinfo:
        service.update("missing", answer="x")
    assert excinfo.value.code == ErrorCode.FAQ_NOT_FOUND.value
    assert excinfo.value.http_status == 404


def test_delete_removes_row_and_invalidates_cache(admin_ctx):
    service, _ = admin_ctx
    row = service.create("多久发货", "48 小时")

    service.delete(row.id)

    assert service.list(page=1, size=20)[1] == 0


def test_import_rows_counts_created_updated_skipped(admin_ctx):
    service, _ = admin_ctx
    service.create("多久发货", "48 小时")

    result = service.import_rows([
        {"question": "多久发货", "answer": "24 小时"},
        {"question": "发票怎么开", "answer": "备注即可"},
        {"question": "", "answer": "空问题"},
    ])

    assert (result.created, result.updated, result.skipped) == (1, 1, 1)


def test_import_rows_over_limit_raises(admin_ctx):
    _, factory = admin_ctx
    service = FaqAdminService(factory, FaqService(factory), max_rows=2)

    with pytest.raises(AppError) as excinfo:
        service.import_rows([{"question": f"q{i}", "answer": "a"} for i in range(3)])
    assert excinfo.value.code == ErrorCode.FAQ_IMPORT_INVALID.value


def test_import_rows_duplicate_within_batch_updates_instead_of_raising(admin_ctx):
    service, _ = admin_ctx

    result = service.import_rows([
        {"question": "多久发货", "answer": "48 小时"},
        {"question": "多久发货", "answer": "24 小时"},
    ])

    assert (result.created, result.updated, result.skipped) == (1, 1, 0)
    assert service.list(page=1, size=20)[0][0].answer == "24 小时"


def test_update_blank_answer_raises(admin_ctx):
    service, _ = admin_ctx
    row = service.create("多久发货", "48 小时")

    with pytest.raises(AppError) as excinfo:
        service.update(row.id, answer="   ")
    assert excinfo.value.code == ErrorCode.FAQ_IMPORT_INVALID.value


def test_update_blank_question_raises(admin_ctx):
    service, _ = admin_ctx
    row = service.create("多久发货", "48 小时")

    with pytest.raises(AppError) as excinfo:
        service.update(row.id, question="")
    assert excinfo.value.code == ErrorCode.FAQ_IMPORT_INVALID.value


def test_stats_returns_totals_and_top_hits(admin_ctx):
    service, _ = admin_ctx
    row = service.create("多久发货", "48 小时")
    for _ in range(3):
        service.faq_service.record_hit(row.id)

    stats = service.stats()

    assert stats["total"] == 1 and stats["enabled"] == 1 and stats["hit_total"] == 3
    assert stats["top"][0]["question"] == "多久发货"
