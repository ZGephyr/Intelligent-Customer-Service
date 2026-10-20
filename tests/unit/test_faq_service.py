"""FAQ 匹配服务测试：阈值、margin、别名、缓存与命中计数。"""

from __future__ import annotations

import pytest

from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
from rag_service.models.tables import FaqPair
from rag_service.services.faq_service import FaqService


@pytest.fixture
def faq_ctx():
    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)

    def seed(rows):
        with factory() as session:
            for row in rows:
                session.add(FaqPair(**row))
            session.commit()

    return factory, seed


def build(factory, **kwargs):
    return FaqService(factory, **kwargs)


def test_match_exact_question_returns_answer(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "多久发货", "answer": "48 小时内发出"}])

    hit = build(factory, threshold=0.5).match("多久发货")

    assert hit is not None
    assert hit.faq_id == "f1" and hit.answer == "48 小时内发出" and hit.matched_text == "多久发货"


def test_match_alias_line_returns_same_faq(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "多久发货", "answer": "48 小时", "aliases": "几天能到\n什么时候发货"}])

    hit = build(factory, threshold=0.5).match("几天能到")

    assert hit is not None and hit.faq_id == "f1" and hit.matched_text == "几天能到"


def test_match_below_threshold_returns_none(faq_ctx):
    factory, seed = faq_ctx
    seed([
        {"id": "f1", "question": "多久发货", "answer": "A"},
        {"id": "f2", "question": "发货时间", "answer": "B"},
    ])

    assert build(factory, threshold=0.55).match("发货") is None


def test_match_probability_split_returns_none(faq_ctx):
    factory, seed = faq_ctx
    seed([
        {"id": "f1", "question": "多久发货", "answer": "A"},
        {"id": "f2", "question": "发货时间", "answer": "B", "aliases": "多久发货"},
    ])

    assert build(factory, threshold=0.55).match("多久发货") is None


def test_match_ignores_disabled_rows(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "多久发货", "answer": "A", "enabled": False}])

    assert build(factory, threshold=0.1).match("多久发货") is None


def test_match_empty_table_returns_none(faq_ctx):
    factory, _ = faq_ctx

    assert build(factory).match("多久发货") is None


def test_match_disabled_service_returns_none(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "多久发货", "answer": "A"}])

    assert build(factory, enabled=False).match("多久发货") is None


def test_record_hit_increments_count_and_last_hit(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "多久发货", "answer": "48 小时"}])

    build(factory).record_hit("f1")

    with factory() as session:
        row = session.get(FaqPair, "f1")
        assert row.hit_count == 1 and row.last_hit_at is not None


def test_record_hit_unknown_id_does_not_raise(faq_ctx):
    factory, _ = faq_ctx

    build(factory).record_hit("missing")


def test_warmup_tokenizes_without_touching_database(faq_ctx, monkeypatch):
    import rag_service.services.faq_service as module

    calls = []
    monkeypatch.setattr(module, "tokenize_hybrid", lambda text: calls.append(text) or ["预热"])

    factory, _ = faq_ctx
    build(factory).warmup()

    assert calls == ["预热"]


def test_invalidate_picks_up_new_rows(faq_ctx):
    factory, seed = faq_ctx
    service = build(factory, threshold=0.5)

    assert service.match("发票怎么开") is None
    seed([{"id": "f2", "question": "发票怎么开", "answer": "下单备注即可"}])
    service.invalidate()

    hit = service.match("发票怎么开")
    assert hit is not None and hit.faq_id == "f2"


def test_last_top_scores_exposes_best_and_second(faq_ctx):
    factory, seed = faq_ctx
    seed([
        {"id": "f1", "question": "多久发货", "answer": "A"},
        {"id": "f2", "question": "发票怎么开", "answer": "B"},
    ])
    service = build(factory, threshold=0.5)

    assert service.match("多久发货") is not None
    best, second = service.last_top_scores
    assert best > second >= 0.0


def test_match_rejects_weak_single_candidate_by_coverage(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "耳机保修多久", "answer": "保修一年"}])

    hit = build(factory, threshold=0.5, min_coverage=0.3).match("店里有哪几款蓝牙耳机")

    assert hit is None, "只共享一个词、覆盖率极低时应拒绝"


def test_match_accepts_exact_question_under_coverage_gate(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "耳机保修多久", "answer": "保修一年"}])

    hit = build(factory, threshold=0.5, min_coverage=0.3).match("耳机保修多久")

    assert hit is not None and hit.faq_id == "f1"


def test_match_accepts_alias_under_coverage_gate(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "多久发货", "answer": "48 小时", "aliases": "几天能到"}])

    hit = build(factory, threshold=0.5, min_coverage=0.3).match("几天能到")

    assert hit is not None and hit.matched_text == "几天能到"


def test_coverage_gate_disabled_when_zero(faq_ctx):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "耳机保修多久", "answer": "保修一年"}])

    hit = build(factory, threshold=0.5, min_coverage=0.0).match("店里有哪几款蓝牙耳机")

    assert hit is not None, "门槛为 0 时应保持旧行为"


def test_match_logs_coverage_values(faq_ctx, caplog):
    factory, seed = faq_ctx
    seed([{"id": "f1", "question": "耳机保修多久", "answer": "保修一年"}])

    with caplog.at_level("INFO"):
        build(factory, threshold=0.5, min_coverage=0.3).match("耳机保修多久")

    assert any("覆盖率" in record.message for record in caplog.records)