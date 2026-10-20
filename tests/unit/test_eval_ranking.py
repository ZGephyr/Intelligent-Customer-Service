"""Recall@k / MRR / top_k 选择逻辑测试。"""

from __future__ import annotations

import pytest

from rag_service.evaluation.ranking import (
    RetrievalRecord,
    document_recall_at_k,
    mean_reciprocal_rank,
    rank_of,
    recall_at_k,
    recall_curve,
    select_top_k,
)


def record(gold: str, ranked: tuple[str, ...], docs: tuple[str, ...] = ()) -> RetrievalRecord:
    return RetrievalRecord(
        question_id="q1",
        gold_parent_id=gold,
        gold_document_id=docs[0] if docs else "d1",
        ranked_parent_ids=ranked,
        ranked_document_ids=docs,
    )


def test_rank_of_returns_one_based_position():
    assert rank_of("p2", ("p1", "p2", "p3")) == 2


def test_rank_of_returns_none_when_missing():
    assert rank_of("p9", ("p1", "p2")) is None


def test_recall_at_k_hits_only_when_gold_in_top_k():
    records = [record("p2", ("p1", "p2", "p3")), record("p9", ("p1", "p2", "p3"))]

    assert recall_at_k(records, 1) == 0.0
    assert recall_at_k(records, 2) == 0.5
    assert recall_at_k(records, 3) == 0.5


def test_recall_at_k_empty_records_returns_zero():
    assert recall_at_k([], 3) == 0.0


def test_recall_at_k_invalid_k_raises():
    with pytest.raises(ValueError):
        recall_at_k([record("p1", ("p1",))], 0)


def test_document_recall_at_k_accepts_any_parent_from_gold_document():
    records = [
        RetrievalRecord("q1", "p9", "d1", ("p1", "p2"), ("d1", "d2")),
        RetrievalRecord("q2", "p9", "d3", ("p3",), ("d2",)),
    ]

    assert document_recall_at_k(records, 1) == 0.5
    assert document_recall_at_k(records, 2) == 0.5


def test_mean_reciprocal_rank_averages_reciprocals_and_ignores_misses():
    records = [record("p1", ("p1", "p2")), record("p2", ("p1", "p2")), record("p9", ("p1",))]

    assert mean_reciprocal_rank(records) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_recall_curve_contains_every_k():
    curve = recall_curve([record("p2", ("p1", "p2"))], max_k=3)

    assert curve == {1: 0.0, 2: 1.0, 3: 1.0}


def test_select_top_k_returns_smallest_k_meeting_target():
    decision = select_top_k({1: 0.6, 2: 0.8, 3: 0.9, 4: 0.95}, target=0.9, min_k=3)

    assert decision.top_k == 3
    assert decision.achieved == 0.9


def test_select_top_k_respects_min_k_even_if_target_met_earlier():
    decision = select_top_k({1: 1.0, 2: 1.0, 3: 1.0}, target=0.9, min_k=3)

    assert decision.top_k == 3


def test_select_top_k_falls_back_to_max_when_target_unreachable():
    decision = select_top_k({3: 0.5, 4: 0.7}, target=0.9, min_k=3, max_k=4)

    assert decision.top_k == 4
    assert decision.achieved == 0.7
    assert "未达到目标" in decision.reason


def test_select_top_k_rejects_empty_curve():
    with pytest.raises(ValueError):
        select_top_k({})


def test_select_top_k_rejects_missing_k_in_curve():
    with pytest.raises(ValueError):
        select_top_k({3: 0.9}, min_k=3, max_k=4)


def test_select_top_k_rejects_max_k_less_than_min_k():
    with pytest.raises(ValueError):
        select_top_k({3: 0.9}, min_k=5, max_k=4)
