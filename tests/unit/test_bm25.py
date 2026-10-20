"""BM25 打分、softmax 归一化与中文分词测试。"""

from __future__ import annotations

import pytest

from rag_service.services.bm25 import Bm25Index, tokenize, tokenize_hybrid

CORPUS = ["发票怎么开", "多久发货", "怎么退货", "耳机保修多久"]


def test_tokenize_keeps_content_words_and_drops_stopwords():
    tokens = tokenize("这个耳机的保修多久？")
    assert "耳机" in tokens and "保修" in tokens
    assert "这个" not in tokens and "的" not in tokens


def test_normalized_scores_exact_match_dominates_distribution():
    index = Bm25Index(CORPUS)

    scores = index.normalized_scores("多久发货")

    assert scores[1] == max(scores) and scores[1] > 0.7
    assert scores[1] > scores[3] > 0
    assert sum(scores) == pytest.approx(1.0, abs=1e-6)


def test_normalized_scores_zero_when_no_overlap():
    index = Bm25Index(["耳机保修多久"])

    assert index.normalized_scores("发票怎么开") == [0.0]


def test_normalized_scores_prefers_more_overlap():
    index = Bm25Index(["耳机保修多久", "耳机"])

    scores = index.normalized_scores("耳机保修")

    assert scores[0] > scores[1]


def test_normalized_scores_empty_query_returns_zeros():
    assert Bm25Index(["耳机"]).normalized_scores("   ") == [0.0]


def test_normalized_scores_single_document_corpus_is_one():
    scores = Bm25Index(["发货时间"]).normalized_scores("发货")

    assert len(scores) == 1 and scores[0] == 1.0


def test_normalized_scores_higher_temperature_flattens_distribution():
    sharp = Bm25Index(CORPUS, temperature=1.0).normalized_scores("多久发货")
    flat = Bm25Index(CORPUS, temperature=2.0).normalized_scores("多久发货")

    assert sharp[1] > flat[1]


def test_tokenize_hybrid_adds_char_bigrams():
    tokens = tokenize_hybrid("退货运费谁承担")

    assert "运费" in tokens
    assert "货运" in tokens and "费谁" in tokens


def test_hybrid_index_matches_when_jieba_word_boundary_differs():
    index = Bm25Index(["退货运费谁承担", "多久发货"], tokenizer=tokenize_hybrid)

    assert index.normalized_scores("运费谁出")[0] > 0.0


def test_default_index_still_uses_pure_jieba():
    index = Bm25Index(["退货运费谁承担"])

    assert index.normalized_scores("运费谁出")[0] == 0.0


def test_unknown_query_terms_do_not_win_the_distribution():
    index = Bm25Index(CORPUS, tokenizer=tokenize_hybrid)

    exact = index.normalized_scores("发票怎么开")
    off_topic = index.normalized_scores("怎么学英语")

    assert exact[0] > 0.9
    assert max(off_topic) < 0.55


def test_coverage_scores_full_for_identical_text():
    index = Bm25Index(["耳机保修多久"], tokenizer=tokenize_hybrid)

    assert index.coverage_scores("耳机保修多久")[0] == pytest.approx(1.0)


def test_coverage_scores_penalize_unrelated_query():
    index = Bm25Index(["耳机保修多久"], tokenizer=tokenize_hybrid)

    coverage = index.coverage_scores("店里有哪几款蓝牙耳机")[0]

    assert 0 < coverage < 0.3