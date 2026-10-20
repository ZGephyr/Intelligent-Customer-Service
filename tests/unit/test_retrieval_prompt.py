"""检索与提示词测试。"""

from __future__ import annotations

from rag_service.core.types import ChatTurn, RetrievedChunk
from rag_service.services.prompt import build_context_block, build_messages, build_system_prompt
from rag_service.services.retrieval_service import RetrievalService

HITS = [
    RetrievedChunk("c1", "d1", "无关", "p.md", 0, 0.9),
    RetrievedChunk("c2", "d1", "保修一年", "p.md", 1, 0.8),
]


class FakeEmbedding:
    dimension = 8

    def embed_query(self, text):
        return [0.0] * 8

    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]


class FakeStore:
    def __init__(self, hits=None, error=None):
        self.hits = hits or []
        self.error = error
        self.calls = []

    def search(self, vector, top_k, score_threshold):
        self.calls.append(top_k)
        if self.error:
            raise self.error
        return self.hits[:top_k]


class FakeReranker:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def rerank(self, query, chunks, top_n):
        self.calls.append((query, top_n))
        if self.error:
            raise self.error
        return list(reversed(chunks))[:top_n]


def test_retrieve_uses_candidate_k_and_reranks_all_candidates():
    store, reranker = FakeStore(HITS), FakeReranker()
    out = RetrievalService(
        FakeEmbedding(), store, reranker, candidate_k=10, top_k=1
    ).retrieve("保修")
    assert store.calls == [10]
    assert reranker.calls == [("保修", 10)]
    assert [chunk.id for chunk in out] == ["c2"]


def test_retrieve_dedupes_hits_from_same_parent():
    same_parent = [
        RetrievedChunk("p1", "d1", "父块", "p.md", 0, 0.9, snippet="子块一"),
        RetrievedChunk("p1", "d1", "父块", "p.md", 1, 0.8, snippet="子块二"),
        RetrievedChunk("p2", "d1", "父块二", "p.md", 2, 0.7, snippet="子块三"),
    ]
    out = RetrievalService(FakeEmbedding(), FakeStore(same_parent), None, top_k=5).retrieve("保修")
    assert [chunk.id for chunk in out] == ["p1", "p2"]
    assert out[0].snippet == "子块一"


class FakeHybridEmbedding(FakeEmbedding):
    def embed_query_hybrid(self, text):
        return [0.0] * 8, {1: 0.5, 2: 0.3}


class FakeHybridStore(FakeStore):
    def __init__(self, hits=None):
        super().__init__(hits)
        self.hybrid_calls = []

    def hybrid_search(self, dense, sparse, branch_limit, limit, rrf_k):
        self.hybrid_calls.append((branch_limit, limit, rrf_k))
        return self.hits[:limit]


def test_retrieve_uses_hybrid_search_when_sparse_available():
    store = FakeHybridStore(HITS)
    out = RetrievalService(FakeHybridEmbedding(), store, None, candidate_k=20).retrieve("保修")
    assert store.hybrid_calls == [(20, 40, 60)]
    assert [chunk.id for chunk in out] == ["c1", "c2"]


def test_retrieve_filters_by_rerank_score_threshold():
    low = [RetrievedChunk("c1", "d1", "无关", "p.md", 0, 0.1)]
    svc = RetrievalService(FakeEmbedding(), FakeStore(low), FakeReranker(), rerank_score_threshold=0.3)
    assert svc.retrieve("保修") == []


def test_retrieve_without_reranker_truncates_recall_to_top_k():
    out = RetrievalService(FakeEmbedding(), FakeStore(HITS), None, top_k=1).retrieve("保修")
    assert [chunk.id for chunk in out] == ["c1"]


def test_retrieve_reranker_error_falls_back_to_recall_order():
    svc = RetrievalService(
        FakeEmbedding(), FakeStore(HITS), FakeReranker(error=RuntimeError("cuda oom")),
        candidate_k=10, top_k=2,
    )
    assert [chunk.id for chunk in svc.retrieve("保修")] == ["c1", "c2"]


def test_retrieve_store_error_returns_empty_list_for_fallback():
    svc = RetrievalService(FakeEmbedding(), FakeStore(error=RuntimeError("milvus down")))
    assert svc.retrieve("保修") == []


CHUNK = RetrievedChunk("c1", "d1", "保修一年", "policy.md", 0, 0.9)


def test_build_context_block_numbers_sources():
    assert "[1] 来源：policy.md" in build_context_block([CHUNK])


def test_build_system_prompt_contains_constraints_and_shop_name():
    prompt = build_system_prompt("酷玩店", "建议联系人工客服")
    assert "酷玩店" in prompt
    assert "不得使用外部知识" in prompt
    assert "建议联系人工客服" in prompt


def test_build_context_block_truncates_over_budget():
    big = RetrievedChunk("c2", "d1", "长" * 100, "big.md", 1, 0.8)
    assert "big.md" not in build_context_block([big], max_chars=30)


def test_build_messages_keeps_history_and_question():
    messages = build_messages(
        "保修多久",
        [CHUNK],
        [ChatTurn("user", "耳机"), ChatTurn("assistant", "199")],
        "店",
        "转人工",
    )
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "保修多久"}
    assert any(message["content"] == "耳机" for message in messages)
