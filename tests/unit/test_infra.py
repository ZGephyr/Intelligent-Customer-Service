"""Embedding / LLM / Milvus / 重排 / 历史缓存 的单元测试。"""

from __future__ import annotations

import pytest

from rag_service.core.exceptions import AppError
from rag_service.core.types import IndexedChunk, RetrievedChunk
from rag_service.infra.embedding import BGEM3Embedding, build_embedding
from rag_service.infra.llm import OpenAIChatProvider
from rag_service.infra.milvus_store import MilvusChunkStore
from rag_service.infra.reranker import CrossEncoderReranker, build_reranker


class FakeBGEModel:
    def __init__(self, dim: int = 8, sparse: dict | None = None):
        self.dim = dim
        self.sparse = {1: 0.5, 2: 0.25} if sparse is None else sparse

    def encode(self, texts, **kwargs):
        return {
            "dense_vecs": [[3.0, 4.0] + [0.0] * (self.dim - 2) for _ in texts],
            "lexical_weights": [dict(self.sparse) for _ in texts],
        }


def test_embed_query_returns_normalized_dense_and_int_sparse():
    provider = BGEM3Embedding("unused", "cpu", 32, 8, model=FakeBGEModel())
    dense, sparse = provider.embed_query_hybrid("耳机")
    assert provider.dimension == 8
    assert dense[:2] == pytest.approx([0.6, 0.8])
    assert sparse == {1: 0.5, 2: 0.25}
    assert all(isinstance(token, int) for token in sparse)


def test_embed_documents_empty_input_returns_empty_list():
    provider = BGEM3Embedding("unused", "cpu", 32, 8, model=FakeBGEModel())
    assert provider.embed_documents([]) == []
    assert provider.embed_documents_hybrid([]) == ([], [])


def test_embedding_dimension_mismatch_raises_app_error():
    provider = BGEM3Embedding("unused", "cpu", 32, 8, model=FakeBGEModel(dim=4))
    with pytest.raises(AppError):
        provider.embed_query("x")


def test_build_embedding_returns_bge_m3_with_configured_dimension(settings_factory):
    provider = build_embedding(settings_factory())
    assert isinstance(provider, BGEM3Embedding)
    assert provider.dimension == 1024


def test_embedding_model_load_failure_raises_app_error(monkeypatch):
    import FlagEmbedding

    def boom(*args, **kwargs):
        raise RuntimeError("model not found")

    monkeypatch.setattr(FlagEmbedding, "BGEM3FlagModel", boom)
    provider = BGEM3Embedding("missing-model-path", "cpu", 32, 8)
    with pytest.raises(AppError):
        provider.embed_query("x")


class FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def stream(self, messages):
        yield type("C", (), {"content": "你"})()
        yield type("C", (), {"content": "好"})()

    def invoke(self, messages):
        return type("C", (), {"content": "改写后的问题"})()


def test_llm_stream_concatenates_chunk_contents(monkeypatch):
    monkeypatch.setattr("rag_service.infra.llm.ChatOpenAI", FakeChatModel)
    provider = OpenAIChatProvider("k", "http://x", "m", 0.2, 60)
    assert "".join(provider.stream([{"role": "user", "content": "hi"}])) == "你好"


def test_llm_invoke_returns_text(monkeypatch):
    monkeypatch.setattr("rag_service.infra.llm.ChatOpenAI", FakeChatModel)
    provider = OpenAIChatProvider("k", "http://x", "m", 0.2, 60)
    assert provider.invoke([{"role": "user", "content": "hi"}]) == "改写后的问题"


class RecordingChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invoke_kwargs = {}

    def invoke(self, messages, **kwargs):
        self.invoke_kwargs = kwargs
        return type("C", (), {"content": "ok"})()

    def stream(self, messages):
        yield type("C", (), {"content": "ok"})()


def test_llm_invoke_forwards_max_tokens_to_client(monkeypatch):
    monkeypatch.setattr("rag_service.infra.llm.ChatOpenAI", RecordingChatModel)
    provider = OpenAIChatProvider("k", "http://x", "m", 0.2, 60)

    provider.invoke([{"role": "user", "content": "hi"}], max_tokens=16)

    assert provider._client.invoke_kwargs["max_tokens"] == 16


def test_build_chat_provider_honours_temperature_retries_and_timeout(monkeypatch, settings_factory):
    from rag_service.infra.llm import build_chat_provider

    monkeypatch.setattr("rag_service.infra.llm.ChatOpenAI", RecordingChatModel)
    provider = build_chat_provider(
        settings_factory(), timeout_seconds=1.5, temperature=0.0, max_retries=0
    )

    assert provider._client.kwargs["temperature"] == 0.0
    assert provider._client.kwargs["max_retries"] == 0
    assert provider._client.kwargs["timeout"] == 1.5


def test_build_chat_provider_disables_thinking_when_configured(monkeypatch, settings_factory):
    from rag_service.infra.llm import build_chat_provider

    monkeypatch.setattr("rag_service.infra.llm.ChatOpenAI", RecordingChatModel)
    provider = build_chat_provider(settings_factory(), enable_thinking=False)

    assert provider._client.kwargs["extra_body"] == {"enable_thinking": False}


def test_build_chat_provider_omits_extra_body_when_thinking_unset(monkeypatch, settings_factory):
    from rag_service.infra.llm import build_chat_provider

    monkeypatch.setattr("rag_service.infra.llm.ChatOpenAI", RecordingChatModel)
    provider = build_chat_provider(settings_factory())

    assert "extra_body" not in provider._client.kwargs


class FakeMilvusClient:
    def __init__(self):
        self.deleted = []
        self.upserted = []
        self.hybrid_used = False

    def has_collection(self, name):
        return True

    def describe_collection(self, name):
        return {"fields": [{"name": "embedding", "params": {"dim": 8}}]}

    def upsert(self, collection_name, data):
        self.upserted.extend(data)
        return {"upsert_count": len(data)}

    def search(self, collection_name, data, limit, output_fields, search_params):
        return self._hits()

    def hybrid_search(self, collection_name, reqs, ranker, limit, output_fields):
        self.hybrid_used = True
        return self._hits()

    def _hits(self):
        return [
            [
                {"id": "c1", "distance": 0.9, "entity": {"document_id": "d1", "chunk_index": 0,
                                                        "content": "保修一年", "parent_id": "p1",
                                                        "parent_content": "父块一", "source": "policy.md"}},
                {"id": "c2", "distance": 0.1, "entity": {"document_id": "d1", "chunk_index": 1,
                                                        "content": "无关", "parent_id": "p2",
                                                        "parent_content": "父块二", "source": "policy.md"}},
            ]
        ]

    def delete(self, collection_name, filter):
        self.deleted.append(filter)

    def close(self):
        pass


def test_milvus_search_maps_parent_context_and_child_snippet():
    store = MilvusChunkStore("http://x", None, "kb_chunks", 8, client=FakeMilvusClient())
    hits = store.search([0.0] * 8, top_k=5, score_threshold=0.35)
    assert [hit.id for hit in hits] == ["p1"]
    assert hits[0].content == "父块一"
    assert hits[0].snippet == "保修一年"
    assert hits[0].score == 0.9


def test_milvus_delete_document_uses_document_id_filter():
    client = FakeMilvusClient()
    MilvusChunkStore("http://x", None, "kb_chunks", 8, client=client).delete_document("d1")
    assert 'document_id == "d1"' in client.deleted[0]


def test_milvus_upsert_maps_indexed_chunks_to_rows():
    client = FakeMilvusClient()
    store = MilvusChunkStore("http://x", None, "kb_chunks", 8, client=client)
    store.upsert([IndexedChunk("c1", "d1", 0, "x", "a.md", "md", 1, [0.0] * 8)])
    assert client.upserted[0]["id"] == "c1"


def test_milvus_hybrid_search_maps_parent_context():
    client = FakeMilvusClient()
    store = MilvusChunkStore("http://x", None, "kb_chunks", 8, client=client)
    hits = store.hybrid_search([0.0] * 8, {1: 0.5}, branch_limit=20, limit=40)
    assert client.hybrid_used is True
    assert hits[0].id == "p1"
    assert hits[0].snippet == "保修一年"


class FakeRerankEncoder:
    def predict(self, pairs):
        return [0.1 if "无关" in content else 0.9 for _, content in pairs]


def _chunk(cid, content, score):
    return RetrievedChunk(cid, "d1", content, "p.md", 0, score)


def test_rerank_orders_by_cross_encoder_score_and_truncates():
    chunks = [_chunk("c1", "无关内容", 0.9), _chunk("c2", "保修一年", 0.8)]
    out = CrossEncoderReranker("unused", "cpu", encoder=FakeRerankEncoder()).rerank(
        "保修", chunks, top_n=1
    )
    assert [chunk.id for chunk in out] == ["c2"]


def test_build_reranker_returns_none_when_disabled(settings_factory):
    assert build_reranker(settings_factory(rerank_enabled=False)) is None
