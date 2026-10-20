"""重排模型加载路径、退化分数保护与预热测试。"""

from __future__ import annotations

import pytest

from rag_service.core.exceptions import AppError
from rag_service.core.types import RetrievedChunk
from rag_service.infra.reranker import CrossEncoderReranker


def chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(cid, "d1", text, "policy.md", 0, 0.5, text)


class FakeEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0
        self.pairs = None

    def predict(self, pairs):
        self.calls += 1
        self.pairs = list(pairs)
        return list(self.scores)


def test_uses_cross_encoder_when_modules_json_present(monkeypatch, tmp_path):
    (tmp_path / "modules.json").write_text("[]", encoding="utf-8")
    used = {}

    def fake_cross_encoder(path, device):
        used["path"] = path
        return FakeEncoder([2.0, -1.0])

    monkeypatch.setattr("rag_service.infra.reranker._load_cross_encoder", fake_cross_encoder)
    reranker = CrossEncoderReranker(str(tmp_path), "cpu")

    out = reranker.rerank("q", [chunk("c1", "a"), chunk("c2", "b")], top_n=2)

    assert used["path"] == str(tmp_path)
    assert [item.id for item in out] == ["c1", "c2"]
    assert out[0].score > 0.8 and out[1].score < 0.3


def test_falls_back_to_transformers_when_modules_json_missing(monkeypatch, tmp_path):
    used = {}

    def fake_cross_encoder(path, device):
        raise AssertionError("缺少 modules.json 时不应使用 CrossEncoder")

    def fake_transformers(path, device):
        used["path"] = path
        return FakeEncoder([1.0, -1.0])

    monkeypatch.setattr("rag_service.infra.reranker._load_cross_encoder", fake_cross_encoder)
    monkeypatch.setattr("rag_service.infra.reranker._load_transformers_scorer", fake_transformers)
    reranker = CrossEncoderReranker(str(tmp_path), "cpu")

    out = reranker.rerank("q", [chunk("c1", "a"), chunk("c2", "b")], top_n=1)

    assert used["path"] == str(tmp_path)
    assert [item.id for item in out] == ["c1"]


def test_rerank_returns_empty_when_scores_have_no_discrimination(monkeypatch, tmp_path):
    (tmp_path / "modules.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "rag_service.infra.reranker._load_cross_encoder",
        lambda path, device: FakeEncoder([0.0, 0.0, 0.0]),
    )
    reranker = CrossEncoderReranker(str(tmp_path), "cpu")

    out = reranker.rerank("q", [chunk(f"c{i}", "x") for i in range(3)], top_n=3)

    assert out == []


def test_warmup_loads_encoder_and_predicts_once(monkeypatch, tmp_path):
    (tmp_path / "modules.json").write_text("[]", encoding="utf-8")
    encoder = FakeEncoder([0.5])
    monkeypatch.setattr(
        "rag_service.infra.reranker._load_cross_encoder", lambda path, device: encoder
    )
    reranker = CrossEncoderReranker(str(tmp_path), "cpu")

    reranker.warmup()

    assert encoder.calls == 1


def test_load_failure_raises_app_error(monkeypatch, tmp_path):
    (tmp_path / "modules.json").write_text("[]", encoding="utf-8")

    def boom(path, device):
        raise RuntimeError("模型文件损坏")

    monkeypatch.setattr("rag_service.infra.reranker._load_cross_encoder", boom)
    reranker = CrossEncoderReranker(str(tmp_path), "cpu")

    with pytest.raises(AppError):
        reranker.warmup()
