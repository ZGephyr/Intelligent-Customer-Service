"""重排：CrossEncoder 优先；缺 modules.json 时用 transformers 直接加载。"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.core.types import RetrievedChunk

logger = get_logger(__name__)

DEGENERATE_EPSILON = 1e-6


class RerankerProvider(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...


def _load_cross_encoder(model_path: str, device: str):
    """sentence-transformers 格式（有 modules.json）走这里。"""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_path, device=device)


def _load_transformers_scorer(model_path: str, device: str):
    """裸 transformers 权重（只有 config + safetensors）走这里。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    if device and device != "cpu":
        model.to(device)
    return _TransformersScorer(tokenizer, model, device)


class _TransformersScorer:
    """给裸 transformers 重排模型套一个和 CrossEncoder 一致的 predict 接口。"""

    def __init__(self, tokenizer: Any, model: Any, device: str) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        queries = [pair[0] for pair in pairs]
        documents = [pair[1] for pair in pairs]
        with torch.no_grad():
            inputs = self._tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            if self._device and self._device != "cpu":
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
            logits = self._model(**inputs).logits
        return logits.view(-1).float().cpu().tolist()


class CrossEncoderReranker:
    def __init__(self, model_path: str, device: str = "auto", encoder: Any | None = None) -> None:
        self._model_path = model_path
        self._device = self._resolve_device(device)
        self._encoder = encoder

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _ensure_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        has_st_layout = (Path(self._model_path) / "modules.json").exists()
        try:
            if has_st_layout:
                self._encoder = _load_cross_encoder(self._model_path, self._device)
                logger.info("重排模型加载完成（CrossEncoder）: %s", self._model_path)
            else:
                logger.warning(
                    "重排模型缺少 modules.json，改用 transformers 直接加载: %s", self._model_path
                )
                self._encoder = _load_transformers_scorer(self._model_path, self._device)
                logger.info("重排模型加载完成（transformers）: %s", self._model_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("重排模型加载失败: %s", self._model_path)
            raise AppError(ErrorCode.INTERNAL_ERROR, f"重排模型加载失败: {self._model_path}") from exc
        return self._encoder

    def warmup(self) -> None:
        """启动时加载并跑一次推理，把模型问题暴露在启动阶段而不是首个请求。"""
        encoder = self._ensure_encoder()
        try:
            encoder.predict([("预热", "预热)")])
        except Exception as exc:  # noqa: BLE001
            logger.exception("重排模型预热失败: %s", self._model_path)
            raise AppError(ErrorCode.INTERNAL_ERROR, "重排模型预热失败") from exc
        logger.info("重排模型预热完成: %s", self._model_path)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        encoder = self._ensure_encoder()
        try:
            scores = encoder.predict(
                [(query, chunk.snippet or chunk.content) for chunk in chunks]
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("重排推理失败，条数=%s", len(chunks))
            raise AppError(ErrorCode.INTERNAL_ERROR, "重排推理失败") from exc

        values = [float(score) for score in scores]
        if len(values) > 1 and (max(values) - min(values)) < DEGENERATE_EPSILON:
            logger.warning(
                "重排分数无区分度（全部=%.6f），视为无有效片段，避免垃圾资料进入提示词", values[0]
            )
            return []

        ranked = sorted(zip(chunks, values), key=lambda pair: pair[1], reverse=True)
        selected = ranked[:top_n]
        logger.info(
            "重排完成: 候选=%s 返回=%s top_score=%.4f",
            len(chunks),
            len(selected),
            selected[0][1] if selected else 0.0,
        )
        return [
            replace(chunk, score=1.0 / (1.0 + math.exp(-raw)))
            for chunk, raw in selected
        ]


def build_reranker(settings) -> CrossEncoderReranker | None:
    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(settings.rerank_model_path, settings.embedding_device)
