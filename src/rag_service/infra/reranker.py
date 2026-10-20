"""重排：sentence-transformers CrossEncoder。"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Protocol

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.core.types import RetrievedChunk

logger = get_logger(__name__)


class RerankerProvider(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]: ...


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
        if self._encoder is None:
            try:
                from sentence_transformers import CrossEncoder

                self._encoder = CrossEncoder(self._model_path, device=self._device)
                logger.info("重排模型加载完成: %s（device=%s）", self._model_path, self._device)
            except Exception as exc:  # noqa: BLE001
                logger.exception("重排模型加载失败: %s", self._model_path)
                raise AppError(
                    ErrorCode.INTERNAL_ERROR, f"重排模型加载失败: {self._model_path}"
                ) from exc
        return self._encoder

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

        ranked = sorted(zip(chunks, scores), key=lambda pair: float(pair[1]), reverse=True)
        selected = ranked[:top_n]
        logger.info(
            "重排完成: 候选=%s 返回=%s top_score=%.4f",
            len(chunks),
            len(selected),
            float(selected[0][1]) if selected else 0.0,
        )
        # CrossEncoder 输出是 logits，转成 0-1 概率供阈值判定与前端展示
        return [
            replace(chunk, score=1.0 / (1.0 + math.exp(-float(raw))))
            for chunk, raw in selected
        ]


def build_reranker(settings) -> CrossEncoderReranker | None:
    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(settings.rerank_model_path, settings.embedding_device)
