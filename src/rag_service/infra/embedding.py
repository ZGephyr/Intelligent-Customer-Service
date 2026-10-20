"""Embedding 适配层：本地 BGE-M3 混合编码（稠密 + 稀疏）。"""

from __future__ import annotations

import math
from typing import Any, Protocol

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents_hybrid(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]: ...

    def embed_query_hybrid(self, text: str) -> tuple[list[float], dict[int, float]]: ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - 没有 torch 时退回 CPU
        return "cpu"


class BGEM3Embedding:
    """BGE-M3 混合编码：dense（1024 维）+ sparse（词权重）。"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        batch_size: int = 32,
        dimension: int = 1024,
        model: Any | None = None,
    ) -> None:
        self._model_path = model_path
        self._device = _resolve_device(device)
        self._batch_size = batch_size
        self._dimension = dimension
        self._model = model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(
                    self._model_path, use_fp16=False, device=self._device
                )
                logger.info(
                    "BGE-M3 混合模型加载完成: %s（device=%s）", self._model_path, self._device
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("BGE-M3 模型加载失败: %s", self._model_path)
                raise AppError(
                    ErrorCode.INTERNAL_ERROR, f"BGE-M3 模型加载失败: {self._model_path}"
                ) from exc
        return self._model

    def _encode(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        model = self._ensure_model()
        try:
            output = model.encode(
                texts,
                batch_size=self._batch_size,
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("BGE-M3 编码失败，条数=%s", len(texts))
            raise AppError(ErrorCode.INTERNAL_ERROR, "BGE-M3 编码失败") from exc

        dense: list[list[float]] = []
        for vector in output["dense_vecs"]:
            values = _normalize([float(value) for value in vector])
            if len(values) != self._dimension:
                raise AppError(
                    ErrorCode.INTERNAL_ERROR,
                    f"Embedding 维度不一致: 期望 {self._dimension}，实际 {len(values)}",
                )
            dense.append(values)

        sparse = [
            {
                int(token): float(weight)
                for token, weight in weights.items()
                if float(weight) > 0
            }
            for weights in output["lexical_weights"]
        ]
        return dense, sparse

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)[0]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0][0]

    def embed_documents_hybrid(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        if not texts:
            return [], []
        return self._encode(texts)

    def embed_query_hybrid(self, text: str) -> tuple[list[float], dict[int, float]]:
        dense, sparse = self._encode([text])
        return dense[0], sparse[0]

    def warmup(self) -> None:
        self._encode(["预热"])


def build_embedding(settings) -> EmbeddingProvider:
    """构建本地 BGE-M3 混合编码器。"""
    return BGEM3Embedding(
        settings.embedding_model_path,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        dimension=settings.embedding_dim,
    )
