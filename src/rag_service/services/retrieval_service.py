"""检索服务：稠密 + 稀疏 RRF 融合 → 父块去重 → 子块重排 → 阈值过滤。"""

from __future__ import annotations

from rag_service.core.logging import get_logger
from rag_service.core.types import RetrievedChunk
from rag_service.infra.embedding import EmbeddingProvider
from rag_service.infra.milvus_store import MilvusChunkStore
from rag_service.infra.reranker import RerankerProvider

logger = get_logger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedding: EmbeddingProvider,
        store: MilvusChunkStore,
        reranker: RerankerProvider | None = None,
        candidate_k: int = 20,
        rerank_candidates: int = 10,
        top_k: int = 3,
        score_threshold: float = 0.35,
        rerank_score_threshold: float = 0.3,
        rrf_k: int = 60,
    ) -> None:
        self._embedding = embedding
        self._store = store
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._rerank_candidates = rerank_candidates
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._rerank_score_threshold = rerank_score_threshold
        self._rrf_k = rrf_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        try:
            dense_vector, sparse_vector = self._embed_query(query)
            if sparse_vector:
                hits = self._store.hybrid_search(
                    dense_vector,
                    sparse_vector,
                    branch_limit=self._candidate_k,
                    limit=self._candidate_k * 2,
                    rrf_k=self._rrf_k,
                )
            else:
                hits = self._store.search(
                    dense_vector,
                    top_k=self._candidate_k,
                    score_threshold=self._score_threshold,
                )
        except Exception:  # noqa: BLE001 - 检索失败统一走兜底
            logger.exception("召回失败，降级为空结果: query=%s", query)
            return []

        if not hits:
            return []

        # RRF 顺序下去重父块，同一父块只保留排名最高的子块
        candidates = self._unique_parents(hits, self._candidate_k)
        if not candidates:
            return []

        if self._reranker is None:
            return candidates[: self._top_k]

        try:
            ranked = self._reranker.rerank(
                query, candidates[: self._rerank_candidates], top_n=self._rerank_candidates
            )
        except Exception:  # noqa: BLE001 - 重排失败退回 RRF 顺序
            logger.exception("重排失败，退回召回结果: query=%s", query)
            return candidates[: self._top_k]

        filtered = [hit for hit in ranked if hit.score >= self._rerank_score_threshold]
        if not filtered:
            logger.info(
                "重排分数全部低于阈值 %.2f，走兜底: query=%s",
                self._rerank_score_threshold,
                query,
            )
            return []
        return filtered[: self._top_k]

    def _embed_query(self, query: str) -> tuple[list[float], dict[int, float]]:
        hybrid = getattr(self._embedding, "embed_query_hybrid", None)
        if callable(hybrid):
            return hybrid(query)
        return self._embedding.embed_query(query), {}

    @staticmethod
    def _unique_parents(hits: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        seen: set[str] = set()
        selected: list[RetrievedChunk] = []
        for hit in hits:
            if hit.id in seen:
                continue
            seen.add(hit.id)
            selected.append(hit)
            if len(selected) >= limit:
                break
        return selected
