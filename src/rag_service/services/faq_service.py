"""高频问答对匹配：内存 BM25 快照 + 阈值判定 + 命中计数。

匹配始终使用用户原始问题（发生在改写之前），命中后答案原样返回。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_service.core.logging import get_logger
from rag_service.core.types import FaqHit
from rag_service.models.tables import FaqPair
from rag_service.services.bm25 import Bm25Index, tokenize_hybrid

logger = get_logger(__name__)


@dataclass
class _Snapshot:
    """一次全量加载的结果：语料文档与 index 下标到 FAQ 的映射。"""

    index: Bm25Index = field(default_factory=lambda: Bm25Index([]))
    documents: list[tuple[str, str, str, str]] = field(default_factory=list)


class FaqService:
    def __init__(
        self,
        session_factory,
        *,
        enabled: bool = True,
        threshold: float = 0.55,
        margin: float = 0.0,
        cache_ttl_seconds: int = 60,
        temperature: float = 1.0,
        min_coverage: float = 0.0,
    ) -> None:
        self._session_factory = session_factory
        self._enabled = enabled
        self._threshold = threshold
        self._margin = margin
        self._cache_ttl_seconds = cache_ttl_seconds
        self._temperature = temperature
        self._min_coverage = min_coverage
        self._snapshot: _Snapshot | None = None
        self._loaded_at = 0.0
        self.last_top_scores: tuple[float, float] = (0.0, 0.0)

    # ---------- 匹配 ----------

    def match(self, question: str) -> FaqHit | None:
        if not self._enabled:
            return None
        snapshot = self._get_snapshot()
        if snapshot is None or not snapshot.documents:
            self.last_top_scores = (0.0, 0.0)
            return None

        try:
            scores = snapshot.index.normalized_scores(question)
        except Exception:  # noqa: BLE001 - 匹配失败不阻断问答
            logger.exception("FAQ 打分失败，按未命中处理")
            return None

        try:
            coverages = snapshot.index.coverage_scores(question)
        except Exception:  # noqa: BLE001 - 覆盖率失败时退回旧行为
            logger.exception("FAQ 覆盖率计算失败，按未命中处理")
            return None

        best_by_faq: dict[str, tuple[float, float, tuple[str, str, str, str]]] = {}
        for score, coverage, entry in zip(scores, coverages, snapshot.documents, strict=False):
            faq_id = entry[0]
            current = best_by_faq.get(faq_id)
            if current is None or score > current[0]:
                best_by_faq[faq_id] = (score, coverage, entry)

        ranked = sorted(best_by_faq.values(), key=lambda item: item[0], reverse=True)
        best = ranked[0][0]
        best_coverage = ranked[0][1]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        self.last_top_scores = (best, second)

        logger.info(
            "FAQ 匹配: query=%s 候选=%s top1=%.4f top2=%.4f 覆盖率=%.2f 阈值=%.2f 覆盖率门槛=%.2f",
            question,
            len(ranked),
            best,
            second,
            best_coverage,
            self._threshold,
            self._min_coverage,
        )

        if best < self._threshold:
            return None
        if self._min_coverage > 0 and best_coverage < self._min_coverage:
            logger.info(
                "FAQ 命中因覆盖率不足被放弃: top1=%.4f 覆盖率=%.2f 门槛=%.2f",
                best,
                best_coverage,
                self._min_coverage,
            )
            return None
        if self._margin > 0 and len(ranked) > 1 and best - second < self._margin:
            logger.info("FAQ 命中因分差不足被放弃: top1=%.4f top2=%.4f margin=%.2f", best, second, self._margin)
            return None

        faq_id, question_text, answer, matched_text = ranked[0][2]
        return FaqHit(
            faq_id=faq_id,
            question=question_text,
            answer=answer,
            score=best,
            matched_text=matched_text,
        )

    def record_hit(self, faq_id: str) -> None:
        try:
            with self._session_factory() as session:
                row = session.get(FaqPair, faq_id)
                if row is None:
                    return
                row.hit_count = (row.hit_count or 0) + 1
                row.last_hit_at = datetime.now()
                session.commit()
        except Exception:  # noqa: BLE001 - 统计失败不影响回答
            logger.exception("写入 FAQ 命中次数失败: faq_id=%s", faq_id)

    def invalidate(self) -> None:
        self._snapshot = None
        self._loaded_at = 0.0

    def warmup(self) -> None:
        """预热分词器：jieba 首次加载词典要几百毫秒，放在启动期而不是首个请求。"""
        started = time.perf_counter()
        tokenize_hybrid("预热")
        logger.info("FAQ 分词器预热完成: 耗时=%.0fms", (time.perf_counter() - started) * 1000)

    # ---------- 快照 ----------

    def _get_snapshot(self) -> _Snapshot | None:
        now = time.monotonic()
        if self._snapshot is not None and now - self._loaded_at <= self._cache_ttl_seconds:
            return self._snapshot
        try:
            with self._session_factory() as session:
                rows = (
                    session.query(FaqPair)
                    .filter(FaqPair.enabled.is_(True))
                    .order_by(FaqPair.created_at.asc())
                    .all()
                )
                entries: list[tuple[str, str, str, str]] = []
                for row in rows:
                    entries.append((row.id, row.question, row.answer, row.question))
                    for alias in (row.aliases or "").splitlines():
                        alias_text = alias.strip()
                        if alias_text:
                            entries.append((row.id, row.question, row.answer, alias_text))
        except Exception:  # noqa: BLE001 - FAQ 库不可用时继续走 RAG
            logger.exception("加载 FAQ 失败，本轮跳过 FAQ 匹配")
            return None

        self._snapshot = _Snapshot(
            index=Bm25Index(
                [entry[3] for entry in entries],
                tokenizer=tokenize_hybrid,
                temperature=self._temperature,
            ),
            documents=entries,
        )
        self._loaded_at = now
        logger.debug("FAQ 快照加载完成: 文档数=%s", len(entries))
        return self._snapshot
