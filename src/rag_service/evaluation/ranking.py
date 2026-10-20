"""检索排序指标：Recall@k、文档级 Recall@k、MRR 与 top_k 选择。

这里只放纯函数，不依赖 Milvus、模型或 RAGAS，便于单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalRecord:
    """一道题的检索结果：按排名保存父块 ID 与文档 ID。"""

    question_id: str
    gold_parent_id: str
    gold_document_id: str
    ranked_parent_ids: tuple[str, ...]
    ranked_document_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopKDecision:
    """top_k 选择结果，保留完整曲线便于写报告。"""

    top_k: int
    target: float
    achieved: float
    reason: str
    curve: dict[int, float] = field(default_factory=dict)


def rank_of(gold_id: str, ranked_ids: tuple[str, ...]) -> int | None:
    """返回 gold 的 1-based 排名；未召回返回 None。"""
    try:
        return ranked_ids.index(gold_id) + 1
    except ValueError:
        return None


def recall_at_k(records: list[RetrievalRecord], k: int) -> float:
    """父块级 Recall@k：gold 父块是否出现在前 k 条。

    这是一个单金标问题集，因此 Recall@k 等价于 Hit@k。
    """
    if k < 1:
        raise ValueError("k 必须 >= 1")
    if not records:
        return 0.0
    hits = 0
    for record in records:
        rank = rank_of(record.gold_parent_id, record.ranked_parent_ids)
        if rank is not None and rank <= k:
            hits += 1
    return hits / len(records)


def document_recall_at_k(records: list[RetrievalRecord], k: int) -> float:
    """文档级 Recall@k：只要 gold 文档的任一父块进入前 k 即命中。"""
    if k < 1:
        raise ValueError("k 必须 >= 1")
    if not records:
        return 0.0
    hits = 0
    for record in records:
        ranked_docs = record.ranked_document_ids or tuple()
        if record.gold_document_id in ranked_docs[:k]:
            hits += 1
    return hits / len(records)


def mean_reciprocal_rank(records: list[RetrievalRecord]) -> float:
    """MRR：gold 父块排名倒数的平均值；未召回记 0。"""
    if not records:
        return 0.0
    total = 0.0
    for record in records:
        rank = rank_of(record.gold_parent_id, record.ranked_parent_ids)
        if rank is not None:
            total += 1.0 / rank
    return total / len(records)


def recall_curve(records: list[RetrievalRecord], max_k: int = 10) -> dict[int, float]:
    """返回 1..max_k 的父块级 Recall 曲线。"""
    if max_k < 1:
        raise ValueError("max_k 必须 >= 1")
    return {k: recall_at_k(records, k) for k in range(1, max_k + 1)}


def select_top_k(
    curve: dict[int, float],
    *,
    target: float = 0.90,
    min_k: int = 3,
    max_k: int | None = None,
) -> TopKDecision:
    """选择满足 Recall 目标的最小 k。

    规则（与用户确认一致）：
    - k 从 min_k 开始，避免上下文过少；
    - 取第一个 Recall@k >= target 的 k；
    - 全部不达标时取最大 k，并在 reason 中明确标注未达标。
    """
    if not curve:
        raise ValueError("召回曲线不能为空")
    if not 0.0 <= target <= 1.0:
        raise ValueError("target 必须在 [0, 1] 内")
    if min_k < 1:
        raise ValueError("min_k 必须 >= 1")

    upper = max_k if max_k is not None else max(curve)
    if upper < min_k:
        raise ValueError("max_k 不能小于 min_k")

    for k in range(min_k, upper + 1):
        if k not in curve:
            raise ValueError(f"召回曲线缺少 k={k}")

    for k in range(min_k, upper + 1):
        if curve[k] >= target:
            return TopKDecision(
                top_k=k,
                target=target,
                achieved=curve[k],
                reason=f"Recall@{k}={curve[k]:.4f} 首次达到目标 {target:.2f}",
                curve=dict(curve),
            )

    upper_value = curve[upper]
    return TopKDecision(
        top_k=upper,
        target=target,
        achieved=upper_value,
        reason=f"Recall@{upper}={upper_value:.4f} 未达到目标 {target:.2f}，取上限 k={upper}",
        curve=dict(curve),
    )
