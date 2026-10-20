"""BM25 打分与中文分词：FAQ 匹配的纯计算层，不碰数据库。"""

from __future__ import annotations

import math
from collections import Counter

import jieba

from rag_service.core.logging import get_logger

logger = get_logger(__name__)

K1 = 1.5
B = 0.75

STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "吗", "呢", "啊", "吧", "呀", "哦", "嗯", "是", "在", "有", "和", "与", "及",
        "这个", "那个", "这些", "那些", "这样", "那样", "这里", "那里",
        "什么", "怎么", "怎样", "如何", "多少", "哪些", "哪个", "为什么",
        "可以", "能否", "是否", "需要", "请问", "麻烦", "谢谢",
        "我们", "你们", "他们", "咱们", "我", "你", "他", "她", "它",
        "一下", "一个", "一些", "还是", "就是", "不是", "没有", "以及", "然后",
        "你好", "您好", "亲", "呢", "哈",
    }
)


def tokenize(text: str) -> list[str]:
    """jieba 精确模式分词 → 小写 → 去标点/停用词/单字。"""
    tokens: list[str] = []
    for raw in jieba.lcut(text or "", cut_all=False):
        token = raw.strip().lower()
        if not token or token in STOPWORDS:
            continue
        if not any(char.isalnum() for char in token):
            continue
        if len(token) < 2 and not token.isascii():
            continue
        tokens.append(token)
    return tokens


def char_bigrams(text: str) -> list[str]:
    """把连续的中英文/数字串切成字符二元组，中文短问句更鲁棒。"""
    tokens: list[str] = []
    buffer = ""
    for char in (text or "").lower():
        if char.isalnum() or "\u4e00" <= char <= "\u9fff":
            buffer += char
            continue
        if buffer:
            tokens.extend([buffer[i : i + 2] for i in range(len(buffer) - 1)] or [buffer])
            buffer = ""
    if buffer:
        tokens.extend([buffer[i : i + 2] for i in range(len(buffer) - 1)] or [buffer])
    return tokens


def tokenize_hybrid(text: str) -> list[str]:
    """jieba 词 + 字符二元组取并集，兼顾词义与词边界不一致的短问句。"""
    merged: list[str] = []
    for token in tokenize(text) + char_bigrams(text):
        if token not in merged:
            merged.append(token)
    return merged


class Bm25Index:
    """一组文档的 BM25 索引，原始分用 softmax 归一化成「候选概率」。

    分工：``_score`` 给原始 BM25 分；``normalized_scores`` 对同一 query 的全部文档做
    ``softmax(s / temperature)``，输出即「这条文档是正确答案的概率」，总和为 1。
    全部文档原始分都为 0（query 一个词都没命中）时返回全 0，避免均匀分布混入噪声。
    """

    def __init__(
        self,
        documents: list[str],
        k1: float = K1,
        b: float = B,
        tokenizer=None,
        temperature: float = 1.0,
    ) -> None:
        self._tokenize = tokenizer or tokenize
        self._k1 = k1
        self._b = b
        self._temperature = max(temperature, 1e-6)
        self._docs = [self._tokenize(document) for document in documents]
        self._doc_sets = [set(doc) for doc in self._docs]
        self._n = len(self._docs)
        self._avgdl = (sum(len(doc) for doc in self._docs) / self._n) if self._n else 0.0
        self._df: Counter[str] = Counter()
        for tokens in self._docs:
            self._df.update(set(tokens))
        self._idf = {
            term: math.log(1 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in self._df.items()
        }
        self._term_counts = [Counter(tokens) for tokens in self._docs]
        logger.debug("BM25 索引构建完成: 文档数=%s 平均长度=%.2f", self._n, self._avgdl)

    @property
    def size(self) -> int:
        return self._n

    def normalized_scores(self, query: str) -> list[float]:
        """返回 softmax 概率：只对「有命中的候选」归一化，未命中的文档得 0。

        把零分文档排除在分母外，概率才衡量「候选之间的相对可信度」；否则语料越大，
        固定命中下 top1 的概率越低，阈值就失去意义。
        """
        if not self._n:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens or self._avgdl <= 0:
            return [0.0] * self._n

        raw_scores = [self._score(query_tokens, doc) for doc in self._docs]
        matched = [index for index, score in enumerate(raw_scores) if score > 0]
        if not matched:
            return [0.0] * self._n

        peak = max(raw_scores)
        weights = {index: math.exp((raw_scores[index] - peak) / self._temperature) for index in matched}
        total = sum(weights.values())
        if total <= 0:  # pragma: no cover - 指数溢出兜底
            return [0.0] * self._n
        return [round(weights.get(index, 0.0) / total, 6) for index in range(self._n)]

    def coverage_scores(self, query: str) -> list[float]:
        """查询覆盖率：命中词 IDF 占全查询词 IDF 的比重（0-1）。

        与 softmax 概率互补：概率衡量「候选之间谁更像」，覆盖率衡量「这篇文档到底覆盖了
        问题多少内容」。单候选时概率恒为 1，覆盖率仍能识别出弱匹配。
        """
        if not self._n:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens or self._avgdl <= 0:
            return [0.0] * self._n
        unseen_idf = math.log(1 + (self._n + 0.5) / 0.5)
        weights = [self._idf.get(token, unseen_idf) for token in query_tokens]
        total = sum(weights)
        if total <= 0:
            return [0.0] * self._n
        return [
            round(
                sum(weight for token, weight in zip(query_tokens, weights) if token in doc_set)
                / total,
                6,
            )
            for doc_set in self._doc_sets
        ]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        counts = Counter(doc_tokens)
        length = len(doc_tokens)
        score = 0.0
        for term in query_tokens:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = self._idf.get(term)
            if idf is None:
                continue
            denominator = frequency + self._k1 * (1 - self._b + self._b * length / self._avgdl)
            score += idf * frequency * (self._k1 + 1) / denominator
        return score
