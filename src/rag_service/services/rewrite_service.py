"""查询改写：历史上下文改写 + 指代消解 + 关键词扩展 + 缩写展开 + 假设答案。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rag_service.core.logging import get_logger
from rag_service.core.types import ChatTurn
from rag_service.infra.llm import ChatModelProvider

logger = get_logger(__name__)

PRONOUNS = ("它", "这个", "那个", "这款", "那款", "还有", "呢")
STARTERS = ("那", "那么")
MAX_REWRITE_CHARS = 100
MAX_HYPOTHESIS_CHARS = 200
MAX_KEYWORD_CHARS = 12
HISTORY_TURNS_FOR_REWRITE = 6

MINIMAL_SYSTEM_PROMPT = (
    "你是检索查询改写助手。根据对话历史，把用户最后一句话改写成不依赖上下文、"
    "可以独立检索的问题。只输出改写后的问题本身，不要解释，不超过 50 字。"
)

FULL_SYSTEM_PROMPT = (
    "你是电商客服的检索查询改写助手。基于对话历史与缩写表，完成三件事：\n"
    "1) 把用户最后一个问题改写为不依赖上下文、可独立检索的问题，缩写展开为全称；\n"
    "2) 给出用于检索的关键词（名词为主，包含商品名与属性）；\n"
    "3) 写一个假设中的标准答案片段，用于假设问题检索。\n"
    '只输出 JSON：{"rewritten_query":"...","keywords":["..."],"hypothetical_answer":"..."}。\n'
    "约束：rewritten_query 不超过 50 字；keywords 最多 5 个、每个不超过 12 字；"
    "hypothetical_answer 不超过 200 字；不要输出解释或多余文本。"
)


@dataclass(frozen=True)
class RewriteOutcome:
    """改写结果；keywords 与 hypothetical_answer 仅在 full 模式下非空。"""

    query: str
    rewritten: bool
    keywords: tuple[str, ...] = ()
    hypothetical_answer: str = ""
    degraded: bool = False


def should_rewrite(question: str, history: list[ChatTurn]) -> bool:
    if not history:
        return False
    text = question.strip()
    if not text:
        return False
    if len(text) <= 12:
        return True
    if any(pronoun in text for pronoun in PRONOUNS):
        return True
    return text.startswith(STARTERS)


class RewriteService:
    def __init__(
        self,
        llm: ChatModelProvider,
        *,
        enabled: bool = True,
        mode: str = "full",
        max_keywords: int = 5,
        abbreviations: dict[str, str] | None = None,
    ) -> None:
        self._llm = llm
        self._enabled = enabled
        self._mode = mode if mode in {"full", "minimal"} else "full"
        self._max_keywords = max_keywords
        self._abbreviations = abbreviations or {}

    def rewrite(self, question: str, history: list[ChatTurn]) -> RewriteOutcome:
        if not self._enabled:
            return RewriteOutcome(question, False)
        if self._mode == "minimal":
            return self._rewrite_minimal(question, history)
        return self._rewrite_full(question, history)

    # ---------- full ----------

    def _rewrite_full(self, question: str, history: list[ChatTurn]) -> RewriteOutcome:
        messages = [
            {"role": "system", "content": FULL_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(question, history)},
        ]

        try:
            raw = self._llm.invoke(messages, max_tokens=400)
        except Exception:  # noqa: BLE001 - 改写失败不阻断问答
            logger.warning("查询改写调用失败，使用原问题: %s", _summary(question))
            return RewriteOutcome(question, False, degraded=True)

        payload = _parse_payload(raw)
        query = self._expand(str(payload.get("rewritten_query") or "").strip())
        if not query or len(query) > MAX_REWRITE_CHARS:
            logger.warning("改写输出不可用（长度=%s），使用原问题", len(query))
            return RewriteOutcome(question, False, degraded=True)

        # 强制映射：原问题里的缩写即使被模型改写掉，也要把全称补回检索 query
        missing = [
            full
            for short, full in self._abbreviations.items()
            if short in question and full not in query
        ]
        if missing:
            query = f"{query} {' '.join(missing)}".strip()

        keywords = _clean_keywords(payload.get("keywords"), self._max_keywords)
        hypothesis = str(payload.get("hypothetical_answer") or "").strip()
        if len(hypothesis) > MAX_HYPOTHESIS_CHARS:
            logger.warning("假设答案超过 %s 字，已丢弃", MAX_HYPOTHESIS_CHARS)
            hypothesis = ""

        logger.info(
            "查询改写: %s -> %s 关键词=%s 假设答案长度=%s",
            _summary(question),
            _summary(query),
            list(keywords),
            len(hypothesis),
        )
        return RewriteOutcome(
            query=query,
            rewritten=query != question,
            keywords=keywords,
            hypothetical_answer=hypothesis,
        )

    # ---------- minimal ----------

    def _rewrite_minimal(self, question: str, history: list[ChatTurn]) -> RewriteOutcome:
        if not should_rewrite(question, history):
            return RewriteOutcome(question, False)

        recent = history[-HISTORY_TURNS_FOR_REWRITE:]
        history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)
        messages = [
            {"role": "system", "content": MINIMAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"对话历史：\n{history_text}\n\n最后一句话：{question}"},
        ]

        try:
            output = self._expand(self._llm.invoke(messages, max_tokens=100).strip())
        except Exception:  # noqa: BLE001 - 改写失败不阻断问答
            logger.warning("指代改写失败，使用原问题: %s", question)
            return RewriteOutcome(question, False, degraded=True)

        if not output or len(output) > MAX_REWRITE_CHARS:
            logger.warning("指代改写输出不可用（长度=%s），使用原问题", len(output))
            return RewriteOutcome(question, False, degraded=True)

        logger.info("指代改写: %s -> %s", _summary(question), _summary(output))
        return RewriteOutcome(output, output != question)

    # ---------- 内部 ----------

    def _build_user_prompt(self, question: str, history: list[ChatTurn]) -> str:
        recent = history[-HISTORY_TURNS_FOR_REWRITE:]
        history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in recent) or "（无历史）"
        abbreviations = (
            "、".join(f"{short}={full}" for short, full in self._abbreviations.items()) or "（无）"
        )
        return (
            f"对话历史：\n{history_text}\n\n"
            f"缩写表：{abbreviations}\n\n"
            f"用户问题：{question}"
        )

    def _expand(self, text: str) -> str:
        expanded = text
        for short, full in self._abbreviations.items():
            expanded = expanded.replace(short, full)
        return expanded


def _parse_payload(raw: Any) -> dict:
    text = str(raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    return {"rewritten_query": text}


def _clean_keywords(value: Any, limit: int) -> tuple[str, ...]:
    if isinstance(value, str):
        items: list[Any] = [part for part in value.replace("，", ",").split(",")]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = []

    seen: list[str] = []
    for item in items:
        keyword = str(item or "").strip()
        if not keyword or len(keyword) > MAX_KEYWORD_CHARS or keyword in seen:
            continue
        seen.append(keyword)
        if len(seen) >= limit:
            break
    return tuple(seen)


def _summary(text: str, limit: int = 40) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else f"{cleaned[:limit]}…"
