"""指代改写：把追问改成可独立检索的问题，失败降级用原问题。"""

from __future__ import annotations

from dataclasses import dataclass

from rag_service.core.logging import get_logger
from rag_service.core.types import ChatTurn
from rag_service.infra.llm import ChatModelProvider

logger = get_logger(__name__)

PRONOUNS = ("它", "这个", "那个", "这款", "那款", "还有", "呢")
STARTERS = ("那", "那么")
MAX_REWRITE_CHARS = 100
HISTORY_TURNS_FOR_REWRITE = 6

SYSTEM_PROMPT = (
    "你是检索查询改写助手。根据对话历史，把用户最后一句话改写成不依赖上下文、"
    "可以独立检索的问题。只输出改写后的问题本身，不要解释，不超过 50 字。"
)


@dataclass(frozen=True)
class RewriteOutcome:
    query: str
    rewritten: bool


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
    def __init__(self, llm: ChatModelProvider) -> None:
        self._llm = llm

    def rewrite(self, question: str, history: list[ChatTurn]) -> RewriteOutcome:
        if not should_rewrite(question, history):
            return RewriteOutcome(question, False)

        recent = history[-HISTORY_TURNS_FOR_REWRITE:]
        history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"对话历史：\n{history_text}\n\n最后一句话：{question}"},
        ]

        try:
            output = self._llm.invoke(messages, max_tokens=100).strip()
        except Exception:  # noqa: BLE001 - 改写失败不阻断问答
            logger.warning("指代改写失败，使用原问题: %s", question)
            return RewriteOutcome(question, False)

        if not output or len(output) > MAX_REWRITE_CHARS:
            logger.warning("指代改写输出不可用（长度=%s），使用原问题", len(output))
            return RewriteOutcome(question, False)

        logger.info("指代改写: %s -> %s", question, output)
        return RewriteOutcome(output, True)
