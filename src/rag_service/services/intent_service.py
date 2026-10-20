"""意图二分类：判断问题是否属于店铺业务范围，失败一律降级为 in_scope。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag_service.core.logging import get_logger
from rag_service.core.types import ChatTurn
from rag_service.infra.llm import ChatModelProvider

logger = get_logger(__name__)

IN_SCOPE = "in_scope"
CHITCHAT = "chitchat"
OUT_OF_SCOPE = "out_of_scope"
UNKNOWN = "unknown"
LABELS = (IN_SCOPE, CHITCHAT, OUT_OF_SCOPE)

MAX_HISTORY_TURNS = 6
SMALLTALK_MAX_CHARS = 30

GREETING_WORDS = (
    "你好", "您好", "在吗", "在不在", "早上好", "中午好", "下午好", "晚上好",
    "早安", "晚安", "谢谢", "多谢", "感谢", "再见", "拜拜", "哈喽", "hello", "hi",
)
WEATHER_WORDS = (
    "天气", "下雨", "晴天", "阴天", "刮风", "下雪", "冷不冷", "热不热", "降温", "升温",
)
BUSINESS_WORDS = (
    "商品", "耳机", "手机", "衣服", "鞋", "尺码", "颜色", "价格", "多少钱", "优惠", "活动",
    "库存", "现货", "发货", "快递", "物流", "运费", "包邮", "退货", "换货", "退款", "保修",
    "维修", "发票", "订单", "付款", "支付", "质量",
)


def _looks_like_smalltalk(question: str) -> bool:
    """明显的问候/天气寒暄直接判闲聊，不等模型；带业务词时交给模型。"""
    text = (question or "").strip().lower()
    if not text or len(text) > SMALLTALK_MAX_CHARS:
        return False
    if any(word in text for word in BUSINESS_WORDS):
        return False
    return any(word in text for word in GREETING_WORDS + WEATHER_WORDS)
_JSON_BLOCK = re.compile(r"\{.*?\}", re.S)

SYSTEM_PROMPT = (
    "你是电商店铺客服的意图分类器。判断用户最后一条问题属于哪一类，只输出 JSON。\n"
    "1) in_scope：商品（参数、规格、材质、颜色、尺码、兼容性、使用方法、库存、价格）；"
    "购买政策（退换货规则、运费承担、发票、发货时效、保修期限、优惠活动）；"
    "售后（退货换货流程、维修、补寄配件、质量问题）；物流（快递公司、配送范围、时效、查件）。\n"
    "2) chitchat：问候与寒暄（你好、在吗、早上好、吃了吗）、天气与日常闲聊（今天天气不错）、"
    "道谢与告别、闲聊玩笑、情感陪伴。\n"
    "3) out_of_scope：与店铺业务无关的通识问题（新闻、股票、健康、学习、科普）、"
    "询问店主或员工身份、要求写代码/写文章/翻译等任务、以及无法归入前两类的其他问题。\n"
    "判断原则：能靠店铺资料回答的业务问题一律 in_scope；只是在打招呼或闲聊用 chitchat；"
    "其余信息类需求用 out_of_scope。\n"
    '只输出 {"intent":"in_scope"}、{"intent":"chitchat"} 或 {"intent":"out_of_scope"}，不要解释。'
)


@dataclass(frozen=True)
class IntentResult:
    intent: str
    degraded: bool = False


class IntentService:
    def __init__(self, llm: ChatModelProvider, *, enabled: bool = True, max_tokens: int = 128) -> None:
        self._llm = llm
        self._enabled = enabled
        self._max_tokens = max_tokens

    def classify(self, question: str, history: list[ChatTurn]) -> IntentResult:
        if not self._enabled:
            return IntentResult(IN_SCOPE)

        if _looks_like_smalltalk(question):
            logger.info("意图规则命中闲聊，跳过模型: question=%s", _summary(question))
            return IntentResult(CHITCHAT)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(question, history)},
        ]

        try:
            raw = self._llm.invoke(messages, max_tokens=self._max_tokens)
        except Exception:  # noqa: BLE001 - 意图失败不能阻断问答
            logger.exception("意图识别调用失败，降级为 in_scope: question=%s", _summary(question))
            return IntentResult(IN_SCOPE, degraded=True)

        intent = _extract_intent(raw)
        if intent is None:
            logger.warning("意图识别输出无法解析，降级为 in_scope: output=%r", _summary(raw))
            return IntentResult(IN_SCOPE, degraded=True)

        logger.info("意图识别: intent=%s question=%s", intent, _summary(question))
        return IntentResult(intent)

    @staticmethod
    def _build_user_prompt(question: str, history: list[ChatTurn]) -> str:
        recent = history[-MAX_HISTORY_TURNS:]
        if not recent:
            return f"用户问题：{question}"
        history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)
        return f"对话历史：\n{history_text}\n\n用户问题：{question}"


def _extract_intent(raw: str) -> str | None:
    """只接受确切的标签：JSON 里的 intent 字段，或整段文本恰好是一个标签。

    带解释、否定、双标签、截断 JSON 一律返回 None（由调用方降级为 in_scope）。
    """
    text = (raw or "").strip()
    if not text:
        return None

    match = _JSON_BLOCK.search(text)
    if match is not None:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            value = str(payload.get("intent", "")).strip().lower()
            return value if value in LABELS else None
        return None

    normalized = text.strip().strip("\"'`*。.，, ").lower()
    return normalized if normalized in LABELS else None


def _summary(text: str, limit: int = 40) -> str:
    cleaned = (text or "").replace("\n", " ").strip()
    return cleaned if len(cleaned) <= limit else f"{cleaned[:limit]}…"
