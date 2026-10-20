"""提示词组装：系统约束 + 编号资料 + 历史 + 当前问题。"""

from __future__ import annotations

from rag_service.core.types import ChatTurn, RetrievedChunk

MAX_HISTORY_TURNS = 6
EMPTY_CONTEXT = "（无相关资料）"

SYSTEM_PROMPT_TEMPLATE = """你是「{shop_name}」的在线客服，只回答与商品、订单政策、售后相关的问题。

必须遵守：
1. 只能依据下方「资料」回答，不得使用外部知识，也不要补充资料中没有的细节。
2. 如果资料不足，直接说明「店铺资料中没有找到」，并{handoff_hint}，不要猜测。
3. 禁止编造价格、库存、发货时效、保修期限等具体承诺。
4. 用中文回答，简洁分点，不超过 300 字。
5. 不要只写开场白：必须直接给出具体规则要点。
6. 即使历史对话里出现过问候、闲聊或兜底话术，本轮也必须依据资料作答；资料不足就说明没有找到，不要重复历史原话。"""


def build_system_prompt(shop_name: str, handoff_hint: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(shop_name=shop_name, handoff_hint=handoff_hint)


def build_context_block(chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
    if not chunks:
        return EMPTY_CONTEXT

    lines: list[str] = []
    used = 0
    for index, chunk in enumerate(chunks, start=1):
        block = f"[{index}] 来源：{chunk.source}\n内容：{chunk.content}"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block) + 2

    return "\n\n".join(lines) if lines else EMPTY_CONTEXT


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[ChatTurn],
    shop_name: str,
    handoff_hint: str,
    max_context_chars: int = 6000,
) -> list[dict[str, str]]:
    context = build_context_block(chunks, max_chars=max_context_chars)
    system = f"{build_system_prompt(shop_name, handoff_hint)}\n\n资料：\n{context}"

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": question})
    return messages
