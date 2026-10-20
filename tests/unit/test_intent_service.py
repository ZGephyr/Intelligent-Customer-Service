"""意图二分类测试：正常解析与各类降级。"""

from __future__ import annotations

from rag_service.core.types import ChatTurn
from rag_service.services.intent_service import (
    CHITCHAT,
    IN_SCOPE,
    OUT_OF_SCOPE,
    IntentResult,
    IntentService,
)


class FakeLLM:
    def __init__(self, output: str = "", error: Exception | None = None):
        self.output = output
        self.error = error
        self.messages: list[list[dict[str, str]]] = []
        self.max_tokens: int | None = None

    def invoke(self, messages, max_tokens=None):
        self.messages.append(messages)
        self.max_tokens = max_tokens
        if self.error:
            raise self.error
        return self.output

    def stream(self, messages):
        yield self.output


def test_classify_parses_json_out_of_scope():
    result = IntentService(FakeLLM('{"intent": "out_of_scope"}')).classify("你们老板是谁", [])

    assert result.intent == OUT_OF_SCOPE and result.degraded is False


def test_classify_accepts_bare_word():
    result = IntentService(FakeLLM("in_scope")).classify("耳机保修多久", [])

    assert result.intent == IN_SCOPE and result.degraded is False


def test_classify_tolerates_markdown_fence():
    result = IntentService(FakeLLM('```json\n{"intent":"in_scope"}\n```')).classify("退货怎么办", [])

    assert result.intent == IN_SCOPE and result.degraded is False


def test_classify_degrades_to_in_scope_on_exception():
    result = IntentService(FakeLLM(error=RuntimeError("timeout"))).classify("耳机保修多久", [])

    assert result.intent == IN_SCOPE and result.degraded is True


def test_classify_degrades_on_unexpected_label():
    result = IntentService(FakeLLM("业务相关")).classify("耳机保修多久", [])

    assert result.intent == IN_SCOPE and result.degraded is True


def test_classify_degrades_on_empty_output():
    result = IntentService(FakeLLM("")).classify("耳机保修多久", [])

    assert result.intent == IN_SCOPE and result.degraded is True


def test_classify_disabled_skips_llm():
    llm = FakeLLM('{"intent":"out_of_scope"}')

    result = IntentService(llm, enabled=False).classify("今天天气", [])

    assert result.intent == IN_SCOPE and llm.messages == []


def test_classify_passes_history_into_prompt():
    llm = FakeLLM('{"intent":"in_scope"}')

    IntentService(llm).classify(
        "那它呢", [ChatTurn("user", "耳机保修多久"), ChatTurn("assistant", "一年")]
    )

    assert "耳机保修多久" in str(llm.messages[0])


def test_intent_result_defaults_to_not_degraded():
    assert IntentResult(IN_SCOPE).degraded is False


def test_classify_degrades_on_negated_label():
    result = IntentService(FakeLLM("这不是 out_of_scope 问题，应判 in_scope")).classify("保修多久", [])

    assert result.intent == IN_SCOPE and result.degraded is True


def test_classify_degrades_when_both_labels_present():
    result = IntentService(FakeLLM("in_scope 或 out_of_scope 都可以")).classify("保修多久", [])

    assert result.intent == IN_SCOPE and result.degraded is True


def test_classify_degrades_on_truncated_json():
    result = IntentService(FakeLLM('{"intent": "out_of_scope"')).classify("保修多久", [])

    assert result.intent == IN_SCOPE and result.degraded is True


def test_classify_uses_configured_max_tokens():
    llm = FakeLLM('{"intent":"in_scope"}')

    IntentService(llm).classify("保修多久", [])

    assert llm.max_tokens == 128


def test_classify_respects_custom_max_tokens():
    llm = FakeLLM('{"intent":"in_scope"}')

    IntentService(llm, max_tokens=64).classify("保修多久", [])

    assert llm.max_tokens == 64


def test_classify_parses_chitchat_label():
    result = IntentService(FakeLLM('{"intent":"chitchat"}')).classify("你好呀", [])

    assert result.intent == CHITCHAT and result.degraded is False


def test_rule_marks_weather_as_chitchat_without_llm():
    llm = FakeLLM('{"intent":"out_of_scope"}')

    result = IntentService(llm).classify("今天天气不错", [])

    assert result.intent == CHITCHAT and result.degraded is False
    assert llm.messages == []


def test_rule_marks_greeting_as_chitchat_without_llm():
    llm = FakeLLM('{"intent":"in_scope"}')

    result = IntentService(llm).classify("你好呀", [])

    assert result.intent == CHITCHAT and llm.messages == []


def test_rule_does_not_handle_business_questions():
    llm = FakeLLM('{"intent":"in_scope"}')

    IntentService(llm).classify("耳机保修多久", [])

    assert llm.messages, "业务问题应该交给模型判断"


def test_rule_skips_weather_when_business_word_present():
    llm = FakeLLM('{"intent":"in_scope"}')

    IntentService(llm).classify("下雨天耳机还能用吗", [])

    assert llm.messages, "带业务词时不能直接判闲聊"


def test_system_prompt_lists_weather_as_chitchat():
    llm = FakeLLM('{"intent":"chitchat"}')

    IntentService(llm).classify("这款耳机支持主动降噪吗", [])

    system_prompt = llm.messages[0][0]["content"]
    assert "天气" in system_prompt and "chitchat" in system_prompt


def test_classify_prompt_lists_three_labels():
    llm = FakeLLM('{"intent":"chitchat"}')

    IntentService(llm).classify("这款耳机支持主动降噪吗", [])

    system_prompt = llm.messages[0][0]["content"]
    for label in ("in_scope", "chitchat", "out_of_scope"):
        assert label in system_prompt
