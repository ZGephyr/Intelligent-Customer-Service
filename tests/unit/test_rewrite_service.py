"""改写服务测试。"""

from __future__ import annotations

import json

from rag_service.core.types import ChatTurn
from rag_service.services.rewrite_service import RewriteService, should_rewrite

HISTORY = [ChatTurn("user", "这款耳机多少钱"), ChatTurn("assistant", "199 元")]


class FakeLLM:
    def __init__(self, out):
        self.out = out

    def invoke(self, messages, max_tokens=None):
        if isinstance(self.out, Exception):
            raise self.out
        return self.out

    def stream(self, messages):
        yield self.out


def test_should_rewrite_short_follow_up_with_history():
    assert should_rewrite("那它保修多久", HISTORY) is True


def test_should_rewrite_without_history_is_false():
    assert should_rewrite("那它保修多久", []) is False


def test_should_rewrite_long_independent_question_is_false():
    assert should_rewrite("七天无理由退货需要自己承担运费吗", HISTORY) is False


def test_rewrite_returns_model_output():
    outcome = RewriteService(FakeLLM("XX 耳机的保修期是多久")).rewrite("那它保修多久", HISTORY)
    assert outcome.query == "XX 耳机的保修期是多久"
    assert outcome.rewritten is True


def test_rewrite_model_error_degrades_to_original():
    outcome = RewriteService(FakeLLM(RuntimeError("timeout"))).rewrite("那它保修多久", HISTORY)
    assert outcome.query == "那它保修多久"
    assert outcome.rewritten is False


def test_rewrite_overlong_output_degrades_to_original():
    outcome = RewriteService(FakeLLM("长" * 101)).rewrite("那它保修多久", HISTORY)
    assert outcome.rewritten is False


class CountingLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    def invoke(self, messages, max_tokens=None):
        self.calls += 1
        return self.payload

    def stream(self, messages):
        yield ""


def test_rewrite_full_mode_parses_structured_output():
    payload = json.dumps(
        {
            "rewritten_query": "XX 耳机保修多久",
            "keywords": ["耳机", "保修"],
            "hypothetical_answer": "该耳机保修一年。",
        },
        ensure_ascii=False,
    )

    outcome = RewriteService(CountingLLM(payload)).rewrite("那它保修多久", HISTORY)

    assert outcome.query == "XX 耳机保修多久"
    assert outcome.keywords == ("耳机", "保修")
    assert outcome.hypothetical_answer.startswith("该耳机保修一年")
    assert outcome.degraded is False


def test_rewrite_full_mode_accepts_bare_text():
    outcome = RewriteService(CountingLLM("XX 耳机保修多久")).rewrite("那它保修多久", HISTORY)

    assert outcome.query == "XX 耳机保修多久" and outcome.keywords == ()


def test_rewrite_truncates_keywords_to_max():
    payload = json.dumps({"rewritten_query": "q", "keywords": ["a", "b", "c", "d", "e", "f"]})

    outcome = RewriteService(CountingLLM(payload), max_keywords=5).rewrite("那它多久", HISTORY)

    assert len(outcome.keywords) == 5


def test_rewrite_drops_overlong_hypothesis_but_keeps_query():
    payload = json.dumps({"rewritten_query": "q", "keywords": ["a"], "hypothetical_answer": "长" * 201})

    outcome = RewriteService(CountingLLM(payload)).rewrite("那它多久", HISTORY)

    assert outcome.query == "q" and outcome.hypothetical_answer == "" and outcome.keywords == ("a",)


def test_rewrite_applies_abbreviations():
    outcome = RewriteService(CountingLLM("噪音控制怎么样"), abbreviations={"降噪": "主动降噪"}).rewrite(
        "那它的降噪怎么样", HISTORY
    )

    assert "主动降噪" in outcome.query


def test_rewrite_minimal_mode_skips_when_rule_not_triggered():
    llm = CountingLLM("改写")

    outcome = RewriteService(llm, mode="minimal").rewrite("这款耳机的颜色有几种可选", [])

    assert outcome.rewritten is False and llm.calls == 0


def test_rewrite_disabled_returns_original_without_calling_llm():
    llm = CountingLLM("改写")

    outcome = RewriteService(llm, enabled=False).rewrite("那它多久", HISTORY)

    assert outcome.query == "那它多久" and llm.calls == 0
