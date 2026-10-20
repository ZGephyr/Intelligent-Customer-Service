"""改写服务测试。"""

from __future__ import annotations

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
