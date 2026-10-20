"""问答编排测试。"""

from __future__ import annotations

import pytest

from rag_service.core.exceptions import AppError
from rag_service.core.types import ChatTurn, RetrievedChunk
from rag_service.infra.db import build_engine, build_session_factory, create_all
from rag_service.services.chat_service import ChatService
from rag_service.services.rewrite_service import RewriteOutcome

HIT = RetrievedChunk("c1", "d1", "保修一年", "policy.md", 0, 0.9)
HIT2 = RetrievedChunk("c2", "d1", "七天无理由", "policy.md", 1, 0.8)


class FakeHistory:
    def __init__(self):
        self.data = {}
        self.appended = []

    def get(self, session_id):
        return list(self.data.get(session_id, []))

    def append(self, session_id, turns):
        self.appended.extend(turns)
        self.data.setdefault(session_id, []).extend(turns)


class FakeRetrieval:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.hits)


class FakeRewrite:
    def __init__(self, query="", rewritten=False):
        self.query = query
        self.rewritten = rewritten

    def rewrite(self, question, history):
        return RewriteOutcome(self.query or question, self.rewritten)


class FakeLLM:
    def __init__(self, tokens=None, error=None):
        self.tokens = tokens or []
        self.error = error
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        for token in self.tokens:
            yield token

    def invoke(self, messages, max_tokens=None):
        return ""


def build_chat(hits, tokens=None, error=None):
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)
    history = FakeHistory()
    llm = FakeLLM(tokens=tokens, error=error)
    service = ChatService(
        factory,
        history,
        FakeRetrieval(hits),
        FakeRewrite(),
        llm,
        settings_factory_stub(),
    )
    return service, factory, llm, history


def settings_factory_stub():
    class Settings:
        shop_name = "测试店"
        handoff_hint = "建议您联系人工客服"
        max_context_chars = 6000

    return Settings()


def test_stream_answer_happy_path_emits_meta_sources_delta_done():
    service, factory, llm, history = build_chat([HIT], tokens=["保修", "一年"])
    events = list(service.stream_answer(None, "保修多久"))

    assert [event.type for event in events] == ["meta", "sources", "delta", "delta", "done"]
    assert events[0].data["session_id"]
    assert events[-1].data["status"] == "ok"

    messages = service.list_messages(events[0].data["session_id"])
    assistant = [m for m in messages if m.role == "assistant"][-1]
    assert assistant.content == "保修一年"
    assert assistant.status == "ok"
    assert history.appended


def test_stream_answer_without_hits_returns_fallback_and_skips_llm():
    service, factory, llm, _ = build_chat([], tokens=["不该被调用"])
    events = list(service.stream_answer(None, "老板是谁"))

    assert events[-1].data["status"] == "fallback"
    assert llm.calls == 0
    assistant = [m for m in service.list_messages(events[0].data["session_id"]) if m.role == "assistant"][-1]
    assert "没有找到" in assistant.content
    assert assistant.status == "fallback"


def test_stream_answer_empty_message_raises_invalid_message():
    service, _, _, _ = build_chat([HIT])
    with pytest.raises(AppError):
        list(service.stream_answer(None, "   "))


def test_stream_answer_overlong_message_raises_invalid_message():
    service, _, _, _ = build_chat([HIT])
    with pytest.raises(AppError):
        list(service.stream_answer(None, "长" * 2001))


def test_stream_answer_filters_sources_by_citation():
    service, factory, llm, _ = build_chat([HIT, HIT2], tokens=["保修一年 [2]"])
    events = list(service.stream_answer(None, "保修多久"))
    sources_event = [event for event in events if event.type == "sources"][0]
    assert len(sources_event.data["sources"]) == 2

    assistant = [m for m in service.list_messages(events[0].data["session_id"]) if m.role == "assistant"][-1]
    assert [source["index"] for source in assistant.sources] == [1]
    assert assistant.sources[0]["source"] == "policy.md"


def test_stream_answer_llm_error_marks_message_error_and_emits_error_event():
    service, factory, llm, _ = build_chat([HIT], error=RuntimeError("boom"))
    events = list(service.stream_answer(None, "保修多久"))

    assert events[-1].type == "error"
    assistant = [m for m in service.list_messages(events[0].data["session_id"]) if m.role == "assistant"][-1]
    assert assistant.status == "error"
