"""问答编排测试。"""

from __future__ import annotations

import logging

import pytest

from rag_service.core.exceptions import AppError
from rag_service.core.types import ChatTurn, FaqHit, RetrievedChunk
from rag_service.infra.db import build_engine, build_session_factory, create_all
from rag_service.services.chat_service import ChatService
from rag_service.services.intent_service import CHITCHAT, IN_SCOPE, OUT_OF_SCOPE, IntentResult
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
        chitchat_reply = "您好呀，闲聊这块我还不太会～不过店铺的事我都能帮您看,有什么想要了解的?"
        out_of_scope_reply = "不好意思呀，这个我确实答不上来～我只懂店里的事"
        hyde_enabled = True

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


# ---------- 1.0 路由：意图 / FAQ / 关键词 / HyDE ----------


class FakeIntent:
    def __init__(self, intent=IN_SCOPE, degraded=False):
        self.result = IntentResult(intent, degraded)
        self.calls = 0

    def classify(self, question, history):
        self.calls += 1
        return self.result


class FakeFaq:
    def __init__(self, hit=None):
        self.hit = hit
        self.hits = []

    def match(self, question):
        return self.hit

    def record_hit(self, faq_id):
        self.hits.append(faq_id)


class FakeRetrievalQueue:
    """按调用顺序返回预设结果，用于验证 HyDE 的两次检索。"""

    def __init__(self, *results):
        self.results = list(results)
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return list(self.results.pop(0)) if self.results else []


class FakeRewriteFull:
    def __init__(self, query, keywords=(), hypothetical_answer=""):
        self.outcome = RewriteOutcome(
            query, True, keywords=tuple(keywords), hypothetical_answer=hypothetical_answer
        )

    def rewrite(self, question, history):
        return self.outcome


def build_chat_v2(*, intent=None, faq=None, rewrite=None, retrieval=None, tokens=None, error=None):
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)
    history = FakeHistory()
    llm = FakeLLM(tokens=tokens or [], error=error)
    service = ChatService(
        factory,
        history,
        retrieval or FakeRetrieval([]),
        rewrite or FakeRewrite(),
        llm,
        settings_factory_stub(),
        intent=intent,
        faq=faq,
    )
    return service, factory, llm, history


def test_stream_answer_out_of_scope_skips_retrieval_and_generation():
    retrieval = FakeRetrievalQueue([HIT])
    service, _, llm, _ = build_chat_v2(
        intent=FakeIntent(OUT_OF_SCOPE), retrieval=retrieval, tokens=["不该出现"]
    )

    events = list(service.stream_answer(None, "今天天气怎么样"))

    assert [event.type for event in events] == ["meta", "delta", "done"]
    assert events[-1].data["status"] == "out_of_scope"
    assert retrieval.queries == [] and llm.calls == 0


def test_stream_answer_chitchat_uses_fixed_reply_without_retrieval():
    retrieval = FakeRetrievalQueue([HIT])
    service, _, llm, _ = build_chat_v2(
        intent=FakeIntent(CHITCHAT), retrieval=retrieval, tokens=["不该出现"]
    )

    events = list(service.stream_answer(None, "你好呀"))

    assert [event.type for event in events] == ["meta", "delta", "done"]
    assert events[-1].data["status"] == "chitchat"
    assert events[1].data["text"].startswith("您好呀，闲聊这块我还不太会")
    assert retrieval.queries == [] and llm.calls == 0


def test_stream_answer_out_of_scope_reply_includes_handoff_hint():
    service, _, _, _ = build_chat_v2(intent=FakeIntent(OUT_OF_SCOPE), tokens=[])

    events = list(service.stream_answer(None, "你们老板是谁"))

    reply = events[1].data["text"]
    assert reply.startswith("不好意思呀，这个我确实答不上来")
    assert "建议您联系人工客服" in reply


def test_stream_answer_faq_hit_returns_faq_answer_without_generation():
    hit = FaqHit("f1", "多久发货", "48 小时内发出", 0.82, "多久发货")
    faq = FakeFaq(hit)
    service, _, llm, _ = build_chat_v2(faq=faq, tokens=["不该出现"])

    events = list(service.stream_answer(None, "多久发货"))

    assert events[-1].data["status"] == "faq"
    assert events[1].type == "sources" and events[1].data["sources"][0]["type"] == "faq"
    assert events[2].data["text"] == "48 小时内发出"
    assert llm.calls == 0 and faq.hits == ["f1"]


def test_stream_answer_faq_miss_continues_to_rag():
    service, _, _, _ = build_chat_v2(
        faq=FakeFaq(None), retrieval=FakeRetrievalQueue([HIT]), tokens=["保修一年"]
    )

    events = list(service.stream_answer(None, "保修多久"))

    assert events[-1].data["status"] == "ok"


def test_stream_answer_meta_contains_intent():
    service, _, _, _ = build_chat_v2(
        intent=FakeIntent(IN_SCOPE), retrieval=FakeRetrievalQueue([HIT]), tokens=["x"]
    )

    events = list(service.stream_answer(None, "保修多久"))

    assert events[0].data["intent"] == IN_SCOPE


def test_stream_answer_appends_keywords_to_retrieval_query():
    retrieval = FakeRetrievalQueue([HIT])
    rewrite = FakeRewriteFull("耳机保修多久", ("耳机", "保修"))
    service, _, _, _ = build_chat_v2(retrieval=retrieval, rewrite=rewrite, tokens=["x"])

    list(service.stream_answer(None, "那它保修多久"))

    assert "耳机" in retrieval.queries[0] and "保修" in retrieval.queries[0]


def test_stream_answer_hyde_retries_when_first_recall_empty():
    retrieval = FakeRetrievalQueue([], [HIT])
    rewrite = FakeRewriteFull("耳机保修多久", (), "该耳机保修一年")
    service, _, _, _ = build_chat_v2(retrieval=retrieval, rewrite=rewrite, tokens=["保修一年"])

    events = list(service.stream_answer(None, "那它保修多久"))

    assert len(retrieval.queries) == 2
    assert events[-1].data["status"] == "ok"


def test_stream_answer_hyde_not_used_when_first_recall_hits():
    retrieval = FakeRetrievalQueue([HIT], [HIT])
    rewrite = FakeRewriteFull("耳机保修多久", (), "该耳机保修一年")
    service, _, _, _ = build_chat_v2(retrieval=retrieval, rewrite=rewrite, tokens=["x"])

    list(service.stream_answer(None, "保修多久"))

    assert len(retrieval.queries) == 1


def test_stream_answer_intent_degraded_still_answers():
    service, _, _, _ = build_chat_v2(
        intent=FakeIntent(IN_SCOPE, degraded=True),
        retrieval=FakeRetrievalQueue([HIT]),
        tokens=["x"],
    )

    events = list(service.stream_answer(None, "保修多久"))

    assert events[-1].data["status"] == "ok"
    assert events[0].data["intent"] == "unknown"


def test_stream_answer_logs_all_segments_with_request_id(caplog):
    with caplog.at_level(logging.INFO, logger="rag_service.services.chat_service"):
        service, _, _, _ = build_chat_v2(retrieval=FakeRetrievalQueue([HIT]), tokens=["x"])
        list(service.stream_answer(None, "保修多久", request_id="req-42"))

    text = "\n".join(record.getMessage() for record in caplog.records)
    for segment in ("intent:", "faq:", "rewrite:", "retrieve:", "generate:"):
        assert segment in text
    assert "req-42" in text


def test_stream_answer_faq_branch_reports_measured_latency():
    hit = FaqHit("f1", "多久发货", "48 小时内发出", 0.82, "多久发货")
    service, _, _, _ = build_chat_v2(faq=FakeFaq(hit), tokens=["不该出现"])

    events = list(service.stream_answer(None, "多久发货"))

    assert isinstance(events[-1].data["latency_ms"], int)
    assert events[-1].data["latency_ms"] >= 0


class ExplodingFaq:
    def match(self, question):
        raise RuntimeError("faq db down")


def test_stream_answer_faq_error_falls_back_to_rag():
    service, _, _, _ = build_chat_v2(
        faq=ExplodingFaq(), retrieval=FakeRetrievalQueue([HIT]), tokens=["保修一年"]
    )

    events = list(service.stream_answer(None, "保修多久"))

    assert events[-1].data["status"] == "ok"


def test_stream_answer_assigns_owner_and_title():
    from rag_service.models.tables import ChatSession

    service, factory, _, _ = build_chat_v2(retrieval=FakeRetrievalQueue([HIT]), tokens=["x"])

    events = list(service.stream_answer(None, "耳机保修多久，另外发票怎么开", user_id="u1"))

    session_id = events[0].data["session_id"]
    with factory() as session:
        row = session.get(ChatSession, session_id)
        assert row.user_id == "u1"
        assert row.title == "耳机保修多久，另外发票怎么开"[:20]


def test_get_or_create_session_rejects_foreign_session():
    from rag_service.core.exceptions import AppError, ErrorCode

    service, _, _, _ = build_chat_v2(retrieval=FakeRetrievalQueue([HIT]), tokens=["x"])
    events = list(service.stream_answer(None, "保修多久", user_id="u1"))
    foreign_id = events[0].data["session_id"]

    with pytest.raises(AppError) as excinfo:
        list(service.stream_answer(foreign_id, "保修多久", user_id="u2"))

    assert excinfo.value.code == ErrorCode.CHAT_SESSION_NOT_FOUND.value


def test_stream_answer_assigns_increasing_session_seq():
    from rag_service.models.tables import ChatMessage

    service, factory, _, _ = build_chat_v2(retrieval=FakeRetrievalQueue([HIT]), tokens=["x"])

    events = list(service.stream_answer(None, "保修多久", user_id="u1"))

    session_id = events[0].data["session_id"]
    with factory() as session:
        rows = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.seq)
            .all()
        )
    assert [row.role for row in rows] == ["user", "assistant"]
    assert [row.seq for row in rows] == [1, 2]