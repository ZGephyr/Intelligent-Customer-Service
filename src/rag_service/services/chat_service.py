"""问答编排：意图 → FAQ → 改写 → 召回（必要时 HyDE）→ 重排 → 生成 → 落库。"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator

from sqlalchemy import func

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.core.types import ChatEvent, ChatTurn, RetrievedChunk
from rag_service.infra.llm import ChatModelProvider
from rag_service.infra.redis_store import HistoryStore
from rag_service.models.tables import ChatMessage, ChatSession
from rag_service.services.intent_service import CHITCHAT, IN_SCOPE, OUT_OF_SCOPE, UNKNOWN
from rag_service.services.prompt import build_messages
from rag_service.services.retrieval_service import RetrievalService
from rag_service.services.session_service import SessionService
from rag_service.services.rewrite_service import RewriteService

logger = get_logger(__name__)

MAX_MESSAGE_CHARS = 2000
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


class ChatService:
    def __init__(
        self,
        session_factory,
        history_store: HistoryStore,
        retrieval: RetrievalService,
        rewrite: RewriteService,
        llm: ChatModelProvider,
        settings,
        *,
        intent=None,
        faq=None,
        sessions=None,
    ) -> None:
        self._session_factory = session_factory
        self._history = history_store
        self._retrieval = retrieval
        self._rewrite = rewrite
        self._llm = llm
        self._settings = settings
        self._intent = intent
        self._faq = faq
        self._sessions = sessions or SessionService(session_factory)

    @property
    def retrieval(self) -> RetrievalService:
        return self._retrieval

    @property
    def faq(self):
        return self._faq

    @property
    def intent(self):
        return self._intent

    # ---------- 会话 ----------

    def create_session(self, user_id: str | None = None) -> str:
        return self._sessions.create(user_id)

    def get_or_create_session(self, session_id: str | None, user_id: str | None = None) -> str:
        if session_id:
            with self._session_factory() as session:
                row = session.get(ChatSession, session_id)
            if row is not None and (user_id is None or row.user_id == user_id):
                return session_id
            if user_id is not None:
                raise AppError(ErrorCode.CHAT_SESSION_NOT_FOUND, "会话不存在", http_status=404)
        return self.create_session(user_id)

    def clear_history(self, session_id: str) -> None:
        """删除会话时同步清掉 Redis 历史缓存（尽力而为）。"""
        clear = getattr(self._history, "clear", None)
        if callable(clear):
            clear(session_id)

    def list_messages(
        self, session_id: str, limit: int = 50, user_id: str | None = None
    ) -> list[ChatMessage]:
        if user_id is not None:
            return self._sessions.messages(session_id, user_id, limit=limit)
        with self._session_factory() as session:
            rows = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
                .all()
            )
            return list(rows)

    # ---------- 问答 ----------

    def stream_answer(
        self,
        session_id: str | None,
        message: str,
        request_id: str = "",
        user_id: str | None = None,
    ) -> Iterator[ChatEvent]:
        question = (message or "").strip()
        if not question:
            raise AppError(ErrorCode.CHAT_INVALID_MESSAGE, "请输入有效问题")
        if len(question) > MAX_MESSAGE_CHARS:
            raise AppError(ErrorCode.CHAT_INVALID_MESSAGE, "问题不能超过 2000 字")

        turn_started = time.perf_counter()
        session_id = self.get_or_create_session(session_id, user_id)
        history = self._history.get(session_id)

        user_message_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        user_seq = self._next_seq(session_id)
        self._save_message(
            ChatMessage(
                id=user_message_id,
                session_id=session_id,
                role="user",
                content=question,
                seq=user_seq,
            )
        )
        self._sessions.set_title_if_empty(session_id, question)
        self._save_message(
            ChatMessage(
                id=assistant_id,
                session_id=session_id,
                role="assistant",
                content="",
                status="error",
                seq=self._next_seq(session_id),
            )
        )

        intent = IN_SCOPE
        intent_degraded = False
        intent_started = time.perf_counter()
        if self._intent is not None:
            result = self._intent.classify(question, history)
            intent_degraded = result.degraded
            intent = UNKNOWN if result.degraded else result.intent
            self._set_intent(user_message_id, intent)
        logger.info(
            "intent: request=%s session=%s intent=%s degraded=%s cost=%dms",
            request_id,
            session_id,
            intent,
            intent_degraded,
            _elapsed_ms(intent_started),
        )

        yield ChatEvent(
            "meta",
            {
                "session_id": session_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_id,
                "intent": intent,
            },
        )

        # 1) 闲聊：固定话术，不检索、不生成
        if intent == CHITCHAT:
            answer = self._settings.chitchat_reply
            total_ms = _elapsed_ms(turn_started)
            self._finish_message(
                assistant_id,
                answer,
                status="chitchat",
                sources=[],
                latency_ms=total_ms,
                route="chitchat",
            )
            logger.info(
                "generate: request=%s session=%s status=chitchat total=%dms",
                request_id,
                session_id,
                total_ms,
            )
            yield ChatEvent("delta", {"text": answer})
            yield ChatEvent("done", {"status": "chitchat", "latency_ms": total_ms})
            self._remember(session_id, question, answer)
            return

        # 2) 非业务问题：兜底话术 + 转人工提示，不检索、不生成
        if intent == OUT_OF_SCOPE:
            answer = f"{self._settings.out_of_scope_reply}。{self._settings.handoff_hint}。"
            total_ms = _elapsed_ms(turn_started)
            self._finish_message(
                assistant_id,
                answer,
                status="out_of_scope",
                sources=[],
                latency_ms=total_ms,
                route="out_of_scope",
            )
            logger.info(
                "generate: request=%s session=%s status=out_of_scope total=%dms",
                request_id,
                session_id,
                total_ms,
            )
            yield ChatEvent("delta", {"text": answer})
            yield ChatEvent("done", {"status": "out_of_scope", "latency_ms": total_ms})
            self._remember(session_id, question, answer)
            return

        # 3) 高频问答：命中即原样返回；FAQ 故障视为未命中
        faq_started = time.perf_counter()
        try:
            hit = self._faq.match(question) if self._faq is not None else None
        except Exception:  # noqa: BLE001 - FAQ 故障不能阻断问答
            logger.exception("FAQ 匹配异常，按未命中继续: session=%s request=%s", session_id, request_id)
            hit = None
        top1, top2 = getattr(self._faq, "last_top_scores", (0.0, 0.0)) if self._faq else (0.0, 0.0)
        faq_ms = _elapsed_ms(faq_started)
        logger.info(
            "faq: request=%s session=%s hit=%s top1=%.4f top2=%.4f cost=%dms",
            request_id,
            session_id,
            hit is not None,
            top1,
            top2,
            faq_ms,
        )
        if hit is not None:
            self._faq.record_hit(hit.faq_id)
            sources = [self._faq_payload(hit)]
            total_ms = _elapsed_ms(turn_started)
            self._finish_message(
                assistant_id,
                hit.answer,
                status="faq",
                sources=sources,
                latency_ms=total_ms,
                route="faq",
            )
            logger.info(
                "generate: request=%s session=%s status=faq total=%dms faq_segment=%dms",
                request_id,
                session_id,
                total_ms,
                faq_ms,
            )
            yield ChatEvent("sources", {"sources": sources})
            yield ChatEvent("delta", {"text": hit.answer})
            yield ChatEvent("done", {"status": "faq", "latency_ms": total_ms})
            self._remember(session_id, question, hit.answer)
            return

        # 3) 改写 → 检索（空召回可走一次 HyDE）
        rewrite_started = time.perf_counter()
        outcome = self._rewrite.rewrite(question, history)
        query = outcome.query
        if outcome.keywords:
            query = f"{query} {' '.join(outcome.keywords)}".strip()
        if outcome.rewritten:
            self._set_rewritten_query(user_message_id, outcome.query)
        logger.info(
            "rewrite: request=%s session=%s query=%s keywords=%s degraded=%s cost=%dms",
            request_id,
            session_id,
            outcome.query,
            list(outcome.keywords),
            outcome.degraded,
            _elapsed_ms(rewrite_started),
        )

        retrieve_started = time.perf_counter()
        chunks = self._retrieval.retrieve(query)
        hyde_used = False
        if not chunks and outcome.hypothetical_answer and getattr(self._settings, "hyde_enabled", True):
            hyde_used = True
            logger.info("首轮召回为空，使用假设答案二次检索: session=%s", session_id)
            chunks = self._retrieval.retrieve(outcome.hypothetical_answer)
        logger.info(
            "retrieve: request=%s session=%s hyde=%s hits=%s cost=%dms",
            request_id,
            session_id,
            hyde_used,
            len(chunks),
            _elapsed_ms(retrieve_started),
        )
        yield ChatEvent("sources", {"sources": [self._source_payload(i, c) for i, c in enumerate(chunks, 1)]})

        if not chunks:
            answer = self._fallback_text()
            total_ms = _elapsed_ms(turn_started)
            self._finish_message(
                assistant_id, answer, status="fallback", sources=[], latency_ms=total_ms, route="rag"
            )
            logger.info(
                "generate: request=%s session=%s status=fallback total=%dms",
                request_id,
                session_id,
                total_ms,
            )
            yield ChatEvent("delta", {"text": answer})
            yield ChatEvent("done", {"status": "fallback", "latency_ms": total_ms})
            self._remember(session_id, question, answer)
            return

        messages = build_messages(
            question,
            chunks,
            history,
            shop_name=self._settings.shop_name,
            handoff_hint=self._settings.handoff_hint,
            max_context_chars=self._settings.max_context_chars,
        )

        started = time.perf_counter()
        first_token_ms: int | None = None
        collected: list[str] = []
        try:
            for token in self._llm.stream(messages):
                if first_token_ms is None:
                    first_token_ms = _elapsed_ms(turn_started)
                collected.append(token)
                yield ChatEvent("delta", {"text": token})
        except Exception:  # noqa: BLE001 - 生成失败要落状态并通知前端
            logger.exception("生成回答失败: session=%s request=%s", session_id, request_id)
            latency = _elapsed_ms(started)
            self._finish_message(
                assistant_id, "".join(collected), status="error", sources=[], latency_ms=latency, route="rag"
            )
            yield ChatEvent("error", {"code": ErrorCode.LLM_ERROR.value, "message": "模型响应失败，请重试"})
            return

        answer = "".join(collected)
        latency = _elapsed_ms(started)
        if not answer.strip():
            logger.error("模型返回空内容: session=%s request=%s", session_id, request_id)
            self._finish_message(
                assistant_id, "", status="error", sources=[], latency_ms=latency, route="rag"
            )
            yield ChatEvent(
                "error",
                {"code": ErrorCode.LLM_ERROR.value, "message": "模型没有返回内容，请重试"},
            )
            return

        total_ms = _elapsed_ms(turn_started)
        logger.info(
            "generate: request=%s session=%s status=ok first_token=%sms total=%dms chars=%d",
            request_id,
            session_id,
            first_token_ms if first_token_ms is not None else "n/a",
            total_ms,
            len(answer),
        )
        cited = self._filter_cited(answer, chunks)
        self._finish_message(
            assistant_id,
            answer,
            status="ok",
            sources=[self._source_payload(i, c) for i, c in enumerate(cited, 1)],
            latency_ms=latency,
            route="rag",
        )
        yield ChatEvent("done", {"status": "ok", "latency_ms": latency})
        self._remember(session_id, question, answer)

    # ---------- 内部 ----------

    def _fallback_text(self) -> str:
        return (
            "抱歉，我在店铺资料里没有找到相关信息。"
            f"{self._settings.handoff_hint}确认，工作时间为 9:00-21:00。"
        )

    @staticmethod
    def _source_payload(index: int, chunk: RetrievedChunk) -> dict:
        return {
            "index": index,
            "type": "rag",
            "document_id": chunk.document_id,
            "source": chunk.source,
            "snippet": (chunk.snippet or chunk.content)[:80],
            "score": round(chunk.score, 4),
        }

    @staticmethod
    def _faq_payload(hit) -> dict:
        return {
            "index": 1,
            "type": "faq",
            "question": hit.question,
            "source": "高频问答",
            "snippet": hit.answer[:80],
            "score": round(hit.score, 4),
        }

    @staticmethod
    def _filter_cited(answer: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        indexes = {int(match) for match in CITATION_PATTERN.findall(answer)}
        if not indexes:
            return chunks
        return [chunk for index, chunk in enumerate(chunks, 1) if index in indexes]

    def _next_seq(self, session_id: str) -> int:
        """会话内递增序号：同一秒写入多条时也能稳定排序。"""
        with self._session_factory() as session:
            current = (
                session.query(func.max(ChatMessage.seq))
                .filter(ChatMessage.session_id == session_id)
                .scalar()
            )
        return int(current or 0) + 1

    def _save_message(self, message: ChatMessage) -> None:
        with self._session_factory() as session:
            session.add(message)
            session.commit()

    def _set_rewritten_query(self, message_id: str, query: str) -> None:
        with self._session_factory() as session:
            row = session.get(ChatMessage, message_id)
            if row is not None:
                row.rewritten_query = query[:512]
                session.commit()

    def _set_intent(self, message_id: str, intent: str) -> None:
        with self._session_factory() as session:
            row = session.get(ChatMessage, message_id)
            if row is not None:
                row.intent = intent[:16]
                session.commit()

    def _finish_message(
        self,
        message_id: str,
        content: str,
        status: str,
        sources: list[dict],
        latency_ms: int,
        route: str | None = None,
    ) -> None:
        with self._session_factory() as session:
            row = session.get(ChatMessage, message_id)
            if row is None:
                return
            row.content = content
            row.status = status
            row.sources = sources
            row.latency_ms = latency_ms
            row.route = route
            session.commit()

    def _remember(self, session_id: str, question: str, answer: str) -> None:
        self._history.append(
            session_id,
            [ChatTurn("user", question), ChatTurn("assistant", answer)],
        )
