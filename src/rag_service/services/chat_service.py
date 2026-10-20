"""问答编排：改写 → 召回 → 重排 → 提示词 → 流式生成 → 落库。"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.core.types import ChatEvent, ChatTurn, RetrievedChunk
from rag_service.infra.llm import ChatModelProvider
from rag_service.infra.redis_store import HistoryStore
from rag_service.models.tables import ChatMessage, ChatSession
from rag_service.services.prompt import build_messages
from rag_service.services.retrieval_service import RetrievalService
from rag_service.services.rewrite_service import RewriteService, should_rewrite

logger = get_logger(__name__)

MAX_MESSAGE_CHARS = 2000
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class ChatService:
    def __init__(
        self,
        session_factory,
        history_store: HistoryStore,
        retrieval: RetrievalService,
        rewrite: RewriteService,
        llm: ChatModelProvider,
        settings,
    ) -> None:
        self._session_factory = session_factory
        self._history = history_store
        self._retrieval = retrieval
        self._rewrite = rewrite
        self._llm = llm
        self._settings = settings

    # ---------- 会话 ----------

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._session_factory() as session:
            session.add(ChatSession(id=session_id))
            session.commit()
        return session_id

    def get_or_create_session(self, session_id: str | None) -> str:
        if session_id:
            with self._session_factory() as session:
                if session.get(ChatSession, session_id) is not None:
                    return session_id
        return self.create_session()

    def list_messages(self, session_id: str, limit: int = 50) -> list[ChatMessage]:
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

    def stream_answer(self, session_id: str | None, message: str) -> Iterator[ChatEvent]:
        question = (message or "").strip()
        if not question:
            raise AppError(ErrorCode.CHAT_INVALID_MESSAGE, "请输入有效问题")
        if len(question) > MAX_MESSAGE_CHARS:
            raise AppError(ErrorCode.CHAT_INVALID_MESSAGE, "问题不能超过 2000 字")

        session_id = self.get_or_create_session(session_id)
        history = self._history.get(session_id)

        user_message_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        self._save_message(
            ChatMessage(id=user_message_id, session_id=session_id, role="user", content=question)
        )
        self._save_message(
            ChatMessage(
                id=assistant_id,
                session_id=session_id,
                role="assistant",
                content="",
                status="error",
            )
        )

        yield ChatEvent(
            "meta",
            {
                "session_id": session_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_id,
            },
        )

        query = question
        if should_rewrite(question, history):
            outcome = self._rewrite.rewrite(question, history)
            query = outcome.query
            if outcome.rewritten:
                self._set_rewritten_query(user_message_id, query)

        chunks = self._retrieval.retrieve(query)
        yield ChatEvent("sources", {"sources": [self._source_payload(i, c) for i, c in enumerate(chunks, 1)]})

        if not chunks:
            answer = self._fallback_text()
            self._finish_message(assistant_id, answer, status="fallback", sources=[], latency_ms=0)
            yield ChatEvent("delta", {"text": answer})
            yield ChatEvent("done", {"status": "fallback", "latency_ms": 0})
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
        collected: list[str] = []
        try:
            for token in self._llm.stream(messages):
                collected.append(token)
                yield ChatEvent("delta", {"text": token})
        except Exception:  # noqa: BLE001 - 生成失败要落状态并通知前端
            logger.exception("生成回答失败: session=%s", session_id)
            latency = int((time.perf_counter() - started) * 1000)
            self._finish_message(
                assistant_id, "".join(collected), status="error", sources=[], latency_ms=latency
            )
            yield ChatEvent("error", {"code": ErrorCode.LLM_ERROR.value, "message": "模型响应失败，请重试"})
            return

        answer = "".join(collected)
        latency = int((time.perf_counter() - started) * 1000)
        if not answer.strip():
            logger.error("模型返回空内容: session=%s", session_id)
            self._finish_message(assistant_id, "", status="error", sources=[], latency_ms=latency)
            yield ChatEvent(
                "error",
                {"code": ErrorCode.LLM_ERROR.value, "message": "模型没有返回内容，请重试"},
            )
            return

        cited = self._filter_cited(answer, chunks)
        self._finish_message(
            assistant_id,
            answer,
            status="ok",
            sources=[self._source_payload(i, c) for i, c in enumerate(cited, 1)],
            latency_ms=latency,
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
            "document_id": chunk.document_id,
            "source": chunk.source,
            "snippet": (chunk.snippet or chunk.content)[:80],
            "score": round(chunk.score, 4),
        }

    @staticmethod
    def _filter_cited(answer: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        indexes = {int(match) for match in CITATION_PATTERN.findall(answer)}
        if not indexes:
            return chunks
        return [chunk for index, chunk in enumerate(chunks, 1) if index in indexes]

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

    def _finish_message(
        self,
        message_id: str,
        content: str,
        status: str,
        sources: list[dict],
        latency_ms: int,
    ) -> None:
        with self._session_factory() as session:
            row = session.get(ChatMessage, message_id)
            if row is None:
                return
            row.content = content
            row.status = status
            row.sources = sources
            row.latency_ms = latency_ms
            session.commit()

    def _remember(self, session_id: str, question: str, answer: str) -> None:
        self._history.append(
            session_id,
            [ChatTurn("user", question), ChatTurn("assistant", answer)],
        )
