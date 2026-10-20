"""FAQ 内容维护：增删改查、批量导入与统计，供管理端使用。"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.models.tables import FaqPair
from rag_service.services.faq_service import FaqService

logger = get_logger(__name__)


@dataclass(frozen=True)
class FaqImportResult:
    created: int
    updated: int
    skipped: int


class FaqAdminService:
    def __init__(self, session_factory, faq_service: FaqService, *, max_rows: int = 2000) -> None:
        self._session_factory = session_factory
        self.faq_service = faq_service
        self._max_rows = max_rows

    # ---------- 查询 ----------

    def list(
        self,
        page: int,
        size: int,
        keyword: str = "",
        category: str = "",
        enabled: bool | None = None,
    ) -> tuple[list[FaqPair], int]:
        with self._session_factory() as session:
            query = session.query(FaqPair)
            if keyword:
                pattern = f"%{keyword}%"
                query = query.filter(
                    or_(FaqPair.question.like(pattern), FaqPair.aliases.like(pattern))
                )
            if category:
                query = query.filter(FaqPair.category == category)
            if enabled is not None:
                query = query.filter(FaqPair.enabled.is_(enabled))
            total = query.count()
            items = (
                query.order_by(FaqPair.updated_at.desc())
                .offset((page - 1) * size)
                .limit(size)
                .all()
            )
            return list(items), total

    def stats(self) -> dict:
        with self._session_factory() as session:
            total = session.query(func.count(FaqPair.id)).scalar() or 0
            enabled = (
                session.query(func.count(FaqPair.id)).filter(FaqPair.enabled.is_(True)).scalar() or 0
            )
            hit_total = session.query(func.coalesce(func.sum(FaqPair.hit_count), 0)).scalar() or 0
            top_rows = (
                session.query(FaqPair)
                .filter(FaqPair.hit_count > 0)
                .order_by(FaqPair.hit_count.desc())
                .limit(10)
                .all()
            )
            return {
                "total": int(total),
                "enabled": int(enabled),
                "hit_total": int(hit_total),
                "top": [{"question": row.question, "hit_count": row.hit_count} for row in top_rows],
            }

    # ---------- 写入 ----------

    def create(
        self,
        question: str,
        answer: str,
        category: str | None = None,
        aliases: str | None = None,
    ) -> FaqPair:
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            raise AppError(ErrorCode.FAQ_IMPORT_INVALID, "问题与答案都不能为空")

        with self._session_factory() as session:
            if session.query(FaqPair).filter(FaqPair.question == question).first() is not None:
                raise AppError(ErrorCode.FAQ_DUPLICATE, f"问法已存在: {question}")
            row = FaqPair(
                id=str(uuid4()),
                question=question,
                answer=answer,
                category=category,
                aliases=aliases,
                enabled=True,
                hit_count=0,
            )
            session.add(row)
            try:
                session.commit()
            except IntegrityError as exc:  # 并发下的唯一键冲突
                session.rollback()
                raise AppError(ErrorCode.FAQ_DUPLICATE, f"问法已存在: {question}") from exc

        self.faq_service.invalidate()
        logger.info("新增 FAQ: id=%s question=%s", row.id, question)
        return row

    def update(
        self,
        faq_id: str,
        *,
        question: str | None = None,
        answer: str | None = None,
        category: str | None = None,
        aliases: str | None = None,
        enabled: bool | None = None,
    ) -> FaqPair:
        if question is not None:
            question = question.strip()
            if not question:
                raise AppError(ErrorCode.FAQ_IMPORT_INVALID, "问题不能为空")
            if len(question) > 500:
                raise AppError(ErrorCode.FAQ_IMPORT_INVALID, "问题不能超过 500 字")
        if answer is not None:
            answer = answer.strip()
            if not answer:
                raise AppError(ErrorCode.FAQ_IMPORT_INVALID, "答案不能为空")

        with self._session_factory() as session:
            row = session.get(FaqPair, faq_id)
            if row is None:
                raise AppError(ErrorCode.FAQ_NOT_FOUND, f"FAQ 不存在: {faq_id}", http_status=404)
            if question is not None:
                row.question = question
            if answer is not None:
                row.answer = answer
            if category is not None:
                row.category = category
            if aliases is not None:
                row.aliases = aliases
            if enabled is not None:
                row.enabled = enabled
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise AppError(ErrorCode.FAQ_DUPLICATE, "问法已存在") from exc

        self.faq_service.invalidate()
        logger.info("更新 FAQ: id=%s", faq_id)
        return row

    def delete(self, faq_id: str) -> None:
        with self._session_factory() as session:
            row = session.get(FaqPair, faq_id)
            if row is None:
                raise AppError(ErrorCode.FAQ_NOT_FOUND, f"FAQ 不存在: {faq_id}", http_status=404)
            session.delete(row)
            session.commit()

        self.faq_service.invalidate()
        logger.info("删除 FAQ: id=%s", faq_id)

    def import_rows(self, rows: list[dict[str, str]]) -> FaqImportResult:
        if len(rows) > self._max_rows:
            raise AppError(
                ErrorCode.FAQ_IMPORT_INVALID,
                f"单次导入不能超过 {self._max_rows} 行",
                http_status=400,
            )

        created = updated = skipped = 0
        pending: dict[str, FaqPair] = {}
        with self._session_factory() as session:
            for row in rows:
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                if not question or not answer:
                    skipped += 1
                    continue

                duplicate = pending.get(question)
                if duplicate is not None:
                    duplicate.answer = answer
                    updated += 1
                    continue

                existing = session.query(FaqPair).filter(FaqPair.question == question).first()
                if existing is None:
                    entity = FaqPair(
                        id=str(uuid4()),
                        question=question,
                        answer=answer,
                        category=(row.get("category") or "").strip() or None,
                        aliases=(row.get("aliases") or "").strip() or None,
                        enabled=True,
                        hit_count=0,
                    )
                    session.add(entity)
                    pending[question] = entity
                    created += 1
                else:
                    existing.answer = answer
                    if row.get("category") is not None:
                        existing.category = (row.get("category") or "").strip() or None
                    if row.get("aliases") is not None:
                        existing.aliases = (row.get("aliases") or "").strip() or None
                    pending[question] = existing
                    updated += 1
            session.commit()

        self.faq_service.invalidate()
        logger.info("导入 FAQ: 新增=%s 更新=%s 跳过=%s", created, updated, skipped)
        return FaqImportResult(created=created, updated=updated, skipped=skipped)
