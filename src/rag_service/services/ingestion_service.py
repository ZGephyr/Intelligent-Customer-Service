"""知识库导入：校验、去重、解析、切分、向量化、入库、删除、重建。"""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import aliased

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.core.types import IndexedChunk
from rag_service.infra.embedding import EmbeddingProvider
from rag_service.infra.milvus_store import MilvusChunkStore
from rag_service.infra.parsers import SUPPORTED_EXTENSIONS, parse_file
from rag_service.models.tables import KbChunk, KbDocument
from rag_service.services.chunking import split_parent_child

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 32


class IngestionService:
    def __init__(
        self,
        session_factory,
        embedding: EmbeddingProvider,
        store: MilvusChunkStore,
        upload_dir: Path,
        max_upload_mb: int = 20,
    ) -> None:
        self._session_factory = session_factory
        self._embedding = embedding
        self._store = store
        self._upload_dir = Path(upload_dir)
        self._max_bytes = max_upload_mb * 1024 * 1024

    # ---------- 查询 ----------

    def get_document(self, document_id: str) -> KbDocument:
        with self._session_factory() as session:
            document = session.get(KbDocument, document_id)
            if document is None:
                raise AppError(ErrorCode.KB_DOCUMENT_NOT_FOUND, "文档不存在", http_status=404)
            return document

    def list_documents(self, page: int = 1, size: int = 20) -> tuple[list[KbDocument], int]:
        offset = max(page - 1, 0) * size
        with self._session_factory() as session:
            total = session.scalar(select(func.count()).select_from(KbDocument)) or 0
            rows = (
                session.execute(
                    select(KbDocument).order_by(KbDocument.created_at.desc()).offset(offset).limit(size)
                )
                .scalars()
                .all()
            )
            return list(rows), total

    # ---------- 写入 ----------

    def _embed_hybrid(self, texts: list[str]):
        hybrid = getattr(self._embedding, "embed_documents_hybrid", None)
        if callable(hybrid):
            return hybrid(texts)
        return self._embedding.embed_documents(texts), [{} for _ in texts]

    def create_document(self, filename: str, data: bytes) -> KbDocument:
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in SUPPORTED_EXTENSIONS:
            raise AppError(
                ErrorCode.KB_FILE_TYPE_UNSUPPORTED,
                f"不支持的文件类型: .{extension}",
            )
        if len(data) > self._max_bytes:
            raise AppError(
                ErrorCode.KB_FILE_TOO_LARGE,
                f"文件超过 {self._max_bytes // 1024 // 1024}MB",
                http_status=413,
            )

        file_hash = hashlib.sha256(data).hexdigest()
        with self._session_factory() as session:
            existing = session.scalar(select(KbDocument).where(KbDocument.file_hash == file_hash))
            if existing is not None and existing.status != "failed":
                logger.info("命中重复文件，复用已有文档: %s", existing.id)
                return existing
            if existing is not None:
                session.execute(delete(KbChunk).where(KbChunk.document_id == existing.id))
                session.delete(existing)
                session.flush()

            document_id = str(uuid.uuid4())
            suffix = f".{extension}"
            self._upload_dir.mkdir(parents=True, exist_ok=True)
            storage_path = self._upload_dir / f"{document_id}{suffix}"
            storage_path.write_bytes(data)

            document = KbDocument(
                id=document_id,
                filename=filename,
                file_ext=extension,
                file_size=len(data),
                file_hash=file_hash,
                storage_path=str(storage_path),
                status="processing",
                chunk_count=0,
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            logger.info("文档已登记: id=%s name=%s size=%s", document_id, filename, len(data))
            return document

    def process_document(self, document_id: str) -> None:
        with self._session_factory() as session:
            document = session.get(KbDocument, document_id)
            if document is None:
                raise AppError(ErrorCode.KB_DOCUMENT_NOT_FOUND, "文档不存在", http_status=404)

        try:
            blocks = parse_file(Path(document.storage_path))
            parents = split_parent_child(blocks)
            if not parents:
                raise AppError(ErrorCode.KB_EMPTY_CONTENT, "文件未切分出有效内容")

            parent_rows: list[KbChunk] = []
            child_specs: list[tuple[str, str, object, int]] = []
            next_index = len(parents)
            for parent in parents:
                parent_id = str(uuid.uuid4())
                parent_rows.append(
                    KbChunk(
                        id=parent_id,
                        document_id=document_id,
                        parent_id=None,
                        level="parent",
                        chunk_index=parent.index,
                        content=parent.content,
                        char_count=len(parent.content),
                        meta=parent.meta,
                    )
                )
                for child in parent.children:
                    child_specs.append((parent_id, parent.content, child, next_index))
                    next_index += 1

            vectors, sparse_vectors = self._embed_hybrid(
                [spec[2].content for spec in child_specs]
            )
            now = int(time.time())
            indexed: list[IndexedChunk] = []
            child_rows: list[KbChunk] = []
            for (parent_id, parent_content, child, global_index), vector, sparse in zip(
                child_specs, vectors, sparse_vectors
            ):
                child_id = str(uuid.uuid4())
                indexed.append(
                    IndexedChunk(
                        id=child_id,
                        document_id=document_id,
                        chunk_index=global_index,
                        content=child.content,
                        source=document.filename,
                        doc_type=document.file_ext,
                        created_at=now,
                        embedding=vector,
                        parent_id=parent_id,
                        parent_content=parent_content,
                        sparse=sparse,
                    )
                )
                child_rows.append(
                    KbChunk(
                        id=child_id,
                        document_id=document_id,
                        parent_id=parent_id,
                        level="child",
                        chunk_index=global_index,
                        content=child.content,
                        char_count=len(child.content),
                        meta=child.meta,
                    )
                )

            self._store.upsert(indexed)

            with self._session_factory() as session:
                session.execute(delete(KbChunk).where(KbChunk.document_id == document_id))
                session.add_all(parent_rows)
                session.add_all(child_rows)
                target = session.get(KbDocument, document_id)
                target.status = "ready"
                target.chunk_count = len(child_rows)
                target.error_message = None
                session.commit()

            logger.info(
                "文档入库完成: id=%s parents=%s children=%s",
                document_id,
                len(parent_rows),
                len(child_rows),
            )
        except Exception as exc:  # noqa: BLE001 - 失败要落状态，不能吞掉日志
            logger.exception("文档入库失败: id=%s", document_id)
            with self._session_factory() as session:
                target = session.get(KbDocument, document_id)
                if target is not None:
                    target.status = "failed"
                    target.error_message = str(exc)[:500]
                    session.commit()

    def delete_document(self, document_id: str) -> None:
        document = self.get_document(document_id)
        self._store.delete_document(document_id)
        with self._session_factory() as session:
            session.execute(delete(KbChunk).where(KbChunk.document_id == document_id))
            target = session.get(KbDocument, document_id)
            if target is not None:
                session.delete(target)
            session.commit()

        try:
            Path(document.storage_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("删除本地文件失败: %s", document.storage_path)
        logger.info("文档已删除: id=%s", document_id)

    def reindex(self) -> int:
        self._store.drop_and_recreate()
        total = 0
        parent = aliased(KbChunk)
        with self._session_factory() as session:
            rows = session.execute(
                select(KbChunk, KbDocument, parent.content)
                .join(KbDocument, KbChunk.document_id == KbDocument.id)
                .join(parent, KbChunk.parent_id == parent.id)
                .where(KbChunk.level == "child")
                .order_by(KbChunk.created_at)
            ).all()

        for start in range(0, len(rows), EMBED_BATCH_SIZE):
            batch = rows[start : start + EMBED_BATCH_SIZE]
            vectors, sparse_vectors = self._embed_hybrid(
                [child.content for child, _, _ in batch]
            )
            now = int(time.time())
            self._store.upsert(
                [
                    IndexedChunk(
                        id=child.id,
                        document_id=child.document_id,
                        chunk_index=child.chunk_index,
                        content=child.content,
                        source=document.filename,
                        doc_type=document.file_ext,
                        created_at=now,
                        embedding=vector,
                        parent_id=child.parent_id or "",
                        parent_content=parent_content,
                        sparse=sparse,
                    )
                    for (child, document, parent_content), vector, sparse in zip(
                        batch, vectors, sparse_vectors
                    )
                ]
            )
            total += len(batch)

        logger.info("重建索引完成: children=%s", total)
        return total
