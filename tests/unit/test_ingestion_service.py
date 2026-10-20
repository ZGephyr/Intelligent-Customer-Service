"""导入服务测试。"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.infra.db import build_engine, build_session_factory, create_all
from rag_service.models.tables import KbChunk, KbDocument
from rag_service.services.ingestion_service import IngestionService


class FakeEmbedding:
    dimension = 8

    def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


class FakeStore:
    def __init__(self):
        self.upserted = []
        self.deleted = []
        self.dropped = False

    def upsert(self, chunks):
        self.upserted.extend(chunks)

    def delete_document(self, document_id):
        self.deleted.append(document_id)

    def drop_and_recreate(self):
        self.dropped = True


def build_service(tmp_path):
    engine = build_engine("sqlite://")
    create_all(engine)
    factory = build_session_factory(engine)
    store = FakeStore()
    service = IngestionService(factory, FakeEmbedding(), store, tmp_path, max_upload_mb=1)
    return service, factory, store


def test_create_document_rejects_unsupported_extension(tmp_path):
    service, _, _ = build_service(tmp_path)
    with pytest.raises(AppError) as excinfo:
        service.create_document("a.png", b"x")
    assert excinfo.value.code == ErrorCode.KB_FILE_TYPE_UNSUPPORTED


def test_create_document_rejects_file_over_limit(tmp_path):
    service, _, _ = build_service(tmp_path)
    with pytest.raises(AppError) as excinfo:
        service.create_document("big.md", b"x" * (1024 * 1024 + 1))
    assert excinfo.value.code == ErrorCode.KB_FILE_TOO_LARGE


def test_create_document_same_hash_returns_existing_without_duplicate(tmp_path):
    service, factory, _ = build_service(tmp_path)
    first = service.create_document("a.md", "政策".encode())
    second = service.create_document("copy.md", "政策".encode())
    assert second.id == first.id
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(KbDocument)) == 1


def test_process_document_marks_failed_when_no_text(tmp_path):
    service, _, _ = build_service(tmp_path)
    document = service.create_document("empty.md", b"   ")
    service.process_document(document.id)
    saved = service.get_document(document.id)
    assert saved.status == "failed"
    assert saved.error_message


def test_process_document_ready_writes_parent_and_child_chunks(tmp_path):
    service, factory, store = build_service(tmp_path)
    document = service.create_document("policy.md", "七天无理由退货，运费由卖家承担。".encode())
    service.process_document(document.id)

    saved = service.get_document(document.id)
    assert saved.status == "ready"
    assert saved.chunk_count >= 1
    assert store.upserted
    assert store.upserted[0].parent_id
    assert store.upserted[0].parent_content
    with factory() as session:
        parents = session.scalar(
            select(func.count()).select_from(KbChunk).where(KbChunk.level == "parent")
        )
        children = session.scalar(
            select(func.count()).select_from(KbChunk).where(KbChunk.level == "child")
        )
    assert parents >= 1
    assert children == saved.chunk_count


def test_delete_document_removes_rows_and_vectors(tmp_path):
    service, factory, store = build_service(tmp_path)
    document = service.create_document("policy.md", "退货政策".encode())
    service.process_document(document.id)

    service.delete_document(document.id)

    assert store.deleted == [document.id]
    with factory() as session:
        assert session.get(KbDocument, document.id) is None
        assert session.scalar(select(func.count()).select_from(KbChunk)) == 0


def test_reindex_rebuilds_all_chunks(tmp_path):
    service, _, store = build_service(tmp_path)
    document = service.create_document("policy.md", "退货政策。保修一年。".encode())
    service.process_document(document.id)

    count = service.reindex()

    assert store.dropped is True
    assert count == service.get_document(document.id).chunk_count
