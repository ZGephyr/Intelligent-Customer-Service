"""管理端接口：文档上传 / 列表 / 详情 / 删除 / 重建索引。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, Request, UploadFile

from rag_service.api.deps import require_admin
from rag_service.api.schemas import DocumentOut, PageOut
from rag_service.infra.parsers import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


@router.get("/config")
def admin_config(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "code": 0,
        "data": {
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "max_upload_mb": settings.max_upload_mb,
            "admin_auth_enabled": settings.admin_auth_enabled,
        },
        "request_id": _request_id(request),
    }


@router.post("/documents", status_code=202)
async def upload_document(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict:
    service = request.app.state.services.ingestion
    data = await file.read()
    document = service.create_document(file.filename or "unnamed", data)
    background.add_task(service.process_document, document.id)
    return {
        "code": 0,
        "data": DocumentOut.model_validate(document).model_dump(mode="json"),
        "request_id": _request_id(request),
    }


@router.get("/documents")
def list_documents(
    request: Request,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
) -> dict:
    service = request.app.state.services.ingestion
    items, total = service.list_documents(page=page, size=size)
    payload = PageOut(
        items=[DocumentOut.model_validate(item) for item in items],
        total=total,
        page=page,
        size=size,
    )
    return {
        "code": 0,
        "data": payload.model_dump(mode="json"),
        "request_id": _request_id(request),
    }


@router.get("/documents/{document_id}")
def get_document(request: Request, document_id: str) -> dict:
    document = request.app.state.services.ingestion.get_document(document_id)
    return {
        "code": 0,
        "data": DocumentOut.model_validate(document).model_dump(mode="json"),
        "request_id": _request_id(request),
    }


@router.delete("/documents/{document_id}")
def delete_document(request: Request, document_id: str) -> dict:
    request.app.state.services.ingestion.delete_document(document_id)
    return {"code": 0, "data": {"deleted": document_id}, "request_id": _request_id(request)}


@router.post("/reindex", status_code=202)
def reindex(request: Request) -> dict:
    total = request.app.state.services.ingestion.reindex()
    return {"code": 0, "data": {"chunks": total}, "request_id": _request_id(request)}
