"""FastAPI 应用装配。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from rag_service.api import routes_admin, routes_auth, routes_chat, routes_faq, routes_health, ws_chat
from rag_service.core.config import Settings, get_settings
from rag_service.core.exceptions import AppError
from rag_service.core.logging import get_logger, setup_logging
from rag_service.infra.db import build_engine, build_session_factory, ensure_database, ensure_schema
from rag_service.infra.security import load_or_create_secret
from rag_service.infra.embedding import build_embedding
from rag_service.infra.llm import build_chat_provider
from rag_service.infra.milvus_store import MilvusChunkStore
from rag_service.infra.redis_store import HistoryStore, build_redis_client
from rag_service.infra.reranker import build_reranker
from rag_service.services.auth_service import AttemptLimiter, AuthService
from rag_service.services.chat_service import ChatService
from rag_service.services.faq_admin_service import FaqAdminService
from rag_service.services.faq_service import FaqService
from rag_service.services.health_service import HealthService
from rag_service.services.ingestion_service import IngestionService
from rag_service.services.intent_service import IntentService
from rag_service.services.retrieval_service import RetrievalService
from rag_service.services.session_service import SessionService
from rag_service.services.rewrite_service import RewriteService

logger = get_logger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass
class AppServices:
    chat: Any
    ingestion: Any
    health: Any
    faq_admin: Any = None
    auth: Any = None
    sessions: Any = None


def build_services(settings: Settings) -> AppServices:
    engine = build_engine(settings.mysql_url)
    ensure_database(settings.mysql_url)
    ensure_schema(engine)
    factory = build_session_factory(engine)

    auth = AuthService(
        factory,
        load_or_create_secret(settings.jwt_secret_path),
        bcrypt_rounds=settings.bcrypt_rounds,
        captcha_ttl_seconds=settings.captcha_ttl_seconds,
        allow_admin_signup=settings.allow_admin_signup,
        limiter=AttemptLimiter(settings.auth_max_attempts, settings.auth_lock_seconds),
    )

    redis_client = build_redis_client(settings.redis_url)
    history = HistoryStore(
        redis_client,
        ttl_seconds=settings.history_ttl_seconds,
        max_messages=settings.history_max_messages,
    )

    store = MilvusChunkStore(
        settings.milvus_uri,
        settings.milvus_token,
        settings.milvus_collection,
        settings.embedding_dim,
        database=settings.milvus_database,
    )
    store.ensure_collection()

    embedding = build_embedding(settings)
    if hasattr(embedding, "warmup"):
        embedding.warmup()

    reranker = build_reranker(settings)
    if reranker is not None:
        reranker.warmup()

    retrieval = RetrievalService(
        embedding,
        store,
        reranker,
        candidate_k=settings.candidate_k,
        rerank_candidates=settings.rerank_candidates,
        top_k=settings.top_k,
        score_threshold=settings.score_threshold,
        rerank_score_threshold=settings.rerank_score_threshold,
        rrf_k=settings.rrf_k,
    )
    rewrite = RewriteService(
        build_chat_provider(
            settings,
            timeout_seconds=settings.rewrite_timeout_seconds,
            max_retries=0,
            enable_thinking=settings.llm_enable_thinking,
        ),
        enabled=settings.rewrite_enabled,
        mode=settings.rewrite_mode,
        max_keywords=settings.rewrite_max_keywords,
        abbreviations=settings.abbreviations_map,
    )
    intent = IntentService(
        build_chat_provider(
            settings,
            timeout_seconds=settings.intent_timeout_seconds,
            temperature=0.0,
            max_retries=0,
            enable_thinking=settings.llm_enable_thinking,
        ),
        enabled=settings.intent_enabled,
        max_tokens=settings.intent_max_tokens,
    )
    faq = FaqService(
        factory,
        enabled=settings.faq_enabled,
        threshold=settings.faq_score_threshold,
        margin=settings.faq_margin,
        min_coverage=settings.faq_min_coverage,
        cache_ttl_seconds=settings.faq_cache_ttl_seconds,
        temperature=settings.faq_softmax_temperature,
    )
    faq.warmup()
    faq_admin = FaqAdminService(factory, faq, max_rows=settings.faq_import_max_rows)
    sessions = SessionService(factory)
    chat = ChatService(
        factory,
        history,
        retrieval,
        rewrite,
        build_chat_provider(settings, enable_thinking=settings.llm_enable_thinking),
        settings,
        intent=intent,
        faq=faq,
        sessions=sessions,
    )
    ingestion = IngestionService(
        factory,
        embedding,
        store,
        Path(settings.upload_dir),
        max_upload_mb=settings.max_upload_mb,
    )
    health = HealthService(engine, redis_client, store, embedding)
    return AppServices(chat=chat, ingestion=ingestion, health=health, faq_admin=faq_admin, auth=auth, sessions=sessions)


def create_app(settings: Settings | None = None, services: AppServices | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)
    services = services or build_services(settings)

    app = FastAPI(title="知店智能客服", version="1.0.0")
    app.state.settings = settings
    app.state.auth_secret = getattr(services.auth, "secret", "")
    app.state.services = services

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "code": exc.code,
                "message": exc.message,
                "request_id": getattr(request.state, "request_id", uuid.uuid4().hex),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "request_id": getattr(request.state, "request_id", uuid.uuid4().hex),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未捕获异常: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_ERROR",
                "message": "服务内部错误",
                "request_id": getattr(request.state, "request_id", uuid.uuid4().hex),
            },
        )

    app.include_router(routes_health.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_auth.router)
    app.include_router(routes_admin.router)
    app.include_router(routes_faq.router)
    app.include_router(ws_chat.router)

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app
