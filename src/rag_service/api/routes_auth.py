"""登录、注册、登出与当前用户。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request, Response

from rag_service.api.deps import current_user
from rag_service.api.schemas import AuthOut, LoginIn, RegisterIn
from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.infra.security import create_token

logger = get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def _client_ip(request: Request) -> str:
    return (request.client.host if request.client else "") or "unknown"


def _set_auth_cookie(response: Response, request: Request, user_id: str, role: str) -> None:
    settings = request.app.state.settings
    token = create_token(
        user_id,
        role,
        request.app.state.auth_secret,
        ttl_seconds=settings.auth_token_ttl_days * 86400,
    )
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=settings.auth_token_ttl_days * 86400,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


@router.get("/captcha")
def create_captcha(request: Request) -> dict:
    service = request.app.state.services.auth
    captcha_id, question = service.create_captcha(_client_ip(request))
    return {
        "code": 0,
        "data": {
            "captcha_id": captcha_id,
            "question": question,
            "allow_admin_signup": request.app.state.settings.allow_admin_signup,
        },
        "request_id": _request_id(request),
    }


@router.post("/register")
def register(request: Request, response: Response, body: RegisterIn) -> dict:
    service = request.app.state.services.auth
    user = service.register(
        body.username,
        body.password,
        body.captcha_id,
        body.captcha_answer,
        role=body.role,
    )
    _set_auth_cookie(response, request, user.id, user.role)
    return {
        "code": 0,
        "data": AuthOut.model_validate(user).model_dump(mode="json"),
        "request_id": _request_id(request),
    }


@router.post("/login")
def login(request: Request, response: Response, body: LoginIn) -> dict:
    service = request.app.state.services.auth
    user = service.login(body.username, body.password, _client_ip(request))
    _set_auth_cookie(response, request, user.id, user.role)
    return {
        "code": 0,
        "data": AuthOut.model_validate(user).model_dump(mode="json"),
        "request_id": _request_id(request),
    }


@router.post("/logout")
def logout(request: Request, response: Response) -> dict:
    settings = request.app.state.settings
    response.delete_cookie(key=settings.auth_cookie_name, path="/")
    return {"code": 0, "data": {"logged_out": True}, "request_id": _request_id(request)}


@router.get("/me")
def me(request: Request) -> dict:
    user = current_user(request)
    if user is None:
        raise AppError(ErrorCode.AUTH_INVALID_CREDENTIALS, "请先登录", http_status=401)
    return {
        "code": 0,
        "data": AuthOut.model_validate(user).model_dump(mode="json"),
        "request_id": _request_id(request),
    }
