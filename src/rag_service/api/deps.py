"""请求级依赖：登录用户、管理员校验。

兼容保留：`ADMIN_AUTH_ENABLED=true` 时，管理接口仍接受 `X-Admin-Token`（便于脚本调用）。
"""

from __future__ import annotations

from fastapi import Header, Request

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.infra.security import decode_token
from rag_service.models.tables import User


def _resolve_user(request: Request) -> User | None:
    services = getattr(request.app.state, "services", None)
    auth = getattr(services, "auth", None)
    secret = getattr(request.app.state, "auth_secret", "")
    if auth is None or not secret:
        return None

    cookie_name = request.app.state.settings.auth_cookie_name
    payload = decode_token(request.cookies.get(cookie_name, ""), secret)
    if not payload:
        return None
    return auth.get_user(payload.get("sub", ""))


def current_user(request: Request) -> User | None:
    return _resolve_user(request)


def require_user(request: Request) -> User:
    user = _resolve_user(request)
    if user is None:
        raise AppError(ErrorCode.AUTH_INVALID_CREDENTIALS, "请先登录", http_status=401)
    return user


def require_admin(
    request: Request,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> User | None:
    settings = request.app.state.settings
    user = _resolve_user(request)

    if user is not None and user.role == "admin":
        return user

    if getattr(settings, "admin_auth_enabled", False) and x_admin_token:
        if x_admin_token == settings.admin_token:
            return None

    if user is None:
        raise AppError(ErrorCode.AUTH_INVALID_CREDENTIALS, "请先登录", http_status=401)
    raise AppError(ErrorCode.AUTH_FORBIDDEN, "需要管理员权限", http_status=403)
