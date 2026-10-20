"""请求级依赖：管理令牌校验。"""

from __future__ import annotations

from fastapi import Header, Request

from rag_service.core.exceptions import AppError, ErrorCode


def require_admin(
    request: Request,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    settings = request.app.state.settings
    if not getattr(settings, "admin_auth_enabled", False):
        return
    if not x_admin_token or x_admin_token != settings.admin_token:
        raise AppError(ErrorCode.UNAUTHORIZED, "管理令牌无效", http_status=401)
