"""健康检查。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> JSONResponse:
    checks = request.app.state.services.health.check()
    healthy = all(value == "ok" for value in checks.values())
    body = {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
        "request_id": getattr(request.state, "request_id", uuid.uuid4().hex),
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)
