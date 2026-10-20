"""统一业务异常与错误码。"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """对外错误码。前端按 code 做分支，message 只做展示。"""

    KB_FILE_TYPE_UNSUPPORTED = "KB_FILE_TYPE_UNSUPPORTED"
    KB_FILE_TOO_LARGE = "KB_FILE_TOO_LARGE"
    KB_EMPTY_CONTENT = "KB_EMPTY_CONTENT"
    KB_DOCUMENT_NOT_FOUND = "KB_DOCUMENT_NOT_FOUND"
    KB_DUPLICATE = "KB_DUPLICATE"
    FAQ_DUPLICATE = "FAQ_DUPLICATE"
    FAQ_NOT_FOUND = "FAQ_NOT_FOUND"
    FAQ_IMPORT_INVALID = "FAQ_IMPORT_INVALID"
    AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"
    AUTH_USERNAME_TAKEN = "AUTH_USERNAME_TAKEN"
    AUTH_CAPTCHA_INVALID = "AUTH_CAPTCHA_INVALID"
    AUTH_CAPTCHA_EXPIRED = "AUTH_CAPTCHA_EXPIRED"
    AUTH_FORBIDDEN = "AUTH_FORBIDDEN"
    AUTH_TOO_MANY_ATTEMPTS = "AUTH_TOO_MANY_ATTEMPTS"
    CHAT_SESSION_NOT_FOUND = "CHAT_SESSION_NOT_FOUND"
    CHAT_INVALID_MESSAGE = "CHAT_INVALID_MESSAGE"
    CHAT_BUSY = "CHAT_BUSY"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_ERROR = "LLM_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """可预期的业务异常，由 API 层统一转成响应。"""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code.value if isinstance(code, ErrorCode) else code
        self.message = message
        self.http_status = http_status

    def __str__(self) -> str:  # pragma: no cover - 直接返回消息
        return self.message
