"""账号体系：验证码、注册、登录与失败限流。"""

from __future__ import annotations

import re
import secrets
import time
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.infra.security import (
    hash_captcha_answer,
    hash_password,
    verify_password,
)
from rag_service.models.tables import AuthCaptcha, User

logger = get_logger(__name__)

USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,32}$")
MIN_PASSWORD_LENGTH = 8
VALID_ROLES = {"user", "admin"}


class AttemptLimiter:
    """按 key（IP）计数的失败限流；进程内实现，单 worker 有效。"""

    def __init__(self, max_attempts: int = 10, lock_seconds: int = 60) -> None:
        self._max_attempts = max_attempts
        self._lock_seconds = lock_seconds
        self._failures: dict[str, int] = {}
        self._locked_until: dict[str, float] = {}

    def check(self, key: str) -> None:
        until = self._locked_until.get(key)
        if until is not None and time.monotonic() < until:
            raise AppError(ErrorCode.AUTH_TOO_MANY_ATTEMPTS, "尝试过于频繁，请稍后再试", http_status=429)

    def fail(self, key: str) -> None:
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count >= self._max_attempts:
            self._locked_until[key] = time.monotonic() + self._lock_seconds
            self._failures[key] = 0
            logger.warning("登录失败次数过多，已临时锁定: key=%s", _mask(key))

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)
        self._locked_until.pop(key, None)


class AuthService:
    def __init__(
        self,
        session_factory,
        secret: str,
        *,
        bcrypt_rounds: int = 12,
        captcha_ttl_seconds: int = 300,
        allow_admin_signup: bool = True,
        limiter: AttemptLimiter | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._secret = secret
        self._bcrypt_rounds = bcrypt_rounds
        self._captcha_ttl_seconds = captcha_ttl_seconds
        self._allow_admin_signup = allow_admin_signup
        self._limiter = limiter or AttemptLimiter()
        self._rng = secrets.SystemRandom()

    # ---------- 验证码 ----------

    @property
    def secret(self) -> str:
        """JWT 与验证码 HMAC 共用的密钥，供 API 层签发令牌。"""
        return self._secret

    def create_captcha(self, ip: str = "") -> tuple[str, str]:
        left = self._rng.randint(1, 9)
        right = self._rng.randint(1, 9)
        answer = str(left + right)
        captcha_id = str(uuid4())
        expires_at = datetime.now() + timedelta(seconds=self._captcha_ttl_seconds)

        with self._session_factory() as session:
            session.add(
                AuthCaptcha(
                    id=captcha_id,
                    answer_hash=hash_captcha_answer(answer, self._secret),
                    expires_at=expires_at,
                )
            )
            session.commit()

        logger.debug("下发验证码: id=%s ip=%s", captcha_id, _mask(ip))
        return captcha_id, f"{left} + {right} = ?"

    def _consume_captcha(self, captcha_id: str, answer: str) -> None:
        with self._session_factory() as session:
            row = session.get(AuthCaptcha, captcha_id)
            if row is None or row.used_at is not None:
                raise AppError(ErrorCode.AUTH_CAPTCHA_INVALID, "验证码不正确，请刷新后重试")
            if row.expires_at < datetime.now():
                raise AppError(ErrorCode.AUTH_CAPTCHA_EXPIRED, "验证码已过期，请刷新后重试")

            expected = hash_captcha_answer((answer or "").strip(), self._secret)
            if not secrets.compare_digest(expected, row.answer_hash):
                row.used_at = datetime.now()
                session.commit()
                raise AppError(ErrorCode.AUTH_CAPTCHA_INVALID, "验证码不正确，请刷新后重试")

            row.used_at = datetime.now()
            session.commit()

    # ---------- 注册 / 登录 ----------

    def register(
        self,
        username: str,
        password: str,
        captcha_id: str,
        captcha_answer: str,
        role: str = "user",
    ) -> User:
        normalized = (username or "").strip().lower()
        if not USERNAME_PATTERN.match(normalized):
            raise AppError(
                ErrorCode.AUTH_INVALID_CREDENTIALS,
                "用户名需为 3–32 位字母、数字或下划线",
            )
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise AppError(ErrorCode.AUTH_INVALID_CREDENTIALS, "密码至少 8 位")
        if role not in VALID_ROLES:
            raise AppError(ErrorCode.AUTH_FORBIDDEN, "非法的角色", http_status=403)
        if role == "admin" and not self._allow_admin_signup:
            raise AppError(ErrorCode.AUTH_FORBIDDEN, "当前不允许注册管理员", http_status=403)

        self._consume_captcha(captcha_id, captcha_answer)

        with self._session_factory() as session:
            if session.query(User).filter(User.username == normalized).first() is not None:
                raise AppError(ErrorCode.AUTH_USERNAME_TAKEN, "用户名已被占用", http_status=409)
            user = User(
                id=str(uuid4()),
                username=normalized,
                password_hash=hash_password(password, rounds=self._bcrypt_rounds),
                role=role,
            )
            session.add(user)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise AppError(ErrorCode.AUTH_USERNAME_TAKEN, "用户名已被占用", http_status=409) from exc

        logger.info("注册成功: user_id=%s role=%s", user.id, role)
        return user

    def login(self, username: str, password: str, ip: str = "") -> User:
        key = ip or "unknown"
        self._limiter.check(key)

        normalized = (username or "").strip().lower()
        with self._session_factory() as session:
            user = session.query(User).filter(User.username == normalized).first()
            if user is None or not verify_password(password, user.password_hash):
                self._limiter.fail(key)
                logger.warning("登录失败: username=%s ip=%s", normalized, _mask(key))
                raise AppError(ErrorCode.AUTH_INVALID_CREDENTIALS, "账号或密码不正确", http_status=401)

        self._limiter.reset(key)
        logger.info("登录成功: user_id=%s", user.id)
        return user

    def get_user(self, user_id: str) -> User | None:
        with self._session_factory() as session:
            return session.get(User, user_id)


def _mask(value: str) -> str:
    text = value or ""
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"
