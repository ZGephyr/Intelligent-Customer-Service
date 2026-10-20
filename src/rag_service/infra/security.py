"""密码哈希、JWT 签发与验证码答案哈希。

密码先做 SHA-256 预哈希再交给 bcrypt：规避 bcrypt 的 72 字节上限，
也让超长中文密码能稳定校验。
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from pathlib import Path

import bcrypt
import jwt

from rag_service.core.logging import get_logger

logger = get_logger(__name__)

ALGORITHM = "HS256"


def _prepare(password: str) -> bytes:
    return hashlib.sha256((password or "").encode("utf-8")).digest()


def hash_password(password: str, rounds: int = 12) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(rounds=rounds)).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(password), hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def create_token(user_id: str, role: str, secret: str, ttl_seconds: int) -> str:
    now = int(time.time())
    payload = {"sub": user_id, "role": role, "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str) -> dict | None:
    if not token:
        return None
    try:
        return jwt.decode(token, secret, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


def load_or_create_secret(path: Path) -> str:
    key_path = Path(path)
    if key_path.exists():
        existing = key_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing

    secret = secrets.token_hex(32)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(secret, encoding="utf-8")
    logger.info("已生成新的 JWT 密钥文件: %s", key_path)
    return secret


def hash_captcha_answer(answer: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), (answer or "").encode("utf-8"), hashlib.sha256).hexdigest()
