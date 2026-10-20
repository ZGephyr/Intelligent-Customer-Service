"""数据库连接与初始化工具。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.models.tables import Base

logger = get_logger(__name__)

_SAFE_DB_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def build_engine(url: str) -> Engine:
    """创建 Engine；sqlite 内存库用 StaticPool 保证多连接同库。"""
    parsed = make_url(url)
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if parsed.get_backend_name() == "sqlite" and not parsed.database:
        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def database_name_from_url(database_url: str) -> str:
    return make_url(database_url).database or ""


def server_url_without_database(database_url: str) -> URL:
    """去掉库名的连接串，用于 CREATE DATABASE。"""
    url = make_url(database_url)
    return URL.create(
        drivername=url.drivername,
        username=url.username,
        password=url.password,
        host=url.host,
        port=url.port,
        query=url.query,
    )


def ensure_database(database_url: str) -> None:
    """MySQL 库不存在时自动创建；非 MySQL（测试用 sqlite）直接跳过。"""
    url = make_url(database_url)
    if url.get_backend_name() != "mysql":
        return

    name = url.database or ""
    if not _SAFE_DB_NAME.match(name):
        raise AppError(ErrorCode.INTERNAL_ERROR, f"非法的数据库名: {name!r}")

    import pymysql

    try:
        conn = pymysql.connect(
            host=url.host or "localhost",
            port=url.port or 3306,
            user=url.username or "root",
            password=url.password or "",
            charset="utf8mb4",
            autocommit=True,
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{name}` DEFAULT CHARACTER SET utf8mb4"
                )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - 统一转成启动失败
        logger.exception("创建数据库失败: %s", name)
        raise AppError(ErrorCode.INTERNAL_ERROR, f"创建数据库失败: {name}") from exc

    logger.info("数据库就绪: %s", name)
