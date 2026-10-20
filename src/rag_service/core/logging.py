"""集中式日志：控制台 + 按天轮转文件，全项目共用。"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from rag_service.core.config import Settings, get_settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE_NAME = "app.log"
BACKUP_DAYS = 14


def _ensure_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，中文日志会乱码或抛错。"""
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 特殊终端不支持时忽略
            pass


def setup_logging(settings: Settings | None = None) -> None:
    """初始化根日志器；重复调用会重建 handler，不会重复输出。"""
    settings = settings or get_settings()

    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _ensure_utf8_stdout()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        filename=log_dir / LOG_FILE_NAME,
        when="midnight",
        backupCount=BACKUP_DAYS,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(settings.log_level.upper())
    root.addHandler(console_handler)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """模块获取 logger 的唯一入口。"""
    return logging.getLogger(name)
