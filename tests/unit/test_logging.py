"""日志模块测试。"""

from __future__ import annotations

import logging

from rag_service.core.logging import get_logger, setup_logging


def test_setup_logging_writes_utf8_file_with_pipe_format(tmp_path, settings_factory):
    settings = settings_factory(log_dir=tmp_path)
    setup_logging(settings)

    get_logger("tests.demo").info("中文日志 %s", "ok")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "中文日志 ok" in content
    assert "| INFO |" in content


def test_setup_logging_is_idempotent_and_keeps_two_handlers(tmp_path, settings_factory):
    settings = settings_factory(log_dir=tmp_path)
    setup_logging(settings)
    setup_logging(settings)

    handlers = logging.getLogger().handlers
    assert len(handlers) == 2
