"""PDF 解析：逐页取文本，扫描件会因无文本而失败。"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.types import ParsedBlock


def parse(path: Path) -> list[ParsedBlock]:
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - 统一转成业务异常
        raise AppError(ErrorCode.KB_EMPTY_CONTENT, f"PDF 打开失败: {path.name}") from exc

    blocks: list[ParsedBlock] = []
    try:
        for index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                blocks.append(ParsedBlock(text=text, meta={"page": index + 1}))
    finally:
        document.close()

    if not blocks:
        raise AppError(
            ErrorCode.KB_EMPTY_CONTENT,
            "PDF 未解析出可用文本，扫描件请先做 OCR",
        )
    return blocks
