"""文本类文件解析。"""

from __future__ import annotations

from pathlib import Path

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.types import ParsedBlock


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AppError(ErrorCode.KB_EMPTY_CONTENT, "文件编码无法识别，请转成 UTF-8 后重试")


def parse(path: Path) -> list[ParsedBlock]:
    text = _decode(path.read_bytes()).strip()
    if not text:
        raise AppError(ErrorCode.KB_EMPTY_CONTENT, "文件内容为空")
    return [ParsedBlock(text=text)]
