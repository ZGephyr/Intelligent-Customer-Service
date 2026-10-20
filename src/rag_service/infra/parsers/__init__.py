"""解析器分发入口。"""

from __future__ import annotations

from pathlib import Path

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.types import ParsedBlock
from rag_service.infra.parsers import docx_parser, pdf_parser, table_parser, text_parser

SUPPORTED_EXTENSIONS = {"md", "txt", "csv", "xlsx", "pdf", "docx"}

_PARSERS = {
    "md": text_parser.parse,
    "txt": text_parser.parse,
    "pdf": pdf_parser.parse,
    "docx": docx_parser.parse,
    "csv": table_parser.parse_csv,
    "xlsx": table_parser.parse_xlsx,
}


def parse_file(path: Path) -> list[ParsedBlock]:
    suffix = path.suffix.lower().lstrip(".")
    parser = _PARSERS.get(suffix)
    if parser is None:
        raise AppError(
            ErrorCode.KB_FILE_TYPE_UNSUPPORTED,
            f"不支持的文件类型: .{suffix}，支持 {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    blocks = [block for block in parser(path) if block.text.strip()]
    if not blocks:
        raise AppError(ErrorCode.KB_EMPTY_CONTENT, "文件未解析出可用文本")
    return blocks
