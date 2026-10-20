"""清空并重建知识库：删除 MySQL 文档/分块表与 Milvus 集合。

用法：
    uv run python scripts/reset_kb.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_service.core.config import get_settings  # noqa: E402
from rag_service.infra.db import build_engine, create_all  # noqa: E402
from rag_service.infra.milvus_store import MilvusChunkStore  # noqa: E402
from rag_service.models.tables import Base, KbChunk, KbDocument  # noqa: E402


def main() -> None:
    settings = get_settings()
    engine = build_engine(settings.mysql_url)

    Base.metadata.drop_all(engine, tables=[KbChunk.__table__, KbDocument.__table__])
    create_all(engine)
    print("MySQL: 已删除 kb_chunks / kb_documents 并重建")

    store = MilvusChunkStore(
        settings.milvus_uri,
        settings.milvus_token,
        settings.milvus_collection,
        settings.embedding_dim,
        database=settings.milvus_database,
    )
    store.drop_and_recreate()
    print(f"Milvus: 已重建集合 {settings.milvus_collection}")

    upload_dir = Path(settings.upload_dir)
    if upload_dir.is_dir():
        shutil.rmtree(upload_dir, ignore_errors=True)
    print("已清空上传文件目录")


if __name__ == "__main__":
    main()
