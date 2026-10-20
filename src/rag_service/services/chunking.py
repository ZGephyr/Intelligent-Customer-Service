"""父子分块：递归中文分割器，父块提供上下文，子块用于向量化。"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_service.core.types import ParsedBlock, ParsedChunk, ParsedParent

PARENT_SIZE = 1200
PARENT_OVERLAP = 100
CHILD_SIZE = 300
CHILD_OVERLAP = 50

# 中文优先：段落 → 换行 → 句号/问号/感叹号/分号 → 逗号/顿号 → 空格
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
    )


def split_parent_child(
    blocks: list[ParsedBlock],
    parent_size: int = PARENT_SIZE,
    parent_overlap: int = PARENT_OVERLAP,
    child_size: int = CHILD_SIZE,
    child_overlap: int = CHILD_OVERLAP,
) -> list[ParsedParent]:
    """先按父块粒度切分，再把每个父块切成子块。"""
    parent_splitter = _splitter(parent_size, parent_overlap)
    child_splitter = _splitter(child_size, child_overlap)

    parents: list[ParsedParent] = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        for parent_text in parent_splitter.split_text(text):
            parent_content = parent_text.strip()
            if not parent_content:
                continue

            children: list[ParsedChunk] = []
            for child_text in child_splitter.split_text(parent_content):
                child_content = child_text.strip()
                if not child_content:
                    continue
                children.append(
                    ParsedChunk(
                        content=child_content,
                        meta=dict(block.meta),
                        index=len(children),
                    )
                )
            if not children:
                continue

            parents.append(
                ParsedParent(
                    content=parent_content,
                    meta=dict(block.meta),
                    index=len(parents),
                    children=tuple(children),
                )
            )
    return parents
