"""切分器测试：父子分块 + 递归中文分割。"""

from __future__ import annotations

from rag_service.core.types import ParsedBlock
from rag_service.services.chunking import split_parent_child


def test_split_parent_child_short_text_single_parent_single_child():
    parents = split_parent_child([ParsedBlock(text="短文本")])
    assert len(parents) == 1
    assert parents[0].index == 0
    assert [child.content for child in parents[0].children] == ["短文本"]


def test_split_parent_child_keeps_child_inside_parent_and_sequential_indexes():
    text = "".join(f"第{i}段内容，说明退换货规则。" for i in range(120))
    parents = split_parent_child(
        [ParsedBlock(text=text)], parent_size=400, parent_overlap=50, child_size=100, child_overlap=20
    )
    assert len(parents) > 1
    assert [parent.index for parent in parents] == list(range(len(parents)))
    for parent in parents:
        assert [child.index for child in parent.children] == list(range(len(parent.children)))
        for child in parent.children:
            assert child.content in parent.content


def test_split_parent_child_empty_blocks_returns_empty_list():
    assert split_parent_child([]) == []


def test_split_parent_child_skips_blank_block_and_keeps_meta():
    parents = split_parent_child(
        [ParsedBlock(text="   "), ParsedBlock(text="政策内容", meta={"page": 2})]
    )
    assert len(parents) == 1
    assert parents[0].meta == {"page": 2}
