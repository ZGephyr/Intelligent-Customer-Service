"""合成评测语料的规模、唯一性与切分行为测试。"""

from __future__ import annotations

from collections import Counter

from rag_service.evaluation.corpus import CORPUS_VERSION, DOCUMENTS
from rag_service.core.types import ParsedBlock
from rag_service.services.chunking import split_parent_child


def test_corpus_has_thirty_documents_across_ten_categories():
    assert len(DOCUMENTS) == 30
    categories = Counter(document.category for document in DOCUMENTS)

    assert len(categories) == 10
    assert set(categories.values()) == {3}


def test_corpus_slugs_and_filenames_are_unique():
    assert len({document.slug for document in DOCUMENTS}) == len(DOCUMENTS)
    assert len({document.filename for document in DOCUMENTS}) == len(DOCUMENTS)


def test_each_document_fits_parent_chunk_size():
    assert all(len(document.content) <= 1200 for document in DOCUMENTS)
    assert all(CORPUS_VERSION in document.content for document in DOCUMENTS)


def test_each_document_splits_into_exactly_one_parent():
    for document in DOCUMENTS:
        parents = split_parent_child([ParsedBlock(text=document.content)])

        assert len(parents) == 1, document.slug
        assert parents[0].children, document.slug
