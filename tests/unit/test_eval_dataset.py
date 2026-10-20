"""评测集校验、JSONL 读取与模型 JSON 解析测试。"""

from __future__ import annotations

import pytest

from rag_service.evaluation.dataset import (
    load_jsonl,
    parse_json_object,
    validate_dataset,
    validate_question_text,
    validate_sample,
)


def sample(**overrides):
    value = {
        "id": "q001",
        "question": "保温杯能放洗碗机吗？",
        "reference_answer": "不建议放入洗碗机。",
        "gold_parent_id": "p1",
        "gold_document_id": "d1",
        "gold_source": "cup.md",
    }
    value.update(overrides)
    return value


def test_validate_sample_accepts_complete_sample():
    assert validate_sample(sample()) == []


def test_validate_sample_reports_missing_field():
    errors = validate_sample(sample(gold_source=""))

    assert any("gold_source" in error for error in errors)


def test_validate_sample_rejects_short_question():
    errors = validate_sample(sample(question="保修"))

    assert any("长度" in error for error in errors)


def test_validate_dataset_rejects_duplicate_id_and_question():
    with pytest.raises(ValueError) as exc_info:
        validate_dataset([sample(), sample(question="保温杯能放洗碗机吗？")])

    assert "id 重复" in str(exc_info.value)
    assert "question 重复" in str(exc_info.value)


def test_validate_dataset_rejects_empty_list():
    with pytest.raises(ValueError, match="评测集为空"):
        validate_dataset([])


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id":"1"}\n\n{"id":"2"}\n', encoding="utf-8")

    assert [row["id"] for row in load_jsonl(path)] == ["1", "2"]


def test_load_jsonl_reports_bad_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"1"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="第 2 行"):
        load_jsonl(path)


def test_parse_json_object_handles_code_fence_and_explanation():
    text = '结果如下：\n```json\n{"question":"q","reference_answer":"a"}\n```\n'

    assert parse_json_object(text) == {"question": "q", "reference_answer": "a"}


def test_parse_json_object_rejects_non_object_json():
    with pytest.raises(ValueError):
        parse_json_object("[1, 2]")


def test_validate_question_text_rejects_doc_tone_and_title_copy():
    errors = validate_question_text(
        "根据资料，商品参数 · 保温杯的容量是多少？",
        title="商品参数 · 保温杯",
    )

    assert any("资料口吻" in error for error in errors)
    assert any("标题" in error for error in errors)


def test_validate_question_text_accepts_customer_style_question():
    assert validate_question_text("这个杯子能装咖啡吗？") == []
