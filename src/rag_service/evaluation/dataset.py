"""评测集读取与校验：不依赖 RAGAS，主环境也能跑单元测试。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "id",
    "question",
    "reference_answer",
    "gold_parent_id",
    "gold_document_id",
    "gold_source",
)

_WHITESPACE = re.compile(r"\s+")
_BANNED_QUESTION_TERMS = (
    "根据资料",
    "资料显示",
    "资料中",
    "根据文档",
    "文档中",
    "根据上文",
    "上文提到",
    "本文",
    "文中",
)


def parse_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象，兼容 ```json 代码围栏与前后说明文字。"""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型输出中没有 JSON 对象")
    value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型输出的 JSON 顶层必须是对象")
    return value


def validate_question_text(question: str, *, title: str = "", reference: str = "") -> list[str]:
    """检查生成的问题是否像顾客提问，而不是在复述资料标题或提示词。"""
    text = (question or "").strip()
    errors: list[str] = []
    if not 6 <= len(text) <= 120:
        errors.append(f"问题长度应在 6-120 字，实际 {len(text)}")
    for term in _BANNED_QUESTION_TERMS:
        if term in text:
            errors.append(f"问题不应包含资料口吻词: {term}")
    if title and title.strip() and title.strip() in text:
        errors.append("问题不应照抄资料标题")
    if reference and normalize_question(text) == normalize_question(reference):
        errors.append("问题不能与参考答案相同")
    return errors


def normalize_question(text: str) -> str:
    """用于去重与相似检查：去空白、统一小写。"""
    return _WHITESPACE.sub("", (text or "").strip()).lower()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，跳过空行，任何一行解析失败都抛出带行号的信息。"""
    rows: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path} 第 {line_no} 行必须是 JSON 对象")
        rows.append(value)
    return rows


def validate_sample(sample: dict[str, Any]) -> list[str]:
    """返回单条样本的校验错误；空列表表示通过。"""
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        value = sample.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"缺少或为空: {field}")

    question = str(sample.get("question") or "").strip()
    reference = str(sample.get("reference_answer") or "").strip()
    if question and not 6 <= len(question) <= 120:
        errors.append(f"question 长度应在 6-120 字，实际 {len(question)}")
    if reference and len(reference) > 300:
        errors.append(f"reference_answer 过长: {len(reference)} 字")

    gold_parent = str(sample.get("gold_parent_id") or "").strip()
    gold_doc = str(sample.get("gold_document_id") or "").strip()
    if gold_parent and gold_doc and gold_parent == gold_doc:
        errors.append("gold_parent_id 不能等于 gold_document_id")
    return errors


def validate_dataset(samples: list[dict[str, Any]]) -> None:
    """整批校验，并检查 ID 与问题重复；失败时一次性抛出全部问题。"""
    if not samples:
        raise ValueError("评测集为空")

    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for index, sample in enumerate(samples, start=1):
        errors = validate_sample(sample)
        if errors:
            problems.append(f"第 {index} 条: " + "; ".join(errors))
            continue

        sample_id = str(sample["id"]).strip()
        if sample_id in seen_ids:
            problems.append(f"第 {index} 条: id 重复 {sample_id}")
        seen_ids.add(sample_id)

        normalized = normalize_question(str(sample["question"]))
        if normalized in seen_questions:
            problems.append(f"第 {index} 条: question 重复")
        seen_questions.add(normalized)

    if problems:
        raise ValueError("评测集校验失败:\n" + "\n".join(problems))
