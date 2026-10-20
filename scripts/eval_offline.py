"""离线评测：意图准确率与 FAQ 命中率，用来给 1.0 验收出证据。

用法：
    python -m scripts.eval_offline faq --threshold 0.6 --margin 0.05
    python -m scripts.eval_offline intent --live
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

DEFAULT_DIR = Path("data/eval")

IN_SCOPE = "in_scope"
OUT_OF_SCOPE = "out_of_scope"


def load_cases(path: Path) -> list[dict[str, str]]:
    """读取 CSV（兼容 BOM），丢掉全空行，键值都 strip。"""
    text = Path(path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for raw in reader:
        cleaned = {
            str(key).strip(): ("" if value is None else str(value).strip())
            for key, value in raw.items()
            if key is not None
        }
        if any(cleaned.values()):
            rows.append(cleaned)
    return rows


def intent_accuracy(cases: list[dict[str, str]], classifier) -> dict:
    total = len(cases)
    correct = 0
    false_block = 0
    for case in cases:
        label = case.get("label", "")
        predicted = classifier(case.get("query", ""))
        if predicted == label:
            correct += 1
        elif label == IN_SCOPE and predicted == OUT_OF_SCOPE:
            false_block += 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "false_block": false_block,
    }


def faq_hit_rate(cases: list[dict[str, str]], matcher) -> dict:
    total = len(cases)
    hit = 0
    for case in cases:
        if matcher(case.get("query", "")):
            hit += 1
    return {"total": total, "hit": hit, "rate": round(hit / total, 4) if total else 0.0}


def evaluate_faq(
    faq_path: Path,
    cases_path: Path,
    threshold: float,
    margin: float,
    min_coverage: float = 0.0,
) -> dict:
    """把种子表灌进内存库，直接量 FaqService 的命中行为（不另写一套匹配逻辑）。"""
    from uuid import uuid4

    from rag_service.infra.db import build_engine, build_session_factory, ensure_schema
    from rag_service.models.tables import FaqPair
    from rag_service.services.faq_service import FaqService

    faq_rows = load_cases(faq_path)
    cases = load_cases(cases_path)

    engine = build_engine("sqlite://")
    ensure_schema(engine)
    factory = build_session_factory(engine)
    with factory() as session:
        for row in faq_rows:
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue
            session.add(
                FaqPair(
                    id=str(uuid4()),
                    question=question,
                    answer=answer,
                    category=(row.get("category") or "").strip() or None,
                    aliases=(row.get("aliases") or "").strip() or None,
                )
            )
        session.commit()

    service = FaqService(
        factory, threshold=threshold, margin=margin, min_coverage=min_coverage
    )
    correct = 0
    wrong = 0
    false_hit = 0
    expected_rows = 0
    for case in cases:
        expected = case.get("expected_faq", "")
        hit = service.match(case.get("query", ""))
        matched = hit.question if hit else ""
        if expected:
            expected_rows += 1
            if matched == expected:
                correct += 1
            elif matched:
                wrong += 1
        elif matched:
            false_hit += 1

    total = len(cases)
    return {
        "total": total,
        "expected_rows": expected_rows,
        "correct": correct,
        "wrong": wrong,
        "false_hit": false_hit,
        "recall": round(correct / expected_rows, 4) if expected_rows else 0.0,
        "precision_samples": round(correct / (correct + wrong + false_hit), 4)
        if (correct + wrong + false_hit)
        else 0.0,
    }


def _print_table(title: str, payload: dict) -> None:
    print(f"\n### {title}\n")
    print("| 指标 | 数值 |")
    print("| --- | --- |")
    for key, value in payload.items():
        print(f"| {key} | {value} |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="1.0 离线评测")
    parser.add_argument("mode", choices=["faq", "intent"])
    parser.add_argument("--faqs", type=Path, default=DEFAULT_DIR / "faq_seed.csv")
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-coverage", type=float, default=0.3)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--live", action="store_true", help="intent 模式需要真实 LLM 配置")
    args = parser.parse_args(argv)

    if args.mode == "faq":
        cases = args.cases or DEFAULT_DIR / "faq_cases.csv"
        report = evaluate_faq(
            args.faqs, cases, args.threshold, args.margin, args.min_coverage
        )
        _print_table(
            f"FAQ 命中率（threshold={args.threshold}, margin={args.margin}, min_coverage={args.min_coverage}）",
            report,
        )
        return 0

    cases = args.cases or DEFAULT_DIR / "intent_cases.csv"
    if not args.live:
        print("intent 模式需要 --live 且 .env 中的 LLM 可用；未跑真实分类。", file=sys.stderr)
        return 2

    from rag_service.core.config import get_settings
    from rag_service.infra.llm import build_chat_provider
    from rag_service.services.intent_service import IntentService

    settings = get_settings()
    service = IntentService(
        build_chat_provider(settings, timeout_seconds=settings.intent_timeout_seconds)
    )
    report = intent_accuracy(load_cases(cases), lambda query: service.classify(query, []).intent)
    _print_table("意图二分类准确率", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
