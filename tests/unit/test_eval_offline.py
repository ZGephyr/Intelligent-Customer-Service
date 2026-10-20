"""离线评测脚本测试：CSV 读取与指标计算。"""

from __future__ import annotations

from scripts.eval_offline import faq_hit_rate, intent_accuracy, load_cases


def test_load_cases_reads_csv_rows(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text("query,label\n你好,out_of_scope\n耳机保修多久,in_scope\n", encoding="utf-8")

    rows = load_cases(path)

    assert rows[0]["query"] == "你好" and rows[1]["label"] == "in_scope"


def test_load_cases_strips_bom_and_blank_lines(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("\ufeffquery,label\n你好,out_of_scope\n\n".encode("utf-8"))

    rows = load_cases(path)

    assert len(rows) == 1 and rows[0]["query"] == "你好"


def test_intent_accuracy_counts_correct_and_false_block():
    cases = [{"query": "a", "label": "in_scope"}, {"query": "b", "label": "out_of_scope"}]

    report = intent_accuracy(cases, lambda q: "in_scope")

    assert report == {"total": 2, "correct": 1, "accuracy": 0.5, "false_block": 0}


def test_intent_accuracy_counts_false_block():
    cases = [{"query": "保修多久", "label": "in_scope"}]

    report = intent_accuracy(cases, lambda q: "out_of_scope")

    assert report["correct"] == 0 and report["false_block"] == 1


def test_intent_accuracy_empty_cases_returns_zero_accuracy():
    report = intent_accuracy([], lambda q: "in_scope")

    assert report == {"total": 0, "correct": 0, "accuracy": 0.0, "false_block": 0}


def test_faq_hit_rate_counts_hits():
    cases = [
        {"query": "q1", "expected_faq": "多久发货"},
        {"query": "q2", "expected_faq": "发票怎么开"},
    ]

    report = faq_hit_rate(cases, lambda q: "多久发货" if q == "q1" else None)

    assert report["total"] == 2 and report["hit"] == 1 and report["rate"] == 0.5
