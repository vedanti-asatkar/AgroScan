import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import render_results_table as r  # noqa: E402


def test_tables_render_and_missing_files_are_graceful(tmp_path):
    assert "not found" in r.metrics_table(None) and "not found" in r.per_class_table(None)
    t = r.metrics_table({"test_accuracy": 0.5833, "plantwild": {"top1": 0.605}, "name": "x"})
    assert "| test_accuracy | 0.5833 |" in t and "plantwild.top1" in t and "name" not in t
    rep = {"a": {"precision": 1, "recall": .5, "f1-score": .6667, "support": 4}, "accuracy": .5,
           "macro avg": {"precision": 1, "recall": .5, "f1-score": .6667, "support": 4}}
    p = r.per_class_table(rep)
    assert "| a | 1.000 | 0.500 | 0.667 | 4 |" in p and "macro avg" in p and "| accuracy" not in p


def test_nested_report_by_split_and_unknown_format():
    rep = {"a": {"precision": 1, "recall": .5, "f1-score": .6667, "support": 4}}
    nested = {"val": rep, "test": rep, "note": "x"}
    out = r.per_class_table(nested)
    assert "**val**" in out and "**test**" in out and out.count("| a |") == 2
    assert "unrecognised" in r.per_class_table({"foo": 1})