import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_sample_logs import make_records  # noqa: E402
from src import monitor_logs as ml  # noqa: E402

CFG = ml.load_config(Path(__file__).resolve().parents[1] / "configs" / "monitoring.yaml")


def _write(path, recs, extra_lines=()):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        for line in extra_lines:
            f.write(line + "\n")
    return path


def test_percentile_and_mean_std():
    assert ml.percentile([1, 2, 3, 4, 5], 50) == 3
    assert ml.percentile([], 50) is None
    assert ml.percentile([7], 95) == 7
    assert ml.mean_std([2, 4]) == (3, 1)


def test_read_jsonl_skips_bad_lines_and_missing_file(tmp_path):
    p = _write(tmp_path / "l.jsonl", make_records(3, False), ["not json", "[1,2]", ""])
    recs, bad = ml.read_jsonl(p)
    assert len(recs) == 3 and bad == 2
    assert ml.read_jsonl(tmp_path / "nope.jsonl") == ([], 0)


def test_confidence_histogram_counts_everything():
    recs = [{"confidence": c} for c in (0.0, 0.05, 0.55, 0.99, 1.0)]
    s = ml.confidence_stats(recs)
    assert sum(s["histogram"]["counts"]) == 5
    assert s["histogram"]["counts"][9] == 2  # 0.99 and 1.0 share the top bin


def test_class_mix_sums_to_one_and_low_conf_rate():
    recs = [{"top1_class": "a", "low_confidence": True}, {"top1_class": "a", "low_confidence": False},
            {"top1_class": "b", "low_confidence": False}, {"top1_class": "b", "low_confidence": False}]
    assert sum(ml.class_mix(recs).values()) == pytest.approx(1.0, abs=1e-3)
    assert ml.low_confidence_rate(recs) == 0.25
    assert ml.low_confidence_rate([{"x": 1}]) is None


def test_window_last_n_and_days():
    recs = make_records(100, False)
    assert len(ml.apply_window(recs, {"last_n": 10})) == 10
    old = dict(recs[0], ts="2000-01-01T00:00:00+00:00")
    assert old not in ml.apply_window([old] + recs, {"last_days": 7})


def test_healthy_logs_are_healthy(tmp_path):
    ref = ml.build_reference(make_records(300, False, seed=1))
    out = tmp_path / "summary.json"
    _write(tmp_path / "l.jsonl", make_records(300, False, seed=2))
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    s = ml.run(tmp_path / "l.jsonl", CFG, out, reference_path=tmp_path / "ref.json",
               metrics_path=tmp_path / "m.json", drift_path=tmp_path / "d.json",
               params_path=tmp_path / "p.yaml")
    assert s["status"] == "healthy" and s["retrain_triggers"] == []
    assert json.loads(out.read_text())["n_predictions"] == 300


def test_drifted_logs_trigger_retrain(tmp_path):
    ref = ml.build_reference(make_records(300, False, seed=1))
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    _write(tmp_path / "l.jsonl", make_records(300, True, seed=3))
    s = ml.run(tmp_path / "l.jsonl", CFG, tmp_path / "s.json", reference_path=tmp_path / "ref.json",
               metrics_path=tmp_path / "m.json", drift_path=tmp_path / "d.json",
               params_path=tmp_path / "p.yaml")
    ids = {t["id"] for t in s["retrain_triggers"]}
    assert {"low_confidence_rate", "brightness_drift", "class_mix_shift"} <= ids
    assert ml.branch_for_status(s) == "retrain_decision"


def test_too_few_predictions_cannot_trigger_live_alerts(tmp_path):
    _write(tmp_path / "l.jsonl", make_records(5, True))
    s = ml.run(tmp_path / "l.jsonl", CFG, tmp_path / "s.json", reference_path=tmp_path / "none.json",
               metrics_path=tmp_path / "m.json", drift_path=tmp_path / "d.json",
               params_path=tmp_path / "p.yaml")
    assert s["status"] == "insufficient_data"
    assert ml.branch_for_status(s) == "healthy_report"


def test_missing_log_file_does_not_crash(tmp_path):
    s = ml.run(tmp_path / "missing.jsonl", CFG, tmp_path / "s.json", reference_path=tmp_path / "r.json",
               metrics_path=tmp_path / "m.json", drift_path=tmp_path / "d.json", params_path=tmp_path / "p.yaml")
    assert s["n_predictions"] == 0 and s["status"] == "insufficient_data"


def test_accuracy_below_params_minimum_triggers_even_without_logs(tmp_path):
    (tmp_path / "params.yaml").write_text(yaml.safe_dump({"evaluate": {"min_test_accuracy": 0.5}}))
    (tmp_path / "m.json").write_text(json.dumps({"test_accuracy": 0.444}))
    s = ml.run(tmp_path / "missing.jsonl", CFG, tmp_path / "s.json", reference_path=tmp_path / "r.json",
               metrics_path=tmp_path / "m.json", drift_path=tmp_path / "d.json",
               params_path=tmp_path / "params.yaml")
    assert s["accuracy_floor"] == 0.5
    assert [t["id"] for t in s["retrain_triggers"]] == ["accuracy_below_minimum"]
    assert s["status"] == "retrain"


def test_accuracy_above_minimum_and_percent_tolerated(tmp_path):
    (tmp_path / "params.yaml").write_text(yaml.safe_dump({"evaluate": {"min_test_accuracy": 0.5}}))
    (tmp_path / "m.json").write_text(json.dumps({"test_accuracy": 58.3}))  # percentage form
    s = ml.run(tmp_path / "missing.jsonl", CFG, tmp_path / "s.json", reference_path=tmp_path / "r.json",
               metrics_path=tmp_path / "m.json", drift_path=tmp_path / "d.json",
               params_path=tmp_path / "params.yaml")
    assert s["retrain_triggers"] == []


def test_offline_drift_report_triggers(tmp_path):
    (tmp_path / "d.json").write_text(json.dumps({"drift_detected": True}))
    s = ml.run(tmp_path / "missing.jsonl", CFG, tmp_path / "s.json", reference_path=tmp_path / "r.json",
               metrics_path=tmp_path / "m.json", drift_path=tmp_path / "d.json", params_path=tmp_path / "p.yaml")
    assert s["retrain_triggers"][0]["id"] == "offline_drift"


def test_accuracy_floor_falls_back_when_params_missing(tmp_path):
    assert ml.resolve_accuracy_floor(CFG, tmp_path / "nope.yaml") == CFG["retrain"]["accuracy_floor_fallback"]


def test_write_decision_files(tmp_path):
    p = ml.write_decision({"status": "retrain", "retrain_triggers": [{"id": "x"}]}, "retrain_decision", tmp_path)
    doc = json.loads(p.read_text())
    assert doc["decision"] == "retrain" and doc["next_steps"]
    p = ml.write_decision({"status": "healthy"}, "healthy_report", tmp_path)
    assert json.loads(p.read_text())["decision"] == "no_action"


def test_cli_build_reference_and_fail_on_retrain(tmp_path, capsys):
    ok = _write(tmp_path / "ok.jsonl", make_records(200, False, seed=1))
    bad = _write(tmp_path / "bad.jsonl", make_records(200, True, seed=2))
    ref = tmp_path / "ref.json"
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "monitoring.yaml")
    assert ml.main(["--config", cfg, "--log", str(ok), "--reference", str(ref), "--build-reference"]) == 0
    common = ["--config", cfg, "--reference", str(ref), "--metrics", str(tmp_path / "m.json"),
              "--drift-report", str(tmp_path / "d.json"), "--params", str(tmp_path / "p.yaml")]
    assert ml.main(common + ["--log", str(ok), "--out", str(tmp_path / "a.json"), "--fail-on-retrain"]) == 0
    assert ml.main(common + ["--log", str(bad), "--out", str(tmp_path / "b.json"), "--fail-on-retrain"]) == 2
