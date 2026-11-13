"""Live-log monitoring for AgroScan (task B1, thresholds from B3).

Reads the API's ``logs/predictions.jsonl`` and writes
``reports/monitoring/summary.json`` with:

* confidence distribution
* predicted-class mix
* low-confidence rate
* latency (mean / p50 / p95)
* brightness drift against a reference set
* alerts (warn) and retrain triggers (act), using ``configs/monitoring.yaml``

Frozen log fields (owned by A): ts, model_version, top1_class, confidence,
top3, low_confidence, latency_ms, brightness.

Usage
-----
    python -m src.monitor_logs                       # real logs, default config
    python -m src.monitor_logs --log tests/data/sample_predictions.jsonl
    python -m src.monitor_logs --build-reference     # (re)build baseline from current log
    python scripts/make_sample_logs.py --out logs/sample.jsonl --drift   # demo data

Exit code is 0 unless ``--fail-on-retrain`` is given and a retrain is triggered.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULT_CONFIG = Path("configs/monitoring.yaml")


# --------------------------------------------------------------------------- IO
def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: str | Path) -> tuple[list[dict], int]:
    """Return (records, n_bad_lines). Missing file -> ([], 0). Bad lines are skipped."""
    p = Path(path)
    if not p.exists():
        return [], 0
    records, bad = [], 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                bad += 1
    return records, bad


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def apply_window(records: list[dict], window: dict, now: datetime | None = None) -> list[dict]:
    """Keep records from the last N days, then the last N rows."""
    window = window or {}
    out = records
    days = window.get("last_days")
    if days:
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)
        kept = []
        for r in out:
            ts = _parse_ts(r.get("ts"))
            if ts is None or ts >= cutoff:  # unparseable timestamp: keep rather than hide data
                kept.append(r)
        out = kept
    last_n = window.get("last_n")
    if last_n:
        out = out[-int(last_n):]
    return out


# ------------------------------------------------------------------------ stats
def _floats(records: Iterable[dict], key: str) -> list[float]:
    vals = []
    for r in records:
        v = r.get(key)
        if isinstance(v, bool) or v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    return vals


def percentile(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, math.sqrt(var)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def confidence_stats(records: list[dict]) -> dict:
    vals = sorted(_floats(records, "confidence"))
    if not vals:
        return {"n": 0}
    m, s = mean_std(vals)
    edges = [i / 10 for i in range(11)]
    hist = [0] * 10
    for v in vals:
        hist[min(int(max(v, 0.0) * 10), 9)] += 1
    return {
        "n": len(vals),
        "mean": _r(m),
        "std": _r(s),
        "p10": _r(percentile(vals, 10)),
        "median": _r(percentile(vals, 50)),
        "p90": _r(percentile(vals, 90)),
        "histogram": {"bin_edges": edges, "counts": hist},
    }


def class_mix(records: list[dict]) -> dict[str, float]:
    counts = Counter(r.get("top1_class") for r in records if r.get("top1_class"))
    total = sum(counts.values())
    if not total:
        return {}
    return {k: _r(v / total) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def low_confidence_rate(records: list[dict]) -> float | None:
    flags = [r["low_confidence"] for r in records if isinstance(r.get("low_confidence"), bool)]
    if not flags:
        return None
    return sum(flags) / len(flags)


def latency_stats(records: list[dict]) -> dict:
    vals = sorted(_floats(records, "latency_ms"))
    if not vals:
        return {"n": 0}
    m, _ = mean_std(vals)
    return {
        "n": len(vals),
        "mean_ms": _r(m, 2),
        "p50_ms": _r(percentile(vals, 50), 2),
        "p95_ms": _r(percentile(vals, 95), 2),
        "max_ms": _r(vals[-1], 2),
    }


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


# -------------------------------------------------------------------- reference
def build_reference(records: list[dict]) -> dict:
    """Baseline from a known-good period (or a reference image set's log)."""
    b = _floats(records, "brightness")
    m, s = mean_std(b)
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(records),
        "brightness_mean": _r(m),
        "brightness_std": _r(s),
        "class_mix": class_mix(records),
    }


def load_reference(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def brightness_drift(records: list[dict], ref: dict | None) -> dict:
    vals = _floats(records, "brightness")
    m, _ = mean_std(vals)
    out: dict[str, Any] = {"n": len(vals), "mean": _r(m)}
    if ref is None or ref.get("brightness_mean") is None:
        out.update(reference_available=False, zscore=None)
        return out
    ref_m = ref["brightness_mean"]
    ref_s = ref.get("brightness_std") or 0.0
    out.update(reference_available=True, reference_mean=ref_m, reference_std=ref_s)
    if m is None:
        out["zscore"] = None
    elif ref_s > 0:
        out["zscore"] = _r(abs(m - ref_m) / ref_s)
    else:
        out["zscore"] = 0.0 if m == ref_m else float("inf")
    return out


# ------------------------------------------------------------------- thresholds
def _first_key(d: dict, keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _read_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def resolve_accuracy_floor(cfg: dict, params_path: str | Path | None = None) -> float:
    rcfg = cfg.get("retrain", {})
    src = rcfg.get("accuracy_floor_source", "params")
    fallback = float(rcfg.get("accuracy_floor_fallback", 0.5))
    if isinstance(src, (int, float)):
        return float(src)
    path = params_path or cfg.get("paths", {}).get("params", "params.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        return float(params["evaluate"]["min_test_accuracy"])
    except (OSError, KeyError, TypeError, ValueError):
        return fallback


def evaluate_thresholds(
    summary: dict,
    cfg: dict,
    metrics: dict | None = None,
    drift_report: dict | None = None,
    accuracy_floor: float | None = None,
) -> dict:
    """Return {"alerts": [...], "retrain_triggers": [...], "status": str}.

    Triggers (retrain policy, docs/retraining_policy.md):
      T1 offline drift detected (drift.json)
      T2 accuracy below evaluate.min_test_accuracy
      T3 low-confidence rate too high
      T4 live brightness drift / class-mix shift (extra, config-driven)
    Live-log based triggers need at least window.min_predictions records.
    """
    alerts_cfg = cfg.get("alerts", {})
    rt_cfg = cfg.get("retrain", {})
    keys = cfg.get("report_keys", {})
    alerts: list[dict] = []
    triggers: list[dict] = []

    enough = summary.get("n_predictions", 0) >= cfg.get("window", {}).get("min_predictions", 0)

    # T1: offline drift (does not need live logs)
    if rt_cfg.get("retrain_on_offline_drift", True) and drift_report:
        flag = _first_key(drift_report, keys.get("drift_flag", ["drift_detected"]))
        if flag is True:
            triggers.append({"id": "offline_drift", "detail": "reports/drift.json reports drift"})

    # T2: accuracy below the gate (does not need live logs)
    if metrics is not None and accuracy_floor is not None:
        acc = _first_key(metrics, keys.get("accuracy", ["test_accuracy"]))
        if isinstance(acc, (int, float)):
            if acc > 1.0:  # tolerate percentages
                acc = acc / 100.0
            if acc < accuracy_floor:
                triggers.append({
                    "id": "accuracy_below_minimum",
                    "detail": f"test accuracy {acc:.3f} < minimum {accuracy_floor:.3f}",
                })

    if enough:
        lcr = summary.get("low_confidence_rate")
        if lcr is not None:
            if lcr > rt_cfg.get("low_confidence_rate_max", 1.1):
                triggers.append({
                    "id": "low_confidence_rate",
                    "detail": f"{lcr:.3f} > {rt_cfg['low_confidence_rate_max']}",
                })
            elif lcr > alerts_cfg.get("low_confidence_rate_warn", 1.1):
                alerts.append({"id": "low_confidence_rate",
                               "detail": f"{lcr:.3f} > {alerts_cfg['low_confidence_rate_warn']}"})

        z = (summary.get("brightness") or {}).get("zscore")
        if z is not None:
            if z > rt_cfg.get("brightness_drift_zscore_max", math.inf):
                triggers.append({"id": "brightness_drift",
                                 "detail": f"z={z} > {rt_cfg['brightness_drift_zscore_max']}"})
            elif z > alerts_cfg.get("brightness_drift_zscore_warn", math.inf):
                alerts.append({"id": "brightness_drift",
                               "detail": f"z={z} > {alerts_cfg['brightness_drift_zscore_warn']}"})

        tv = summary.get("class_mix_tv_distance")
        if tv is not None and tv > rt_cfg.get("class_mix_tv_distance_max", math.inf):
            triggers.append({"id": "class_mix_shift",
                             "detail": f"TV={tv} > {rt_cfg['class_mix_tv_distance_max']}"})

        p95 = (summary.get("latency") or {}).get("p95_ms")
        if p95 is not None and p95 > alerts_cfg.get("latency_p95_ms_warn", math.inf):
            alerts.append({"id": "latency_p95",
                           "detail": f"{p95}ms > {alerts_cfg['latency_p95_ms_warn']}ms"})

    if triggers:
        status = "retrain"
    elif not enough:
        status = "insufficient_data"
    else:
        status = "warning" if alerts else "healthy"
    return {"alerts": alerts, "retrain_triggers": triggers, "status": status}


# ------------------------------------------------------------------- orchestration
def summarize(records: list[dict], cfg: dict, ref: dict | None, bad_lines: int = 0) -> dict:
    mix = class_mix(records)
    summary: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_predictions": len(records),
        "n_malformed_lines": bad_lines,
        "model_versions": dict(Counter(r.get("model_version") for r in records if r.get("model_version"))),
        "confidence": confidence_stats(records),
        "class_mix": mix,
        "low_confidence_rate": _r(low_confidence_rate(records)),
        "latency": latency_stats(records),
        "brightness": brightness_drift(records, ref),
    }
    ts = [t for t in (_parse_ts(r.get("ts")) for r in records) if t]
    if ts:
        summary["period"] = {"start": min(ts).isoformat(), "end": max(ts).isoformat()}
    if ref and ref.get("class_mix") and mix:
        summary["class_mix_tv_distance"] = _r(total_variation(mix, ref["class_mix"]))
    return summary


def run(
    log_path: str | Path,
    cfg: dict,
    out_path: str | Path,
    reference_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    drift_path: str | Path | None = None,
    params_path: str | Path | None = None,
) -> dict:
    paths = cfg.get("paths", {})
    records, bad = read_jsonl(log_path)
    records = apply_window(records, cfg.get("window", {}))
    ref = load_reference(reference_path or paths.get("reference", "reports/monitoring/reference.json"))
    summary = summarize(records, cfg, ref, bad)
    summary["log_file"] = str(log_path)

    metrics = _read_json(metrics_path or paths.get("metrics", "reports/metrics.json"))
    drift = _read_json(drift_path or paths.get("drift_report", "reports/drift.json"))
    floor = resolve_accuracy_floor(cfg, params_path)
    summary["accuracy_floor"] = floor
    summary["inputs_found"] = {"metrics": metrics is not None, "drift_report": drift is not None,
                               "reference": ref is not None}
    summary.update(evaluate_thresholds(summary, cfg, metrics, drift, floor))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def branch_for_status(summary: dict) -> str:
    """Airflow branch: task_id to follow after check_thresholds."""
    return "retrain_decision" if summary.get("status") == "retrain" else "healthy_report"


def write_decision(summary: dict, kind: str, out_dir: str | Path = "reports/monitoring") -> Path:
    """Write healthy_report.json or retrain_decision.json next to summary.json."""
    assert kind in ("healthy_report", "retrain_decision")
    doc = {
        "decision": "retrain" if kind == "retrain_decision" else "no_action",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": summary.get("status"),
        "n_predictions": summary.get("n_predictions"),
        "retrain_triggers": summary.get("retrain_triggers", []),
        "alerts": summary.get("alerts", []),
        "next_steps": (["dvc repro --single-item train", "dvc repro --single-item evaluate",
                        "dvc repro --single-item export"] if kind == "retrain_decision" else []),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{kind}.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--log", help="prediction log (default: from config)")
    ap.add_argument("--out", help="summary path (default: from config)")
    ap.add_argument("--reference", help="reference json (default: from config)")
    ap.add_argument("--metrics", help="metrics.json (default: from config)")
    ap.add_argument("--drift-report", help="drift.json (default: from config)")
    ap.add_argument("--params", help="params.yaml (default: from config)")
    ap.add_argument("--build-reference", action="store_true",
                    help="write the reference baseline from the log and exit")
    ap.add_argument("--fail-on-retrain", action="store_true", help="exit 2 if a retrain is triggered")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    log = args.log or paths.get("prediction_log", "logs/predictions.jsonl")
    ref_path = args.reference or paths.get("reference", "reports/monitoring/reference.json")

    if args.build_reference:
        records, _ = read_jsonl(log)
        if not records:
            print(f"No records in {log}; cannot build reference.", file=sys.stderr)
            return 1
        ref = build_reference(records)
        Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ref_path).write_text(json.dumps(ref, indent=2), encoding="utf-8")
        print(f"Reference written to {ref_path} ({ref['n']} records)")
        return 0

    summary = run(log, cfg, args.out or paths.get("summary_out", "reports/monitoring/summary.json"),
                  reference_path=ref_path, metrics_path=args.metrics,
                  drift_path=args.drift_report, params_path=args.params)
    print(f"status={summary['status']} n={summary['n_predictions']} "
          f"triggers={[t['id'] for t in summary['retrain_triggers']]} "
          f"alerts={[a['id'] for a in summary['alerts']]}")
    return 2 if (args.fail_on_retrain and summary["status"] == "retrain") else 0


if __name__ == "__main__":
    sys.exit(main())
