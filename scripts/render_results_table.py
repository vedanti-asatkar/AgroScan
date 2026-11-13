"""Print markdown results tables from A's measured reports, so README / MODEL_CARD / slides
use real numbers instead of hand-copied ones (merge-order step 4).

    python scripts/render_results_table.py                 # to stdout
    python scripts/render_results_table.py --out docs/results_tables.md

Reads reports/metrics.json (any flat or one-level-nested numeric values) and
reports/classification_report.json (scikit-learn ``output_dict=True`` format).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def metrics_table(m: dict | None) -> str:
    if not m:
        return "_reports/metrics.json not found. Run `dvc pull` / `dvc repro` (Person A's stages)._\n"
    rows = []
    for k, v in m.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            rows.append((k, v))
        elif isinstance(v, dict):
            rows += [(f"{k}.{kk}", vv) for kk, vv in v.items()
                     if isinstance(vv, (int, float)) and not isinstance(vv, bool)]
    out = ["| Metric | Value |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    return "\n".join(out) + "\n"


def _is_report(d) -> bool:
    """True for a scikit-learn style report: {class: {precision, recall, f1-score, support}, ...}."""
    return isinstance(d, dict) and any(
        isinstance(v, dict) and "precision" in v and "support" in v for v in d.values())


def _report_rows(r: dict) -> list[str]:
    skip = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    out = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]

    def row(label, d):
        return (f"| {label} | {d['precision']:.3f} | {d['recall']:.3f} | "
                f"{d['f1-score']:.3f} | {int(d['support'])} |")

    for name, d in r.items():
        if name not in skip and isinstance(d, dict) and "precision" in d:
            out.append(row(name, d))
    for name in ("macro avg", "weighted avg"):
        d = r.get(name)
        if isinstance(d, dict) and "precision" in d:
            out.append(row(f"**{name}**", d))
    return out


def per_class_table(r: dict | None) -> str:
    """Handles a flat report, or one nested by split (e.g. {"val": {...}, "test": {...}})."""
    if not r:
        return "_reports/classification_report.json not found._\n"
    if _is_report(r):
        return "\n".join(_report_rows(r)) + "\n"
    parts = []
    for split, rep in r.items():
        if _is_report(rep):
            parts.append(f"**{split}**\n\n" + "\n".join(_report_rows(rep)) + "\n")
    return "\n".join(parts) if parts else "_classification_report.json has an unrecognised format._\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.reports)
    text = ("### Overall metrics\n\n" + metrics_table(_load(d / "metrics.json")) +
            "\n### Per-class results\n\n" + per_class_table(_load(d / "classification_report.json")))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print("wrote", a.out)
    else:
        print(text)


if __name__ == "__main__":
    main()