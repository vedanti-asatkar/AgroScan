"""Generate synthetic logs/predictions.jsonl-style data for demos, tests and the Airflow DAG.

Matches the frozen log fields: ts, model_version, top1_class, confidence, top3,
low_confidence, latency_ms, brightness.

    python scripts/make_sample_logs.py --out logs/sample.jsonl --n 300
    python scripts/make_sample_logs.py --out logs/sample_drift.jsonl --n 300 --drift
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Placeholder class names; replace with the real ones from models/classes.json if you like.
CLASSES = ["pepper_healthy", "pepper_bacterial_spot", "potato_healthy", "potato_early_blight",
           "potato_late_blight", "tomato_healthy", "tomato_early_blight", "tomato_late_blight"]


def make_records(n: int, drift: bool, seed: int = 0, threshold: float = 0.7,
                 model_version: str = "sample-v5") -> list[dict]:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    recs = []
    for i in range(n):
        if drift:
            conf = min(max(rng.gauss(0.55, 0.18), 0.05), 0.99)      # less sure
            bright = rng.gauss(0.72, 0.05)                           # brighter images
            weights = [1, 1, 1, 1, 1, 1, 1, 8]                      # one class dominates
        else:
            conf = min(max(rng.gauss(0.82, 0.12), 0.05), 0.99)
            bright = rng.gauss(0.45, 0.05)
            weights = [1] * len(CLASSES)
        top1 = rng.choices(CLASSES, weights=weights)[0]
        rest = rng.sample([c for c in CLASSES if c != top1], 2)
        p2 = (1 - conf) * 0.6
        recs.append({
            "ts": (now - timedelta(minutes=(n - i) * 5)).isoformat(timespec="seconds"),
            "model_version": model_version,
            "top1_class": top1,
            "confidence": round(conf, 4),
            "top3": [{"class": top1, "confidence": round(conf, 4)},
                     {"class": rest[0], "confidence": round(p2, 4)},
                     {"class": rest[1], "confidence": round((1 - conf) - p2, 4)}],
            "low_confidence": conf < threshold,
            "latency_ms": round(max(rng.gauss(180, 40), 20), 1),
            "brightness": round(bright, 4),
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--drift", action="store_true", help="simulate domain shift")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in make_records(a.n, a.drift, a.seed):
            f.write(json.dumps(r) + "\n")
    print(f"wrote {a.n} records to {out}")


if __name__ == "__main__":
    main()
