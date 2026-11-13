# Retraining policy

Owner: Person B. Thresholds live in [`configs/monitoring.yaml`](../configs/monitoring.yaml), not in code.
This page says what the numbers *mean* and what to do; change the YAML to change the numbers.

## Triggers (any one starts a retrain decision)

| # | Trigger | Source | Default | Why |
|---|---|---|---|---|
| T1 | **Drift detected** (offline) | `reports/drift.json` from the `drift` DVC stage | `drift_detected == true` | Training and test image statistics differ (the committed report already flags brightness / RGB shifts for PlantDoc). |
| T2 | **Accuracy below the minimum** | `reports/metrics.json` vs `evaluate.min_test_accuracy` in `params.yaml` | currently 0.50 | The same gate the export stage uses, so monitoring and delivery never disagree. |
| T3 | **Low-confidence rate too high** | `logs/predictions.jsonl` | > 0.45 of recent predictions | Users are uploading images the model is unsure about, a proxy for real-world shift when there are no labels. |
| T4 | Live **brightness drift** or **class-mix shift** | `logs/predictions.jsonl` vs `reports/monitoring/reference.json` | z > 3.0, TV distance > 0.30 | Cheap input-drift signals that need no labels. |

Warnings (shown in `summary.json`, **no** retrain): low-confidence rate > 0.30, p95 latency > 1500 ms, brightness z > 2.0.

Live-log triggers (T3, T4) are ignored until at least `window.min_predictions` (50) predictions exist, so a handful of odd uploads cannot trigger anything.

## What happens

1. The weekly Airflow DAG runs `src.monitor_logs`, which writes `reports/monitoring/summary.json` with a `status`: `healthy`, `warning`, `insufficient_data` or `retrain`.
2. `retrain` -> `reports/monitoring/retrain_decision.json` lists the triggers, then the DAG reruns `train -> evaluate -> export`.
3. Anything else -> `reports/monitoring/healthy_report.json`.

## Rules that keep retraining safe

- **A retrain never auto-deploys.** `export` only publishes a candidate that passes the accuracy gate; otherwise the previous model stays.
- **Never train on or edit the PlantDoc test set**, even when it triggered the retrain.
- Retraining uses the data already versioned by DVC; adding new labelled data is a human decision (Person A).
- After any retrain, re-measure and update `reports/`, then regenerate the README table (`python scripts/render_results_table.py`).
- Build a fresh reference baseline after a deliberate model/data change: `python -m src.monitor_logs --build-reference`.

## Limits

Small test sets (72 / ~300 images) mean accuracy differences of a few points are noise. Live triggers have no ground truth, so they signal *possible* degradation, not confirmed errors.
