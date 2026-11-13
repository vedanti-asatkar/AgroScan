# Presentation outline (about 12 slides)

Build the deck from this after the real numbers land in `reports/` (merge order, step 4). Lead with measured results, and call v5 numbers "reported by the training notebook" until the pipeline reproduces them.

1. **Title and problem:** why leaf-disease triage matters; who it is for.
2. **Scope:** 8 classes (pepper, potato, tomato), what it is not (not a replacement for an expert).
3. **Lifecycle diagram:** the architecture diagram from `docs/architecture.md`.
4. **Data and experiments:** PlantVillage vs PlantDoc, DVC, MLflow, validation stage, leakage guard.
5. **Pipeline and CI:** DVC stages, quality gate, CI (lint, tests, smoke, Docker).
6. **Results (measured):** table from `scripts/render_results_table.py`; top-1 and top-3; small-sample caveat.
7. **The honest finding:** domain shift (validation vs field accuracy); confusion matrix highlights.
8. **Serving:** FastAPI, UI, model_meta contract, low-confidence message; live demo.
9. **Monitoring and retraining policy:** `summary.json`, thresholds, the three triggers.
10. **Automation:** Airflow weekly DAG with both branches (screenshot of the graph view for each run) and `release.yml` to GHCR.
11. **Responsible AI:** LIME and SHAP examples, model card highlights, privacy of uploaded and logged images.
12. **Cloud and industry:** platform comparison, agriculture case study, what we would do next and what we left out on purpose (cloud host, dashboards, fairness, auth).

Demo checklist: API running, one good leaf, one hard/field image, one non-leaf image (shows low confidence), `monitor_logs` output, Airflow graph view.
