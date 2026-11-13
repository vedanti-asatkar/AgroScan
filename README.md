# AgroScan

End-to-end MLOps project: classify **pepper, potato and tomato leaf diseases (8 classes)** from a photo with MobileNetV2, tracked with DVC and MLflow, served with FastAPI + a browser UI, monitored from live logs, and automated with CI, Airflow and a container release workflow.

> **Status:** TODO(B): update the results table from `reports/` before submitting (see "Results").

## Quickstart

```bash
git clone <repo-url> && cd AgroScan
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

make test        # unit tests (no data needed)
make smoke       # synthetic-data pipeline smoke test (no data needed)
```

Real data and models are versioned with DVC and are **not** in Git:

```bash
dvc pull         # needs DVC remote access; ask the repo owner
dvc repro        # prepare -> validate -> train -> evaluate -> export -> drift
```

Run the app:

```bash
docker compose up --build        # then open http://localhost:8000
# or: uvicorn api.main:app --reload
```

Try it without the UI:

```bash
curl -fsS localhost:8000/health
curl -F "file=@leaf.jpg" localhost:8000/predict      # top-3 classes + low-confidence flag
```

(Check exact endpoint paths and `make` target names against `api/main.py` and the `Makefile`.)

## What is where

| Path | What |
|---|---|
| `dvc.yaml`, `params.yaml` | pipeline stages and parameters |
| `src/` | prepare, validate, train, evaluate, export, drift, **monitor_logs** |
| `api/`, `ui/` | FastAPI service and upload page |
| `models/model_meta.json` | the serving contract (image size, value range, crop, threshold, top-k) |
| `reports/` | measured metrics, per-class report, confusion matrix, drift, data validation |
| `reports/monitoring/` | live-log summary and retrain / healthy decisions |
| `configs/monitoring.yaml` | alert and retrain thresholds |
| `airflow/dags/agroscan_pipeline.py` | weekly automation DAG |
| `scripts/batch_predict.py` | folder of images -> CSV |
| `notebooks/` | LIME and SHAP examples |
| `MODEL_CARD.md`, `docs/` | model card, policy, architecture, runbook, syllabus mapping |
| `.github/workflows/` | `ci.yml` (lint, tests, smoke, Docker build), `release.yml` (publish image) |

Architecture diagram: [docs/architecture.md](docs/architecture.md).

## Results

Numbers must come from `reports/`, not from memory. Generate the tables with:

```bash
python scripts/render_results_table.py
```

and paste them here.

<!-- RESULTS:START -->
TODO(B): paste measured tables (PlantDoc and PlantWild top-1 / top-3, per class).
<!-- RESULTS:END -->

Caveats that belong next to any table:

- Test sets are small (about 72 PlantDoc and 300 PlantWild images), so differences of a few points are noise.
- The largest finding is **domain shift**: roughly 88% validation accuracy for the pipeline candidate versus 44-60% on field images.
- Earlier reports disagree on v5 (62.5% vs 58.3% on PlantDoc, 60.9% vs 60.5% on PlantWild); the re-measured values in `reports/` are the authority.

## Batch predictions

```bash
python scripts/batch_predict.py --images path/to/leaves --out reports/batch_predictions.csv
# images in sub-folders named after classes also get true_label / correct columns and an accuracy printout
```

## Monitoring and automation

```bash
python -m src.monitor_logs                 # summarise logs/predictions.jsonl -> reports/monitoring/summary.json
python -m src.monitor_logs --build-reference   # set the brightness / class-mix baseline from known-good logs
python scripts/make_sample_logs.py --out logs/sample.jsonl --drift   # demo data
```

- Thresholds: `configs/monitoring.yaml`. Policy: [docs/retraining_policy.md](docs/retraining_policy.md).
- Airflow DAG (weekly; Linux / WSL2 / Docker): [docs/airflow_setup.md](docs/airflow_setup.md).
- Deploy and health checks: [docs/deployment_runbook.md](docs/deployment_runbook.md).

## Explainability and responsible AI

```bash
pip install -r requirements-extras.txt
jupyter nbconvert --to notebook --execute notebooks/lime_example.ipynb
jupyter nbconvert --to notebook --execute notebooks/shap_example.ipynb
```

Read [MODEL_CARD.md](MODEL_CARD.md) for intended use, limits and privacy.

## Project rules

- Branch per person, PRs reviewed by the other person, no direct pushes to `main`.
- Only the pipeline owner touches DVC (`dvc add`, `dvc push`, `*.dvc`).
- **Never** train on or edit the PlantDoc test set. **Never** commit secrets (Kaggle tokens, OAuth secrets, DVC credentials).

## Course mapping

See [docs/syllabus_mapping.md](docs/syllabus_mapping.md) for unit-by-unit coverage and the command that reproduces each stage; platform overview in [docs/mlops_platforms.md](docs/mlops_platforms.md).
