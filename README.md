# AgroScan

AgroScan is an MLOps course project for classifying pepper, potato, and tomato leaf photos into eight disease/healthy classes. It uses a MobileNetV2 model, DVC pipeline and data versioning, MLflow tracking, a FastAPI inference service, a browser UI, offline and prediction-log monitoring, Airflow scheduling, and GitHub Actions CI/container delivery.

## Quickstart

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
make test                                  # pytest -q
make smoke                                 # synthetic-data pipeline; writes .smoke/, reports/, and models/
```

The `make` targets require GNU Make. If Make is unavailable, `make test` runs `pytest -q`; `make lint` runs `ruff check .`. The `smoke` target runs `scripts/make_synthetic_data.py .smoke` followed by `src.prepare_data`, `src.validate_data`, `src.train`, `src.evaluate`, `src.export_model`, and `src.drift_check` with `params.ci.yaml`.

Real datasets and the v5 model are stored through DVC. Pull them only when you have access to the configured remote:

```bash
dvc pull
dvc repro
```

The DVC stages are `clean_prepared_archive`, `prepare`, `validate`, `train`, `evaluate`, `evaluate_v5_baseline`, `export`, and `drift`. Individual stages can be reproduced with `dvc repro <stage>`; `dvc repro` runs the dependency graph. Training uses the PlantDoc train split alongside PlantVillage. The PlantDoc test split remains held out.

## Run the service

The API listens on port 8000. Docker Compose also starts MLflow on port 5000:

```bash
docker compose up --build
# or, with the API dependencies and model files available:
make serve                                  # uvicorn api.main:app --port 8000
```

Open the UI at <http://localhost:8000/ui/>. Available API routes are `/` (status), `/health` (liveness), `/ready` (model loaded and version), and `POST /predict` (multipart field `file`). Example:

For predictions, pull the DVC-managed model before building/running the API. Without a model artifact, the service can answer `/health`, but `/ready` returns 503 and `/predict` is unavailable.

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
curl -F "file=@leaf.jpg" http://localhost:8000/predict
```

The API logs prediction summaries to `logs/predictions.jsonl`. To summarize those logs, use `python -m src.monitor_logs`; to generate example logs, use `python scripts/make_sample_logs.py --out logs/sample.jsonl --drift`.

## Repository map

| Path | Contents |
|---|---|
| `dvc.yaml`, `params.yaml`, `params.ci.yaml` | Pipeline stages and production/smoke parameters |
| `src/` | Data preparation and validation, training, evaluation, export, offline drift and prediction-log monitoring |
| `api/main.py`, `ui/index.html` | FastAPI routes and browser interface |
| `models/model_meta.v5.json`, `models/model_meta.json` | v5 metadata and generated serving metadata; `model_meta.json` is produced by `export` |
| `scripts/batch_predict.py` | Folder inference: `python scripts/batch_predict.py --images <dir> --out <csv>` |
| `airflow/dags/agroscan_pipeline.py` | Weekly `agroscan_pipeline` DAG |
| `configs/monitoring.yaml` | Prediction-log monitoring thresholds |
| `reports/` | Committed metrics, classification reports, confusion matrix, validation and audit reports |
| `notebooks/lime_example.ipynb`, `notebooks/shap_example.ipynb` | Explainability examples |
| `MODEL_CARD.md`, `docs/` | Model card, policies, architecture, deployment and course documentation |
| `.github/workflows/ci.yml`, `.github/workflows/release.yml` | CI and GHCR container publishing workflows |

## Results

`python scripts/render_results_table.py` is intended to format `reports/metrics.json` and `reports/classification_report.json`. In the current checkout it errors because the classification report is nested under `val` and `test`, while the renderer expects one flat report. The tables below are transcribed from the named report files.

<!-- RESULTS:START -->
### Pipeline candidate: renderer metrics (`reports/metrics.json`)

| Metric | Value |
|---|---:|
| val_accuracy | 0.8838 |
| val_mean_confidence | 0.9175 |
| val_confident_wrong_rate | 0.0227 |
| val_macro_f1 | 0.8707 |
| test_accuracy | 0.4444 |
| test_mean_confidence | 0.7123 |
| test_confident_wrong_rate | 0.0972 |
| test_macro_f1 | 0.4286 |

### Model and dataset results

| Model / evaluation | Test set | Top-1 | Top-3 | n | Evidence |
|---|---|---:|---:|---:|---|
| Pipeline candidate, current DVC test | PlantDoc | 44.4% | not reported | 72 | measured: `reports/metrics.json` (`test_accuracy`) |
| Existing v5, current DVC test | PlantDoc | 58.3% | 94.4% | 72 | measured: `reports/v5_plantdoc_metrics.json` |
| Existing v5, cleaned prepared archive | PlantDoc | 61.4% | 94.3% | 70 | measured: `reports/v5_plantdoc_prepared_metrics.json` |
| Existing v5, cleaned prepared archive | PlantWild | 60.5% | 89.0% | 299 | measured: `reports/v5_plantwild_metrics.json` |
| Existing v5, all archive test images | PlantWild | 60.9% | 88.8% | 304 | measured: `reports/v5_plantwild_metrics_including_duplicates.json` |
| Pipeline candidate, cleaned prepared archive | PlantDoc | 42.9% | 78.6% | 70 | measured: `reports/pipeline_plantdoc_prepared_metrics.json` |
| Pipeline candidate, cleaned prepared archive | PlantWild | 37.5% | 68.2% | 299 | measured: `reports/pipeline_plantwild_prepared_metrics.json` |

The previously cited PlantDoc notebook top-1 figure has no matching value in the committed report files. **TODO:** provide the source notebook output if that result must be retained. The 60.9% PlantWild result is measured in the all-304 report above; it differs from the cleaned 299-image score because the evaluation sets differ.

### Per-class results (pipeline candidate, PlantDoc test; `reports/classification_report.json`)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Pepper__bell___Bacterial_spot | 0.533 | 0.889 | 0.667 | 9 |
| Potato___Early_blight | 0.556 | 0.625 | 0.588 | 8 |
| Potato___Late_blight | 0.333 | 0.125 | 0.182 | 8 |
| Tomato_Bacterial_spot | 0.125 | 0.111 | 0.118 | 9 |
| Tomato_Early_blight | 0.667 | 0.444 | 0.533 | 9 |
| Tomato_Late_blight | 0.294 | 0.500 | 0.370 | 10 |
| Tomato_Septoria_leaf_spot | 0.800 | 0.364 | 0.500 | 11 |
| Tomato_healthy | 0.444 | 0.500 | 0.471 | 8 |
| **macro avg** | 0.469 | 0.445 | 0.429 | 72 |
| **weighted avg** | 0.477 | 0.444 | 0.430 | 72 |

<!-- RESULTS:END -->

The test samples are small: 72 PlantDoc images in the current pipeline split and 299–304 PlantWild images in the archive reports. The separate prepared-archive audit found exact duplicates across splits and excludes some images, so those scores are not directly interchangeable. Validation performance is higher than field-test performance; this is evidence of domain shift, not a guarantee about all field conditions. See [the audit](reports/prepared_dataset_audit.json), [data validation](reports/data_validation.json), and [the model card](MODEL_CARD.md).

## Monitoring, Airflow, and delivery

```bash
python -m src.monitor_logs
python -m src.monitor_logs --build-reference
airflow dags test agroscan_pipeline 2025-01-05
```

The first monitor command reads `logs/predictions.jsonl` and writes `reports/monitoring/summary.json` (including status and retrain triggers). `--build-reference` writes `reports/monitoring/reference.json` from the current log. With no prediction log, it reports zero live predictions, but offline drift or accuracy reports can still trigger a `retrain` status. Airflow setup and placeholder-mode commands are in [docs/airflow_setup.md](docs/airflow_setup.md). Airflow is run on Linux, WSL2, or Docker as described there.

The CI workflow runs on pushes and pull requests targeting `main`; it installs `requirements-dev.txt`, runs Ruff and pytest, then builds the Docker image. It pulls the model and checks `/ready` only when `GDRIVE_CREDENTIALS_DATA` is configured. The release workflow publishes `latest` and a short-SHA tag to GHCR on pushes to `main` or manual dispatch, then smoke-checks `/health`. See [the deployment runbook](docs/deployment_runbook.md).

## Explainability and responsible use

Run the explainability notebooks in an environment with `requirements-extras.txt` installed. Their existence does not mean this checkout contains saved LIME/SHAP output figures. AgroScan is a learning and decision-support demo, not a substitute for an agronomist or a sole basis for pesticide or treatment decisions. Read [MODEL_CARD.md](MODEL_CARD.md) for scope, limitations and the API's data handling.

## Project rules

- Work on a feature branch and use a reviewed pull request; do not commit or push credentials, tokens, or prediction logs.
- Never train on or edit the PlantDoc test set.
- DVC data/model pointer changes are owned by the pipeline owner. This documentation does not require DVC pointer edits.

## Course mapping

See [docs/syllabus_mapping.md](docs/syllabus_mapping.md) for the six-unit mapping and exact reproduction commands. See [docs/architecture.md](docs/architecture.md) for the DVC graph.
