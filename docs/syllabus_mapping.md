# Syllabus to project

Run each command from the repo root (`dvc pull` first if data / models are missing). Owners: A = pipeline/serving, B = automation/docs.

| Unit | Topic | Where it lives | Reproduce it | Owner |
|---|---|---|---|---|
| 1 | MLOps and ML lifecycle | `README.md`, [architecture](architecture.md), [platforms overview](mlops_platforms.md) | read the docs | B |
| 2 | Data, features, experiments | DVC data + `prepare` / `validate` stages, MLflow tracking and registry | `dvc repro prepare validate` ; `mlflow ui` (view runs) | A |
| 3 | Pipeline automation | `dvc.yaml`, CI, quality gate (`export`), Airflow DAG | `dvc repro` ; `airflow dags test agroscan_pipeline 2025-01-05` ([setup](airflow_setup.md)) | A (DVC, CI), B (Airflow) |
| 4 | Deployment and serving | FastAPI + Docker, batch inference, live-log monitoring, alert thresholds | `docker compose up --build` ; `python scripts/batch_predict.py --images <dir>` ; `python -m src.monitor_logs` | A (API, Docker), B (batch, monitoring) |
| 5 | Responsible and explainable AI | LIME, SHAP, model card | `jupyter nbconvert --to notebook --execute notebooks/lime_example.ipynb notebooks/shap_example.ipynb` ; `MODEL_CARD.md` | B |
| 6 | Cloud MLOps and industry | agriculture case study (below), delivery workflow, deployment runbook | `.github/workflows/release.yml` ; [runbook](deployment_runbook.md) | B |

## Reproduce each pipeline stage

| Stage | Command | Output |
|---|---|---|
| prepare | `dvc repro prepare` | train / val / test manifests |
| validate | `dvc repro validate` | `reports/data_validation.json` |
| train | `dvc repro train` | model + MLflow run |
| evaluate | `dvc repro evaluate` | `reports/metrics.json`, `classification_report.json`, `confusion_matrix.png` |
| export | `dvc repro export` | model + `models/model_meta.json` |
| drift | `dvc repro drift` | `reports/drift.json` |
| everything | `dvc repro` | all of the above |
| monitoring | `python -m src.monitor_logs` | `reports/monitoring/summary.json` |
| tests | `make test` / `pytest` | |

## Unit 6 case study: plant-disease detection in agriculture

Smallholder farmers often lack quick access to a plant pathologist, so phone-based diagnosis is attractive. The MLOps lessons mirror this project:

- **Domain shift is the central risk.** Models trained on clean lab images degrade on field photos (AgroScan: ~88% validation vs ~44-60% on PlantDoc). Continuous monitoring and periodic retraining on field data are the fix, not one-off training.
- **Seasonality and geography:** crops, diseases and lighting change through the year and across regions, so drift monitoring and a retraining schedule matter.
- **Decision support, not decision making:** confidence thresholds, top-3 answers and an "ask an expert" message reduce the cost of wrong predictions.
- **Connectivity and cost:** small models (MobileNetV2) and containers suit low-resource deployment, including on-device inference.
- **Data governance:** photos may reveal farm locations; keep logs free of images and identifiers.
