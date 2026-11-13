# Syllabus-to-project mapping

Commands below run from the repository root. DVC commands require the referenced datasets/model artifacts to be present; use `dvc pull` when you have access to the configured remote. `dvc repro <stage>` uses the exact stage names in `dvc.yaml`.

| Unit | Topic and AgroScan example | Exact project files | Reproduction command |
|---|---|---|---|
| 1 | MLOps lifecycle and system design | `README.md`; `docs/architecture.md`; `docs/mlops_platforms.md`; `dvc.yaml`; `params.yaml` | `dvc dag` displays the configured DVC graph; `dvc repro` executes its available stages. |
| 2 | Data management, validation, experiments and model registry | `data/plantvillage.dvc`; `data/plantdoc.dvc`; `src/prepare_data.py`; `src/validate_data.py`; `src/train.py`; `src/evaluate.py`; `src/evaluate_v5.py`; `reports/data_validation.json`; `reports/metrics.json`; `mlflow.db` (created locally) | `dvc repro prepare`; `dvc repro validate`; `dvc repro train`; `dvc repro evaluate`; `dvc repro evaluate_v5_baseline`; `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| 3 | Automated pipelines, CI and scheduled orchestration | `dvc.yaml`; `params.yaml`; `params.ci.yaml`; `.github/workflows/ci.yml`; `.github/workflows/release.yml`; `airflow/dags/agroscan_pipeline.py`; `docs/airflow_setup.md` | `dvc repro`; `make test`; `make smoke`; on Linux/WSL2 with Airflow configured: `airflow dags test agroscan_pipeline 2025-01-05` |
| 4 | Deployment, API serving and operational monitoring | `api/main.py`; `src/preprocess.py`; `ui/index.html`; `Dockerfile`; `docker-compose.yml`; `scripts/batch_predict.py`; `src/monitor_logs.py`; `configs/monitoring.yaml`; `docs/deployment_runbook.md` | `docker compose up --build`; `python scripts/batch_predict.py --images <dir> --out <csv>`; `python -m src.monitor_logs`; API checks: `curl -fsS http://localhost:8000/health`, `curl -i http://localhost:8000/ready`, `curl -F "file=@leaf.jpg" http://localhost:8000/predict` |
| 5 | Responsible AI and explainability | `MODEL_CARD.md`; `notebooks/lime_example.ipynb`; `notebooks/shap_example.ipynb`; `notebooks/explain_utils.py`; `reports/confusion_matrix.png`; `reports/classification_report.json` | `pip install -r requirements-extras.txt`; `jupyter nbconvert --to notebook --execute notebooks/lime_example.ipynb`; `jupyter nbconvert --to notebook --execute notebooks/shap_example.ipynb` |
| 6 | Cloud delivery and agriculture case study | `docs/mlops_platforms.md`; `docs/deployment_runbook.md`; `.github/workflows/release.yml`; `airflow/dags/agroscan_pipeline.py`; `Dockerfile`; `docker-compose.yml`; `README.md` | `docker build -t agroscan-api .`; `docker compose up --build`; `airflow dags test agroscan_pipeline 2025-01-05` (Airflow setup is in `docs/airflow_setup.md`). GitHub release publication runs from pushes to `main` or manual `workflow_dispatch`; it is not a local shell command. |

## DVC stages and exact outputs

| Stage in `dvc.yaml` | Exact command | Main inputs and outputs |
|---|---|---|
| `clean_prepared_archive` | `python -m src.clean_prepared_archive --input data/agroscan_prepared_v1.zip --output data/agroscan_prepared_v1_clean.zip --audit reports/prepared_dataset_cleaning.json --classes params.yaml` (or `dvc repro clean_prepared_archive`) | Input archive; cleaned archive and `reports/prepared_dataset_cleaning.json` |
| `prepare` | `python -m src.prepare_data` (or `dvc repro prepare`) | `data/plantvillage`, `data/plantdoc`; produces `data/prepared/` manifests and stats |
| `validate` | `python -m src.validate_data` (or `dvc repro validate`) | `data/prepared/`; produces `reports/data_validation.json` |
| `train` | `python -m src.train` (or `dvc repro train`) | Prepared data and validation report; produces `models/model.keras`, `models/classes.json`, `reports/train_metrics.json`, MLflow run/registry record |
| `evaluate` | `python -m src.evaluate` (or `dvc repro evaluate`) | Candidate model and prepared data; produces `reports/metrics.json`, `reports/classification_report.json`, `reports/confusion_matrix.png` |
| `evaluate_v5_baseline` | `python -m src.evaluate_v5 --meta models/model_meta.v5.json --manifest data/prepared/test.csv --output reports/v5_plantdoc_metrics.json` (or `dvc repro evaluate_v5_baseline`) | v5 metadata/model/class names and PlantDoc test manifest; produces `reports/v5_plantdoc_metrics.json` |
| `export` | `python -m src.export_model` (or `dvc repro export`) | Candidate and v5 evaluation results; produces `models/model_meta.json` for the selected model |
| `drift` | `python -m src.drift_check` (or `dvc repro drift`) | Prepared train/test manifests and export metadata; produces `reports/drift.json` |
| All dependencies | `dvc repro` | Runs the available dependency graph. Export selects the passing candidate or the measured v5 baseline. |

`dvc repro train` and later stages require their predecessor outputs. Stage `clean_prepared_archive` is a separate archive-cleaning path; it is not a prerequisite of the `prepare` stage in the current `dvc.yaml`.

## Agriculture case study: leaf-disease triage

Smallholder farmers may not have quick access to a plant pathologist, so mobile-photo triage is useful as decision support. AgroScan makes the central MLOps issues concrete:

- **Domain shift:** the committed pipeline candidate has 88.4% validation accuracy but 44.4% on the 72-image PlantDoc test set (`reports/metrics.json`). The v5 report files record other results on 70- and 72-image PlantDoc samples and 299- and 304-image PlantWild samples; see [README results](../README.md#results) for the exact sources.
- **Data governance:** the dataset audit reports exact duplicate groups across splits, including conflicting labels. The evaluation archive has a documented cleaning policy; the separately prepared DVC validation report checks exact duplicate hashes across its three splits.
- **Safety:** the API returns a top-three list and flags low-confidence predictions. It cannot identify classes outside the eight supported labels and should not make treatment or pesticide decisions.
- **Operations:** the DVC graph records offline training/evaluation; `src.monitor_logs` summarizes API prediction records; the weekly Airflow DAG can create a retrain decision, but a new model still has to pass export's accuracy gate.
- **Deployment constraints:** Docker and small MobileNetV2 models suit modest servers, while intermittent connectivity and the need for agronomist review remain deployment considerations rather than solved features.
