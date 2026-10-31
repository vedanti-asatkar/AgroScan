# AgroScan
Lightweight plant disease classifier (MobileNetV2) trained on PlantVillage + PlantDoc, evaluated for real-world
generalisation on PlantDoc-test, with drift monitoring — MLOps course project.

## MLOps stack
| Concern | Tool |
|---|---|
| Data / model versioning | DVC (Google Drive remote) |
| Pipeline | `dvc.yaml`: prepare → train → evaluate → drift, configured by `params.yaml` |
| Experiment tracking + model registry | MLflow (`sqlite:///mlflow.db`; override with `MLFLOW_TRACKING_URI` in `params.yaml`) |
| Serving | FastAPI + Docker (`/health`, `/ready`, `/predict`, prediction log in `logs/`) |
| CI | GitHub Actions: ruff, pytest (incl. end-to-end smoke on synthetic data), docker build |
| Monitoring | `src/drift_check.py` (KS tests) + `logs/predictions.jsonl` |

## Quickstart
```bash
pip install -r requirements-dev.txt
make smoke        # whole pipeline on synthetic data (CPU, ~1 min), no dataset needed
make test         # pytest + ruff via `make lint`

dvc pull          # needs Google Drive credentials in .dvc/config.local
dvc repro         # full pipeline; metrics land in reports/, runs in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
dvc metrics show && dvc params diff     # compare experiments
make serve        # http://localhost:8000/ui/
docker compose up --build   # MLflow on :5000, API on :8000 (after dvc repro/pull)
```
Full training needs a GPU or lots of patience; run `dvc repro` on Colab/GPU box, then `dvc push`.

## Design notes
- **Preprocessing contract** (`src/preprocess.py`): used by training *and* the API → RGB, resize, [0,1]. The model
  rescales to [-1,1] internally, removing train/serve skew.
- **Class names** are saved with the model (`models/classes.json`), not hardcoded in the API.
- **Leakage guard:** `prepare_data` drops train/val files byte-identical to PlantDoc-test files.
- **Quality gate:** `evaluate.min_test_accuracy` in `params.yaml`; `src.evaluate` exits 1 if below it.
- The legacy `models/mobilenetv2_combined_v2.h5` is still served as a fallback if no new model exists.

## CI secrets
Add `GDRIVE_CREDENTIALS_DATA` (service-account JSON) to let CI `dvc pull` the model for a container smoke test.
