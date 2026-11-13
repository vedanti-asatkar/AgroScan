# Deployment and operations runbook

AgroScan packages an API container and publishes that container to GitHub Container Registry (GHCR). The workflow does not provision a cloud host. Run the container on a Docker host you control.

## 1. Build prerequisites

The Dockerfile copies the whole `models/` directory into the image. The model binaries are DVC-managed, so a local build intended to serve predictions needs the model artifacts present first:

```bash
dvc pull models/mobilenetv2.keras
docker build -t agroscan-api .
```

The v5 metadata is `models/model_meta.v5.json`; the DVC `export` stage can generate `models/model_meta.json` after a successful pipeline run. In the checked-in workflow outputs, `models/model_meta.json` and a local MLflow database are absent. Confirm the model artifact and metadata are present before starting the service.

## 2. Run locally

```bash
docker compose up --build
```

Compose starts the API on port 8000 and MLflow on port 5000. The API container mounts `./logs` at `/app/logs`. Open the UI at <http://localhost:8000/ui/>. For the API alone, use `make serve` (`uvicorn api.main:app --port 8000`) or:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Direct `uvicorn` serving requires the Python dependencies and model artifacts to be available in the current environment.

## 3. Release workflow

`.github/workflows/release.yml` runs for pushes to `main` and manual `workflow_dispatch`. It builds and publishes `ghcr.io/<owner>/<repo>:latest` and a short commit-SHA tag, then checks `/health` on the published image. The workflow does not pull DVC model artifacts. A green release smoke check therefore verifies API liveness, not model readiness or prediction capability. Ensure a published image has the intended model artifacts before relying on it for inference.

If the package is private, authenticate to GHCR with a token that has `read:packages` before pulling it.

## 4. Health and prediction checks

| Check | Command | Expected result |
|---|---|---|
| Root status | `curl -fsS http://localhost:8000/` | JSON service status |
| Liveness | `curl -fsS http://localhost:8000/health` | HTTP 200, `{"status":"ok"}` |
| Readiness | `curl -i http://localhost:8000/ready` | HTTP 200 when a compatible model and class list loaded; otherwise HTTP 503 |
| Prediction | `curl -F "file=@leaf.jpg" http://localhost:8000/predict` | Top class, confidence, low-confidence flag, and top-k list when ready |
| Browser UI | Open <http://localhost:8000/ui/> | Upload form and prediction result |
| Prediction log | `Get-Content logs/predictions.jsonl -Tail 1` (PowerShell) | One JSON summary after a successful prediction |
| Log monitor | `python -m src.monitor_logs` | Prints status; writes `reports/monitoring/summary.json` |

The API accepts uploaded images up to `MAX_UPLOAD_BYTES` (default 10 MiB). It does not persist uploaded image bytes; see [MODEL_CARD.md](../MODEL_CARD.md) for the exact logged fields.

## 5. GHCR run and rollback

After confirming the image has model files, pull and run it. Replace `<short-sha>` with the short SHA tag shown in the release workflow (the tag does not include a `sha-` prefix):

```bash
docker pull ghcr.io/<owner>/<repo>:latest
docker run -d --name agroscan -p 8000:8000 -v "${PWD}/logs:/app/logs" ghcr.io/<owner>/<repo>:latest
```

PowerShell volume example:

```powershell
docker run -d --name agroscan -p 8000:8000 -v "${PWD}/logs:/app/logs" ghcr.io/<owner>/<repo>:latest
```

Rollback by stopping the current container and running the previous short-SHA image tag:

```bash
docker stop agroscan
docker rm agroscan
docker run -d --name agroscan -p 8000:8000 -v "${PWD}/logs:/app/logs" ghcr.io/<owner>/<repo>:<short-sha>
```

## 6. Troubleshooting

| Symptom | Check |
|---|---|
| `/ready` returns 503 | Confirm the metadata-selected model file and class-name file exist in `models/`; pull the DVC model before a local build. |
| `/health` succeeds but prediction is unavailable | `/health` is liveness only. Use `/ready` to confirm a model loaded. |
| Prediction returns 400 | Upload a non-empty decodable JPG or PNG image. |
| Prediction returns 413 | The upload exceeds `MAX_UPLOAD_BYTES`; the default is 10 MiB. |
| No prediction log line | Check that `/predict` completed and that the configured `PREDICTION_LOG` parent directory is writable. |
| Release workflow is green but container is not ready | The release workflow checks `/health`, not `/ready`, and does not pull DVC model artifacts. |
