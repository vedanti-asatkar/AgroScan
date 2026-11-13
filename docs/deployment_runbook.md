# Deployment runbook

There is no free cloud host, so "deployment" = a published container image you can run anywhere with Docker. `release.yml` publishes it on every merge to `main`.

## 1. Release (automatic)

Merge a PR into `main` -> GitHub Actions `release` builds the image, pushes `ghcr.io/<owner>/<repo>:latest` and `:sha-<short>`, and smoke-tests `/health`.
Check: GitHub > Actions > *release* is green; GitHub > Packages shows the image.
If the package is private, make it public in Package settings or `docker login ghcr.io` with a token that has `read:packages`.

## 2. Run

```bash
docker pull ghcr.io/<owner>/<repo>:latest
docker run -d --name agroscan -p 8000:8000 -v "$PWD/logs:/app/logs" ghcr.io/<owner>/<repo>:latest
```
(Adjust the port and `/app/logs` path to match Person A's Dockerfile. Mounting `logs/` keeps predictions for monitoring.)
Local alternative: `docker compose up --build`.

## 3. Health checks

| Check | Command | Healthy |
|---|---|---|
| Liveness | `curl -fsS localhost:8000/health` | HTTP 200 |
| Readiness / model loaded | `curl -fsS localhost:8000/ready` | HTTP 200 and the expected model version (once A2 merges) |
| Prediction works | upload a leaf image in the UI, or `curl -F "file=@leaf.jpg" localhost:8000/predict` | top-3 classes returned |
| Logging works | `tail -n 1 logs/predictions.jsonl` | a new JSON line with `ts`, `top1_class`, `confidence` ... |
| Monitoring | `python -m src.monitor_logs` | `status=healthy` |

(Confirm endpoint names against `api/main.py`.)

## 4. Roll back

```bash
docker stop agroscan && docker rm agroscan
docker run -d --name agroscan -p 8000:8000 ghcr.io/<owner>/<repo>:sha-<previous>
```
Find the previous `sha-` tag under Packages.

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/ready` fails or model not found | model files are DVC-managed: ensure the image was built after `dvc pull`, or mount `models/` |
| Many low-confidence results | domain shift; see `docs/retraining_policy.md` |
| Release job fails at login | repo Settings > Actions > Workflow permissions must allow read/write for packages |
| Smoke test times out | model loading is slow on cold start; raise the retry loop in `release.yml` |
