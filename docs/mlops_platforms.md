# MLOps platforms and cloud services: a short overview

AgroScan uses open-source tools you can run for free. This page shows how they compare with what the big clouds offer.

| Lifecycle step | AgroScan (used) | Managed cloud equivalents | Other open source |
|---|---|---|---|
| Data / model versioning | DVC | S3 / GCS / Azure Blob versioning, SageMaker Feature Store, Vertex AI Feature Store | lakeFS, Delta Lake |
| Experiment tracking, model registry | MLflow | SageMaker Experiments + Model Registry, Vertex AI Experiments + Model Registry, Azure ML | Weights & Biases (SaaS), Neptune, ClearML |
| Pipeline orchestration | DVC stages + Airflow | SageMaker Pipelines, Vertex AI Pipelines (Kubeflow), Azure ML Pipelines, AWS Step Functions | Kubeflow Pipelines, Prefect, Dagster |
| CI / CD | GitHub Actions + GHCR | CodePipeline / CodeBuild, Cloud Build, Azure DevOps | GitLab CI, Jenkins |
| Serving | FastAPI in Docker | SageMaker Endpoints, Vertex AI Endpoints, Azure ML Online Endpoints, Cloud Run, Lambda | KServe, BentoML, Seldon, TorchServe |
| Monitoring | `monitor_logs.py` + thresholds | SageMaker Model Monitor, Vertex AI Model Monitoring, Azure ML data drift | Evidently, WhyLabs, Prometheus + Grafana |
| Explainability | LIME, SHAP | SageMaker Clarify, Vertex Explainable AI | Captum, Alibi |

## Why we chose this stack

- **Free and local**: no cloud account or card needed; the whole project runs on a laptop plus GitHub.
- **Transparent**: every step is a file in the repo, which is better for learning than a managed console.
- **Portable**: the same Docker image and Airflow DAG would run on a managed service with little change.

## What the cloud would add

Autoscaling endpoints, managed GPUs for training, built-in monitoring dashboards, IAM and audit logs, and one-click retraining pipelines. The trade-offs are cost, lock-in and less visibility into internals.

## Where AgroScan would move first

Host the image on a serverless container service (for example Cloud Run or Azure Container Apps), store DVC data in object storage, and replace the weekly Airflow check with the provider's model-monitoring service. Check current pricing and free tiers before choosing, since they change often.
