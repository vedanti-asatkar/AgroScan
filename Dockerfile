FROM node:22-alpine AS frontend

WORKDIR /frontend
ARG VITE_MLFLOW_URL=http://localhost:5000
ARG VITE_AIRFLOW_URL=http://localhost:8080
ENV VITE_MLFLOW_URL=$VITE_MLFLOW_URL VITE_AIRFLOW_URL=$VITE_AIRFLOW_URL
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt requirements-xai.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-xai.txt

COPY src/__init__.py src/preprocess.py ./src/
COPY configs/monitoring.yaml ./configs/monitoring.yaml
COPY api/ ./api/
COPY reports/metrics.json reports/classification_report.json reports/v5_plantdoc_metrics.json ./reports/
COPY --from=frontend /frontend/dist ./ui/dist
# Model artifacts come from `dvc pull` / `dvc repro` (model.keras + classes.json)
COPY models/ ./models/

RUN useradd -m app && mkdir -p logs && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
