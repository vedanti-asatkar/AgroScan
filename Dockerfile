FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/__init__.py src/preprocess.py ./src/
COPY api/ ./api/
COPY ui/ ./ui/
# Model artifacts come from `dvc pull` / `dvc repro` (model.keras + classes.json)
COPY models/ ./models/

RUN useradd -m app && mkdir -p logs && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
