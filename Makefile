.PHONY: install pull repro test lint smoke mlflow-ui serve docker
install:   ; pip install -r requirements-dev.txt
pull:      ; dvc pull
repro:     ; dvc repro            # prepare -> train -> evaluate -> drift
test:      ; pytest -q
lint:      ; ruff check .
smoke:     ; python scripts/make_synthetic_data.py .smoke && for m in prepare_data train evaluate drift_check; do python -m src.$$m --params params.ci.yaml || exit 1; done
mlflow-ui: ; mlflow ui --backend-store-uri sqlite:///mlflow.db
serve:     ; uvicorn api.main:app --port 8000
docker:    ; docker build -t agroscan-api .
