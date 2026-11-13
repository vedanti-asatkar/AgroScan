# Running the Airflow DAG

Airflow runs on Linux. On Windows use **WSL2** (or Docker). The DAG is `airflow/dags/agroscan_pipeline.py`.
Airflow 2.10 needs Python 3.8-3.12 (not 3.13).

## WSL2 quickstart

```bash
cd /mnt/c/Users/<you>/Documents/GitHub/AgroScan      # or clone inside WSL for speed
python3.12 -m venv .venv-airflow && source .venv-airflow/bin/activate
pip install -r requirements.txt
pip install "apache-airflow==2.10.5" -r requirements-extras.txt \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt

export AIRFLOW_HOME=$PWD/.airflow AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/airflow/dags AGROSCAN_HOME=$PWD
airflow db migrate
```

## Demo without Person A's work (placeholder mode)

```bash
export AGROSCAN_DAG_MODE=placeholder
airflow dags test agroscan_pipeline 2025-01-05                 # healthy branch
AGROSCAN_SIMULATE_DRIFT=1 airflow dags test agroscan_pipeline 2025-01-12   # retrain branch
ls reports/monitoring     # healthy_report.json or retrain_decision.json, plus summary.json
```

## Real mode (after A's stages merge)

`unset AGROSCAN_DAG_MODE` (or `export AGROSCAN_DAG_MODE=real`), `dvc pull`, then `airflow dags test agroscan_pipeline 2025-01-05`.
For the web UI: `airflow standalone`, open http://localhost:8080, unpause `agroscan_pipeline`.

Note: do **not** add an `__init__.py` to the repo's `airflow/` folder; it would shadow the installed package.
