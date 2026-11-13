"""AgroScan weekly pipeline DAG (task B2).

validate_data -> run_pipeline -> summarize_live_logs -> check_thresholds
    healthy  -> healthy_report ------------------------------\
    retrain  -> retrain_decision -> train -> evaluate -> export -> done

Modes (env AGROSCAN_DAG_MODE)
  real         (default) real stage commands, real logs/predictions.jsonl
  placeholder  echo commands + sample logs, so the DAG runs before A's work has merged

Other env vars
  AGROSCAN_HOME  repo root (default: two levels above this file)
  AGROSCAN_PYTHON python executable (default: python)
  AGROSCAN_SIMULATE_DRIFT=1  placeholder mode only: sample logs simulate domain shift,
                             so you can demo the retrain branch

Airflow runs on Linux only: use WSL2 or Docker (docs/airflow_setup.md).
All thresholds live in configs/monitoring.yaml, none in this file.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

ROOT = Path(os.environ.get("AGROSCAN_HOME", Path(__file__).resolve().parents[2]))
PY = os.environ.get("AGROSCAN_PYTHON", "python")
MODE = os.environ.get("AGROSCAN_DAG_MODE", "real").lower()
CONFIG = "configs/monitoring.yaml"
SUMMARY = "reports/monitoring/summary.json"

# append (never insert): must not shadow the real airflow package
sys.path.append(str(ROOT))
from src import monitor_logs  # noqa: E402


def _cmd(real: str, placeholder: str) -> str:
    return placeholder if MODE == "placeholder" else real


def _check_thresholds(**_) -> str:
    with open(ROOT / SUMMARY, "r", encoding="utf-8") as f:
        summary = json.load(f)
    print(f"status={summary.get('status')} triggers={summary.get('retrain_triggers')}")
    return monitor_logs.branch_for_status(summary)


def _write(kind: str):
    def fn(**_):
        with open(ROOT / SUMMARY, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print("wrote", monitor_logs.write_decision(summary, kind, ROOT / "reports/monitoring"))
    return fn


_log_arg = ""
if MODE == "placeholder":
    _log_arg = " --log logs/sample_predictions.jsonl"

default_args = {"owner": "person-b", "retries": 1, "retry_delay": timedelta(minutes=5)}


@dag(
    dag_id="agroscan_pipeline",
    description="Weekly: validate data, run pipeline, monitor live logs, then healthy report or retrain",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["agroscan", "mlops"],
)
def agroscan_pipeline():
    kw = dict(cwd=str(ROOT))

    prep_samples = BashOperator(
        task_id="prepare_sample_logs",
        bash_command=_cmd("echo 'real mode: using logs/predictions.jsonl'",
                          f"{PY} scripts/make_sample_logs.py --out logs/sample_predictions.jsonl --n 300"
                          + (" --drift" if os.environ.get("AGROSCAN_SIMULATE_DRIFT") == "1" else "")),
        **kw,
    )
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=_cmd(f"{PY} -m src.validate_data", "echo '[placeholder] validate_data'"), **kw)
    run_pipeline = BashOperator(
        task_id="run_pipeline",
        bash_command=_cmd("dvc repro", "echo '[placeholder] dvc repro'"), **kw)
    summarize = BashOperator(
        task_id="summarize_live_logs",
        bash_command=f"{PY} -m src.monitor_logs --config {CONFIG}{_log_arg}", **kw)
    check = BranchPythonOperator(task_id="check_thresholds", python_callable=_check_thresholds)

    healthy = PythonOperator(task_id="healthy_report", python_callable=_write("healthy_report"))
    decision = PythonOperator(task_id="retrain_decision", python_callable=_write("retrain_decision"))

    train = BashOperator(task_id="train", bash_command=_cmd(
        "dvc repro --force --single-item train", "echo '[placeholder] train'"), **kw)
    evaluate = BashOperator(task_id="evaluate", bash_command=_cmd(
        "dvc repro --single-item evaluate", "echo '[placeholder] evaluate'"), **kw)
    export = BashOperator(task_id="export", bash_command=_cmd(
        "dvc repro --single-item export", "echo '[placeholder] export'"), **kw)

    done = EmptyOperator(task_id="done", trigger_rule="none_failed_min_one_success")

    prep_samples >> validate_data >> run_pipeline >> summarize >> check
    check >> healthy >> done
    check >> decision >> train >> evaluate >> export >> done


agroscan_pipeline()
