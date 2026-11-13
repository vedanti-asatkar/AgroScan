import sys
from pathlib import Path

import pytest

# a bare "airflow" would match our own airflow/ folder, so name a real submodule
pytest.importorskip("airflow.decorators")  # skipped unless requirements-extras.txt is installed
ROOT = Path(__file__).resolve().parents[1]


def test_dag_structure_and_branches(monkeypatch):
    monkeypatch.setenv("AGROSCAN_DAG_MODE", "placeholder")
    monkeypatch.setenv("AGROSCAN_HOME", str(ROOT))
    sys.path.insert(0, str(ROOT / "airflow" / "dags"))
    from agroscan_pipeline import agroscan_pipeline  # noqa: E402

    dag = agroscan_pipeline() if callable(agroscan_pipeline) else agroscan_pipeline
    ids = set(dag.task_ids)
    assert {"validate_data", "run_pipeline", "summarize_live_logs", "check_thresholds",
            "healthy_report", "retrain_decision", "train", "evaluate", "export", "done"} <= ids
    assert dag.get_task("check_thresholds").downstream_task_ids == {"healthy_report", "retrain_decision"}
    assert dag.schedule_interval in ("@weekly", None) or dag.timetable is not None
