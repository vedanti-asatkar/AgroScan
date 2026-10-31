"""End-to-end: synthetic data -> prepare -> train -> evaluate -> drift (CPU, ~1-2 min)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_pipeline(tmp_path):
    params = yaml.safe_load((ROOT / "params.ci.yaml").read_text())
    params["data"].update(plantvillage_dir=str(tmp_path / "plantvillage"), plantdoc_dir=str(tmp_path / "plantdoc"),
                          prepared_dir=str(tmp_path / "prepared"))
    params["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"
    pf = tmp_path / "params.yaml"
    pf.write_text(yaml.safe_dump(params))
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    def run(*cmd):
        subprocess.run([sys.executable, *cmd], cwd=tmp_path, env=env, check=True)

    (tmp_path / "reports").mkdir()
    run(str(ROOT / "scripts/make_synthetic_data.py"), str(tmp_path), "--params", str(pf))
    for mod in ("prepare_data", "train", "evaluate", "drift_check"):
        run("-m", f"src.{mod}", "--params", str(pf))
    metrics = json.loads((tmp_path / "reports/metrics.json").read_text())
    assert metrics["test_accuracy"] > 0.5  # classes are colour-separable
    assert (tmp_path / "models/model.keras").exists()
