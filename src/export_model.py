"""Export a quality-gated model for the API, falling back to the approved v5 model."""
import json
from pathlib import Path

from src.common import load_params, parse_args, write_json


def export(params):
    candidate_model = Path("models/model.keras")
    candidate_classes = Path("models/classes.json")
    candidate_metrics_path = Path("reports/metrics.json")
    baseline_meta_path = Path("models/model_meta.v5.json")
    baseline_metrics_path = Path("reports/v5_plantdoc_metrics.json")
    gate = float(params.get("evaluate", {}).get("min_test_accuracy", 0.0))

    candidate_metrics = json.loads(candidate_metrics_path.read_text()) if candidate_metrics_path.is_file() else None
    candidate_accuracy = candidate_metrics.get("test_accuracy") if candidate_metrics else None
    use_candidate = candidate_accuracy is None or float(candidate_accuracy) >= gate

    if use_candidate:
        model_path, classes_path = candidate_model, candidate_classes
        if not model_path.is_file() or not classes_path.is_file():
            raise SystemExit("Candidate export requires models/model.keras and models/classes.json")
        class_data = json.loads(classes_path.read_text())
        classes = class_data.get("classes", [])
        img_size = int(class_data.get("img_size", params["model"]["img_size"]))
        input_range = "0_1"
        center_crop = False
        version = class_data.get("mlflow_run_id") or class_data.get("git_sha") or "pipeline-model"
        mlflow_model_name = class_data.get("mlflow_model_name")
        mlflow_model_version = class_data.get("mlflow_model_version")
        confidence_threshold = float(params.get("deploy", {}).get("confidence_threshold", 0.6))
        top_k = int(params.get("deploy", {}).get("top_k", 3))
        low_confidence_message = "Not sure. Take a clearer photo of one leaf."
        selected = "candidate"
    else:
        if not baseline_metrics_path.is_file() or not baseline_meta_path.is_file():
            raise SystemExit("Candidate failed the quality gate and no evaluated v5 baseline is available")
        baseline_metrics = json.loads(baseline_metrics_path.read_text())
        baseline_accuracy = float(baseline_metrics["top1_accuracy"])
        if baseline_accuracy < gate:
            raise SystemExit(
                f"Candidate ({candidate_accuracy:.4f}) and v5 baseline "
                f"({baseline_accuracy:.4f}) both fail the {gate:.4f} quality gate"
            )

        baseline = json.loads(baseline_meta_path.read_text())
        model_path = Path("models") / baseline["model_file"]
        classes_path = Path("models") / baseline["class_names_file"]
        if not model_path.is_file() or not classes_path.is_file():
            raise SystemExit(f"v5 baseline export requires {model_path} and {classes_path}")
        class_data = json.loads(classes_path.read_text())
        classes = class_data.get("classes", []) if isinstance(class_data, dict) else class_data
        img_size = int(baseline["img_size"])
        input_range = baseline["input_range"]
        center_crop = bool(baseline["center_crop"])
        mlflow_model_name = baseline.get("mlflow_model_name")
        mlflow_model_version = baseline.get("mlflow_model_version")
        version = baseline.get("mlflow_run_id") or baseline.get("version") or "v5-baseline"
        confidence_threshold = float(
            baseline.get(
                "confidence_threshold",
                params.get("deploy", {}).get("confidence_threshold", 0.6),
            )
        )
        top_k = int(baseline.get("top_k", params.get("deploy", {}).get("top_k", 3)))
        low_confidence_message = baseline.get("low_confidence_message", "Not sure. Take a clearer photo of one leaf.")
        selected = "v5_baseline"

    if not classes:
        raise SystemExit("Selected model metadata has no class names")
    meta = {
        "version": version,
        "model_file": model_path.name,
        "class_names_file": classes_path.name,
        "img_size": img_size,
        "input_range": input_range,
        "center_crop": center_crop,
        "confidence_threshold": confidence_threshold,
        "top_k": min(top_k, len(classes)),
        "low_confidence_message": low_confidence_message,
    }
    if mlflow_model_name:
        meta["mlflow_model_name"] = mlflow_model_name
    if mlflow_model_version:
        meta["mlflow_model_version"] = str(mlflow_model_version)

    write_json(meta, "models/model_meta.json")
    print(f"Exported {selected} model {model_path} at version {version}")
    return meta


def main(params_path):
    export(load_params(params_path))


if __name__ == "__main__":
    main(parse_args("Export the passing model for FastAPI serving").params)
