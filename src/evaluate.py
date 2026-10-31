"""Evaluate on val + PlantDoc test; write metrics/plots; enforce quality gate."""
import sys
from pathlib import Path

import keras
import matplotlib
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.common import load_params, parse_args, read_manifest, write_json
from src.preprocess import load_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def predict_split(model, rows, class_idx, size, bs=32):
    paths = [r["path"] for r in rows]
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda x: load_path(x, size), num_parallel_calls=tf.data.AUTOTUNE)
    probs = model.predict(ds.batch(bs), verbose=0)
    y_true = np.array([class_idx[r["label"]] for r in rows])
    return y_true, probs


def main(params_path):
    p = load_params(params_path)
    classes = p["classes"]
    class_idx = {c: i for i, c in enumerate(classes)}
    prep = Path(p["data"]["prepared_dir"])
    model = keras.models.load_model("models/model.keras")
    size = p["model"]["img_size"]

    metrics, report = {}, {}
    for split in ("val", "test"):
        y, probs = predict_split(model, read_manifest(prep / f"{split}.csv"), class_idx, size)
        pred = probs.argmax(1)
        metrics[f"{split}_accuracy"] = float((pred == y).mean())
        metrics[f"{split}_mean_confidence"] = float(probs.max(1).mean())
        # confident-but-wrong rate (>=0.9): the failure mode seen in the v2 model
        wrong_conf = ((pred != y) & (probs.max(1) >= 0.9)).sum()
        metrics[f"{split}_confident_wrong_rate"] = float(wrong_conf / len(y))
        rep = classification_report(y, pred, labels=range(len(classes)), target_names=classes,
                                    output_dict=True, zero_division=0)
        report[split] = rep
        metrics[f"{split}_macro_f1"] = float(rep["macro avg"]["f1-score"])
        if split == "test":
            cm = confusion_matrix(y, pred, labels=range(len(classes)))
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
            ax.set_yticks(range(len(classes)), classes, fontsize=7)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
            ax.set_title("PlantDoc test confusion matrix")
            fig.tight_layout()
            fig.savefig("reports/confusion_matrix.png", dpi=120)

    write_json(metrics, "reports/metrics.json")
    write_json(report, "reports/classification_report.json")

    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment"])
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts("reports")

    print(metrics)
    gate = p["evaluate"]["min_test_accuracy"]
    if metrics["test_accuracy"] < gate:
        print(f"QUALITY GATE FAILED: test_accuracy {metrics['test_accuracy']:.4f} < {gate}")
        sys.exit(1)


if __name__ == "__main__":
    main(parse_args("Evaluate model").params)
