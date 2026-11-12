"""Re-measure the downloaded v5 model on the fixed PlantDoc test split."""
import argparse
import json
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import classification_report

from src.common import read_manifest, write_json
from src.preprocess import prepare_for_model


def evaluate(meta_path="models/model_meta.v5.json", manifest_path="data/prepared/test.csv",
             output_path="reports/v5_plantdoc_metrics.json", dataset="PlantDoc test"):
    meta = json.loads(Path(meta_path).read_text())
    model_path = Path(meta_path).parent / meta["model_file"]
    classes_path = Path(meta_path).parent / meta["class_names_file"]
    classes = json.loads(classes_path.read_text())
    rows = read_manifest(manifest_path)
    if not rows:
        raise SystemExit(f"No evaluation rows found in {manifest_path}")

    class_idx = {name: index for index, name in enumerate(classes)}
    unknown = sorted({row["label"] for row in rows} - set(class_idx))
    if unknown:
        raise SystemExit(f"Labels in {manifest_path} are not in the v5 class list: {unknown}")

    batch = np.stack([
        prepare_for_model(Path(row["path"]).read_bytes(), meta["img_size"], meta["input_range"],
                          meta["center_crop"]).numpy()
        for row in rows
    ])
    model = keras.models.load_model(model_path, compile=False)
    probabilities = np.asarray(model.predict(batch, batch_size=32, verbose=0))
    if probabilities.shape[1] != len(classes):
        raise SystemExit(f"Model has {probabilities.shape[1]} outputs but metadata lists {len(classes)} classes")

    truth = np.asarray([class_idx[row["label"]] for row in rows])
    predicted = probabilities.argmax(axis=1)
    top_k = np.argsort(probabilities, axis=1)[:, -min(3, len(classes)):]
    top3_accuracy = float(np.any(top_k == truth[:, None], axis=1).mean())
    result = {
        "model_version": meta["version"],
        "dataset": dataset,
        "manifest": str(manifest_path),
        "n_images": len(rows),
        "top1_accuracy": float((predicted == truth).mean()),
        "top3_accuracy": top3_accuracy,
        "classification_report": classification_report(
            truth, predicted, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0
        ),
    }
    write_json(result, output_path)
    print({key: result[key] for key in ("model_version", "dataset", "n_images", "top1_accuracy", "top3_accuracy")})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate v5 on a prepared test split")
    parser.add_argument("--meta", default="models/model_meta.v5.json")
    parser.add_argument("--manifest", default="data/prepared/test.csv")
    parser.add_argument("--output", default="reports/v5_plantdoc_metrics.json")
    parser.add_argument("--dataset", default="PlantDoc test")
    args = parser.parse_args()
    evaluate(args.meta, args.manifest, args.output, args.dataset)
