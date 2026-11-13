"""Batch inference (task B4): a folder of images -> a CSV of predictions.

    python scripts/batch_predict.py --images path/to/folder --out reports/batch_predictions.csv
    python scripts/batch_predict.py --images data/plantdoc_test --recursive --out out.csv

Preprocessing follows models/model_meta.json (img_size, input_range, center_crop,
confidence_threshold, top_k), the same contract the API uses. If the images sit in
sub-folders named after classes, a ``true_label`` and ``correct`` column are added
and top-1 / top-k accuracy is printed.

WIRING NOTE: ``preprocess_pil`` below is a meta-driven implementation of the contract.
Once A's src/preprocess.py is merged, point ``_shared_preprocess`` at the real function so
batch and API can never diverge (see the TODO). Do not rename anything in src/preprocess.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_META = Path("models/model_meta.json")

PredictFn = Callable[[np.ndarray], np.ndarray]  # (N,H,W,3) float32 -> (N,num_classes) probabilities


# ----------------------------------------------------------------- meta / classes
def load_meta(path: str | Path = DEFAULT_META) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for k in ("model_file", "class_names_file", "img_size"):
        if k not in meta:
            raise KeyError(f"model_meta.json is missing required key '{k}'")
    return meta


def load_class_names(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if "class_names" in obj:
            obj = obj["class_names"]
        else:  # {"0": "name", ...}
            obj = [v for _, v in sorted(obj.items(), key=lambda kv: int(kv[0]))]
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError(f"Unrecognised class names format in {path}")
    return obj


# --------------------------------------------------------------- preprocessing
def _scale_for(input_range) -> float:
    """Return the multiplier applied to raw 0-255 pixels. v5 uses raw 0-255, the pipeline model 0-1."""
    if isinstance(input_range, (list, tuple)) and len(input_range) == 2:
        hi = float(input_range[1])
        return hi / 255.0
    s = str(input_range).lower().replace(" ", "")
    if s in {"0_1", "0-1", "[0,1]", "01", "unit", "zero_one", "0to1"}:
        return 1.0 / 255.0
    return 1.0  # "0_255", "0-255", "raw", "[0,255]", None


def preprocess_pil(img: Image.Image, meta: dict) -> np.ndarray:
    """Resize (shorter side) + optional centre crop + scale, per model_meta.json."""
    size = int(meta["img_size"])
    img = img.convert("RGB")
    if meta.get("center_crop", False):
        w, h = img.size
        scale = size / min(w, h)
        img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    else:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return arr * _scale_for(meta.get("input_range"))


def _shared_preprocess(img: Image.Image, meta: dict) -> np.ndarray:
    # TODO(B, after A1/A2 merge): call the real function from src/preprocess.py here, e.g.
    #   from src.preprocess import <existing_function_name>
    #   return <existing_function_name>(img, meta)
    # and delete preprocess_pil if it duplicates it. Until then the meta-driven version is used.
    return preprocess_pil(img, meta)


# ------------------------------------------------------------------ inference
def find_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def topk(probs: np.ndarray, class_names: Sequence[str], k: int) -> list[tuple[str, float]]:
    k = min(k, len(class_names))
    idx = np.argsort(-probs)[:k]
    return [(class_names[i], float(probs[i])) for i in idx]


def predict_folder(
    images: Sequence[Path],
    predict_fn: PredictFn,
    meta: dict,
    class_names: Sequence[str],
    root: Path | None = None,
    batch_size: int = 32,
    infer_labels: bool = True,
) -> list[dict]:
    """Return one row per image. Unreadable images get an ``error`` and no prediction."""
    k = int(meta.get("top_k", 3))
    thr = float(meta.get("confidence_threshold", 0.0))
    rows: list[dict] = []
    batch_arrays: list[np.ndarray] = []
    batch_rows: list[dict] = []

    def flush() -> None:
        if not batch_arrays:
            return
        probs = np.asarray(predict_fn(np.stack(batch_arrays)))
        for row, p in zip(batch_rows, probs):
            res = topk(p, class_names, k)
            for i in range(k):
                row[f"top{i+1}_class"] = res[i][0] if i < len(res) else ""
                row[f"top{i+1}_confidence"] = round(res[i][1], 4) if i < len(res) else ""
            row["low_confidence"] = res[0][1] < thr
        batch_arrays.clear()
        batch_rows.clear()

    for path in images:
        rel = path.relative_to(root) if root else Path(path.name)
        row: dict = {"filename": str(rel).replace("\\", "/"), "error": ""}
        if infer_labels and root and len(rel.parts) > 1 and rel.parts[-2] in class_names:
            row["true_label"] = rel.parts[-2]
        try:
            with Image.open(path) as im:
                batch_arrays.append(_shared_preprocess(im, meta))
        except (UnidentifiedImageError, OSError, ValueError) as e:
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        rows.append(row)
        batch_rows.append(row)
        if len(batch_arrays) >= batch_size:
            flush()
    flush()

    for row in rows:
        if "true_label" in row and row.get("top1_class"):
            row["correct"] = row["true_label"] == row["top1_class"]
    return rows


def write_csv(rows: list[dict], out: str | Path, k: int = 3) -> None:
    cols = ["filename"]
    if any("true_label" in r for r in rows):
        cols += ["true_label"]
    for i in range(1, k + 1):
        cols += [f"top{i}_class", f"top{i}_confidence"]
    cols += ["low_confidence"]
    if any("correct" in r for r in rows):
        cols += ["correct"]
    cols += ["error"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def accuracy_report(rows: list[dict], k: int = 3) -> dict | None:
    labelled = [r for r in rows if "true_label" in r and r.get("top1_class")]
    if not labelled:
        return None
    top1 = sum(r["true_label"] == r["top1_class"] for r in labelled)
    topk_ = sum(r["true_label"] in [r.get(f"top{i}_class") for i in range(1, k + 1)] for r in labelled)
    return {"n": len(labelled), "top1": top1 / len(labelled), f"top{k}": topk_ / len(labelled)}


# ------------------------------------------------------------------------ CLI
def load_keras_predictor(model_path: str | Path) -> PredictFn:
    import tensorflow as tf  # imported lazily so tests / --help work without TensorFlow

    model = tf.keras.models.load_model(model_path, compile=False)
    return lambda batch: model.predict(batch, verbose=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--out", default="reports/batch_predictions.csv")
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--models-dir", default=None, help="default: the folder containing --meta")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-labels", action="store_true", help="do not infer true labels from folder names")
    a = ap.parse_args(argv)

    meta = load_meta(a.meta)
    mdir = Path(a.models_dir) if a.models_dir else Path(a.meta).parent
    class_names = load_class_names(mdir / meta["class_names_file"])
    images_root = Path(a.images)
    images = find_images(images_root, a.recursive)
    if not images:
        print(f"No images found in {images_root}", file=sys.stderr)
        return 1
    predict_fn = load_keras_predictor(mdir / meta["model_file"])
    rows = predict_folder(images, predict_fn, meta, class_names, root=images_root,
                          batch_size=a.batch_size, infer_labels=not a.no_labels)
    k = int(meta.get("top_k", 3))
    write_csv(rows, a.out, k)
    n_err = sum(bool(r["error"]) for r in rows)
    print(f"{len(rows)} images -> {a.out} ({n_err} unreadable)")
    acc = accuracy_report(rows, k)
    if acc:
        print(f"labelled n={acc['n']}  top1={acc['top1']:.3f}  top{k}={acc[f'top{k}']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
