"""Shared helpers for the LIME and SHAP notebooks (task B6).

Everything is driven by models/model_meta.json so the explanations always use the same
preprocessing contract as the API. Read-only: nothing here trains or edits any dataset,
and the PlantDoc test images are only *looked at* (rule 7).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.batch_predict import (IMAGE_EXTS, _scale_for, load_class_names,  # noqa: E402
                                   load_meta)


def load_assets(meta_path: str | Path = ROOT / "models" / "model_meta.json"):
    """Return (model, meta, class_names). Needs TensorFlow and the model files (dvc pull)."""
    import tensorflow as tf

    meta_path = Path(meta_path)
    meta = load_meta(meta_path)
    mdir = meta_path.parent
    class_names = load_class_names(mdir / meta["class_names_file"])
    model = tf.keras.models.load_model(mdir / meta["model_file"], compile=False)
    return model, meta, class_names


def display_image(path: str | Path, meta: dict) -> np.ndarray:
    """uint8 HxWx3 image after resize / centre crop, *before* value scaling (what a human sees)."""
    size = int(meta["img_size"])
    img = Image.open(path).convert("RGB")
    if meta.get("center_crop", False):
        w, h = img.size
        s = size / min(w, h)
        img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BILINEAR)
        w, h = img.size
        left, top = (w - size) // 2, (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    else:
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def make_predict_fn(model, meta: dict):
    """Batch of 0-255 images (any float/uint dtype) -> class probabilities."""
    k = _scale_for(meta.get("input_range"))

    def predict(batch: np.ndarray) -> np.ndarray:
        x = np.asarray(batch, dtype=np.float32) * k
        return model.predict(x, verbose=0)

    return predict


def pick_images(folder: str | Path, n: int = 4, seed: int = 0) -> list[Path]:
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    random.Random(seed).shuffle(paths)
    return paths[:n]
