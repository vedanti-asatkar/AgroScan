"""Generate a tiny synthetic PlantVillage/PlantDoc-shaped dataset for smoke tests.

Each class gets a distinct base colour so a model can learn it quickly.
Usage: python scripts/make_synthetic_data.py OUT_DIR [--per-class N] [--params params.yaml]
"""
import argparse
import colorsys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def make(out, params, per_class, seed=0):
    rng = np.random.default_rng(seed)
    classes = params["classes"]
    inv = {v: k for k, v in params["plantdoc_name_map"].items()}
    out = Path(out)
    for ci, c in enumerate(classes):
        base = np.array(colorsys.hsv_to_rgb(ci / len(classes), 0.8, 0.8)) * 255
        targets = [(out / "plantvillage" / c, per_class), (out / "plantdoc" / "train" / inv[c], 4),
                   (out / "plantdoc" / "test" / inv[c], 4)]
        for d, n in targets:
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                a = np.clip(base + rng.normal(0, 25, (64, 64, 3)), 0, 255).astype("uint8")
                Image.fromarray(a).save(d / f"{i}.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--params", default="params.ci.yaml")
    a = ap.parse_args()
    make(a.out, yaml.safe_load(open(a.params)), a.per_class)
