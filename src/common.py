"""Shared helpers: params, manifests, paths."""
import argparse
import csv
import json
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png"}
MANIFEST_FIELDS = ["path", "label", "source"]


def parse_args(description=""):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--params", default="params.yaml", help="params file (default: params.yaml)")
    return ap.parse_args()


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def write_manifest(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
