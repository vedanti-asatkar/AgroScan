"""Build train/val/test manifests (CSV) from raw PlantVillage + PlantDoc.

- train/val: PlantVillage + PlantDoc-train, stratified split
- test: PlantDoc-test (the real-world generalisation set)
- any train/val file byte-identical to a test file is dropped (leakage guard)
Manifests reference files in place; no image copies are made.
"""
import hashlib
import os
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.common import list_images, load_params, parse_args, write_json, write_manifest


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_plantvillage(root, classes):
    return [(str(p), p.parent.name, "plantvillage") for p in list_images(root) if p.parent.name in classes]


def collect_plantdoc(root, split, name_map):
    out = []
    for p in list_images(Path(root) / split):
        if p.parent.name in name_map:
            out.append((str(p), name_map[p.parent.name], "plantdoc"))
    return out


def main(params):
    d = params["data"]
    classes = params["classes"]
    pv = collect_plantvillage(d["plantvillage_dir"], set(classes))
    pd_train = collect_plantdoc(d["plantdoc_dir"], "train", params["plantdoc_name_map"])
    test = collect_plantdoc(d["plantdoc_dir"], "test", params["plantdoc_name_map"])
    if not (pv and test):
        raise SystemExit(f"No images found (plantvillage={len(pv)}, plantdoc test={len(test)}). Run `dvc pull`?")

    # leakage guard: hash only files whose size matches some test file
    test_sizes = {os.path.getsize(p) for p, _, _ in test}
    test_hashes = {md5(p) for p, _, _ in test}
    pool, removed = [], 0
    for row in pv + pd_train:
        if os.path.getsize(row[0]) in test_sizes and md5(row[0]) in test_hashes:
            removed += 1
        else:
            pool.append(row)

    labels = [r[1] for r in pool]
    train, val = train_test_split(pool, test_size=d["val_fraction"], stratify=labels, random_state=d["seed"])

    out = Path(d["prepared_dir"])
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_manifest([{"path": p, "label": lab, "source": src} for p, lab, src in rows], out / f"{name}.csv")
    write_json({"classes": classes}, out / "classes.json")
    write_json(
        {
            "leaked_duplicates_removed": removed,
            "counts": {n: dict(Counter(r[1] for r in rows)) for n, rows in
                       (("train", train), ("val", val), ("test", test))},
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        },
        out / "stats.json",
    )
    print(f"train={len(train)} val={len(val)} test={len(test)} leaked_removed={removed}")


if __name__ == "__main__":
    main(load_params(parse_args("Prepare manifests").params))
