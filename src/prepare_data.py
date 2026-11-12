"""Build train/val/test manifests (CSV) from raw PlantVillage + PlantDoc.

- train/val: PlantVillage + PlantDoc-train, stratified split
- test: PlantDoc-test (the real-world generalisation set)
- any train/val file byte-identical to a test file is dropped (leakage guard)
Manifests reference files in place; no image copies are made.
"""
import hashlib
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.common import list_images, load_params, parse_args, write_json, write_manifest


def sha256(path):
    h = hashlib.sha256()
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


def deduplicate_pool(rows, test_hashes):
    by_hash = {}
    for row in rows:
        by_hash.setdefault(sha256(row[0]), []).append(row)

    pool = []
    removed = {"leaked_duplicates_removed": 0, "same_label_duplicates_removed": 0,
               "conflicting_label_images_removed": 0}
    for digest, matches in by_hash.items():
        if digest in test_hashes:
            removed["leaked_duplicates_removed"] += len(matches)
        elif len({row[1] for row in matches}) > 1:
            removed["conflicting_label_images_removed"] += len(matches)
        else:
            pool.append(matches[0])
            removed["same_label_duplicates_removed"] += len(matches) - 1
    return pool, removed


def main(params):
    d = params["data"]
    classes = params["classes"]
    pv = collect_plantvillage(d["plantvillage_dir"], set(classes))
    pd_train = collect_plantdoc(d["plantdoc_dir"], "train", params["plantdoc_name_map"])
    test = collect_plantdoc(d["plantdoc_dir"], "test", params["plantdoc_name_map"])
    if not (pv and test):
        raise SystemExit(f"No images found (plantvillage={len(pv)}, plantdoc test={len(test)}). Run `dvc pull`?")

    # Exact-content leakage guard across all sources and splits.
    test_hashes = {sha256(path) for path, _, _ in test}
    pool, removed = deduplicate_pool(pv + pd_train, test_hashes)

    labels = [r[1] for r in pool]
    train, val = train_test_split(pool, test_size=d["val_fraction"], stratify=labels, random_state=d["seed"])

    out = Path(d["prepared_dir"])
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_manifest([{"path": p, "label": lab, "source": src} for p, lab, src in rows], out / f"{name}.csv")
    write_json({"classes": classes}, out / "classes.json")
    write_json(
        {
            **removed,
            "counts": {n: dict(Counter(r[1] for r in rows)) for n, rows in
                       (("train", train), ("val", val), ("test", test))},
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        },
        out / "stats.json",
    )
    print(f"train={len(train)} val={len(val)} test={len(test)} "
          f"leaked_removed={removed['leaked_duplicates_removed']} "
          f"duplicates_removed={removed['same_label_duplicates_removed']} "
          f"conflicts_removed={removed['conflicting_label_images_removed']}")


if __name__ == "__main__":
    main(load_params(parse_args("Prepare manifests").params))
