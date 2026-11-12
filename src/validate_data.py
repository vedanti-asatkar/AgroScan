"""Validate prepared splits before model training."""
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.common import load_params, parse_args, read_manifest, write_json


def validate(params):
    classes = set(params["classes"])
    prepared = Path(params["data"]["prepared_dir"])
    min_width = int(params["validation"]["min_width"])
    min_height = int(params["validation"]["min_height"])
    counts, issues, seen_hashes = {}, [], {}

    for split in ("train", "val", "test"):
        manifest = prepared / f"{split}.csv"
        if not manifest.is_file():
            counts[split] = {}
            issues.append({"split": split, "issue": "missing_manifest", "path": str(manifest)})
            continue
        rows = read_manifest(manifest)
        if not rows:
            issues.append({"split": split, "issue": "empty_split"})
        split_counts = Counter(row.get("label", "") for row in rows)
        for row in rows:
            path = Path(row.get("path", ""))
            label = row.get("label", "")
            if label not in classes:
                issues.append({"split": split, "issue": "unknown_label", "label": label,
                               "path": str(path)})
            if not path.is_file():
                issues.append({"split": split, "issue": "missing_image", "path": str(path)})
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                if width < min_width or height < min_height:
                    issues.append({"split": split, "issue": "image_too_small", "path": str(path),
                                   "width": width, "height": height,
                                   "minimum": [min_width, min_height]})
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                previous = seen_hashes.get(digest)
                if previous:
                    issues.append({"split": split, "issue": "duplicate_image", "path": str(path),
                                   "other_split": previous[0], "other_path": previous[1],
                                   "same_split": previous[0] == split})
                else:
                    seen_hashes[digest] = (split, str(path))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                issues.append({"split": split, "issue": "unreadable_image", "path": str(path),
                               "detail": str(exc)})
        missing = sorted(classes - set(split_counts))
        if missing:
            issues.append({"split": split, "issue": "missing_classes", "classes": missing})
        counts[split] = dict(sorted(split_counts.items()))

    result = {"valid": not issues, "counts": counts, "issues": issues,
              "images_per_split": {split: sum(split_counts.values()) for split, split_counts in counts.items()}}
    write_json(result, "reports/data_validation.json")
    return result


def main(params_path):
    result = validate(load_params(params_path))
    print(result)
    if not result["valid"]:
        raise SystemExit("Data validation failed; see reports/data_validation.json")


if __name__ == "__main__":
    main(parse_args("Validate prepared image splits").params)
