"""Remove exact duplicate and conflicting-label images from a prepared split archive."""
import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from src.common import write_json

SPLITS = ("pd_test", "pd_train", "pd_val", "pv_train", "pv_val", "pw_test", "pw_train", "pw_val")
SPLIT_PRIORITY = {
    "pd_test": 0, "pw_test": 1,
    "pd_val": 2, "pw_val": 3, "pv_val": 4,
    "pd_train": 5, "pw_train": 6, "pv_train": 7,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def write_member(target, name, content, is_dir=False):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_STORED if is_dir else zipfile.ZIP_DEFLATED
    info.external_attr = (0o40755 if is_dir else 0o100644) << 16
    target.writestr(info, content, compress_type=info.compress_type, compresslevel=6)


def sha256_member(archive, info):
    digest = hashlib.sha256()
    with archive.open(info) as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_archive(input_path, output_path, audit_path, classes):
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups, counts, directories, other_files = defaultdict(list), {s: Counter() for s in SPLITS}, set(), []
    source_hash = hashlib.sha256()
    with input_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            source_hash.update(chunk)

    with zipfile.ZipFile(input_path) as source:
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
                raise SystemExit(f"Unsafe archive path: {info.filename}")
            if info.is_dir():
                directories.add(path.as_posix().rstrip("/"))
                continue
            if len(path.parts) == 3 and path.parts[0] in SPLITS and path.suffix.lower() in IMAGE_SUFFIXES:
                split, label, _ = path.parts
                if label not in classes:
                    raise SystemExit(f"Unexpected class folder {label!r} in {info.filename}")
                digest = sha256_member(source, info)
                groups[digest].append((info, split, label))
                counts[split][label] += 1
                for index in range(1, len(path.parts)):
                    directories.add("/".join(path.parts[:index]))
            else:
                other_files.append(info)

        if not groups:
            raise SystemExit(f"No prepared images found in {input_path}")

        kept, conflict_groups, duplicate_groups = [], [], []
        removed_counts = {s: Counter() for s in SPLITS}
        for digest, members in groups.items():
            labels = {label for _, _, label in members}
            if len(labels) > 1:
                conflict_groups.append({
                    "sha256": digest,
                    "files": [{"split": split, "label": label, "path": info.filename}
                              for info, split, label in members],
                })
                for _, split, label in members:
                    removed_counts[split][label] += 1
                continue

            chosen = min(members, key=lambda item: (SPLIT_PRIORITY[item[1]], item[0].filename))
            kept.append(chosen[0])
            if len(members) > 1:
                duplicate_groups.append({
                    "sha256": digest,
                    "kept": {"split": chosen[1], "label": chosen[2], "path": chosen[0].filename},
                    "removed": [{"split": split, "label": label, "path": info.filename}
                                for info, split, label in members if info is not chosen[0]],
                })
                for info, split, label in members:
                    if info is not chosen[0]:
                        removed_counts[split][label] += 1

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
            for directory in sorted(directories, key=lambda x: (x.count("/"), x)):
                write_member(target, directory.rstrip("/") + "/", b"", is_dir=True)
            for info in sorted(kept, key=lambda x: x.filename):
                write_member(target, info.filename, source.read(info))
            for info in sorted(other_files, key=lambda x: x.filename):
                write_member(target, info.filename, source.read(info))

    clean_counts = {s: Counter() for s in SPLITS}
    with zipfile.ZipFile(output_path) as cleaned:
        for info in cleaned.infolist():
            path = PurePosixPath(info.filename)
            if len(path.parts) == 3 and path.parts[0] in clean_counts and path.suffix.lower() in IMAGE_SUFFIXES:
                clean_counts[path.parts[0]][path.parts[1]] += 1

    output_digest = hashlib.sha256()
    with output_path.open("rb") as output:
        for chunk in iter(lambda: output.read(1 << 20), b""):
            output_digest.update(chunk)
    audit = {
        "source_archive": input_path.name,
        "source_archive_sha256": source_hash.hexdigest(),
        "clean_archive": output_path.name,
        "clean_archive_sha256": output_digest.hexdigest(),
        "classes": classes,
        "original_counts": {s: {"total": sum(c.values()), "by_class": dict(sorted(c.items()))}
                            for s, c in counts.items()},
        "clean_counts": {s: {"total": sum(c.values()), "by_class": dict(sorted(c.items()))}
                         for s, c in clean_counts.items()},
        "conflicting_label_groups_removed": len(conflict_groups),
        "conflicting_label_images_removed": sum(len(group["files"]) for group in conflict_groups),
        "exact_duplicate_groups_deduplicated": len(duplicate_groups),
        "exact_duplicate_copies_removed": sum(len(group["removed"]) for group in duplicate_groups),
        "removed_counts_by_split": {s: dict(sorted(c.items())) for s, c in removed_counts.items()},
        "removed_conflicting_label_details": conflict_groups,
        "deduplicated_group_details": duplicate_groups,
        "split_priority": list(SPLIT_PRIORITY),
        "policy": (
            "Drop exact-content groups with conflicting class labels. For same-label copies, "
            "keep one image and prefer test, then validation, then training splits so duplicate "
            "content does not leak into a held-out split."
        ),
    }
    write_json(audit, audit_path)
    print(json.dumps({k: audit[k] for k in (
        "source_archive_sha256", "clean_archive_sha256", "original_counts", "clean_counts",
        "conflicting_label_groups_removed", "conflicting_label_images_removed",
        "exact_duplicate_groups_deduplicated", "exact_duplicate_copies_removed")}, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input prepared dataset ZIP")
    parser.add_argument("--output", required=True, help="Output cleaned dataset ZIP")
    parser.add_argument("--audit", default="reports/prepared_dataset_cleaning.json")
    parser.add_argument("--classes", default="params.yaml")
    args = parser.parse_args()
    params = json.loads(Path(args.classes).read_text()) if args.classes.endswith(".json") else None
    if params is None:
        import yaml
        params = yaml.safe_load(Path(args.classes).read_text())
    clean_archive(args.input, args.output, args.audit, params["classes"])


if __name__ == "__main__":
    main()
