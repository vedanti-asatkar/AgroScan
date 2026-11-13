import json

from PIL import Image

from src.evaluate_v5 import evaluate as evaluate_v5
from src.export_model import export
from src.prepare_data import deduplicate_pool, sha256
from src.validate_data import validate


def write_split(prepared, split, labels, offset):
    rows = ["path,label,source"]
    for index, label in enumerate(labels):
        image_path = prepared / f"{split}_{index}.png"
        Image.new("RGB", (20, 20), (index * 20, offset, 40)).save(image_path)
        rows.append(f"{image_path},{label},plantvillage")
    (prepared / f"{split}.csv").write_text("\n".join(rows) + "\n")


def test_validation_reports_class_counts_and_rejects_cross_split_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    classes = ["healthy", "diseased"]
    for split, offset in (("train", 20), ("val", 80), ("test", 140)):
        write_split(prepared, split, classes, offset)

    result = validate({"classes": classes, "data": {"prepared_dir": str(prepared)},
                       "validation": {"min_width": 16, "min_height": 16}})
    assert result["valid"] is True
    assert result["images_per_split"] == {"train": 2, "val": 2, "test": 2}
    assert json.loads((tmp_path / "reports/data_validation.json").read_text())["valid"] is True

    (prepared / "test.csv").write_text(
        f"path,label,source\n{prepared / 'train_0.png'},healthy,plantvillage\n"
        f"{prepared / 'test_1.png'},diseased,plantvillage\n"
    )
    duplicate_result = validate({"classes": classes, "data": {"prepared_dir": str(prepared)},
                                 "validation": {"min_width": 16, "min_height": 16}})
    assert duplicate_result["valid"] is False
    assert any(issue["issue"] == "duplicate_image" for issue in duplicate_result["issues"])


def test_export_produces_the_api_metadata_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.keras").write_bytes(b"fake model for metadata unit test")
    (models / "classes.json").write_text(json.dumps({"classes": ["healthy", "diseased"], "img_size": 224,
                                                       "mlflow_run_id": "run-123"}))
    result = export({"model": {"img_size": 224}, "deploy": {"confidence_threshold": 0.7, "top_k": 3}})
    assert set(result) == {"version", "model_file", "class_names_file", "img_size", "input_range",
                           "center_crop", "confidence_threshold", "top_k", "low_confidence_message"}
    assert result["version"] == "run-123" and result["top_k"] == 2
    assert json.loads((models / "model_meta.json").read_text()) == result


def test_prepare_drops_test_leaks_and_conflicting_labels(tmp_path):
    same_a, same_b = tmp_path / "same_a.png", tmp_path / "same_b.png"
    conflict_a, conflict_b = tmp_path / "conflict_a.png", tmp_path / "conflict_b.png"
    leaked = tmp_path / "leaked.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(same_a)
    same_b.write_bytes(same_a.read_bytes())
    Image.new("RGB", (16, 16), (40, 50, 60)).save(conflict_a)
    conflict_b.write_bytes(conflict_a.read_bytes())
    Image.new("RGB", (16, 16), (70, 80, 90)).save(leaked)
    rows = [(str(same_a), "healthy", "plantvillage"), (str(same_b), "healthy", "plantdoc"),
            (str(conflict_a), "early_blight", "plantvillage"),
            (str(conflict_b), "late_blight", "plantdoc"), (str(leaked), "healthy", "plantdoc")]
    pool, removed = deduplicate_pool(rows, {sha256(leaked)})
    assert pool == [rows[0]]
    assert removed == {"leaked_duplicates_removed": 1, "same_label_duplicates_removed": 1,
                       "conflicting_label_images_removed": 2}


def test_v5_evaluator_uses_metadata_preprocessing_and_labels(v5_model_dir, tmp_path):
    rows = ["path,label,source"]
    for index, label in enumerate(("Pepper__bell___Bacterial_spot", "Potato___Early_blight")):
        image_path = tmp_path / f"v5_{index}.png"
        Image.new("RGB", (64, 48), (40 + index * 60, 80, 120)).save(image_path)
        rows.append(f"{image_path},{label},plantdoc")
    manifest = tmp_path / "test.csv"
    manifest.write_text("\n".join(rows) + "\n")
    output = tmp_path / "v5_metrics.json"

    result = evaluate_v5(v5_model_dir / "model_meta.json", manifest, output)
    assert result["n_images"] == 2
    assert 0 <= result["top1_accuracy"] <= result["top3_accuracy"] <= 1
    assert json.loads(output.read_text())["model_version"] == "test-v5"
