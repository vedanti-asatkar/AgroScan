import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import batch_predict as bp  # noqa: E402

CLASSES = ["healthy", "blight", "spot"]


def _img(path, size=(40, 30), color=(10, 200, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _meta(**kw):
    m = {"model_file": "m.keras", "class_names_file": "c.json", "img_size": 16, "input_range": "0_255",
         "center_crop": True, "confidence_threshold": 0.7, "top_k": 3}
    m.update(kw)
    return m


def test_preprocess_shape_and_range_contract():
    img = Image.new("RGB", (50, 20), (255, 255, 255))
    raw = bp.preprocess_pil(img, _meta(input_range="0_255"))
    unit = bp.preprocess_pil(img, _meta(input_range="0_1"))
    assert raw.shape == unit.shape == (16, 16, 3)
    assert raw.max() == 255 and unit.max() == pytest.approx(1.0)
    assert bp.preprocess_pil(img, _meta(center_crop=False)).shape == (16, 16, 3)
    assert bp.preprocess_pil(Image.new("L", (30, 30)), _meta()).shape == (16, 16, 3)  # greyscale ok


def test_center_crop_keeps_middle():
    img = Image.new("RGB", (60, 20), (0, 0, 0))
    img.paste((255, 0, 0), (20, 0, 40, 20))  # red middle third
    arr = bp.preprocess_pil(img, _meta(img_size=20))
    assert arr[..., 0].min() == 255 and arr[..., 1].max() == 0


def test_class_names_formats(tmp_path):
    for i, obj in enumerate([CLASSES, {"class_names": CLASSES}, {"0": "healthy", "1": "blight", "2": "spot"}]):
        p = tmp_path / f"c{i}.json"
        p.write_text(json.dumps(obj))
        assert bp.load_class_names(p) == CLASSES
    (tmp_path / "bad.json").write_text("123")
    with pytest.raises(ValueError):
        bp.load_class_names(tmp_path / "bad.json")


def test_meta_missing_key(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps({"model_file": "x"}))
    with pytest.raises(KeyError):
        bp.load_meta(tmp_path / "m.json")


def test_predict_folder_end_to_end_with_labels_and_bad_file(tmp_path):
    root = tmp_path / "imgs"
    _img(root / "healthy" / "a.jpg")
    _img(root / "blight" / "b.png")
    _img(root / "spot" / "c.jpg")
    (root / "healthy" / "broken.jpg").write_bytes(b"not an image")
    (root / "notes.txt").write_text("ignored")

    # fake model: always predicts 'healthy' at 0.9, 'blight' 0.07, 'spot' 0.03
    def fake(batch):
        assert batch.shape[1:] == (16, 16, 3)
        return np.tile(np.array([0.9, 0.07, 0.03]), (len(batch), 1))

    images = bp.find_images(root, recursive=True)
    assert len(images) == 4  # txt ignored
    rows = bp.predict_folder(images, fake, _meta(), CLASSES, root=root, batch_size=2)
    by = {r["filename"]: r for r in rows}
    assert by["healthy/a.jpg"]["correct"] is True
    assert by["blight/b.png"]["correct"] is False and by["blight/b.png"]["top2_class"] == "blight"
    assert by["healthy/broken.jpg"]["error"].startswith("UnidentifiedImageError")
    assert "top1_class" not in by["healthy/broken.jpg"]
    acc = bp.accuracy_report(rows)
    assert acc["n"] == 3 and acc["top1"] == pytest.approx(1 / 3) and acc["top3"] == 1.0

    out = tmp_path / "o.csv"
    bp.write_csv(rows, out)
    got = list(csv.DictReader(open(out)))
    assert len(got) == 4 and "true_label" in got[0] and got[0]["low_confidence"] in ("True", "False", "")


def test_low_confidence_flag_follows_threshold(tmp_path):
    _img(tmp_path / "a.jpg")
    rows = bp.predict_folder([tmp_path / "a.jpg"], lambda b: np.tile([0.5, 0.3, 0.2], (len(b), 1)),
                             _meta(), CLASSES, root=tmp_path)
    assert rows[0]["low_confidence"] is True
    assert "true_label" not in rows[0]


def test_batching_gives_same_result(tmp_path):
    for i in range(5):
        _img(tmp_path / f"{i}.jpg", color=(i * 40, 0, 0))
    calls = []

    def fake(batch):
        calls.append(len(batch))
        return np.tile([0.2, 0.5, 0.3], (len(batch), 1))

    rows = bp.predict_folder(bp.find_images(tmp_path), fake, _meta(), CLASSES, root=tmp_path, batch_size=2)
    assert calls == [2, 2, 1] and len(rows) == 5 and rows[0]["top1_class"] == "blight"
