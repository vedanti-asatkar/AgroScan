import importlib
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.preprocess import prepare_for_model
from tests.conftest import CLASSES


def make_client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "pred.jsonl"))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "100000")
    import api.main as m
    return TestClient(importlib.reload(m).app)


@pytest.fixture()
def client(tiny_model_dir, tmp_path, monkeypatch):
    with make_client(tiny_model_dir, tmp_path, monkeypatch) as c:
        yield c


@pytest.fixture()
def v5_client(v5_model_dir, tmp_path, monkeypatch):
    with make_client(v5_model_dir, tmp_path, monkeypatch) as c:
        yield c


def png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


# ---- fallback layout (model.keras + classes.json) ----
def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_predict(client, jpeg_bytes, tmp_path):
    r = client.post("/predict", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in {"a", "b", "c"}
    assert 0 <= body["confidence"] <= 1
    lines = (tmp_path / "pred.jsonl").read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert set(event) == {"ts", "model_version", "top1_class", "confidence", "top3",
                          "low_confidence", "latency_ms", "brightness"}
    assert len(event["top3"]) == 3 and 0 <= event["brightness"] <= 255


def test_rejects_non_image(client):
    assert client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400


def test_rejects_empty_and_large(client):
    assert client.post("/predict", files={"file": ("e.jpg", b"", "image/jpeg")}).status_code == 400
    assert client.post("/predict", files={"file": ("b.jpg", b"0" * 200000, "image/jpeg")}).status_code == 413


# ---- v5 layout (model_meta.json) ----
def test_v5_ready_reports_config(v5_client):
    r = v5_client.get("/ready").json()
    assert r["classes"] == CLASSES and r["img_size"] == 48 and r["confidence_threshold"] == 0.7
    assert r["model_version"] == "test-v5"


def test_v5_predict_top3_and_probabilities(v5_client, jpeg_bytes):
    body = v5_client.post("/predict", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")}).json()
    assert len(body["top_k"]) == 3
    confs = [t["confidence"] for t in body["top_k"]]
    assert confs == sorted(confs, reverse=True)
    assert body["predicted_class"] == body["top_k"][0]["class"] in CLASSES
    assert body["low_confidence"] == (body["confidence"] < 0.7)


def test_v5_low_confidence_message(v5_client):
    grey = np.full((60, 90, 3), 127, dtype="uint8")  # maps to ~0 after the model's Rescaling -> near-flat softmax
    body = v5_client.post("/predict", files={"file": ("g.png", png(grey), "image/png")}).json()
    assert body["low_confidence"] is True
    assert body["message"].startswith("Not sure")


def test_preprocess_raw_range_and_center_crop():
    # 100 x 200 image: left quarter red, middle half green, right quarter blue
    a = np.zeros((100, 200, 3), dtype="uint8")
    a[:, :50, 0] = 255
    a[:, 50:150, 1] = 255
    a[:, 150:, 2] = 255
    x = prepare_for_model(png(a), 48, "0_255", center_crop=True).numpy()
    assert x.shape == (48, 48, 3) and x.dtype == np.float32
    assert x.max() > 200  # raw 0-255 values, not divided by 255
    assert x[..., 1].mean() > 250 and x[..., 0].max() < 5 and x[..., 2].max() < 5  # only the green centre square
    y = prepare_for_model(png(a), 48, "0_1", center_crop=False).numpy()
    assert y.max() <= 1.0 and y[..., 0].mean() > 0.1  # whole image, scaled to [0, 1]


def test_class_count_mismatch_is_not_served(v5_model_dir, tmp_path, monkeypatch):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(v5_model_dir, bad)
    (bad / "class_names.json").write_text('["only", "two"]')
    with make_client(bad, tmp_path, monkeypatch) as c:
        assert c.get("/ready").status_code == 503
