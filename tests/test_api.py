import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tiny_model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tiny_model_dir / "model.keras"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "pred.jsonl"))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "100000")
    import api.main as m
    m = importlib.reload(m)
    with TestClient(m.app) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200


def test_predict(client, jpeg_bytes, tmp_path):
    r = client.post("/predict", files={"file": ("x.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in {"a", "b", "c"}
    assert 0 <= body["confidence"] <= 1
    assert (tmp_path / "pred.jsonl").read_text().count("\n") == 1


def test_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_empty_and_large(client):
    assert client.post("/predict", files={"file": ("e.jpg", b"", "image/jpeg")}).status_code == 400
    assert client.post("/predict", files={"file": ("b.jpg", b"0" * 200000, "image/jpeg")}).status_code == 413
