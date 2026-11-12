"""AgroScan inference API.

The model and its preprocessing are described by models/model_meta.json (file names, input size,
pixel range, centre crop, confidence threshold). Without that file the API falls back to
models/model.keras + models/classes.json from the repo's own training pipeline.
"""
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.preprocess import prepare_for_model

log = logging.getLogger("agroscan")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
PRED_LOG = os.getenv("PREDICTION_LOG", "logs/predictions.jsonl")
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

state = {"model": None, "classes": None, "cfg": None, "registry_model": None, "registry_version": None}


def read_class_names(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        for key in ("classes", "class_names"):
            if key in data:
                return list(data[key])
        return [data[k] for k in sorted(data, key=int)]  # {"0": "...", "1": "..."}
    return list(data)


def load_config():
    meta = MODEL_DIR / "model_meta.json"
    if not meta.exists():
        meta = MODEL_DIR / "model_meta.v5.json"
    if meta.exists():
        cfg = json.loads(meta.read_text())
        cfg["model_path"] = MODEL_DIR / cfg["model_file"]
        cfg["classes_path"] = MODEL_DIR / cfg["class_names_file"]
        return cfg
    # fallback: model trained by src/train.py ([0, 1] input, size stored in classes.json)
    cfg = {"version": None, "model_path": MODEL_DIR / "model.keras", "classes_path": MODEL_DIR / "classes.json",
           "img_size": 224, "input_range": "0_1", "center_crop": False, "confidence_threshold": 0.6, "top_k": 3,
           "low_confidence_message": "Low confidence. Try a closer, well-lit photo of a single leaf."}
    if cfg["classes_path"].exists():
        extra = json.loads(cfg["classes_path"].read_text())
        if isinstance(extra, dict):
            cfg["img_size"] = extra.get("img_size", 224)
            cfg["version"] = extra.get("mlflow_run_id")
    return cfg


def load_state():
    cfg = load_config()
    state["cfg"] = cfg
    if not cfg["model_path"].exists():
        log.error("Model file not found: %s", cfg["model_path"])
        return
    if not cfg["classes_path"].exists():
        log.error("Class names file not found: %s", cfg["classes_path"])
        return
    model = keras.models.load_model(cfg["model_path"], compile=False)
    classes = read_class_names(cfg["classes_path"])
    n_out = model.output_shape[-1]
    if n_out != len(classes):
        log.error("Model has %d outputs but %d class names; not serving", n_out, len(classes))
        return
    class_data = json.loads(Path(cfg["classes_path"]).read_text())
    state.update(
        model=model,
        classes=classes,
        registry_model=cfg.get("mlflow_model_name")
        or (class_data.get("mlflow_model_name") if isinstance(class_data, dict) else None),
        registry_version=cfg.get("mlflow_model_version")
        or (class_data.get("mlflow_model_version") if isinstance(class_data, dict) else None),
    )
    log.info("Loaded %s (%d classes, %dpx, input %s, centre crop %s)", cfg["model_path"], len(classes),
             cfg["img_size"], cfg["input_range"], cfg["center_crop"])


@asynccontextmanager
async def lifespan(_app):
    load_state()
    yield


app = FastAPI(title="AgroScan API", lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "AgroScan API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")
    cfg = state["cfg"]
    return {"status": "ready", "model_path": str(cfg["model_path"]), "model_version": cfg["version"],
            "classes": state["classes"], "img_size": cfg["img_size"],
            "confidence_threshold": cfg["confidence_threshold"], "mlflow_model_name": state["registry_model"],
            "mlflow_model_version": state["registry_version"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")
    cfg = state["cfg"]
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(data)) as image:
            brightness = float(np.asarray(image.convert("RGB"), dtype=np.float32).mean())
        x = prepare_for_model(data, cfg["img_size"], cfg["input_range"], cfg["center_crop"]).numpy()[None]
    except Exception:
        raise HTTPException(400, "could not decode image (use jpg or png)")

    t0 = time.perf_counter()
    probs = np.asarray(state["model"].predict(x, verbose=0)[0], dtype=np.float64)  # already softmax
    latency_ms = (time.perf_counter() - t0) * 1000
    order = np.argsort(probs)[::-1][: cfg["top_k"]]
    top = [{"class": state["classes"][i], "confidence": round(float(probs[i]), 4)} for i in order]
    low = top[0]["confidence"] < cfg["confidence_threshold"]
    result = {
        "predicted_class": top[0]["class"],
        "confidence": top[0]["confidence"],
        "low_confidence": bool(low),
        "message": cfg["low_confidence_message"] if low else None,
        "top_k": top,
        "model_version": cfg["version"],
    }

    try:  # prediction log feeds drift / quality monitoring
        Path(PRED_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(PRED_LOG, "a") as f:
            f.write(json.dumps({"ts": time.time(), "model_version": cfg["version"],
                                "top1_class": result["predicted_class"], "confidence": result["confidence"],
                                "top3": result["top_k"], "low_confidence": result["low_confidence"],
                                "latency_ms": round(latency_ms, 1), "brightness": round(brightness, 2)})
                    + "\n")
    except OSError:
        log.warning("could not write prediction log")
    return JSONResponse(result)


if Path("ui").is_dir():
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
