"""AgroScan inference API."""
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

from src.preprocess import decode_and_resize

log = logging.getLogger("agroscan")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.keras")
LEGACY_MODEL_PATH = "models/mobilenetv2_combined_v2.h5"
PRED_LOG = os.getenv("PREDICTION_LOG", "logs/predictions.jsonl")
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
LEGACY_CLASSES = [
    "Pepper__bell___Bacterial_spot", "Potato___Early_blight", "Potato___Late_blight",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot", "Tomato_healthy",
]

state = {"model": None, "classes": LEGACY_CLASSES, "img_size": 224, "model_path": None, "version": None}


def load_state():
    path = Path(MODEL_PATH)
    if not path.exists() and Path(LEGACY_MODEL_PATH).exists():
        path = Path(LEGACY_MODEL_PATH)
    if not path.exists():
        log.error("No model found at %s (run `dvc pull` / train first)", MODEL_PATH)
        return
    meta_file = path.with_name("classes.json")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        state.update(classes=meta["classes"], img_size=meta.get("img_size", 224), version=meta.get("mlflow_run_id"))
    state.update(model=keras.models.load_model(path, compile=False), model_path=str(path))
    log.info("Loaded %s (%d classes)", path, len(state["classes"]))


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
    return {"status": "ready", "model_path": state["model_path"], "model_version": state["version"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES} bytes")
    try:
        x = decode_and_resize(data, state["img_size"]).numpy()[None]
    except Exception:
        raise HTTPException(400, "could not decode image (use jpg or png)")

    t0 = time.perf_counter()
    probs = state["model"].predict(x, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    idx = int(np.argmax(probs))
    result = {"predicted_class": state["classes"][idx], "confidence": round(float(probs[idx]), 4)}

    try:  # prediction log feeds drift / quality monitoring
        Path(PRED_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(PRED_LOG, "a") as f:
            f.write(json.dumps({"ts": time.time(), "filename": file.filename, "latency_ms": round(latency_ms, 1),
                                "model_version": state["version"], **result}) + "\n")
    except OSError:
        log.warning("could not write prediction log")
    return JSONResponse(result)


if Path("ui").is_dir():
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
