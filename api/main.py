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
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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
REPORT_DIR = Path(os.getenv("REPORT_DIR", "reports"))
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


@app.get("/dashboard")
def dashboard_data():
    """Small live summary for the dashboard, derived from the same log used by monitoring."""
    rows = deque(maxlen=10000)
    malformed = 0
    total_predictions = 0
    log_path = Path(PRED_LOG)
    if log_path.exists():
        try:
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            rows.append(row)
                            total_predictions += 1
                    except (json.JSONDecodeError, TypeError):
                        malformed += 1
        except OSError:
            log.warning("could not read prediction log %s", log_path)

    now = datetime.now(timezone.utc)
    days = [(now.date() - timedelta(days=offset)) for offset in range(6, -1, -1)]
    daily = {day.isoformat(): 0 for day in days}
    class_counts = Counter()
    confidences, latencies = [], []
    low_count = 0
    for row in rows:
        name = row.get("top1_class")
        if name:
            class_counts[str(name)] += 1
        try:
            value = float(row.get("confidence"))
            if np.isfinite(value):
                confidences.append(value)
        except (TypeError, ValueError):
            pass
        try:
            value = float(row.get("latency_ms"))
            if np.isfinite(value):
                latencies.append(value)
        except (TypeError, ValueError):
            pass
        low_count += row.get("low_confidence") is True
        try:
            stamp = datetime.fromtimestamp(float(row.get("ts")), tz=timezone.utc)
            key = stamp.date().isoformat()
            if key in daily:
                daily[key] += 1
        except (TypeError, ValueError, OSError, OverflowError):
            pass

    metrics = {}
    metrics_path = REPORT_DIR / "metrics.json"
    try:
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("could not read evaluation metrics %s", metrics_path)

    classification = {}
    report_path = REPORT_DIR / "classification_report.json"
    try:
        if report_path.exists():
            classification = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("could not read classification report %s", report_path)

    baseline = {}
    baseline_path = REPORT_DIR / "v5_plantdoc_metrics.json"
    try:
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("could not read v5 baseline metrics %s", baseline_path)

    monitor = {}
    monitor_path = REPORT_DIR / "monitoring" / "summary.json"
    try:
        if monitor_path.exists():
            monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("could not read monitoring summary %s", monitor_path)

    sorted_latency = sorted(latencies)
    p95_index = int(round((len(sorted_latency) - 1) * 0.95)) if sorted_latency else None
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "total_predictions": total_predictions,
        "analytics_window": len(rows),
        "malformed_log_lines": malformed,
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "low_confidence_rate": round(low_count / len(rows), 4) if rows else None,
        "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": round(sorted_latency[p95_index], 2) if p95_index is not None else None,
        "daily_predictions": [{"date": day, "count": daily[day]} for day in days],
        "class_distribution": [
            {"class": name, "count": count, "share": round(count / len(rows), 4)}
            for name, count in class_counts.most_common()
        ],
        "evaluation": {
            "validation_accuracy": metrics.get("val_accuracy"),
            "validation_macro_f1": metrics.get("val_macro_f1"),
            "validation_mean_confidence": metrics.get("val_mean_confidence"),
            "test_accuracy": metrics.get("test_accuracy"),
            "test_macro_f1": metrics.get("test_macro_f1"),
            "test_mean_confidence": metrics.get("test_mean_confidence"),
            "test_report": classification.get("test", {}),
            "validation_samples": (classification.get("val", {}).get("macro avg") or {}).get("support"),
            "test_samples": (classification.get("test", {}).get("macro avg") or {}).get("support"),
            "baseline": {
                "name": "v5",
                "test_accuracy": baseline.get("top1_accuracy"),
                "top3_accuracy": baseline.get("top3_accuracy"),
                "test_macro_f1": (baseline.get("classification_report", {}).get("macro avg") or {}).get("f1-score"),
                "test_samples": baseline.get("n_images"),
                "dataset": baseline.get("dataset"),
            },
        },
        "monitoring": {
            "status": monitor.get("status"),
            "generated": monitor.get("generated"),
            "log_file": monitor.get("log_file"),
            "alerts": monitor.get("alerts", []),
            "retrain_triggers": monitor.get("retrain_triggers", []),
        },
        "model": {
            "loaded": state["model"] is not None,
            "version": (state.get("cfg") or {}).get("version"),
            "img_size": (state.get("cfg") or {}).get("img_size"),
            "confidence_threshold": (state.get("cfg") or {}).get("confidence_threshold"),
            "classes": state.get("classes") or [],
        },
    }


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


UI_DIR = Path(os.getenv("UI_DIR", "ui/dist"))
if UI_DIR.is_dir() and (UI_DIR / "index.html").exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
