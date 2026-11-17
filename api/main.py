"""AgroScan inference API.

The model and its preprocessing are described by models/model_meta.json (file names, input size,
pixel range, centre crop, confidence threshold). Without that file the API falls back to
models/model.keras + models/classes.json from the repo's own training pipeline.
"""
import io
import base64
import json
import logging
import os
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import keras
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.preprocess import prepare_for_model

log = logging.getLogger("agroscan")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
PRED_LOG = os.getenv("PREDICTION_LOG", "logs/predictions.jsonl")
FEEDBACK_LOG = os.getenv("FEEDBACK_LOG", "logs/feedback.jsonl")
REPORT_DIR = Path(os.getenv("REPORT_DIR", "reports"))
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
LIME_NUM_SAMPLES = max(50, int(os.getenv("LIME_NUM_SAMPLES", "200")))
LIME_BATCH_SIZE = max(1, int(os.getenv("LIME_BATCH_SIZE", "32")))
LIME_SUPERPIXELS = max(20, int(os.getenv("LIME_SUPERPIXELS", "80")))

state = {"model": None, "classes": None, "cfg": None, "registry_model": None, "registry_version": None}
history_lock = threading.RLock()


def load_monitor_config():
    path = Path(os.getenv("MONITORING_CONFIG", "configs/monitoring.yaml"))
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        log.exception("could not read monitoring configuration %s", path)
        return {}


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


class PredictionFeedback(BaseModel):
    prediction_id: str
    actual_class: str


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
    predictions_by_id = {}
    for row in rows:
        if row.get("ts") is not None:
            predictions_by_id[repr(row["ts"])] = row
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

    feedback_rows = []
    feedback_ids = []
    feedback_path = Path(FEEDBACK_LOG)
    try:
        if feedback_path.exists():
            with feedback_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prediction = predictions_by_id.get(item.get("prediction_id"))
                    actual = item.get("actual_class")
                    if prediction is None or actual not in (state.get("classes") or []):
                        continue
                    current_version = (state.get("cfg") or {}).get("version")
                    if current_version and prediction.get("model_version") != current_version:
                        continue
                    feedback_ids.append(item["prediction_id"])
                    feedback_rows.append({"predicted": prediction.get("top1_class"), "actual": actual})
    except OSError:
        log.warning("could not read prediction feedback %s", feedback_path)

    evaluated_labels = {row["actual"] for row in feedback_rows}
    evaluated_labels.update(row["predicted"] for row in feedback_rows if row["predicted"])
    evaluated_labels = sorted(evaluated_labels)
    per_class = []
    for label in evaluated_labels:
        tp = sum(row["predicted"] == label and row["actual"] == label for row in feedback_rows)
        fp = sum(row["predicted"] == label and row["actual"] != label for row in feedback_rows)
        fn = sum(row["predicted"] != label and row["actual"] == label for row in feedback_rows)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": label, "precision": round(precision, 4), "recall": round(recall, 4),
                          "f1": round(f1, 4), "support": support})
    correct = sum(row["predicted"] == row["actual"] for row in feedback_rows)
    live_accuracy = correct / len(feedback_rows) if feedback_rows else None
    live_macro_f1 = sum(row["f1"] for row in per_class) / len(per_class) if per_class else None
    sorted_latency = sorted(latencies)
    p95_index = int(round((len(sorted_latency) - 1) * 0.95)) if sorted_latency else None

    # Build explainable, current warnings from configured thresholds and observed logs.
    monitor_cfg = load_monitor_config()
    alert_cfg = monitor_cfg.get("alerts", {})
    window_cfg = monitor_cfg.get("window", {})
    live_alerts = []
    min_labels = int(window_cfg.get("min_labeled_predictions", 20))
    if len(feedback_rows) >= min_labels and live_accuracy is not None:
        limit = float(alert_cfg.get("live_accuracy_warn", 0.60))
        if live_accuracy < limit:
            live_alerts.append({
                "id": "live_accuracy",
                "category": "Verified performance",
                "severity": "warning",
                "title": "Verified accuracy is below its configured floor",
                "explanation": (
                    "This measures correctness on user-verified predictions. A drop can be caused by drift, "
                    "label issues, or model limitations; it does not identify the cause by itself."
                ),
                "evidence": (
                    f"{correct}/{len(feedback_rows)} correct ({live_accuracy:.1%}); "
                    f"configured floor {limit:.1%}; minimum sample {min_labels} met."
                ),
                "action": "Review the verified examples and labels, then investigate input or class-specific changes.",
            })
    elif len(feedback_rows) < min_labels:
        live_alerts.append({
            "id": "live_accuracy_sample",
            "category": "Verified performance",
            "severity": "info",
            "title": "More verified labels are needed",
            "explanation": (
                "Live accuracy is suppressed until the configured minimum sample is reached, "
                "to avoid reacting to a few labels."
            ),
            "evidence": f"{len(feedback_rows)}/{min_labels} verified predictions.",
            "action": "Keep verifying prediction results; confidence alone is not accuracy.",
        })

    min_predictions = int(window_cfg.get("min_predictions", 20))
    reference_path = Path(monitor_cfg.get("paths", {}).get("reference", "reports/monitoring/reference.json"))
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        reference = None
    if len(rows) >= min_predictions:
        low_rate = low_count / len(rows) if rows else None
        low_limit = alert_cfg.get("low_confidence_rate_warn")
        if low_rate is not None and low_limit is not None and low_rate > float(low_limit):
            live_alerts.append({
                "id": "low_confidence_rate", "category": "Prediction signal", "severity": "warning",
                "title": "Low-confidence predictions are elevated",
                "explanation": (
                    "This can indicate unfamiliar or ambiguous inputs. It is an early warning signal, "
                    "not a direct measurement of correctness."
                ),
                "evidence": (
                    f"{low_count}/{len(rows)} predictions ({low_rate:.1%}) are below the model "
                    f"confidence threshold; configured warning {float(low_limit):.1%}."
                ),
                "action": "Review recent low-confidence images and collect verified labels.",
            })

        brightness = [float(r["brightness"]) for r in rows if isinstance(r.get("brightness"), (int, float))]
        if reference and brightness and reference.get("brightness_mean") is not None:
            ref_mean = float(reference["brightness_mean"])
            ref_std = float(reference.get("brightness_std") or 0)
            current_mean = sum(brightness) / len(brightness)
            z = abs(current_mean - ref_mean) / ref_std if ref_std else (
                0 if current_mean == ref_mean else float("inf")
            )
            z_limit = alert_cfg.get("brightness_drift_zscore_warn")
            if z_limit is not None and z > float(z_limit):
                live_alerts.append({
                    "id": "brightness_drift", "category": "Input drift", "severity": "warning",
                    "title": "Image brightness shifted from the reference period",
                    "explanation": (
                        "This detects a change in one input feature. It suggests possible data drift, "
                        "but does not prove model quality has fallen."
                    ),
                    "evidence": (
                        f"Current mean {current_mean:.2f} vs reference {ref_mean:.2f}; "
                        f"z-score {z:.2f}, configured warning {float(z_limit):.2f}; "
                        f"{len(brightness)} images."
                    ),
                    "action": "Check capture conditions and review verified accuracy alongside this input shift.",
                })
        reference_mix = (reference or {}).get("class_mix") or {}
        if reference_mix and class_counts:
            total = sum(class_counts.values())
            current_mix = {name: count / total for name, count in class_counts.items()}
            keys = set(current_mix) | set(reference_mix)
            tv = 0.5 * sum(abs(current_mix.get(name, 0) - float(reference_mix.get(name, 0))) for name in keys)
            tv_limit = monitor_cfg.get("retrain", {}).get("class_mix_tv_distance_max")
            if tv_limit is not None and tv > float(tv_limit):
                live_alerts.append({
                    "id": "class_mix_shift", "category": "Prediction drift", "severity": "warning",
                    "title": "Predicted class mix differs from the reference",
                    "explanation": (
                        "The proportions of predicted classes have shifted. This may reflect changed field "
                        "conditions, traffic mix, or model behavior; predictions are not ground-truth labels."
                    ),
                    "evidence": (
                        f"Total variation distance {tv:.3f}; configured investigation threshold "
                        f"{float(tv_limit):.3f}; {total} predictions."
                    ),
                    "action": (
                        "Compare class-specific verified results and inspect representative images "
                        "before deciding on retraining."
                    ),
                })

        p95_limit = alert_cfg.get("latency_p95_ms_warn")
        if p95_index is not None and p95_limit is not None and sorted_latency[p95_index] > float(p95_limit):
            live_alerts.append({
                "id": "latency_p95", "category": "Service health", "severity": "warning",
                "title": "Inference latency is above its configured limit",
                "explanation": "This is a serving performance signal, separate from prediction quality or model drift.",
                "evidence": (
                    f"p95 latency {sorted_latency[p95_index]:.1f} ms; "
                    f"configured limit {float(p95_limit):.1f} ms; {len(sorted_latency)} timings."
                ),
                "action": "Inspect API resource use and recent deployment or traffic changes.",
            })

    drift_alerts = [item for item in live_alerts if item.get("id") in {"brightness_drift", "class_mix_shift"}]
    has_brightness_reference = bool(reference and reference.get("brightness_mean") is not None)
    has_class_mix_reference = bool((reference or {}).get("class_mix"))
    if len(rows) < min_predictions:
        drift_status = "insufficient_data"
    elif not has_brightness_reference and not has_class_mix_reference:
        drift_status = "baseline_missing"
    elif drift_alerts:
        drift_status = "warning"
    elif not (has_brightness_reference and has_class_mix_reference):
        drift_status = "partial_baseline"
    else:
        drift_status = "watching"

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "total_predictions": total_predictions,
        "analytics_window": len(rows),
        "malformed_log_lines": malformed,
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "low_confidence_rate": round(low_count / len(rows), 4) if rows else None,
        "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": round(sorted_latency[p95_index], 2) if p95_index is not None else None,
        "daily_predictions": [{"date": day.isoformat(), "count": daily[day.isoformat()]} for day in days],
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
        "live_performance": {
            "model_version": (state.get("cfg") or {}).get("version"),
            "labeled_predictions": len(feedback_rows),
            "accuracy": round(live_accuracy, 4) if live_accuracy is not None else None,
            "macro_f1": round(live_macro_f1, 4) if live_macro_f1 is not None else None,
            "correct_predictions": correct,
            "feedback_rate": round(len(feedback_rows) / len(rows), 4) if rows else None,
            "per_class": per_class,
            "prediction_ids": feedback_ids,
        },
        "live_alerts": live_alerts,
        "alert_thresholds": {
            "live_accuracy_warn": alert_cfg.get("live_accuracy_warn"),
            "min_labeled_predictions": min_labels,
            "min_predictions": min_predictions,
        },
        "drift_monitor": {
            "status": drift_status,
            "window_predictions": len(rows),
            "minimum_predictions": min_predictions,
            "reference_path": str(reference_path),
            "brightness_reference_available": has_brightness_reference,
            "class_mix_reference_available": has_class_mix_reference,
            "alerts": drift_alerts,
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
    event_ts = time.time()
    result = {
        "prediction_id": repr(event_ts),
        "predicted_class": top[0]["class"],
        "confidence": top[0]["confidence"],
        "low_confidence": bool(low),
        "message": cfg["low_confidence_message"] if low else None,
        "top_k": top,
        "model_version": cfg["version"],
    }

    try:  # prediction log feeds drift / quality monitoring
        with history_lock:
            Path(PRED_LOG).parent.mkdir(parents=True, exist_ok=True)
            with open(PRED_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": event_ts, "model_version": cfg["version"],
                                    "top1_class": result["predicted_class"], "confidence": result["confidence"],
                                    "top3": result["top_k"], "low_confidence": result["low_confidence"],
                                    "latency_ms": round(latency_ms, 1), "brightness": round(brightness, 2)})
                        + "\n")
    except OSError:
        log.warning("could not write prediction log")
    return JSONResponse(result)


def _explanation_predict(images):
    """Predict XAI perturbations in the same tensor range produced by prepare_for_model."""
    batch = np.asarray(images, dtype=np.float32)
    # The uploaded image is prepared once with the exact /predict preprocessing. XAI
    # masks operate in that already resized/cropped model input space; only restore the
    # configured numeric range after masking when displaying pixels as 0..255 RGB.
    if state["cfg"]["input_range"] == "0_1":
        batch = batch / 255.0
    # Calling the model directly avoids `predict()`'s per-batch data adapter setup,
    # which is costly because LIME invokes this callback several times.
    return np.asarray(state["model"](batch, training=False), dtype=np.float64)


def _overlay_png(image, heat, mode):
    """Create a compact positive/negative heat overlay and return a data URI."""
    from matplotlib import colormaps

    rgb = np.asarray(image.convert("RGB").resize((heat.shape[1], heat.shape[0])), dtype=np.float32) / 255
    scale = float(np.max(np.abs(heat))) or 1.0
    normalized = np.clip(heat / scale, -1, 1)
    if mode == "lime":
        mask = normalized > 0.15
        overlay = rgb.copy()
        overlay[mask] = overlay[mask] * 0.42 + np.array([0.20, 0.80, 0.42]) * 0.58
    else:
        colors = colormaps["coolwarm"]((normalized + 1) / 2)[..., :3]
        strength = np.clip(np.abs(normalized), 0, 1)[..., None] * 0.68
        overlay = rgb * (1 - strength) + colors * strength
    out = Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/explain")
async def explain(file: UploadFile = File(...), method: str = Form(...)):
    """Explain the same model prediction using LIME superpixels or SHAP image masking."""
    if state["model"] is None:
        raise HTTPException(503, "model not loaded")
    method = method.lower().strip()
    if method not in {"lime", "shap"}:
        raise HTTPException(422, "method must be 'lime' or 'shap'")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.verify()
        prepared = prepare_for_model(data, state["cfg"]["img_size"], state["cfg"]["input_range"],
                                     state["cfg"]["center_crop"]).numpy()
        # Keep a display image aligned with the exact resized/cropped model input.
        display_pixels = prepared * 255 if state["cfg"]["input_range"] == "0_1" else prepared
        display = Image.fromarray(np.uint8(np.clip(display_pixels, 0, 255)))
        pixels = np.asarray(display)
    except Exception as exc:
        raise HTTPException(400, "could not decode image (use jpg or png)") from exc

    try:
        baseline = _explanation_predict([pixels])[0]
        class_index = int(np.argmax(baseline))
        if method == "lime":
            try:
                from lime.lime_image import LimeImageExplainer
                from skimage.segmentation import mark_boundaries, slic
            except ImportError as exc:
                raise HTTPException(503, "LIME requires lime and scikit-image; install requirements-xai.txt") from exc
            segments = slic(pixels, n_segments=LIME_SUPERPIXELS, compactness=10, sigma=1, start_label=0,
                            channel_axis=-1)
            explanation = LimeImageExplainer(random_state=42).explain_instance(
                pixels, _explanation_predict, labels=(class_index,), hide_color=0,
                num_samples=LIME_NUM_SAMPLES, batch_size=LIME_BATCH_SIZE,
                segmentation_fn=lambda image: segments)
            weights = dict(explanation.local_exp[class_index])
            heat = np.zeros(segments.shape, dtype=np.float32)
            for segment_id, weight in weights.items():
                heat[segments == segment_id] = weight
            positive = sorted(((int(i), float(w)) for i, w in weights.items() if w > 0), key=lambda row: row[1], reverse=True)[:8]
            boundary = mark_boundaries(np.asarray(display) / 255.0, segments, color=(0.1, 0.9, 0.35), mode="thick")
            result_image = Image.fromarray(np.uint8(boundary * 255))
            # Blend each positive superpixel according to its relative LIME weight.
            positive_scale = max((weight for weight in weights.values() if weight > 0), default=1.0)
            strength = np.zeros(segments.shape, dtype=np.float32)
            for segment_id, weight in weights.items():
                if weight > 0:
                    strength[segments == segment_id] = min(weight / positive_scale, 1.0)
            alpha_mask = Image.fromarray(np.uint8(strength * 110), "L")
            green = Image.new("RGBA", result_image.size, (49, 191, 105, 0))
            green.putalpha(alpha_mask)
            result_image = Image.alpha_composite(result_image.convert("RGBA"), green).convert("RGB")
            buf = io.BytesIO(); result_image.save(buf, format="PNG")
            explanation_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            metadata = {"method": "lime", "superpixels": int(segments.max() + 1),
                        "important_superpixels": [{"segment": i, "weight": round(w, 5)} for i, w in positive],
                        "samples": LIME_NUM_SAMPLES, "batch_size": LIME_BATCH_SIZE}
            description = "Green superpixels support the predicted class; brighter outlined regions have stronger positive LIME weights."
        else:
            try:
                import shap
            except ImportError as exc:
                raise HTTPException(503, "SHAP requires shap; install requirements-xai.txt") from exc
            try:
                masker = shap.maskers.Image("blur(16,16)", pixels.shape)
            except ModuleNotFoundError as exc:
                if exc.name == "cv2":
                    raise HTTPException(
                        503,
                        "SHAP image explanations need OpenCV. In AgroScan's active Python environment, "
                        "run 'python -m pip install opencv-python-headless' and restart the API.",
                    ) from exc
                raise
            explainer = shap.Explainer(_explanation_predict, masker, algorithm="partition",
                                       output_names=state["classes"])
            values = explainer(pixels[None], max_evals=160, batch_size=16)
            raw = np.asarray(values.values)
            # SHAP image outputs are (batch, H, W, RGB, classes).
            if raw.ndim == 5:
                contribution = raw[0, ..., class_index].mean(axis=-1)
            elif raw.ndim == 4:
                contribution = raw[0].mean(axis=-1)
            else:
                raise ValueError(f"unexpected SHAP image output shape {raw.shape}")
            heat = np.asarray(contribution, dtype=np.float32)
            explanation_uri = _overlay_png(display, heat, "shap")
            flat = heat.ravel()
            metadata = {"method": "shap", "max_evals": 160,
                        "positive_pixels": int(np.count_nonzero(flat > 0)),
                        "negative_pixels": int(np.count_nonzero(flat < 0)),
                        "positive_strength": round(float(np.maximum(heat, 0).sum()), 5),
                        "negative_strength": round(float(np.minimum(heat, 0).sum()), 5)}
            description = "Red regions increase the selected class score; blue regions decrease it. Color intensity shows relative contribution."
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("%s explanation failed", method)
        raise HTTPException(500, f"{method.upper()} explanation failed: {exc}") from exc

    return {"predicted_class": state["classes"][class_index],
            "confidence": round(float(baseline[class_index]), 4), "method": method,
            "explanation_image": explanation_uri, "description": description, "metadata": metadata}


@app.delete("/history")
def clear_scan_history():
    """Clear shared live scan and feedback data so every dashboard view resets together."""
    paths = [Path(PRED_LOG), Path(FEEDBACK_LOG), REPORT_DIR / "monitoring" / "summary.json",
             REPORT_DIR / "monitoring" / "healthy_report.json",
             REPORT_DIR / "monitoring" / "retrain_decision.json"]
    try:
        with history_lock:
            for path in paths:
                path.unlink(missing_ok=True)
    except OSError as exc:
        log.exception("could not clear scan history")
        raise HTTPException(500, "could not clear all server scan data") from exc
    return {"status": "cleared", "cleared": [str(path) for path in paths]}


@app.post("/feedback")
def record_feedback(payload: PredictionFeedback):
    """Record a user-confirmed label for a prediction to calculate live accuracy."""
    if state["classes"] is None:
        raise HTTPException(503, "model classes are unavailable")
    if payload.actual_class not in state["classes"]:
        raise HTTPException(422, "actual_class must be one of the model classes")

    prediction = None
    prediction_path = Path(PRED_LOG)
    try:
        if prediction_path.exists():
            with prediction_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if repr(row.get("ts")) == payload.prediction_id:
                        prediction = row
    except OSError as exc:
        log.warning("could not read prediction log %s: %s", prediction_path, exc)
    if prediction is None:
        raise HTTPException(404, "prediction not found; it may have expired from the server log")

    feedback_path = Path(FEEDBACK_LOG)
    try:
        with history_lock:
            if feedback_path.exists():
                with feedback_path.open(encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if row.get("prediction_id") == payload.prediction_id:
                            raise HTTPException(409, "feedback was already recorded for this prediction")
            feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with feedback_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "prediction_id": payload.prediction_id,
                    "predicted_class": prediction.get("top1_class"),
                    "actual_class": payload.actual_class,
                    "model_version": prediction.get("model_version"),
                    "ts": time.time(),
                }) + "\n")
    except HTTPException:
        raise
    except OSError as exc:
        log.error("could not write prediction feedback: %s", exc)
        raise HTTPException(500, "could not save feedback") from exc
    return {"status": "recorded", "prediction_id": payload.prediction_id}


UI_DIR = Path(os.getenv("UI_DIR", "ui/dist"))
if UI_DIR.is_dir() and (UI_DIR / "index.html").exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")
