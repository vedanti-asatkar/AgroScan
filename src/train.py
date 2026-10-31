"""Train MobileNetV2 transfer-learning classifier and track everything in MLflow."""
import hashlib
import subprocess
from pathlib import Path

import keras
import mlflow
import tensorflow as tf

from src.common import load_params, parse_args, read_manifest, write_json
from src.preprocess import load_path


def make_dataset(rows, class_idx, p, training):
    size, bs = p["model"]["img_size"], p["train"]["batch_size"]
    if training:
        rep = p["data"]["plantdoc_oversample"]
        rows = [r for r in rows for _ in range(rep if r["source"] == "plantdoc" else 1)]
        cap = p["train"].get("max_train_images")
        if cap:
            import random
            random.Random(p["train"]["seed"]).shuffle(rows)
            rows = rows[:cap]
    paths = [r["path"] for r in rows]
    labels = [class_idx[r["label"]] for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=p["train"]["seed"], reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (load_path(x, size), y), num_parallel_calls=tf.data.AUTOTUNE).batch(bs)
    if training:
        aug = keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
            keras.layers.RandomContrast(0.2),
        ])
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE), len(paths)


def build_model(p, n_classes):
    s, m = p["model"]["img_size"], p["model"]
    inputs = keras.Input((s, s, 3), name="image_0_1")
    x = keras.layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inputs)  # MobileNetV2 range
    base = keras.applications.MobileNetV2((s, s, 3), include_top=False, weights=m["weights"], alpha=m["alpha"])
    if m.get("bn_momentum"):  # faster-adapting BN statistics for tiny from-scratch runs
        for layer in base.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.momentum = m["bn_momentum"]
    scratch = m["weights"] is None
    base.trainable = scratch  # frozen random weights are useless; only freeze pretrained ones
    x = base(x) if scratch else base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(m["dropout"])(x)
    out = keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, out), base


class MlflowEpochLogger(keras.callbacks.Callback):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=self.offset + epoch)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        elif not isinstance(v, list):
            yield f"{prefix}{k}", v


def main(params_path):
    p = load_params(params_path)
    keras.utils.set_random_seed(p["train"]["seed"])
    classes = p["classes"]
    class_idx = {c: i for i, c in enumerate(classes)}
    prep = Path(p["data"]["prepared_dir"])
    train_ds, n_train = make_dataset(read_manifest(prep / "train.csv"), class_idx, p, True)
    val_ds, n_val = make_dataset(read_manifest(prep / "val.csv"), class_idx, p, False)

    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment"])
    with mlflow.start_run(run_name="train") as run:
        mlflow.log_params(dict(flatten(p)))
        mlflow.set_tags({"git_sha": git_sha(), "data_train_md5": file_md5(prep / "train.csv"),
                         "data_test_md5": file_md5(prep / "test.csv")})
        mlflow.log_metrics({"n_train_samples": n_train, "n_val_samples": n_val})

        model, base = build_model(p, len(classes))
        t = p["train"]

        def es():
            return keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=t["early_stopping_patience"],
                                                 restore_best_weights=True)

        model.compile(optimizer=keras.optimizers.Adam(t["learning_rate"]),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(train_ds, validation_data=val_ds, epochs=t["epochs"],
                      callbacks=[es(), MlflowEpochLogger()])
        done = len(h.history["loss"])
        if t["fine_tune_epochs"] > 0 and p["model"]["weights"]:
            base.trainable = True
            for layer in base.layers[:-t["fine_tune_layers"]]:
                layer.trainable = False
            model.compile(optimizer=keras.optimizers.Adam(t["fine_tune_learning_rate"]),
                          loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            h = model.fit(train_ds, validation_data=val_ds, epochs=t["fine_tune_epochs"],
                          callbacks=[es(), MlflowEpochLogger(done)])

        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        mlflow.log_metrics({"final_val_loss": val_loss, "final_val_accuracy": val_acc})

        Path("models").mkdir(exist_ok=True)
        model.save("models/model.keras")
        write_json({"classes": classes, "img_size": p["model"]["img_size"], "input_range": [0, 1],
                    "git_sha": git_sha(), "mlflow_run_id": run.info.run_id}, "models/classes.json")
        write_json({"val_accuracy": val_acc, "val_loss": val_loss}, "reports/train_metrics.json")

        mlflow.log_artifact("models/classes.json")
        mlflow.log_artifact(params_path)
        try:
            mlflow.keras.log_model(model, name="model", registered_model_name=p["mlflow"]["registered_model"])
        except Exception as e:  # registry needs a DB-backed tracking store
            print("model registry skipped:", e)
            mlflow.log_artifact("models/model.keras")
        print("run", run.info.run_id, "val_acc", round(val_acc, 4))


if __name__ == "__main__":
    main(parse_args("Train model").params)
