import json

import keras
import numpy as np
import pytest
from PIL import Image

CLASSES = ["Pepper__bell___Bacterial_spot", "Potato___Early_blight", "Potato___Late_blight", "Tomato_Bacterial_spot",
           "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Septoria_leaf_spot", "Tomato_healthy"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """Fallback layout from src/train.py: model.keras + classes.json, [0, 1] input."""
    d = tmp_path_factory.mktemp("model")
    inp = keras.Input((32, 32, 3))
    out = keras.layers.Dense(3, activation="softmax")(keras.layers.Flatten()(inp))
    keras.Model(inp, out).save(d / "model.keras")
    (d / "classes.json").write_text(json.dumps({"classes": ["a", "b", "c"], "img_size": 32}))
    return d


@pytest.fixture(scope="session")
def v5_model_dir(tmp_path_factory):
    """Stand-in for the v5 model: raw 0-255 input, Rescaling inside, mixed precision, 8 classes.

    The single Dense layer reads the mean value of the image, so a mid-grey image gives a
    flat (low-confidence) prediction and the output depends on the input range.
    """
    d = tmp_path_factory.mktemp("v5")
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        inp = keras.Input((48, 48, 3))
        x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inp)
        x = keras.layers.RandomFlip("horizontal")(x)          # augmentation inside the model, like v5
        x = keras.layers.GlobalAveragePooling2D()(x)
        out = keras.layers.Dense(8, activation="softmax", dtype="float32")(x)
        model = keras.Model(inp, out)
    finally:
        keras.mixed_precision.set_global_policy("float32")
    model.save(d / "mobilenetv2.keras")
    (d / "class_names.json").write_text(json.dumps(CLASSES))
    meta = {"version": "test-v5", "model_file": "mobilenetv2.keras", "class_names_file": "class_names.json",
            "img_size": 48, "input_range": "0_255", "center_crop": True, "confidence_threshold": 0.7, "top_k": 3,
            "low_confidence_message": "Not sure. Take a clearer photo of one leaf."}
    (d / "model_meta.json").write_text(json.dumps(meta))
    return d


@pytest.fixture()
def jpeg_bytes(tmp_path):
    p = tmp_path / "x.jpg"
    Image.fromarray(np.random.randint(0, 255, (50, 40, 3), dtype="uint8")).save(p)
    return p.read_bytes()
