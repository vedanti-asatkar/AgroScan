import json

import keras
import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    inp = keras.Input((32, 32, 3))
    x = keras.layers.Flatten()(inp)
    out = keras.layers.Dense(3, activation="softmax")(x)
    keras.Model(inp, out).save(d / "model.keras")
    (d / "classes.json").write_text(json.dumps({"classes": ["a", "b", "c"], "img_size": 32}))
    return d


@pytest.fixture()
def jpeg_bytes(tmp_path):
    p = tmp_path / "x.jpg"
    Image.fromarray(np.random.randint(0, 255, (50, 40, 3), dtype="uint8")).save(p)
    return p.read_bytes()
