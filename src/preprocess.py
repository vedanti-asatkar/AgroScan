"""Single source of truth for image preprocessing (used by training AND serving).

Contract: RGB image -> resize to (size, size) -> float32 in [0, 1].
Models built by src/train.py rescale to [-1, 1] internally for MobileNetV2,
so callers never apply model-specific scaling.
"""
import tensorflow as tf


def decode_and_resize(image_bytes, size):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size))
    return tf.cast(img, tf.float32) / 255.0


def load_path(path, size):
    return decode_and_resize(tf.io.read_file(path), size)
