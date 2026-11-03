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


def prepare_for_model(image_bytes, size, input_range="0_1", center_crop=False):
    """Serving preprocessing driven by the model's metadata.

    input_range "0_255": raw pixel values (the model rescales internally);
    input_range "0_1": divided by 255 (models trained by src/train.py).
    center_crop: take the centre square (shorter side) before resizing.
    """
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    if center_crop:
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        side = tf.minimum(h, w)
        img = tf.image.crop_to_bounding_box(img, (h - side) // 2, (w - side) // 2, side, side)
    img = tf.cast(tf.image.resize(img, (size, size), method="bilinear"), tf.float32)
    return img if input_range == "0_255" else img / 255.0
