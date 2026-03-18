from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "saved_model" / "ResNet50V2_COVID-19_Radiography.h5"

# Keras version compatibility fix
# Models saved with Keras 3.3+ store 'quantization_config' in the Dense layer
# config. Older Keras versions raise an error on that unknown key. The patch
# below strips it transparently before deserialization so the model loads on
# any Keras 3.x build.
_original_dense_from_config = tf.keras.layers.Dense.from_config


@classmethod  # type: ignore[misc]
def _patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _original_dense_from_config.__func__(cls, config)


tf.keras.layers.Dense.from_config = _patched_dense_from_config
# ─────────────────────────────────────────────────────────────────────────────

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia",
]


def _target_size_from_model() -> tuple[int, int]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = int(input_shape[1]) if input_shape[1] is not None else 299
    width = int(input_shape[2]) if input_shape[2] is not None else 299
    return (width, height)


TARGET_SIZE = _target_size_from_model()

# Build Grad-CAM subgraph once at startup
# Rebuilding tf.keras.Model on every request is the primary cause of the
# ~16 s latency. We locate the last Conv2D layer and construct the auxiliary
# model a single time, then compile the gradient pass with @tf.function so
# TensorFlow traces it once and runs it as a static graph on every call.
_last_conv_layer = next(
    layer for layer in model.layers[::-1] if isinstance(layer, tf.keras.layers.Conv2D)
)
_grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[_last_conv_layer.output, model.output],
)

@tf.function(input_signature=[tf.TensorSpec(shape=(1, TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=tf.float32)])
def _compute_gradcam(batched: tf.Tensor):
    with tf.GradientTape() as tape:
        tape.watch(batched)
        conv_outputs, predictions = _grad_model(batched, training=False)
        class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]
    grads = tape.gradient(class_channel, conv_outputs)
    return conv_outputs, grads, predictions
# ─────────────────────────────────────────────────────────────────────────────


def preprocess_image(img: Image.Image, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    image = img.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict(img: Image.Image):
    input_tensor = preprocess_image(img)
    # Direct __call__ avoids the batching/logging overhead of model.predict()
    preds = model(input_tensor, training=False).numpy()
    probs = preds[0]
    class_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[class_idx], confidence, prob_dict


def gradcam(img: Image.Image, interpolant: float = 0.5) -> np.ndarray:
    if not 0.0 <= interpolant <= 1.0:
        raise ValueError("interpolant must be between 0.0 and 1.0")

    image_resized = img.convert("RGB").resize(TARGET_SIZE)
    input_tensor = np.array(image_resized).astype("float32") / 255.0
    batched = tf.constant(np.expand_dims(input_tensor, axis=0))

    conv_outputs, grads, _ = _compute_gradcam(batched)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs[0].numpy()

    heatmap = np.sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    heatmap = cv2.resize(heatmap.astype("float32"), TARGET_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_uint8 = np.uint8(np.clip(input_tensor * 255.0, 0, 255))
    overlay = cv2.addWeighted(original_uint8, interpolant, heatmap_color, 1.0 - interpolant, 0)
    return overlay