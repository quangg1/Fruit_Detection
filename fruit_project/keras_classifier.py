"""Keras/TensorFlow fruit classifier for `.keras` MobileNetV2 checkpoints.

This module is intentionally imported lazily (TensorFlow import inside functions)
so the rest of the project can run without TF unless the user enables the Keras backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _find_latest_keras_model(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    candidates = list(model_dir.glob("*.keras"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_labels_json(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    # Common names in notebooks: labels.json, label_map.json, classes.json...
    for name in ["labels.json", "label_map.json", "classes.json", "class_map.json"]:
        p = model_dir / name
        if p.is_file():
            return p
    # fallback: any json containing "label" in name
    candidates = [p for p in model_dir.glob("*.json") if "label" in p.name.lower()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_keras_pack(
    model_dir: Path,
    *,
    fallback_classes: list[str],
) -> dict[str, Any] | None:
    """Load latest `.keras` + a label map if available."""
    model_path = _find_latest_keras_model(model_dir)
    if model_path is None:
        return None

    labels_path = _find_labels_json(model_dir)
    classes: list[str]
    if labels_path is not None:
        raw = json.loads(labels_path.read_text(encoding="utf-8"))
        # notebook saved: {int(i): name for i, name in enumerate(le.classes_)}
        # JSON keys become strings.
        if isinstance(raw, dict):
            # sort by numeric key
            pairs: list[tuple[int, str]] = []
            for k, v in raw.items():
                try:
                    pairs.append((int(k), str(v)))
                except Exception:
                    continue
            pairs.sort(key=lambda t: t[0])
            classes = [name for _, name in pairs]
        elif isinstance(raw, list):
            classes = [str(x) for x in raw]
        else:
            classes = list(fallback_classes)
    else:
        classes = list(fallback_classes)

    # Lazy TF import
    import tensorflow as tf  # type: ignore

    model = tf.keras.models.load_model(str(model_path))
    model.trainable = False
    image_size = None
    if getattr(model, "input_shape", None) is not None:
        # (None, H, W, 3)
        sh = model.input_shape
        if len(sh) == 4:
            image_size = int(sh[1])
    if image_size is None:
        image_size = 128

    # Align label list with softmax size (e.g. Kaggle 12-class model vs local 11-class SVM meta).
    try:
        out_dim = int(model.output_shape[-1])
    except Exception:
        out_dim = len(classes)

    if len(classes) != out_dim:
        if len(fallback_classes) == out_dim:
            classes = list(fallback_classes)
        elif len(classes) < out_dim:
            for i in range(len(classes), out_dim):
                classes.append(f"class_{i}")
        else:
            classes = classes[:out_dim]

    return {"type": "keras", "model": model, "classes": classes, "image_size": image_size, "model_path": str(model_path)}


def predict_roi_keras(roi_bgr: np.ndarray, pack: dict[str, Any]) -> tuple[str, float, np.ndarray]:
    """Predict class probabilities for one ROI using Keras MobileNetV2 checkpoint."""
    model = pack["model"]
    classes: list[str] = pack["classes"]
    image_size = int(pack["image_size"])

    # MobileNetV2 preprocess expects RGB float32 in [0..255]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(np.float32)

    import tensorflow as tf  # type: ignore
    roi_resized = tf.keras.applications.mobilenet_v2.preprocess_input(roi_resized)

    x = roi_resized[None, ...]  # (1,H,W,3)
    probs = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(probs))
    label = classes[idx] if idx < len(classes) else str(idx)
    conf = float(probs[idx])
    return label, conf, np.asarray(probs, dtype=np.float64)

