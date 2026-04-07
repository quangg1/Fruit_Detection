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


def _classes_from_labels_file(labels_path: Path, fallback_classes: list[str]) -> list[str]:
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        pairs: list[tuple[int, str]] = []
        for k, v in raw.items():
            try:
                pairs.append((int(k), str(v)))
            except Exception:
                continue
        pairs.sort(key=lambda t: t[0])
        return [name for _, name in pairs]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return list(fallback_classes)


def load_keras_pack_from_path(
    model_path: Path,
    fallback_classes: list[str],
    labels_path: Path | None = None,
) -> dict[str, Any] | None:
    """Load one `.keras` file; optional `labels.json` path. Dùng cho upload / deploy."""
    if not model_path.is_file():
        return None

    if labels_path is not None and labels_path.is_file():
        classes = _classes_from_labels_file(labels_path, fallback_classes)
    else:
        classes = list(fallback_classes)

    import tensorflow as tf  # type: ignore

    model = tf.keras.models.load_model(str(model_path))
    model.trainable = False
    image_size = None
    if getattr(model, "input_shape", None) is not None:
        sh = model.input_shape
        if len(sh) == 4:
            image_size = int(sh[1])
    if image_size is None:
        image_size = 128

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


def load_keras_pack(
    model_dir: Path,
    *,
    fallback_classes: list[str],
) -> dict[str, Any] | None:
    """Load latest `.keras` trong thư mục + label map nếu có."""
    model_path = _find_latest_keras_model(model_dir)
    if model_path is None:
        return None
    labels_path = _find_labels_json(model_dir)
    return load_keras_pack_from_path(model_path, fallback_classes, labels_path)


def load_keras_pack_from_bytes(
    model_bytes: bytes,
    fallback_classes: list[str],
    labels_json_bytes: bytes | None = None,
) -> dict[str, Any] | None:
    """Load từ bytes (Streamlit upload): ghi file tạm rồi load_model."""
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
        f.write(model_bytes)
        mp = f.name
    lp: str | None = None
    if labels_json_bytes:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(labels_json_bytes)
            lp = f.name
    try:
        return load_keras_pack_from_path(Path(mp), fallback_classes, Path(lp) if lp else None)
    finally:
        try:
            os.unlink(mp)
        except OSError:
            pass
        if lp:
            try:
                os.unlink(lp)
            except OSError:
                pass


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

