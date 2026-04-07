"""Deep learning fruit classifier helpers (PyTorch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import models


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """Create backbone with replaced classification head."""
    name = model_name.lower().strip()
    if name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_f = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_f, num_classes)
        return model
    if name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_f = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Linear(in_f, num_classes)
        return model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_f = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_f, num_classes)
    return model


def preprocess_roi(roi_bgr: np.ndarray, image_size: int = 224) -> torch.Tensor:
    """Convert ROI to normalized tensor [1, 3, H, W]."""
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    x /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def load_dl_model(
    model_path: Path,
    meta_path: Path,
    device: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    """Load model checkpoint and metadata."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing DL model: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing DL metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = meta["classes"]
    model_name = str(meta.get("model_name", "efficientnet_b3"))
    image_size = int(meta.get("image_size", 224))

    dev = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(str(model_path), map_location=dev)
    model = build_model(model_name, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()

    meta["image_size"] = image_size
    return model, meta, dev


def predict_roi_dl(
    roi_bgr: np.ndarray,
    model: torch.nn.Module,
    classes: list[str],
    device: torch.device,
    image_size: int = 224,
) -> tuple[str, float, np.ndarray]:
    """Predict class and confidence for one ROI."""
    x = preprocess_roi(roi_bgr, image_size=image_size).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), probs
