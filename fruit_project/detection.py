"""Shared detection helpers for live and streamlit apps."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU for boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union


def square_crop_with_pad(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    pad_ratio: float,
) -> np.ndarray:
    """Crop ROI as square with padding."""
    H, W = frame.shape[:2]
    pad = int(pad_ratio * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    cw = x2 - x1
    ch = y2 - y1
    if cw <= 1 or ch <= 1:
        return frame[y1:y2, x1:x2].copy()

    if cw > ch:
        y_center = (y1 + y2) / 2.0
        new_h = cw
        y1 = int(max(0, y_center - new_h / 2.0))
        y2 = int(min(H, y1 + new_h))
    else:
        x_center = (x1 + x2) / 2.0
        new_w = ch
        x1 = int(max(0, x_center - new_w / 2.0))
        x2 = int(min(W, x1 + new_w))

    return frame[y1:y2, x1:x2].copy()


def detect_fruit_boxes_contour(
    frame_bgr: np.ndarray,
    min_area_ratio: float,
    max_fruits: int,
) -> list[tuple[int, int, int, int]]:
    """Fallback contour detector."""
    H, W = frame_bgr.shape[:2]
    frame_area = float(H * W)
    min_area = frame_area * float(min_area_ratio)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _t, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_a = bw
    mask_b = cv2.bitwise_not(bw)

    def score_mask(mask_u8: np.ndarray) -> int:
        k = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=2)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = 0
        for cc in contours:
            area = float(cv2.contourArea(cc))
            if min_area <= area <= frame_area * 0.95:
                c += 1
        return c

    chosen = mask_a if score_mask(mask_a) >= score_mask(mask_b) else mask_b
    k = np.ones((3, 3), np.uint8)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_OPEN, k, iterations=1)
    chosen = cv2.morphologyEx(chosen, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int, float]] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > frame_area * 0.95:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        aspect = w / float(h + 1e-9)
        if aspect < 0.55 or aspect > 1.8:
            continue
        extent = area / float((w * h) + 1e-9)
        if extent < 0.35:
            continue
        candidates.append((x, y, x + w, y + h, area))

    candidates.sort(key=lambda z: z[4], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2, _a in candidates:
        if len(kept) >= int(max_fruits):
            break
        b = (x1, y1, x2, y2)
        if any(iou(b, k) > 0.35 for k in kept):
            continue
        kept.append(b)
    return kept


def detect_fruit_boxes_yolo(
    frame_bgr: np.ndarray,
    yolo_model: Any,
    max_fruits: int,
    conf_threshold: float,
) -> list[tuple[int, int, int, int, int, float]]:
    """YOLO detector, keep COCO fruit-like classes."""
    # pylint: disable=import-outside-toplevel
    import numpy as _np

    res = yolo_model.predict(frame_bgr, verbose=False, conf=max(0.15, conf_threshold * 0.5))
    if not res:
        return []
    r0 = res[0]
    if not hasattr(r0, "boxes") or r0.boxes is None:
        return []

    boxes_xyxy = r0.boxes.xyxy.cpu().numpy() if r0.boxes.xyxy is not None else _np.empty((0, 4))
    boxes_cls = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes.cls is not None else _np.empty((0,), dtype=int)
    boxes_conf = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else _np.empty((0,))
    keep_cls = {46, 47, 49}

    candidates: list[tuple[int, int, int, int, int, float]] = []
    for i in range(len(boxes_xyxy)):
        cls_id = int(boxes_cls[i]) if i < len(boxes_cls) else -1
        if cls_id not in keep_cls:
            continue
        score = float(boxes_conf[i]) if i < len(boxes_conf) else 0.0
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        x1i, y1i = int(max(0, x1)), int(max(0, y1))
        x2i, y2i = int(max(x1i + 1, x2)), int(max(y1i + 1, y2))
        candidates.append((x1i, y1i, x2i, y2i, cls_id, score))

    candidates.sort(key=lambda z: z[5], reverse=True)
    kept: list[tuple[int, int, int, int, int, float]] = []
    for x1, y1, x2, y2, cls_id, score in candidates:
        if len(kept) >= int(max_fruits):
            break
        b = (x1, y1, x2, y2)
        if any(iou(b, (k[0], k[1], k[2], k[3])) > 0.45 for k in kept):
            continue
        kept.append((x1, y1, x2, y2, cls_id, score))
    return kept

