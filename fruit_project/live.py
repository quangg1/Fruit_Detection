"""Fruit classification from webcam/IP camera (multi-detection). Primary: Keras `.keras`."""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import joblib
import numpy as np

from fruit_project.config import META_PATH, MODEL_DIR, MODEL_PATH, parse_camera_source
from fruit_project.features import DIPOptions, extract_features, preprocess_for_fruit
from fruit_project.keras_classifier import load_keras_pack, predict_roi_keras


def _fallback_class_names() -> list[str]:
    if not META_PATH.is_file():
        return []
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        return list(meta.get("classes", []))
    except Exception:
        return []


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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


def _square_crop_with_pad(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    pad_ratio: float,
) -> np.ndarray:
    """Crop ROI as a square with padding (keeps fruit centered before features)."""
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


def detect_fruit_boxes(
    frame_bgr: np.ndarray,
    min_area_ratio: float,
    max_fruits: int,
) -> list[tuple[int, int, int, int]]:
    """
    Detect multiple fruit candidates using:
      - Otsu threshold on grayscale
      - morphology cleanup
      - contour filtering by area
      - NMS (IoU)
    """
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
        # Geometric filters to reduce noisy fragments ("over-read")
        aspect = w / float(h + 1e-9)
        if aspect < 0.55 or aspect > 1.8:
            continue
        extent = area / float((w * h) + 1e-9)
        if extent < 0.35:
            continue
        candidates.append((x, y, x + w, y + h, area))

    candidates.sort(key=lambda z: z[4], reverse=True)

    kept: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2, _area in candidates:
        if len(kept) >= int(max_fruits):
            break
        b = (x1, y1, x2, y2)
        if any(_iou(b, k) > 0.35 for k in kept):
            continue
        kept.append(b)

    return kept


def detect_fruit_boxes_yolo(
    frame_bgr: np.ndarray,
    yolo_model: object,
    max_fruits: int,
    conf_threshold: float,
) -> list[tuple[int, int, int, int, int, float]]:
    """
    Detect fruit-like boxes with YOLO (Ultralytics).
    We only keep COCO classes: banana(46), apple(47), orange(49).
    """
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

    candidates.sort(key=lambda z: z[4], reverse=True)
    kept: list[tuple[int, int, int, int, int, float]] = []
    for x1, y1, x2, y2, cls_id, score in candidates:
        if len(kept) >= int(max_fruits):
            break
        b = (x1, y1, x2, y2)
        if any(_iou(b, (k[0], k[1], k[2], k[3])) > 0.45 for k in kept):
            continue
        kept.append((x1, y1, x2, y2, cls_id, score))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area-ratio", type=float, default=0.01)
    ap.add_argument("--max-fruits", type=int, default=6)
    ap.add_argument("--pad-ratio", type=float, default=0.12)
    ap.add_argument(
        "--mode",
        type=str,
        default="detect",
        choices=["detect", "center"],
        help="detect: detect boxes then classify each ROI. center: classify centered crop of full frame (same ROI logic as web app).",
    )
    ap.add_argument("--conf-threshold", type=float, default=0.60)
    ap.add_argument(
        "--classifier",
        type=str,
        default="auto",
        choices=["auto", "keras", "svm"],
        help="ROI classifier: keras (default when .keras in weights/), svm (legacy joblib), auto = keras if present else svm.",
    )
    ap.add_argument("--median-ksize", type=int, default=3)
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument(
        "--detector",
        type=str,
        default="auto",
        choices=["auto", "contour", "yolo"],
        help="Detection backend: contour, yolo, or auto (prefer yolo if available).",
    )
    ap.add_argument(
        "--yolo-weights",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLO weights path/name for detection.",
    )
    ap.add_argument(
        "--yolo-prior",
        action="store_true",
        help="Bias final class toward YOLO class (apple/banana/orange) when close probabilities.",
    )
    ap.add_argument(
        "--yolo-prior-margin",
        type=float,
        default=0.15,
        help="Neu top SVM - xac suat lop YOLO <= margin thi uu tien lop YOLO.",
    )
    args = ap.parse_args()

    fb = _fallback_class_names()
    keras_pack = None
    try:
        keras_pack = load_keras_pack(MODEL_DIR, fallback_classes=fb)
    except Exception as exc:  # pragma: no cover
        if args.classifier == "keras":
            print(f"[keras] Failed to load: {exc}", file=sys.stderr)
            sys.exit(1)

    use_keras = keras_pack is not None and args.classifier != "svm"
    if args.classifier == "keras" and keras_pack is None:
        print("[keras] No .keras file in weights/. Add your checkpoint.", file=sys.stderr)
        sys.exit(1)

    clf = None
    le = None
    classes_fusion: list[str] = list(keras_pack.get("classes", fb)) if keras_pack else fb

    if use_keras:
        print(f"[keras] ROI classification: {keras_pack.get('model_path', '')}")
    elif args.classifier in {"auto", "svm"}:
        if not MODEL_PATH.is_file():
            print(
                "No Keras model and no legacy fruit_svm.joblib. Place a .keras file in weights/.",
                file=sys.stderr,
            )
            sys.exit(1)
        pack = joblib.load(MODEL_PATH)
        clf, le = pack["model"], pack["label_encoder"]
        classes_fusion = [str(c) for c in le.classes_]
        print("[svm] Using legacy fruit_svm.joblib (consider switching to Keras only).")
    else:
        print("No classifier available.", file=sys.stderr)
        sys.exit(1)

    src = parse_camera_source()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open camera: {src}", file=sys.stderr)
        sys.exit(1)

    dip_opt = DIPOptions(
        use_median_denoise=True,
        median_ksize=(int(args.median_ksize) | 1),
        use_clahe=True,
        clahe_clip=float(args.clahe_clip),
        crop_size=int(args.crop_size),
    )

    # Optional YOLO detector (more robust than contour on complex backgrounds).
    yolo_model = None
    if args.mode == "detect" and args.detector in {"auto", "yolo"}:
        try:
            # pylint: disable=import-outside-toplevel
            from ultralytics import YOLO  # type: ignore

            yolo_model = YOLO(args.yolo_weights)
            print(f"[detector] YOLO loaded: {args.yolo_weights}")
        except Exception as exc:  # pragma: no cover
            if args.detector == "yolo":
                print(f"[detector] Failed to load YOLO: {exc}", file=sys.stderr)
                print("Install ultralytics: pip install ultralytics", file=sys.stderr)
                sys.exit(1)
            print("[detector] YOLO unavailable, fallback to contour detector.")

    print("q = quit. Detect multiple fruits and classify each one.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mode == "center":
            if use_keras:
                roi = preprocess_for_fruit(frame, size=int(keras_pack.get("image_size", 128)))
                label, conf, proba = predict_roi_keras(roi, keras_pack)
            else:
                feat = extract_features(frame, use_lbp=True, dip_options=dip_opt, apply_dip=True)
                proba = clf.predict_proba(feat.reshape(1, -1))[0]
                pred_idx = int(np.argmax(proba))
                label = le.inverse_transform([pred_idx])[0]
                conf = float(proba[pred_idx])
            if conf >= float(args.conf_threshold):
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (40, 40, 40), -1)
                txt = f"{label} {conf * 100:.1f}%"
                cv2.putText(
                    frame,
                    txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
        else:
            if yolo_model is not None:
                yolo_boxes = detect_fruit_boxes_yolo(
                    frame,
                    yolo_model=yolo_model,
                    max_fruits=int(args.max_fruits),
                    conf_threshold=float(args.conf_threshold),
                )
                boxes = [(b[0], b[1], b[2], b[3]) for b in yolo_boxes]
            else:
                boxes = detect_fruit_boxes(
                    frame,
                    min_area_ratio=float(args.min_area_ratio),
                    max_fruits=int(args.max_fruits),
                )
                yolo_boxes = []

            for idx_box, (x1, y1, x2, y2) in enumerate(boxes):
                w, h = x2 - x1, y2 - y1
                roi = _square_crop_with_pad(
                    frame, x1, y1, w, h, pad_ratio=float(args.pad_ratio)
                )
                classes_used = list(classes_fusion)
                if use_keras:
                    label, conf, proba = predict_roi_keras(roi, keras_pack)
                    pred_idx = int(np.argmax(proba))
                else:
                    feat = extract_features(
                        roi, use_lbp=True, dip_options=dip_opt, apply_dip=True
                    )
                    proba = clf.predict_proba(feat.reshape(1, -1))[0]
                    pred_idx = int(np.argmax(proba))
                    label = le.inverse_transform([pred_idx])[0]
                    conf = float(proba[pred_idx])

                # Fusion YOLO COCO vs SVM (đồng bộ với streamlit_fruit_app)
                if args.yolo_prior and yolo_model is not None and idx_box < len(yolo_boxes):
                    yb = yolo_boxes[idx_box]
                    y_cls = int(yb[4])
                    y_det = float(yb[5]) if len(yb) > 5 else 0.0
                    yolo_to_name = {46: "banana", 47: "apple", 49: "orange"}
                    y_name = yolo_to_name.get(y_cls)
                    if y_name is not None:
                        classes_lower = [str(c).lower() for c in classes_used]
                        if y_name in classes_lower:
                            y_idx = classes_lower.index(y_name)
                            y_prob = float(proba[y_idx])
                            margin = float(args.yolo_prior_margin)
                            if (conf - y_prob) <= margin:
                                pred_idx = y_idx
                                label = str(classes_used[y_idx])
                                conf = y_prob
                            elif y_det >= 0.48 and str(label).lower() != y_name:
                                pred_idx = y_idx
                                label = str(classes_used[y_idx])
                                conf = max(y_prob, y_det * 0.88)

                if conf < float(args.conf_threshold):
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = f"{label} {conf * 100:.1f}%"
                cv2.putText(
                    frame,
                    txt,
                    (x1 + 4, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Fruit classification (multi-detect + DIP)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
