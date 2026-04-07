"""Streamlit app: YOLO detect fruits -> Keras classifier (no SVM). Upload or webcam."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from fruit_project.config import DATA_DIR, DL_META_PATH, DL_MODEL_PATH, META_PATH, MODEL_DIR, ROOT
from fruit_project.detection import detect_fruit_boxes_contour, detect_fruit_boxes_yolo, square_crop_with_pad
from fruit_project.dl_classifier import load_dl_model, predict_roi_dl
from fruit_project.keras_classifier import load_keras_pack, predict_roi_keras
from fruit_project.features import fruit_basic_properties

st.set_page_config(page_title="Fruit YOLO + Keras", page_icon="🍎", layout="wide")


def load_class_fallback() -> list[str]:
    """Class names for Keras label alignment (from fruit_meta.json or labels.json), no SVM model."""
    if META_PATH.is_file():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            return list(meta.get("classes", []))
        except Exception:
            pass
    return []


@st.cache_resource(show_spinner=False)
def load_torch_dl_pack():
    try:
        model, meta, device = load_dl_model(DL_MODEL_PATH, DL_META_PATH)
        return {"model": model, "meta": meta, "device": device}
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_yolo(weights: str):
    from ultralytics import YOLO  # type: ignore

    return YOLO(weights)


def classify_roi_deep(roi: np.ndarray, dl_pack: dict[str, Any]):
    if dl_pack.get("type") == "keras":
        label, conf, proba = predict_roi_keras(roi, dl_pack)
        return label, conf, proba, np.array(dl_pack["classes"])

    meta = dl_pack["meta"]
    classes = list(meta["classes"])
    image_size = int(meta.get("image_size", 224))
    label, conf, proba = predict_roi_dl(
        roi,
        model=dl_pack["model"],
        classes=classes,
        device=dl_pack["device"],
        image_size=image_size,
    )
    return label, conf, proba, np.array(classes)


def resolve_dl_pack():
    fb = load_class_fallback()
    keras_pack = None
    try:
        keras_pack = load_keras_pack(MODEL_DIR, fallback_classes=fb)
    except Exception:
        keras_pack = None
    if keras_pack is not None:
        return keras_pack
    return load_torch_dl_pack()


dl_pack = resolve_dl_pack()

st.title("Fruit detection + Keras classification")
st.caption("YOLO (or contour) → crop ROI → **Keras** deep model. Optional PyTorch fallback if no `.keras` in `weights/`.")

with st.sidebar:
    st.markdown("### Detector")
    yolo_weights = st.text_input("YOLO weights", value="yolov8n.pt")
    detector_mode = st.selectbox("Detector backend", ["yolo", "contour"], index=0)
    max_fruits = st.slider("Max objects", 1, 12, 8)
    yolo_det_conf = st.slider("YOLO detection confidence", 0.05, 0.9, 0.20, 0.05)
    conf_threshold = st.slider("Min classification confidence", 0.3, 0.95, 0.65, 0.05)
    pad_ratio = st.slider("ROI padding ratio", 0.0, 0.4, 0.12, 0.01)
    min_area_ratio = st.slider("Min area ratio (contour fallback)", 0.002, 0.05, 0.01, 0.001)
    fallback_contour = st.checkbox("Fallback contour if YOLO returns no boxes", value=True)
    yolo_prior = st.checkbox("Use YOLO prior (apple/banana/orange)", value=True)
    yolo_prior_margin = st.slider("YOLO prior margin", 0.05, 0.30, 0.15, 0.01)
    show_roi_debug = st.checkbox("Show ROI debug", value=False)
    st.divider()
    if dl_pack is None:
        st.error("No Keras `.keras` in `weights/` (and no PyTorch `fruit_dl.pt`). Add a checkpoint.")
    elif dl_pack.get("type") == "keras":
        st.success(f"Classifier: Keras | classes={len(dl_pack.get('classes', []))}")
    else:
        dmeta = dl_pack["meta"]
        st.warning("Using PyTorch fallback — place a `.keras` model in `weights/` for Keras.")
        st.success(f"Classifier: {dmeta.get('model_name', '?')} | val_acc={dmeta.get('val_accuracy', 0.0):.4f}")
    st.caption(f"Data dir: `{DATA_DIR.relative_to(ROOT)}`")

st.markdown("### Input")
src_tab = st.radio("Nguồn ảnh", ["Upload file", "Webcam (máy tính)"], horizontal=True)
up = None
cam = None
if src_tab == "Upload file":
    up = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp", "bmp"])
else:
    st.caption("Trình duyệt sẽ xin quyền camera; bấm **Take photo** để chụp một khung và chạy nhận dạng.")
    cam = st.camera_input("Webcam")

img_bgr: np.ndarray | None = None
if src_tab == "Upload file":
    if up is None:
        st.info("Upload một ảnh hoặc chuyển sang Webcam.")
        st.stop()
    data = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        try:
            img = Image.open(up).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            st.error("Không đọc được ảnh.")
            st.stop()
else:
    if cam is None:
        st.info("Bật quyền camera và chụp ảnh để phân tích.")
        st.stop()
    data = np.asarray(bytearray(cam.getvalue()), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        try:
            img = Image.open(cam).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            st.error("Không decode được ảnh từ camera.")
            st.stop()

if dl_pack is None:
    st.error("Thiếu model. Thêm file `.keras` vào thư mục `weights/`.")
    st.stop()

yolo_model = None
det_mode = detector_mode
if det_mode == "yolo":
    try:
        yolo_model = load_yolo(yolo_weights)
    except Exception as exc:
        st.warning(f"YOLO load failed ({exc}). Dùng contour.")
        det_mode = "contour"

if det_mode == "yolo" and yolo_model is not None:
    yolo_boxes = detect_fruit_boxes_yolo(
        img_bgr,
        yolo_model=yolo_model,
        max_fruits=int(max_fruits),
        conf_threshold=float(yolo_det_conf),
    )
    boxes = [(b[0], b[1], b[2], b[3]) for b in yolo_boxes]
    if not boxes and fallback_contour:
        boxes = detect_fruit_boxes_contour(
            img_bgr,
            min_area_ratio=float(min_area_ratio),
            max_fruits=int(max_fruits),
        )
else:
    yolo_boxes = []
    boxes = detect_fruit_boxes_contour(
        img_bgr,
        min_area_ratio=float(min_area_ratio),
        max_fruits=int(max_fruits),
    )

vis = img_bgr.copy()
rows: list[dict[str, object]] = []
roi_debug_rows: list[dict[str, object]] = []

for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
    roi = square_crop_with_pad(img_bgr, x1, y1, x2 - x1, y2 - y1, pad_ratio=float(pad_ratio))
    label, conf, proba, classes = classify_roi_deep(roi, dl_pack)

    if yolo_prior and det_mode == "yolo" and i - 1 < len(yolo_boxes):
        y_cls = int(yolo_boxes[i - 1][4])
        y_det = float(yolo_boxes[i - 1][5]) if len(yolo_boxes[i - 1]) > 5 else 0.0
        yolo_to_name = {46: "banana", 47: "apple", 49: "orange"}
        y_name = yolo_to_name.get(y_cls)
        if y_name is not None:
            classes_lower = [str(c).lower() for c in classes]
            if y_name in classes_lower:
                y_idx = classes_lower.index(y_name)
                y_prob = float(proba[y_idx])
                if (conf - y_prob) <= float(yolo_prior_margin):
                    label = classes[y_idx]
                    conf = y_prob
                elif y_det >= 0.48 and str(label).lower() != y_name:
                    label = classes[y_idx]
                    conf = max(y_prob, y_det * 0.88)

    if conf < float(conf_threshold):
        continue

    props = fruit_basic_properties(roi)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    txt = f"{label} {conf * 100:.1f}%"
    cv2.putText(
        vis,
        txt,
        (x1 + 4, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    rows.append(
        {
            "id": i,
            "label": label,
            "confidence": round(conf, 4),
            "dominant_color": props["dominant_color"],
            "shape": props["shape"],
            "shape_ratio": round(float(props["shape_ratio"]), 3),
            "elongation": round(float(props.get("elongation", props["shape_ratio"])), 3),
            "circularity": round(float(props["circularity"]), 3),
            "size_ratio": round(float(props["size_ratio"]), 3),
        }
    )
    top_idx = np.argsort(proba)[::-1][:3]

    def _lbl(j: int) -> str:
        if j >= len(top_idx):
            return ""
        ti = int(top_idx[j])
        name = classes[ti] if ti < len(classes) else f"class_{ti}"
        return f"{name}: {float(proba[ti]):.3f}"

    roi_debug_rows.append(
        {
            "id": i,
            "roi": cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
            "top1": _lbl(0),
            "top2": _lbl(1),
            "top3": _lbl(2),
        }
    )

c1, c2 = st.columns([1.4, 1])
with c1:
    st.subheader("Detected objects")
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.subheader("Summary")
    st.metric("Detected boxes", len(boxes))
    st.metric("Classified objects", len(rows))
    if rows:
        st.metric("Top prediction", rows[0]["label"])

if rows:
    st.subheader("Per-object result")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    prob_rows = [{"Object": row["id"], "Confidence": row["confidence"]} for row in rows]
    st.bar_chart(pd.DataFrame(prob_rows).set_index("Object"), use_container_width=True)
    if show_roi_debug and roi_debug_rows:
        st.subheader("ROI debug")
        cols = st.columns(min(4, len(roi_debug_rows)))
        for idx, item in enumerate(roi_debug_rows):
            with cols[idx % len(cols)]:
                st.image(item["roi"], caption=f"ROI #{item['id']}", use_container_width=True)
                st.caption(item["top1"])
                if item["top2"]:
                    st.caption(item["top2"])
                if item["top3"]:
                    st.caption(item["top3"])
else:
    st.warning("Không có đối tượng đủ độ tin cậy. Hạ ngưỡng confidence hoặc đổi detector.")
