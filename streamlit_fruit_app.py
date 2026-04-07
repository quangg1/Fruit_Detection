"""Streamlit: YOLO + Keras. Upload model từ máy (deploy), ảnh upload/snapshot/live WebRTC."""

from __future__ import annotations

import hashlib
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
from fruit_project.keras_classifier import load_keras_pack, load_keras_pack_from_bytes, predict_roi_keras
from fruit_project.features import fruit_basic_properties

st.set_page_config(page_title="Fruit YOLO + Keras", page_icon="🍎", layout="wide")

try:
    import av

    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # type: ignore[misc, assignment]


def load_class_fallback() -> list[str]:
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


@st.cache_resource(show_spinner=True)
def load_keras_cached_bytes(
    model_bytes: bytes,
    labels_bytes: bytes | None,
    _cache_key: str,
):
    fb = load_class_fallback()
    return load_keras_pack_from_bytes(model_bytes, fb, labels_bytes)


@st.cache_resource(show_spinner=True)
def resolve_dl_pack_disk():
    fb = load_class_fallback()
    try:
        k = load_keras_pack(MODEL_DIR, fallback_classes=fb)
    except Exception:
        k = None
    if k is not None:
        return k
    return load_torch_dl_pack()


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


def process_image_pipeline(
    img_bgr: np.ndarray,
    dl_pack: dict[str, Any],
    *,
    detector_mode: str,
    yolo_weights: str,
    max_fruits: int,
    yolo_det_conf: float,
    conf_threshold: float,
    pad_ratio: float,
    min_area_ratio: float,
    fallback_contour: bool,
    yolo_prior: bool,
    yolo_prior_margin: float,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]], int]:
    det_mode = detector_mode
    yolo_model = None
    if det_mode == "yolo":
        try:
            yolo_model = load_yolo(yolo_weights)
        except Exception:
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

    return vis, rows, roi_debug_rows, len(boxes)


if WEBRTC_AVAILABLE:

    class FruitVideoProcessor(VideoProcessorBase):  # type: ignore[misc]
        def __init__(self) -> None:
            self.detector_mode = "yolo"
            self.yolo_weights = "yolov8n.pt"
            self.max_fruits = 8
            self.yolo_det_conf = 0.2
            self.conf_threshold = 0.65
            self.pad_ratio = 0.12
            self.min_area_ratio = 0.01
            self.fallback_contour = True
            self.yolo_prior = True
            self.yolo_prior_margin = 0.15
            self.process_every = 2
            self.dl_pack: dict[str, Any] | None = None
            self._frame_i = 0
            self._last_vis: np.ndarray | None = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            if self.dl_pack is None:
                return frame
            img = frame.to_ndarray(format="bgr24")
            self._frame_i += 1
            run = self._frame_i % max(1, int(self.process_every)) == 0 or self._last_vis is None
            if run:
                vis, _, _, _ = process_image_pipeline(
                    img,
                    self.dl_pack,
                    detector_mode=self.detector_mode,
                    yolo_weights=self.yolo_weights,
                    max_fruits=self.max_fruits,
                    yolo_det_conf=self.yolo_det_conf,
                    conf_threshold=self.conf_threshold,
                    pad_ratio=self.pad_ratio,
                    min_area_ratio=self.min_area_ratio,
                    fallback_contour=self.fallback_contour,
                    yolo_prior=self.yolo_prior,
                    yolo_prior_margin=self.yolo_prior_margin,
                )
                self._last_vis = vis
            else:
                vis = self._last_vis if self._last_vis is not None else img
            return av.VideoFrame.from_ndarray(vis, format="bgr24")


def _fingerprint(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(str(len(b)).encode())
    n = len(b)
    if n > 0:
        h.update(b[: min(4096, n)])
        h.update(b[max(0, n - 4096) :])
    return h.hexdigest()


# ----- Sidebar: model (upload ưu tiên khi deploy) -----
st.title("Fruit detection + Keras classification")
st.caption(
    "**Deploy:** upload file `.keras` bên dưới — không cần để model trong repo. "
    "Có thể kèm `labels.json`. Live webcam = video (WebRTC), không ảnh hưởng chế độ chụp một ảnh."
)

with st.sidebar:
    st.markdown("### Model classifier")
    keras_up = st.file_uploader(
        "Upload model `.keras` (từ máy bạn)",
        type=["keras"],
        help="Khi deploy Streamlit Cloud: chọn file model trên máy rồi upload; app load trong phiên làm việc.",
    )
    labels_up = st.file_uploader(
        "Tuỳ chọn: `labels.json` / class map",
        type=["json"],
        help="Nếu không có, dùng tên lớp từ fruit_meta.json trên server (có thể lệch số lớp — sẽ pad class_i).",
    )

    dl_pack: dict[str, Any] | None = None
    model_err: str | None = None

    if keras_up is not None:
        mb = keras_up.getvalue()
        lb = labels_up.getvalue() if labels_up is not None else None
        fp_m = _fingerprint(mb)
        fp_l = _fingerprint(lb) if lb else ""
        ver = f"{fp_m}|{fp_l}"
        try:
            dl_pack = load_keras_cached_bytes(mb, lb, ver)
        except Exception as exc:
            model_err = str(exc)
            dl_pack = None
        st.success("Đang dùng model **upload từ máy bạn**.")
    else:
        dl_pack = resolve_dl_pack_disk()
        if dl_pack is None:
            st.warning("Chưa upload `.keras` và chưa có model trong `weights/` trên server.")
        elif dl_pack.get("type") == "keras":
            st.info("Đang dùng model trong thư mục **`weights/`** trên server.")

    st.divider()
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
    live_every = st.slider("Live: xử lý mỗi N khung", 1, 5, 2, 1)
    st.divider()
    if model_err:
        st.error(f"Lỗi load model upload: {model_err}")
    if dl_pack is None:
        st.error("Cần upload `.keras` hoặc cấp model trên server trong `weights/`.")
    elif dl_pack.get("type") == "keras":
        st.success(f"Keras | classes={len(dl_pack.get('classes', []))}")
    else:
        dmeta = dl_pack["meta"]
        st.success(f"PyTorch: {dmeta.get('model_name', '?')}")
    st.caption(f"Data (local): `{DATA_DIR.relative_to(ROOT)}`")

st.markdown("### Input ảnh")
src_tab = st.radio(
    "Nguồn ảnh",
    ["Upload file", "Webcam (một ảnh)", "Live webcam (video)"],
    horizontal=True,
)

if src_tab == "Live webcam (video)":
    if not WEBRTC_AVAILABLE:
        st.error("Cài thêm: `pip install streamlit-webrtc av` để dùng video trực tiếp.")
        st.stop()
    if dl_pack is None:
        st.stop()
    st.caption("Video WebRTC — bật **START** trên player. Khác hẳn tab *Webcam (một ảnh)* (chỉ chụp 1 khung).")
    ctx = webrtc_streamer(
        key="fruit-live",
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_processor_factory=FruitVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )
    if ctx.video_processor:
        vp = ctx.video_processor
        vp.detector_mode = detector_mode
        vp.yolo_weights = yolo_weights
        vp.max_fruits = int(max_fruits)
        vp.yolo_det_conf = float(yolo_det_conf)
        vp.conf_threshold = float(conf_threshold)
        vp.pad_ratio = float(pad_ratio)
        vp.min_area_ratio = float(min_area_ratio)
        vp.fallback_contour = fallback_contour
        vp.yolo_prior = yolo_prior
        vp.yolo_prior_margin = float(yolo_prior_margin)
        vp.process_every = int(live_every)
        vp.dl_pack = dl_pack
    st.info("Live cam dùng **cùng** model đã chọn ở sidebar (upload hoặc weights).")
    st.stop()

# --- Upload ảnh / webcam một ảnh ---
up = None
cam = None
if src_tab == "Upload file":
    up = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "webp", "bmp"])
else:
    st.caption("**Take photo** = một khung tĩnh (không phải live video).")
    cam = st.camera_input("Webcam")

img_bgr: np.ndarray | None = None
if src_tab == "Upload file":
    if up is None:
        st.info("Chọn ảnh hoặc thử Live webcam.")
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
        st.info("Chụp ảnh hoặc đổi nguồn.")
        st.stop()
    data = np.asarray(bytearray(cam.getvalue()), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        try:
            img = Image.open(cam).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            st.error("Không decode được ảnh.")
            st.stop()

if dl_pack is None:
    st.error("Thiếu model.")
    st.stop()

vis, rows, roi_debug_rows, n_boxes = process_image_pipeline(
    img_bgr,
    dl_pack,
    detector_mode=detector_mode,
    yolo_weights=yolo_weights,
    max_fruits=int(max_fruits),
    yolo_det_conf=float(yolo_det_conf),
    conf_threshold=float(conf_threshold),
    pad_ratio=float(pad_ratio),
    min_area_ratio=float(min_area_ratio),
    fallback_contour=fallback_contour,
    yolo_prior=yolo_prior,
    yolo_prior_margin=float(yolo_prior_margin),
)

c1, c2 = st.columns([1.4, 1])
with c1:
    st.subheader("Detected objects")
    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.subheader("Summary")
    st.metric("Detected boxes", n_boxes)
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
                st.caption(str(item["top1"]))
                if item["top2"]:
                    st.caption(str(item["top2"]))
                if item["top3"]:
                    st.caption(str(item["top3"]))
else:
    st.warning("Không có đối tượng đủ độ tin cậy.")
