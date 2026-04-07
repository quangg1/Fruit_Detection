"""Camera & paths — đổi IP cho khớp điện thoại (IP Webcam)."""

import os
from pathlib import Path

# Thư mục gốc project
ROOT = Path(__file__).resolve().parents[1]

# Dữ liệu ảnh: data/<tên_loại>/*.jpg
DATA_DIR = ROOT / "data" / "fruits"

# Checkpoints: ưu tiên Keras `.keras` trong weights/ (web app + live). fruit_svm.joblib = legacy SVM.
MODEL_DIR = ROOT / "weights"
MODEL_PATH = MODEL_DIR / "fruit_svm.joblib"
META_PATH = MODEL_DIR / "fruit_meta.json"
DL_MODEL_PATH = MODEL_DIR / "fruit_dl.pt"
DL_META_PATH = MODEL_DIR / "fruit_dl_meta.json"

# 0 = webcam máy tính; hoặc URL HTTP từ IP Webcam (Android)
# Ví dụ: "http://192.168.1.10:8080/video"
CAMERA_SOURCE = os.environ.get("FRUIT_CAMERA", "0")


def parse_camera_source():
    """Trả về int hoặc str cho cv2.VideoCapture."""
    s = CAMERA_SOURCE.strip()
    if s.isdigit():
        return int(s)
    return s
