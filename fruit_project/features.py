"""Đặc trưng từ ảnh — tiền xử lý DIP + histogram HSV + LBP."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DIPOptions:
    """Bật/tắt các bước DIP trước khi trích đặc trưng."""

    use_median_denoise: bool = True
    median_ksize: int = 3
    use_clahe: bool = True
    clahe_clip: float = 2.0
    crop_size: int = 128


def dip_preprocess(bgr: np.ndarray, opt: DIPOptions | None = None) -> np.ndarray:
    """
    DIP tiền xử lý:
    - Median filter: giảm nhiễu muối tiêu, giữ biên tốt hơn Gaussian cho chi tiết trái.
    - CLAHE trên kênh L (LAB): cân cục bộ độ sáng, dễ tách màu vỏ khi ánh sáng không đều.
    """
    if opt is None:
        opt = DIPOptions()
    x = bgr.copy()
    k = opt.median_ksize | 1
    if opt.use_median_denoise and k >= 3:
        x = cv2.medianBlur(x, k)
    if opt.use_clahe:
        lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=opt.clahe_clip, tileGridSize=(8, 8))
        l2 = clahe.apply(l_ch)
        lab2 = cv2.merge([l2, a_ch, b_ch])
        x = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return x


def preprocess_for_fruit(bgr: np.ndarray, size: int = 128) -> np.ndarray:
    """ROI: cắt vuông giữa khung, resize (giả định trái nằm giữa khung)."""
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = bgr[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _hue_unified_for_hist(h_uint8: np.ndarray) -> np.ndarray:
    """
    Trong OpenCV, màu đỏ nằm ở H~0–15 và ~165–179 (hai đầu thang).
    Gương H qua trục 90°: h' = 180 - h nếu h > 90, ngược lại h' = h
    → hai vùng đỏ gộp cùng cực thấp h', tránh tách histogram ra hai bin đầu/cuối (dễ nhầm với cam).
    """
    h = h_uint8.astype(np.float32)
    return np.where(h > 90.0, 180.0 - h, h)


def extract_hsv_histogram(bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    """Histogram H (đã gộp đỏ) + S + V."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    hu = _hue_unified_for_hist(h_ch)
    hist_h, _ = np.histogram(hu.ravel(), bins=bins, range=(0.0, 90.0), density=False)
    hist_h = hist_h.astype(np.float64)
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()
    feat = np.concatenate([hist_h, hist_s, hist_v])
    n = np.linalg.norm(feat) + 1e-12
    return feat / n


def extract_lab_color_stats(bgr: np.ndarray) -> np.ndarray:
    """Mean/std kênh a,b (LAB) — trục đỏ–xanh / vàng–xanh, bổ trợ khi HSV nhầm (ảnh minh họa, nền đen)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float64) / 255.0
    b = lab[:, :, 2].astype(np.float64) / 255.0
    return np.array(
        [np.mean(a), np.std(a), np.mean(b), np.std(b)], dtype=np.float64
    )


def extract_lbp_hist(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """LBP uniform — histogram cố định 11 bin (P=8 → giá trị 0..10), tránh đổi chiều vector."""
    from skimage.feature import local_binary_pattern

    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    lbp = local_binary_pattern(small, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=11, range=(0, 11), density=False)
    hist = hist.astype(np.float64)
    n = np.linalg.norm(hist) + 1e-12
    return hist / n


def _foreground_mask_from_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """Build a coarse foreground mask robust to bright/dark background."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _t, m1 = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _t2, g = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m2 = g
    m3 = cv2.bitwise_not(g)

    # Pick polarity with medium foreground ratio (avoid almost-all or almost-none).
    def ratio(mask_u8: np.ndarray) -> float:
        return float(np.count_nonzero(mask_u8)) / float(mask_u8.size + 1e-9)

    cand = [m1, m2, m3]
    cand.sort(key=lambda m: abs(ratio(m) - 0.45))
    mask = cand[0]

    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def extract_shape_descriptors(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Shape feature vector from foreground contour:
    [aspect, extent, solidity, circularity, area_ratio, hu1..hu7]
    """
    mask = _foreground_mask_from_roi(roi_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(12, dtype=np.float64)

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    x, y, w, h = cv2.boundingRect(c)
    rect_area = float((w * h) + 1e-9)
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull) + 1e-9)
    mask_area = float(mask.shape[0] * mask.shape[1] + 1e-9)

    aspect = float(w / (h + 1e-9))
    extent = float(area / rect_area)
    solidity = float(area / hull_area)
    circularity = float(4.0 * np.pi * area / ((peri * peri) + 1e-9))
    area_ratio = float(area / mask_area)

    hu = cv2.HuMoments(cv2.moments(c)).flatten()
    # Log transform Hu moments for better numeric stability.
    hu_log = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    vec = np.concatenate(
        [np.array([aspect, extent, solidity, circularity, area_ratio]), hu_log]
    ).astype(np.float64)
    # Per-group normalize to keep stable scale.
    return vec / (np.linalg.norm(vec) + 1e-12)


def extract_features(
    bgr: np.ndarray,
    use_lbp: bool = True,
    dip_options: DIPOptions | None = None,
    apply_dip: bool = True,
) -> np.ndarray:
    """Gom DIP (tùy chọn) → ROI → màu + kết cấu + shape."""
    opt = dip_options or DIPOptions()
    x = dip_preprocess(bgr, opt) if apply_dip else bgr.copy()
    proc = preprocess_for_fruit(x, size=opt.crop_size)
    hsv = extract_hsv_histogram(proc)
    lab_stats = extract_lab_color_stats(proc)
    shape = extract_shape_descriptors(proc)
    if not use_lbp:
        feat = np.concatenate([hsv, lab_stats, shape])
    else:
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        lbp = extract_lbp_hist(gray)
        feat = np.concatenate([hsv, lab_stats, lbp, shape])
    return feat / (np.linalg.norm(feat) + 1e-12)


def pipeline_visualization(
    bgr: np.ndarray,
    dip_options: DIPOptions | None = None,
    apply_dip: bool = True,
) -> dict[str, np.ndarray]:
    """Ảnh từng bước để hiển thị UI / báo cáo."""
    opt = dip_options or DIPOptions()
    out: dict[str, np.ndarray] = {"Ảnh gốc": bgr}
    if apply_dip:
        after = dip_preprocess(bgr, opt)
        out["Sau median + CLAHE (LAB)"] = after
        src = after
    else:
        src = bgr
    roi = preprocess_for_fruit(src, size=opt.crop_size)
    out["ROI (cắt giữa, resize)"] = roi
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    out["Kênh H"] = hsv[:, :, 0]
    out["Kênh S"] = hsv[:, :, 1]
    out["Kênh V"] = hsv[:, :, 2]
    return out


def _color_name_from_unified_hue(h_u: float) -> str:
    """Map unified hue in [0, 90] (see _hue_unified_for_hist) to coarse labels."""
    if h_u < 12:
        return "red"
    if h_u < 22:
        return "orange/yellow"
    if h_u < 32:
        return "yellow"
    if h_u < 45:
        return "yellow-green"
    if h_u < 62:
        return "green"
    if h_u < 78:
        return "cyan/blue"
    return "purple/magenta"


def fruit_basic_properties(roi_bgr: np.ndarray) -> dict[str, float | str]:
    """
    Basic interpretable properties for UI/report:
    - dominant color: mean Hue on **foreground** + **saturated** pixels (tránh nền gỗ/lệch màu),
      với hue đã gộp đỏ (cùng logic histogram).
    - shape: tỷ lệ dài/rộng từ minAreaRect (ổn định hơn boundingRect khi vật xiên).
    - circularity + size_ratio trên mask foreground.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]

    mask = _foreground_mask_from_roi(roi_bgr)
    # Ưu tiên pixel có màu rõ (gỗ thường S thấp → không lấn át táo/chuối)
    sat_min = 28
    sel = (mask > 0) & (s_ch.astype(np.int32) >= sat_min)
    if np.count_nonzero(sel) < max(40, mask.size // 200):
        sel = mask > 0

    hu_flat = _hue_unified_for_hist(h_ch[sel])
    if hu_flat.size == 0:
        hu_flat = _hue_unified_for_hist(h_ch.reshape(-1))

    h_mean = float(np.mean(hu_flat))
    color_name = _color_name_from_unified_hue(h_mean)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    shape_ratio = 1.0
    elongation = 1.0
    fg_ratio = float(np.count_nonzero(mask)) / float(mask.size + 1e-9)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        peri = float(cv2.arcLength(c, True))
        if peri > 1e-6:
            circularity = float(4.0 * np.pi * area / (peri * peri))
        _x, _y, w, h = cv2.boundingRect(c)
        shape_ratio = float(w / (h + 1e-9))
        # Bounding có hướng — ổn định với chuối/xiên so với hộp trục
        _rect = cv2.minAreaRect(c)
        rw, rh = _rect[1]
        rw, rh = float(rw), float(rh)
        if rw < 1e-6 or rh < 1e-6:
            elongation = max(shape_ratio, 1.0 / (shape_ratio + 1e-9))
        else:
            elongation = max(rw, rh) / (min(rw, rh) + 1e-9)

    if elongation >= 1.42:
        shape_name = "elongated"
    elif elongation <= 1.18:
        shape_name = "round/oval"
    else:
        shape_name = "oval / moderate"

    return {
        "dominant_color": color_name,
        "hue_mean": h_mean,
        "shape": shape_name,
        "shape_ratio": shape_ratio,
        "elongation": elongation,
        "circularity": circularity,
        "size_ratio": fg_ratio,
    }
