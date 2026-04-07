"""
Huấn luyện SVM trên thư mục data/fruits/<tên_loại>/... (đọc đệ quy mọi cấp)
Chạy: python -m fruit_project.train
      python -m fruit_project.train --max-per-class 800   (giới hạn mỗi lớp, train nhanh)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fruit_project.config import DATA_DIR, META_PATH, MODEL_PATH, MODEL_DIR
from fruit_project.detection import detect_fruit_boxes_yolo, square_crop_with_pad
from fruit_project.features import extract_features


def safe_imread(path: Path) -> np.ndarray | None:
    """Read image robustly on Windows paths containing Unicode characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def augment_image(bgr: np.ndarray, rng: random.Random) -> list[np.ndarray]:
    """Light augmentations to improve robustness on new backgrounds/lighting."""
    outs: list[np.ndarray] = []

    # 1) brightness/contrast jitter
    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-25, 25)
    bc = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
    outs.append(bc)

    # 2) hsv jitter (hue/sat/value)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.randint(-8, 8)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.randint(-20, 20), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.randint(-20, 20), 0, 255)
    hsv_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    outs.append(hsv_aug)

    # 3) slight blur/noise
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        outs.append(cv2.GaussianBlur(bgr, (k, k), 0))
    else:
        noise = np.random.normal(0, 8, bgr.shape).astype(np.int16)
        noisy = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        outs.append(noisy)

    # 4) slight affine shift/scale
    h, w = bgr.shape[:2]
    tx = rng.uniform(-0.04, 0.04) * w
    ty = rng.uniform(-0.04, 0.04) * h
    sc = rng.uniform(0.94, 1.06)
    M = np.array([[sc, 0, tx], [0, sc, ty]], dtype=np.float32)
    aff = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    outs.append(aff)

    return outs


def load_dataset(
    data_dir: Path,
    max_per_class: int | None = None,
    seed: int = 42,
    roi_mode: str = "center",
    yolo_model: object | None = None,
    yolo_pad_ratio: float = 0.12,
    aug_per_image: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    classes: list[str] = []
    rng = random.Random(seed)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục dữ liệu: {data_dir}")

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        classes.append(label)
        paths = [
            p
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        ]
        rng.shuffle(paths)
        if max_per_class is not None:
            paths = paths[: max_per_class]

        n = 0
        for img_path in paths:
            bgr = safe_imread(img_path)
            if bgr is None:
                continue
            img_for_feat = bgr
            if roi_mode == "yolo" and yolo_model is not None:
                yb = detect_fruit_boxes_yolo(
                    bgr,
                    yolo_model=yolo_model,
                    max_fruits=1,
                    conf_threshold=0.2,
                )
                if yb:
                    x1, y1, x2, y2 = yb[0][0], yb[0][1], yb[0][2], yb[0][3]
                    img_for_feat = square_crop_with_pad(
                        bgr, x1, y1, x2 - x1, y2 - y1, pad_ratio=float(yolo_pad_ratio)
                    )
            feat = extract_features(img_for_feat)
            X_list.append(feat)
            y_list.append(label)
            n += 1

            # Optional augmentation (same label)
            if aug_per_image > 0:
                aug_imgs = augment_image(img_for_feat, rng)
                for aug_im in aug_imgs[:aug_per_image]:
                    feat_aug = extract_features(aug_im)
                    X_list.append(feat_aug)
                    y_list.append(label)
                    n += 1
        if n == 0:
            print(f"Canh bao: {class_dir} khong co anh hop le.", file=sys.stderr)

    if len(X_list) < 4:
        raise RuntimeError("Can it nhat 4 anh va 2 loai tro len.")

    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y, classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Toi da so anh moi lop (mac dinh: het tat ca; nen dat 500-2000 neu dataset lon)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--roi-mode",
        type=str,
        default="center",
        choices=["center", "yolo"],
        help="center: keep old training style (center crop). yolo: detect ROI by YOLO before extracting features.",
    )
    ap.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    ap.add_argument("--yolo-pad-ratio", type=float, default=0.12)
    ap.add_argument(
        "--aug-per-image",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4],
        help="Number of augmented variants per image (0 disables augmentation).",
    )
    ap.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Tỷ lệ tập validation (0–0.4). In accuracy/F1 trên val trước khi fit lại trên toàn bộ dữ liệu.",
    )
    args = ap.parse_args()

    yolo_model = None
    if args.roi_mode == "yolo":
        try:
            # pylint: disable=import-outside-toplevel
            from ultralytics import YOLO  # type: ignore

            yolo_model = YOLO(args.yolo_weights)
            print(f"[train] YOLO loaded: {args.yolo_weights}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO for roi-mode=yolo: {exc}. Install ultralytics first."
            ) from exc

    X, y, class_names = load_dataset(
        DATA_DIR,
        max_per_class=args.max_per_class,
        seed=args.seed,
        roi_mode=args.roi_mode,
        yolo_model=yolo_model,
        yolo_pad_ratio=args.yolo_pad_ratio,
        aug_per_image=args.aug_per_image,
    )
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    vs = float(args.val_split)
    val_metrics: dict[str, float] = {}
    if 0.0 < vs < 0.45 and len(y_enc) >= 8:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y_enc, test_size=vs, random_state=args.seed, stratify=y_enc
        )
        clf.fit(X_tr, y_tr)
        pred_va = clf.predict(X_va)
        acc_va = accuracy_score(y_va, pred_va)
        f1_macro_va = f1_score(y_va, pred_va, average="macro", zero_division=0)
        f1_weighted_va = f1_score(y_va, pred_va, average="weighted", zero_division=0)
        val_metrics = {
            "val_accuracy": float(acc_va),
            "val_f1_macro": float(f1_macro_va),
            "val_f1_weighted": float(f1_weighted_va),
        }
        print(
            f"[metrics] Validation ({vs:.0%} holdout, stratified): "
            f"accuracy={acc_va:.4f}  F1_macro={f1_macro_va:.4f}  F1_weighted={f1_weighted_va:.4f}"
        )
        clf.fit(X, y_enc)
    else:
        clf.fit(X, y_enc)

    pred_tr = clf.predict(X)
    acc_tr = accuracy_score(y_enc, pred_tr)
    f1_macro_tr = f1_score(y_enc, pred_tr, average="macro", zero_division=0)
    f1_weighted_tr = f1_score(y_enc, pred_tr, average="weighted", zero_division=0)
    train_metrics = {
        "train_accuracy": float(acc_tr),
        "train_f1_macro": float(f1_macro_tr),
        "train_f1_weighted": float(f1_weighted_tr),
    }
    print(
        f"[metrics] Training (full fit, in-sample): "
        f"accuracy={acc_tr:.4f}  F1_macro={f1_macro_tr:.4f}  F1_weighted={f1_weighted_tr:.4f}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le}, MODEL_PATH)

    meta = {
        "classes": le.classes_.tolist(),
        "n_samples": int(len(y)),
        "feature_dim": int(X.shape[1]),
        "max_per_class": args.max_per_class,
        "roi_mode": args.roi_mode,
        "yolo_weights": args.yolo_weights if args.roi_mode == "yolo" else None,
        "aug_per_image": args.aug_per_image,
        **train_metrics,
        **val_metrics,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Da luu model: {MODEL_PATH}")
    print(f"Lop: {meta['classes']}, so mau: {meta['n_samples']}")


if __name__ == "__main__":
    main()
