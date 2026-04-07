"""Train deep learning classifier for fruit categories (transfer learning)."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fruit_project.config import DATA_DIR, DL_META_PATH, DL_MODEL_PATH, MODEL_DIR
from fruit_project.dl_classifier import build_model


def safe_imread(path: Path) -> np.ndarray | None:
    """Read image robustly on Windows paths containing Unicode characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@dataclass
class Sample:
    path: Path
    label_idx: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_samples(data_dir: Path, max_per_class: int | None, seed: int) -> tuple[list[Sample], list[str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rng = random.Random(seed)
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: list[Sample] = []
    for cname in classes:
        paths = [p for p in (data_dir / cname).rglob("*") if p.is_file() and p.suffix.lower() in exts]
        rng.shuffle(paths)
        if max_per_class is not None:
            paths = paths[:max_per_class]
        for p in paths:
            samples.append(Sample(path=p, label_idx=class_to_idx[cname]))
    return samples, classes


class FruitDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int,
        train_mode: bool,
        aug_strength: float = 1.0,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.train_mode = train_mode
        self.aug_strength = max(0.0, aug_strength)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, bgr: np.ndarray) -> np.ndarray:
        if not self.train_mode:
            return bgr
        x = bgr
        # Horizontal flip
        if random.random() < 0.5:
            x = cv2.flip(x, 1)
        # Light rotation / scale
        h, w = x.shape[:2]
        angle = random.uniform(-18, 18) * self.aug_strength
        scale = random.uniform(0.90, 1.10)
        mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
        x = cv2.warpAffine(x, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        # Color jitter in HSV
        hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(random.uniform(-10, 10) * self.aug_strength)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(random.uniform(-30, 30) * self.aug_strength), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + int(random.uniform(-30, 30) * self.aug_strength), 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # Blur/noise
        if random.random() < 0.35:
            x = cv2.GaussianBlur(x, (3, 3), 0)
        if random.random() < 0.25:
            noise = np.random.normal(0, 6, x.shape).astype(np.int16)
            x = np.clip(x.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return x

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        bgr = safe_imread(s.path)
        if bgr is None:
            # return black image if read fails
            bgr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        bgr = self._augment(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img), s.label_idx


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_true.extend(yb.numpy().tolist())
    acc = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    return float(acc), float(f1_macro), float(f1_weighted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-name", type=str, default="efficientnet_b3", choices=["efficientnet_b3", "convnext_tiny", "efficientnet_b0"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--aug-strength", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(int(args.seed))

    samples, classes = collect_samples(DATA_DIR, args.max_per_class, args.seed)
    if len(samples) < 10 or len(classes) < 2:
        raise RuntimeError("Not enough data. Need >=10 images and >=2 classes.")

    y = np.array([s.label_idx for s in samples], dtype=np.int64)
    idx_all = np.arange(len(samples))
    idx_tr, idx_va = train_test_split(
        idx_all,
        test_size=float(args.val_split),
        random_state=int(args.seed),
        stratify=y,
    )
    train_samples = [samples[i] for i in idx_tr]
    val_samples = [samples[i] for i in idx_va]

    ds_tr = FruitDataset(train_samples, image_size=int(args.image_size), train_mode=True, aug_strength=float(args.aug_strength))
    ds_va = FruitDataset(val_samples, image_size=int(args.image_size), train_mode=False)
    dl_tr = DataLoader(ds_tr, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available())
    dl_va = DataLoader(ds_va, batch_size=max(16, int(args.batch_size)), shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, num_classes=len(classes)).to(device)

    # Class weights for imbalance
    cls_counts = np.bincount([s.label_idx for s in train_samples], minlength=len(classes)).astype(np.float32)
    cls_weights = (cls_counts.sum() / (cls_counts + 1e-9))
    cls_weights = cls_weights / cls_weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(cls_weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(3, int(args.epochs)))

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics = {"val_accuracy": 0.0, "val_f1_macro": 0.0, "val_f1_weighted": 0.0}

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        run_loss = 0.0
        n_seen = 0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += float(loss.item()) * int(xb.size(0))
            n_seen += int(xb.size(0))
        scheduler.step()
        tr_loss = run_loss / max(1, n_seen)
        va_acc, va_f1m, va_f1w = evaluate(model, dl_va, device)
        print(
            f"[epoch {epoch:02d}/{int(args.epochs)}] loss={tr_loss:.4f} "
            f"val_acc={va_acc:.4f} val_f1_macro={va_f1m:.4f} val_f1_weighted={va_f1w:.4f}"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_accuracy": float(va_acc),
                "val_f1_macro": float(va_f1m),
                "val_f1_weighted": float(va_f1w),
            }

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)

    # train metrics on train split with best checkpoint
    dl_tr_eval = DataLoader(ds_tr, batch_size=max(16, int(args.batch_size)), shuffle=False, num_workers=int(args.num_workers))
    tr_acc, tr_f1m, tr_f1w = evaluate(model, dl_tr_eval, device)
    print(
        "[metrics] Best checkpoint: "
        f"train_acc={tr_acc:.4f} train_f1_macro={tr_f1m:.4f} train_f1_weighted={tr_f1w:.4f} | "
        f"val_acc={best_metrics['val_accuracy']:.4f} val_f1_macro={best_metrics['val_f1_macro']:.4f} "
        f"val_f1_weighted={best_metrics['val_f1_weighted']:.4f}"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, str(DL_MODEL_PATH))
    meta = {
        "backend": "pytorch",
        "model_name": args.model_name,
        "image_size": int(args.image_size),
        "classes": classes,
        "n_samples_total": int(len(samples)),
        "n_train": int(len(train_samples)),
        "n_val": int(len(val_samples)),
        "max_per_class": args.max_per_class,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_accuracy": float(tr_acc),
        "train_f1_macro": float(tr_f1m),
        "train_f1_weighted": float(tr_f1w),
        **best_metrics,
    }
    DL_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved DL model: {DL_MODEL_PATH}")
    print(f"Saved DL meta:  {DL_META_PATH}")


if __name__ == "__main__":
    main()
