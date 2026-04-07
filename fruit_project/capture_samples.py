"""
Chup mau tu camera/phone: nhan phim 1-9 de luu vao data/fruits/<ten_loai>/
Chay: python -m fruit_project.capture_samples apple
      (moi lan bam SPACE luu 1 anh)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import cv2

from fruit_project.config import DATA_DIR, parse_camera_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("class_name", help="Ten loai trai cay (thu muc con trong data/fruits)")
    ap.add_argument("--camera", default=None, help="0 hoac URL IP Webcam")
    args = ap.parse_args()

    if args.camera is not None:
        import os

        os.environ["FRUIT_CAMERA"] = args.camera

    out_dir = DATA_DIR / args.class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(parse_camera_source())
    if not cap.isOpened():
        print("Khong mo duoc camera.", file=sys.stderr)
        sys.exit(1)

    print(f"Luu vao: {out_dir}")
    print("SPACE = luu anh, q = thoat")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Capture (SPACE=save)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{idx}.jpg"
            path = out_dir / name
            cv2.imwrite(str(path), frame)
            print("Saved", path)
            idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
