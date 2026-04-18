import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ocr import EMNIST_LABELS

ROOT = Path(__file__).parent.parent / "Simulated Noisy Office"
OUT = Path(__file__).parent.parent / "data" / "noisyoffice_crops"

_EMNIST_SET = set(EMNIST_LABELS)
_NOISE_TYPES = list("cfpw")


def _to_label(ch: str) -> str | None:
    if ch in _EMNIST_SET:
        return ch
    # EMNIST merges 15 visually identical lowercase/uppercase pairs into uppercase
    if ch.isalpha() and ch.upper() in _EMNIST_SET:
        return ch.upper()
    return None


def _save_crop(arr: np.ndarray, label: str, split: str, name: str):
    h, w = arr.shape
    side = max(h, w)
    padded = np.full((side, side), 255, dtype=np.uint8)
    padded[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = arr
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    # invert to match EMNIST format: white character on black background
    out_dir = OUT / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(255 - resized).save(out_dir / f"{name}.png")


def extract(split: str):
    tag = {"train": "TR", "val": "VA", "test": "TE"}[split]
    clean_dir = ROOT / "clean_images_grayscale"
    noisy_dir = ROOT / "simulated_noisy_images_grayscale"

    clean_files = sorted(f for f in clean_dir.glob("*.png") if tag in f.stem)
    print(f"[{split}] {len(clean_files)} clean images")

    saved = 0
    for clean_path in clean_files:
        noisy_paths = []
        for nt in _NOISE_TYPES:
            p = noisy_dir / re.sub(r"_Clean_", f"_Noise{nt}_", clean_path.name)
            if p.exists():
                noisy_paths.append(p)

        clean_arr = np.array(Image.open(clean_path).convert("L"))
        H, W = clean_arr.shape

        try:
            # image_to_boxes: char left bottom right top page  (y from bottom-left)
            box_str = pytesseract.image_to_boxes(
                Image.fromarray(clean_arr), config="--psm 6 --oem 3"
            )
        except Exception as e:
            print(f"  skipping {clean_path.name}: {e}")
            continue

        noisy_arrs = [np.array(Image.open(p).convert("L")) for p in noisy_paths]

        for line in box_str.strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            ch, l, b, r, t = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

            label = _to_label(ch)
            if label is None:
                continue

            # tesseract y-axis is bottom-left origin; convert to top-left
            y0, y1, x0, x1 = H - t, H - b, l, r
            if x1 <= x0 or y1 <= y0 or (x1 - x0) < 5 or (y1 - y0) < 8:
                continue

            stem = f"{clean_path.stem}_{l}_{b}"
            _save_crop(clean_arr[y0:y1, x0:x1], label, split, f"clean_{stem}")
            for i, noisy_arr in enumerate(noisy_arrs):
                _save_crop(noisy_arr[y0:y1, x0:x1], label, split, f"noisy{i}_{stem}")
            saved += 1

    print(f"[{split}] saved {saved} character instances")


if __name__ == "__main__":
    if not ROOT.exists():
        print(f"NoisyOffice not found at {ROOT}")
        sys.exit(1)
    for split in ("train", "val", "test"):
        extract(split)
    print(f"\nDone. Crops at {OUT}")