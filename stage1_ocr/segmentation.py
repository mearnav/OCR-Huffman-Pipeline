import cv2
import numpy as np
import pytesseract
from PIL import Image

_TESS_MAX_WIDTH = 1000

def _make_crop(gray_arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    crop = gray_arr[y0:y1, x0:x1]
    h, w = crop.shape
    side = max(h, w)
    padded = np.full((side, side), 255, dtype=np.uint8)
    padded[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = crop
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    return Image.fromarray(255 - resized)


def segment_chars(img: Image.Image) -> list[Image.Image | str]:
    """Returns crops interleaved with ' ' (word gap) and '\\n' (line break) sentinels."""
    gray = img.convert("L")
    gray_arr = np.array(gray)
    H, W = gray_arr.shape

    scale = min(1.0, _TESS_MAX_WIDTH / W)
    if scale < 1.0:
        tess_w, tess_h = int(W * scale), int(H * scale)
        tess_img = gray.resize((tess_w, tess_h), Image.LANCZOS)
    else:
        tess_img = gray
        tess_h = H

    try:
        box_str = pytesseract.image_to_boxes(tess_img, config="--psm 6 --oem 1")
    except Exception:
        return []

    boxes = []
    for line in box_str.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        l, b, r, t = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
        # Convert from Tesseract bottom-left origin to top-left, then scale to original size
        x0 = int(l / scale)
        y0 = int((tess_h - t) / scale)
        x1 = int(r / scale)
        y1 = int((tess_h - b) / scale)
        if x1 > x0 and y1 > y0 and (x1 - x0) >= 3 and (y1 - y0) >= 5:
            boxes.append((x0, y0, x1, y1))

    if not boxes:
        return []

    boxes.sort(key=lambda b: (b[1] + b[3]) / 2)
    lines: list[list[tuple]] = [[boxes[0]]]
    for box in boxes[1:]:
        line = lines[-1]
        avg_cy = sum((b[1] + b[3]) / 2 for b in line) / len(line)
        avg_h = sum(b[3] - b[1] for b in line) / len(line)
        if abs((box[1] + box[3]) / 2 - avg_cy) > avg_h * 0.6:
            lines.append([box])
        else:
            line.append(box)

    for line in lines:
        line.sort(key=lambda b: b[0])
    lines = [line for line in lines if len(line) > 1]

    result: list[Image.Image | str] = []
    for line_idx, line_boxes in enumerate(lines):
        if line_idx > 0:
            result.append('\n')

        avg_h = sum(b[3] - b[1] for b in line_boxes) / len(line_boxes)
        word_gap = avg_h * 0.4

        for i, (x0, y0, x1, y1) in enumerate(line_boxes):
            if i > 0 and (x0 - line_boxes[i - 1][2]) >= word_gap:
                result.append(' ')
            result.append(_make_crop(gray_arr, x0, y0, x1, y1))

    return result