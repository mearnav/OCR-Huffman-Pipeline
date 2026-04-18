import base64
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent / "stage1_ocr"))
sys.path.insert(0, str(Path(__file__).parent / "stage2_huffman"))

from denoiser import UNet, denoise
from ocr import OCRNet, predict_batch, EMNIST_MEAN, EMNIST_STD
from segmentation import segment_chars
from huffman import encode, decode, AdaptiveHuffmanTree
from metrics import compression_ratio, shannon_entropy, avg_code_length, encoding_efficiency

app = FastAPI(title="OCR + Huffman Pipeline")

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
weights = Path(__file__).parent / "stage1_ocr" / "weights"

denoiser_model = UNet(base_ch=32).to(device)
denoiser_model.load_state_dict(torch.load(weights / "denoiser.pth", map_location=device, weights_only=True))
denoiser_model.eval()

ocr_model = OCRNet().to(device)
ocr_model.load_state_dict(torch.load(weights / "ocr.pth", map_location=device, weights_only=True))
ocr_model.eval()

_metrics = json.loads((weights / "metrics.json").read_text()) if (weights / "metrics.json").exists() else {}

_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
])


class CompressRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/ocr")
async def ocr_endpoint(image: UploadFile = File(...)):
    t0 = time.perf_counter()

    img = Image.open(io.BytesIO(await image.read()))
    denoised = denoise(denoiser_model, img, device)
    crops = segment_chars(denoised)

    img_tensors = [_to_tensor(crop.convert("L")) for crop in crops if not isinstance(crop, str)]
    predictions = predict_batch(ocr_model, img_tensors, device)

    pred_iter = iter(predictions)
    chars = [next(pred_iter) if not isinstance(crop, str) else crop for crop in crops]
    text = " ".join("".join(chars).splitlines())

    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "text": text,
        "char_count": len(text),
        "latency_ms": round(latency_ms, 2),
        "emnist_digit_val_acc": _metrics.get("emnist_digit_val_acc"),
    }


@app.post("/huffman")
def compress(req: CompressRequest):
    t0 = time.perf_counter()

    compressed = encode(req.text)
    recovered = decode(compressed)

    original_bytes = len(req.text.encode("utf-8"))
    comp_bytes = len(compressed)

    tree = AdaptiveHuffmanTree()
    sym_total_bits: dict[int, int] = {}
    payload_bytes = req.text.encode("utf-8")
    for b in payload_bytes:
        sym_total_bits[b] = sym_total_bits.get(b, 0) + len(tree.encode_symbol(b))

    counts = Counter(payload_bytes)
    bits_per_sym = {b: sym_total_bits[b] / counts[b] for b in sym_total_bits}

    entropy = shannon_entropy(req.text) if req.text else 0.0
    avg_len = avg_code_length(bits_per_sym, req.text) if req.text else 0.0
    efficiency = encoding_efficiency(entropy, avg_len)
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "compressed_b64": base64.b64encode(compressed).decode(),
        "recovered_text": recovered,
        "lossless": recovered == req.text,
        "original_bytes": original_bytes,
        "compressed_bytes": comp_bytes,
        "compression_ratio": round(compression_ratio(original_bytes, comp_bytes), 4),
        "entropy_bits_per_symbol": round(entropy, 4),
        "avg_code_length": round(avg_len, 4),
        "encoding_efficiency": round(efficiency, 4),
        "latency_ms": round(latency_ms, 2),
    }