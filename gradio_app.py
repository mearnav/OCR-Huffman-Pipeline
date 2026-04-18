import sys
import base64
from collections import Counter
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent / "stage1_ocr"))
sys.path.insert(0, str(Path(__file__).parent / "stage2_huffman"))

from denoiser import UNet, denoise
from ocr import OCRNet, predict_batch, EMNIST_MEAN, EMNIST_STD
from segmentation import segment_chars
from huffman import encode, decode, AdaptiveHuffmanTree
from metrics import compression_ratio, shannon_entropy, avg_code_length, encoding_efficiency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = Path(__file__).parent / "stage1_ocr" / "weights"

denoiser_model = UNet(base_ch=32).to(device)
denoiser_model.load_state_dict(torch.load(weights / "denoiser.pth", map_location=device, weights_only=True))
denoiser_model.eval()

ocr_model = OCRNet().to(device)
ocr_model.load_state_dict(torch.load(weights / "ocr.pth", map_location=device, weights_only=True))
ocr_model.eval()

# EMNIST_MEAN / EMNIST_STD are precomputed dataset statistics (fixed)
_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
])


def run_pipeline(uploaded_img):
    if uploaded_img is None:
        return None, "No image uploaded."

    img = Image.fromarray(uploaded_img)

    denoised = denoise(denoiser_model, img, device)
    crops = segment_chars(denoised)

    img_tensors = [_to_tensor(crop.convert("L")) for crop in crops if not isinstance(crop, str)]
    predictions = predict_batch(ocr_model, img_tensors, device)

    pred_iter = iter(predictions)
    chars = [next(pred_iter) if not isinstance(crop, str) else crop for crop in crops]

    text = "".join(chars)
    compressed_text = " ".join(text.splitlines())

    compressed = encode(compressed_text)
    comp_b64 = base64.b64encode(compressed).decode()
    recovered = decode(compressed)

    original_bytes = len(compressed_text.encode("utf-8"))
    comp_bytes = len(compressed)

    tree = AdaptiveHuffmanTree()
    sym_total_bits: dict[int, int] = {}
    payload_bytes = compressed_text.encode("utf-8")
    for b in payload_bytes:
        sym_total_bits[b] = sym_total_bits.get(b, 0) + len(tree.encode_symbol(b))

    counts = Counter(payload_bytes)
    bits_per_sym = {b: sym_total_bits[b] / counts[b] for b in sym_total_bits}

    entropy = shannon_entropy(compressed_text) if compressed_text else 0.0
    avg_len = avg_code_length(bits_per_sym, compressed_text) if compressed_text else 0.0
    efficiency = encoding_efficiency(entropy, avg_len)
    ratio = compression_ratio(original_bytes, comp_bytes)

    match_icon = "LOSSLESS RECOVERY VERIFIED" if recovered == compressed_text else "RECOVERY MISMATCH"

    stats = (
        f"Extracted text: {text}\n\n"
        f"Compressed (base64 preview): {comp_b64[:80]}...\n\n"
        f"Original: {original_bytes} bytes  |  Compressed: {comp_bytes} bytes\n"
        f"Compression ratio: {ratio:.3f}x\n"
        f"Shannon entropy: {entropy:.3f} bits/symbol\n"
        f"Avg code length: {avg_len:.3f} bits/symbol\n"
        f"Encoding efficiency: {efficiency:.3f}\n\n"
        f"{match_icon} Lossless: recovered = {recovered}"
    )

    return denoised, stats


with gr.Blocks(title="OCR & Huffman Pipeline") as demo:
    gr.Markdown("## Noisy Document OCR - Adaptive Huffman Compression")

    with gr.Row():
        inp = gr.Image(label="Upload noisy image", type="numpy", sources=["upload"])
        out_denoised = gr.Image(label="Denoised")

    btn = gr.Button("Run Pipeline", variant="primary")

    out_stats = gr.Textbox(label="Metrics", lines=12)

    btn.click(
        run_pipeline,
        inputs=[inp],
        outputs=[out_denoised, out_stats],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)