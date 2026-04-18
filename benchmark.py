import sys
import csv
import time
import statistics
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent / "stage1_ocr"))
sys.path.insert(0, str(Path(__file__).parent / "stage2_huffman"))

from denoiser import UNet, denoise, NoisyOfficeDataset
from ocr import OCRNet, predict_batch, _eval_accuracy, _noise_transform, NoisyOfficeCharDataset, EMNIST_MEAN, EMNIST_STD
from segmentation import segment_chars
from huffman import encode, decode, AdaptiveHuffmanTree
from metrics import compression_ratio, shannon_entropy, avg_code_length, encoding_efficiency

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
weights = Path(__file__).parent / "stage1_ocr" / "weights"
data_root = Path(__file__).parent / "data"


def load_models():
    d = UNet(base_ch=32).to(device)
    d.load_state_dict(torch.load(weights / "denoiser.pth", map_location=device, weights_only=True))
    d.eval()

    o = OCRNet().to(device)
    o.load_state_dict(torch.load(weights / "ocr.pth", map_location=device, weights_only=True))
    o.eval()
    return d, o


def ocr_accuracy_benchmarks(ocr_model):
    norm = transforms.Normalize(EMNIST_MEAN, EMNIST_STD)
    results = {}

    for noise_type, label, display in [
        (None, "emnist_mnist_no_noise", "EMNIST-MNIST, no noise"),
        ("gaussian", "emnist_mnist_gaussian", "EMNIST-MNIST, gaussian σ=0.15"),
        ("salt_pepper", "emnist_mnist_saltpepper", "EMNIST-MNIST, salt-pepper r=0.05"),
    ]:
        tf = transforms.Compose([_noise_transform(noise_type), norm])
        ds = datasets.EMNIST(data_root, split="mnist", train=False, download=True, transform=tf)
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)
        acc = _eval_accuracy(ocr_model, loader, device)
        results[label] = acc
        print(f"OCR accuracy [{display}]: {acc:.4f}")

    crops_root = data_root / "noisyoffice_crops"
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(EMNIST_MEAN, EMNIST_STD)])
    no_val = NoisyOfficeCharDataset("val", crops_root, val_tf)
    if len(no_val) > 0:
        loader = DataLoader(no_val, batch_size=512, shuffle=False, num_workers=4)
        acc = _eval_accuracy(ocr_model, loader, device)
        results["noisyoffice_crops_47class"] = acc
        print(f"OCR accuracy [NoisyOffice crops, 47-class]: {acc:.4f}")
    else:
        print("NoisyOffice crops not found — skipping 47-class val benchmark")

    return results


def denoiser_psnr(denoiser_model):
    root = Path(__file__).parent / "Simulated Noisy Office"
    test_ds = NoisyOfficeDataset("test", root, patches_per_img=8)
    loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)
    mse_vals = []
    denoiser_model.eval()
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            mse_vals.append(F.mse_loss(denoiser_model(noisy), clean).item())
    avg_mse = statistics.mean(mse_vals)
    psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else float("inf")
    print(f"Denoiser test MSE: {avg_mse:.6f}  PSNR: {psnr:.2f} dB")
    return avg_mse, psnr


def pipeline_latency(denoiser_model, ocr_model, n_runs: int = 20):
    noisy_img = _get_test_image()

    times_denoise, times_segment, times_ocr = [], [], []

    _to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(EMNIST_MEAN, EMNIST_STD),
    ])

    for _ in range(n_runs):
        t0 = time.perf_counter()
        denoised = denoise(denoiser_model, noisy_img, device)
        times_denoise.append((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        crops = segment_chars(denoised)
        times_segment.append((time.perf_counter() - t1) * 1000)

        t2 = time.perf_counter()
        img_tensors = [_to_tensor(crop.convert("L")) for crop in crops if not isinstance(crop, str)]
        if img_tensors:
            predict_batch(ocr_model, img_tensors, device)
        times_ocr.append((time.perf_counter() - t2) * 1000)

    n_crops = len([c for c in crops if not isinstance(c, str)])
    print(f"  (test image yielded {n_crops} character crops)")

    d_mean = statistics.mean(times_denoise)
    s_mean = statistics.mean(times_segment)
    o_mean = statistics.mean(times_ocr)
    total_mean = d_mean + s_mean + o_mean

    print(f"\nLatency (ms)       mean")
    print(f"  Denoiser:       {d_mean:6.1f}")
    print(f"  Segmentation:   {s_mean:6.1f}")
    print(f"  OCR:            {o_mean:6.1f}")
    return d_mean, s_mean, o_mean


def compression_benchmarks(n_runs: int = 50):
    samples = [
        "3141592653589793",
        "hello world this is a test",
        "0" * 100,
        "the quick brown fox jumps over the lazy dog",
    ]
    ratios, entropies, efficiencies = [], [], []
    encode_times, decode_times = [], []

    for text in samples:
        t0 = time.perf_counter()
        for _ in range(n_runs):
            compressed = encode(text)
        encode_times.append((time.perf_counter() - t0) / n_runs * 1000)

        t1 = time.perf_counter()
        for _ in range(n_runs):
            decode(compressed)
        decode_times.append((time.perf_counter() - t1) / n_runs * 1000)

        orig = len(text.encode("utf-8"))
        comp = len(compressed)
        ratio = compression_ratio(orig, comp)
        entropy = shannon_entropy(text)

        tree = AdaptiveHuffmanTree()
        sym_total_bits: dict[int, int] = {}
        payload_bytes = text.encode("utf-8")
        for b in payload_bytes:
            sym_total_bits[b] = sym_total_bits.get(b, 0) + len(tree.encode_symbol(b))
        counts = Counter(payload_bytes)
        bps = {b: sym_total_bits[b] / counts[b] for b in sym_total_bits}

        avg_len = avg_code_length(bps, text)
        eff = encoding_efficiency(entropy, avg_len)
        ratios.append(ratio)
        entropies.append(entropy)
        efficiencies.append(eff)
        print(f"  '{text[:40]}' ratio={ratio:.3f} entropy={entropy:.3f} eff={eff:.3f}")

    enc_mean = statistics.mean(encode_times)
    dec_mean = statistics.mean(decode_times)
    print(f"\nAvg compression ratio: {statistics.mean(ratios):.3f}")
    print(f"\nHuffman latency (mean over {n_runs} runs per sample)")
    print(f"  Encode: {enc_mean:.3f} ms")
    print(f"  Decode: {dec_mean:.3f} ms")
    return ratios, entropies, efficiencies, enc_mean, dec_mean


def _get_test_image() -> Image.Image:
    noisy_dir = Path(__file__).parent / "Simulated Noisy Office" / "simulated_noisy_images_grayscale"
    return Image.open(sorted(noisy_dir.glob("*TE*.png"))[0]).convert("L")


if __name__ == "__main__":
    print(f"Device: {device}")
    print("Loading models...")
    denoiser_model, ocr_model = load_models()

    print("\nOCR Accuracy")
    ocr_results = ocr_accuracy_benchmarks(ocr_model)

    print("\nDenoiser PSNR")
    mse, psnr = denoiser_psnr(denoiser_model)

    print("\nPipeline Latency")
    latency = pipeline_latency(denoiser_model, ocr_model)

    print("\nCompression Metrics")
    comp_results = compression_benchmarks()

    d_mean, s_mean, o_mean = latency
    total_mean = d_mean + s_mean + o_mean
    ratios, entropies, efficiencies, enc_mean, dec_mean = comp_results
    e2e_mean = total_mean + enc_mean + dec_mean

    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in ocr_results.items():
            w.writerow([k, f"{v:.4f}"])
        w.writerow(["denoiser_mse", f"{mse:.6f}"])
        w.writerow(["denoiser_psnr_db", f"{psnr:.2f}"])
        w.writerow(["latency_denoise_mean_ms", f"{d_mean:.1f}"])
        w.writerow(["latency_segment_mean_ms", f"{s_mean:.1f}"])
        w.writerow(["latency_ocr_mean_ms", f"{o_mean:.1f}"])
        w.writerow(["compression_avg_ratio", f"{statistics.mean(ratios):.3f}"])
        w.writerow(["compression_avg_entropy", f"{statistics.mean(entropies):.3f}"])
        w.writerow(["compression_avg_efficiency", f"{statistics.mean(efficiencies):.3f}"])
        w.writerow(["latency_huffman_encode_mean_ms", f"{enc_mean:.3f}"])
        w.writerow(["latency_huffman_decode_mean_ms", f"{dec_mean:.3f}"])
        w.writerow(["latency_total_mean_ms", f"{e2e_mean:.1f}"])

    print("\nSaved benchmark_results.csv")
