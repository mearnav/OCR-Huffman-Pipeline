import math
from collections import Counter


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes == 0:
        return float("inf")
    return original_bytes / compressed_bytes


def shannon_entropy(text: str) -> float:
    """Theoretical lower bound in bits/symbol over the byte distribution."""
    payload = text.encode("utf-8")
    counts = Counter(payload)
    n = len(payload)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def avg_code_length(code_lengths: dict[int, int], text: str) -> float:
    payload = text.encode("utf-8")
    counts = Counter(payload)
    n = len(payload)
    return sum((counts[sym] / n) * code_lengths.get(sym, 0) for sym in counts)


def encoding_efficiency(entropy: float, avg_len: float) -> float:
    if avg_len == 0:
        return 0.0
    return entropy / avg_len
