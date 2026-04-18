import random
import string
from huffman import encode, decode


def roundtrip(text: str):
    assert decode(encode(text)) == text


def test_empty():
    roundtrip("")


def test_single_char():
    roundtrip("a")


def test_repeated_char():
    roundtrip("aaaaaaaaaa")


def test_all_ascii():
    roundtrip(string.printable)


def test_short_sentence():
    roundtrip("hello world")


def test_digits():
    roundtrip("0123456789012345678901234567890123456789")


def test_lorem_ipsum():
    text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
    )
    roundtrip(text)


def test_unicode():
    roundtrip("café résumé naïve 日本語 العربية")


def test_random_short():
    random.seed(42)
    for _ in range(50):
        length = random.randint(1, 64)
        text = "".join(random.choices(string.ascii_letters + string.digits + " ", k=length))
        roundtrip(text)


def test_random_long():
    random.seed(99)
    text = "".join(random.choices(string.printable, k=2000))
    roundtrip(text)


def test_binary_like_content():
    payload = bytes(range(256))
    from huffman import AdaptiveHuffmanTree, _pack_bits, _unpack_bits
    import struct

    enc_tree = AdaptiveHuffmanTree()
    bits = []
    for b in payload:
        bits.extend(enc_tree.encode_symbol(b))
    compressed = struct.pack(">I", len(payload)) + _pack_bits(bits)

    dec_tree = AdaptiveHuffmanTree()
    unpacked = iter(_unpack_bits(compressed[4:]))
    result = bytearray(dec_tree.decode_symbol(unpacked) for _ in range(len(payload)))
    assert result == payload