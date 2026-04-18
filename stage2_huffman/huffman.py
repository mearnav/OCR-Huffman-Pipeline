"""
Adaptive Huffman coding using the FGK algorithm (Faller-Gallager-Knuth).

Encoded format:
  [4 bytes big-endian: number of input bytes] [packed bitstream]

The decoder reads the byte count from the header and stops after reconstructing
exactly that many bytes, avoiding the need for a special EOF symbol.
"""

import struct
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    weight: int
    symbol: Optional[int]
    order: int
    parent: Optional[Node] = field(default=None, repr=False)
    left: Optional[Node] = field(default=None, repr=False)
    right: Optional[Node] = field(default=None, repr=False)

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class AdaptiveHuffmanTree:
    def __init__(self):
        # NYT starts with highest order number; siblings are ordered by weight then order
        self.nyt = Node(weight=0, symbol=None, order=512)
        self.root = self.nyt
        self.nodes: dict[int, Node] = {}
        self._next_order = 511

    def encode_symbol(self, sym: int) -> list[int]:
        if sym in self.nodes:
            bits = self._path_to(self.nodes[sym])
            self._update(self.nodes[sym])
        else:
            bits = self._path_to(self.nyt) + _int_to_bits(sym, 8)
            self._add_symbol(sym)
        return bits

    def decode_symbol(self, bit_iter) -> int:
        node = self.root
        while not node.is_leaf:
            bit = next(bit_iter)
            node = node.left if bit == 0 else node.right

        if node is self.nyt:
            sym = _bits_to_int([next(bit_iter) for _ in range(8)])
            self._add_symbol(sym)
        else:
            sym = node.symbol
            self._update(node)
        return sym

    def _path_to(self, node: Node) -> list[int]:
        bits: list[int] = []
        while node.parent is not None:
            bits.append(0 if node.parent.left is node else 1)
            node = node.parent
        bits.reverse()
        return bits

    def _add_symbol(self, sym: int):
        new_internal = Node(weight=0, symbol=None, order=self._next_order, parent=self.nyt.parent)
        self._next_order -= 1
        new_leaf = Node(weight=0, symbol=sym, order=self._next_order, parent=new_internal)
        self._next_order -= 1
        new_nyt = Node(weight=0, symbol=None, order=self._next_order, parent=new_internal)
        self._next_order -= 1

        new_internal.left = new_nyt
        new_internal.right = new_leaf

        if self.nyt.parent is None:
            self.root = new_internal
        elif self.nyt.parent.left is self.nyt:
            self.nyt.parent.left = new_internal
        else:
            self.nyt.parent.right = new_internal

        self.nyt = new_nyt
        self.nodes[sym] = new_leaf
        self._update(new_leaf)

    def _update(self, node: Node):
        current = node
        while current is not None:
            leader = self._find_block_leader(current)
            if leader is not current and leader is not current.parent:
                self._swap(current, leader)
            current.weight += 1
            current = current.parent

    def _find_block_leader(self, node: Node) -> Node:
        # sibling property: among all nodes with the same weight, the one with highest order is the leader
        leader = node

        def _walk(n: Node | None):
            nonlocal leader
            if n is None:
                return
            if n.weight == node.weight and n.order > leader.order:
                leader = n
            _walk(n.left)
            _walk(n.right)

        _walk(self.root)
        return leader

    def _swap(self, a: Node, b: Node):
        a_parent, b_parent = a.parent, b.parent
        a_is_left = a_parent and a_parent.left is a
        b_is_left = b_parent and b_parent.left is b

        if a_parent:
            if a_is_left:
                a_parent.left = b
            else:
                a_parent.right = b
        if b_parent:
            if b_is_left:
                b_parent.left = a
            else:
                b_parent.right = a

        a.parent, b.parent = b_parent, a_parent

        if b_parent is None:
            self.root = a
        if a_parent is None:
            self.root = b

        a.order, b.order = b.order, a.order


def _int_to_bits(value: int, n_bits: int) -> list[int]:
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


def _bits_to_int(bits: list[int]) -> int:
    result = 0
    for b in bits:
        result = (result << 1) | b
    return result


def _pack_bits(bits: list[int]) -> bytes:
    pad = (8 - len(bits) % 8) % 8
    bits = bits + [0] * pad
    return bytes(_bits_to_int(bits[i:i + 8]) for i in range(0, len(bits), 8))


def _unpack_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def encode(text: str) -> bytes:
    payload = text.encode("utf-8")
    tree = AdaptiveHuffmanTree()
    bits: list[int] = []
    for byte in payload:
        bits.extend(tree.encode_symbol(byte))
    header = struct.pack(">I", len(payload))
    return header + _pack_bits(bits)


def decode(data: bytes) -> str:
    n_bytes = struct.unpack(">I", data[:4])[0]
    bits = _unpack_bits(data[4:])
    bit_iter = iter(bits)
    tree = AdaptiveHuffmanTree()
    result = bytearray()
    for _ in range(n_bytes):
        result.append(tree.decode_symbol(bit_iter))
    return result.decode("utf-8")