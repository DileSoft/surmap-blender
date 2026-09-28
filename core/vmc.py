"""The compressed ``.vmc`` format — Huffman codec, ported 1:1 from:

* ``src/terra/splay.cpp``    — ``InitSplay`` / ``decompress_line1`` / ``ExpandBuffer``
* ``src/terra/huff1.cpp``    — ``DHuff`` (compression half)
* ``src/terra/compress.cpp`` — ``CompressMap``

Layout::

    int32 offset[y] / int16 size[y]   interleaved, 6 bytes per row
    int32 decomp_tree1[512]            heights
    int32 decomp_tree3[512]            flags
    per-row compressed bytes           size[y] of them, starting at offset[y]

Each row is compressed in two passes over a *delta* stream:

* heights: ``d[i] = h[i] - h[i-1]``  (the decoder adds)
* flags:   ``d[i] = m[i] ^ m[i-1]``  (the decoder xors)

Both trees are shared by every row (like ``CompressMap``), which is what lets
the decoder below run all rows at once through numpy instead of one symbol at
a time in Python — a 33.5M-voxel world would otherwise take hours.
"""

from __future__ import annotations

import numpy as np

from .constants import H_SIZE

TREE_ENTRIES = 512
RECORD_BYTES = 6  # int32 offset + int16 size


class VmcError(ValueError):
    pass


def _read_i32(data, offset: int) -> np.ndarray:
    if offset < 0 or offset + TREE_ENTRIES * 4 > len(data):
        raise VmcError(f"tree at {offset} does not fit in {len(data)} bytes")
    flat = np.frombuffer(data, dtype="<i4", count=TREE_ENTRIES, offset=offset)
    return flat.astype(np.int32)


def read_table(data, size_y: int) -> tuple[np.ndarray, np.ndarray]:
    """The interleaved ``(offset, size)`` record of every row."""
    if len(data) < size_y * RECORD_BYTES:
        raise VmcError(f"VMC too small: {len(data)} < {size_y * RECORD_BYTES}")
    flat = np.frombuffer(data, dtype=np.uint8, count=size_y * RECORD_BYTES)
    rec = flat.reshape(size_y, RECORD_BYTES)
    offsets = rec[:, 0:4].copy().view("<i4").reshape(size_y)
    sizes = rec[:, 4:6].copy().view("<i2").reshape(size_y)
    return np.array(offsets, dtype=np.int64), np.array(sizes, dtype=np.int64)


def write_table(offsets: np.ndarray, sizes: np.ndarray) -> bytes:
    """Interleaved ``int32 offset`` / ``int16 size`` records, little endian."""
    n = int(offsets.size)
    if sizes.size != n:
        raise VmcError("offsets and sizes must have the same length")
    rec = np.empty((n, RECORD_BYTES), dtype=np.uint8)
    # reinterpret as bytes first: assigning int32 into a uint8 view would
    # truncate every value to its low byte
    rec[:, 0:4] = np.asarray(offsets, dtype="<i4").view(np.uint8).reshape(n, 4)
    rec[:, 4:6] = np.asarray(sizes, dtype="<i2").view(np.uint8).reshape(n, 2)
    return rec.tobytes()


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------
class VmcDecoder:
    """``InitSplay``: the two decomp trees plus their delta chars."""

    byte_size = TREE_ENTRIES * 4 * 2

    def __init__(self, data, offset: int) -> None:
        self.tree1 = _read_i32(data, offset)
        self.tree3 = _read_i32(data, offset + TREE_ENTRIES * 4)
        self.delta1 = (-self.tree1).astype(np.uint8)
        self.delta3 = (-self.tree3).astype(np.uint8)
        self._check()

    def _check(self) -> None:
        for name, tree in (("tree1", self.tree1), ("tree3", self.tree3)):
            bad = int(np.count_nonzero((tree <= 0) & (tree < -255)))
            if bad:
                raise VmcError(f"{name}: {bad} leaves outside the byte range")

    # -- reference (scalar) path, used by the tests -------------------------
    def expand_row(self, data, src_offset: int, height, meta, base: int) -> int:
        """``ExpandBuffer`` for a single row; returns the end offset.

        ``height``/``meta`` are flat arrays and ``base`` is the row's offset
        inside them.
        """
        after = self._run_scalar(self.tree1, self.delta1, data, src_offset,
                                 height, base, True)
        return self._run_scalar(self.tree3, self.delta3, data, after,
                                meta, base, False)

    def _run_scalar(self, tree, delta, data, start, dst, base, add) -> int:
        cindex = 1
        last = 0
        out = 0
        in_offset = start
        while out < H_SIZE:
            cur = int(data[in_offset])
            in_offset += 1
            for shift in range(7, -1, -1):
                cindex = (cindex << 1) + ((cur >> shift) & 1)
                if tree[cindex] <= 0:
                    d = int(delta[cindex])
                    last = (last + d) & 0xFF if add else (last ^ d) & 0xFF
                    dst[base + out] = last
                    out += 1
                    if out == H_SIZE:
                        break
                    cindex = 1
                else:
                    cindex = int(tree[cindex])
        return in_offset

    # -- vectorised path ----------------------------------------------------
    def decode_all(self, data, offsets: np.ndarray, sizes: np.ndarray,
                   height: np.ndarray, meta: np.ndarray) -> None:
        """Decodes the given rows at once, since both trees are shared.

        ``offsets``/``sizes`` pick the rows to decode, so a window of a big
        world costs only that window. Rows that finish early simply stop
        consuming, exactly like the scalar loop breaking out once it has
        produced ``H_SIZE`` symbols.
        """
        rows = np.arange(offsets.size, dtype=np.int64)
        buf = np.frombuffer(data, dtype=np.uint8)
        limit = buf.size

        inpos = np.array(offsets, dtype=np.int64)
        cindex = np.ones(offsets.size, dtype=np.int32)
        last = np.zeros(offsets.size, dtype=np.uint8)
        outpos = np.zeros(offsets.size, dtype=np.int64)
        count = np.zeros(offsets.size, dtype=np.int32)

        self._run_vector(tree=self.tree1, delta=self.delta1, buf=buf, limit=limit,
                         rows=rows, inpos=inpos, cindex=cindex, last=last,
                         outpos=outpos, count=count, dst=height, add=True)
        # the second pass starts with fresh per-pass state; only the input
        # position carries over from the first
        cindex[:] = 1
        last[:] = 0
        outpos[:] = 0
        count[:] = 0
        self._run_vector(tree=self.tree3, delta=self.delta3, buf=buf, limit=limit,
                         rows=rows, inpos=inpos, cindex=cindex, last=last,
                         outpos=outpos, count=count, dst=meta, add=False)

        bad = np.nonzero((inpos - offsets) != sizes)[0]
        if bad.size:
            y = int(bad[0])
            raise VmcError(
                f"VMC row {y} decode boundary mismatch: consumed {inpos[y] - offsets[y]} "
                f"of {sizes[y]} bytes"
            )

    @staticmethod
    def _run_vector(*, tree, delta, buf, limit, rows, inpos, cindex, last,
                    outpos, count, dst, add) -> None:
        """One pass over every row, ``add`` selects heights (+) or flags (^)."""
        span = count.size
        if span == 0:
            return
        active = count < H_SIZE
        # H_SIZE symbols of at most 32 bits each can never need more than this
        max_bytes = (H_SIZE * 32) // 8 + 1
        for _ in range(max_bytes):
            if not active.any():
                return
            # A row only consumes the bytes it actually reads, so the pointer
            # of a row that already finished must not keep moving.
            reading = active.copy()
            byte = buf[np.minimum(inpos, limit - 1)]
            for shift in (7, 6, 5, 4, 3, 2, 1, 0):
                bit = ((byte >> shift) & 1).astype(np.int32)
                # the tree walk: descend one bit, then follow the stored link
                shifted = np.minimum((cindex << 1) + bit, TREE_ENTRIES - 1)
                node = tree[shifted]
                leaf = active & (node <= 0)
                if leaf.any():
                    d = delta[shifted]
                    if add:
                        newlast = ((last.astype(np.int16) + d) & 0xFF).astype(np.uint8)
                    else:
                        newlast = np.bitwise_xor(last, d)
                    sel = np.nonzero(leaf)[0]
                    dst[rows[sel], outpos[sel]] = newlast[sel]
                    outpos[sel] += 1
                    last[sel] = newlast[sel]
                    count[sel] += 1
                    cindex[sel] = 1  # back to the root
                cindex[~leaf] = node[~leaf]
                active &= count < H_SIZE
            inpos[reading] += 1
        if active.any():
            raise VmcError("decoder ran out of bits before filling every row")


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------
class HuffEncoder:
    """Port of ``DHuff`` (huff1.cpp) — the compression half."""

    def __init__(self) -> None:
        self.father = np.zeros(512, dtype=np.int32)
        self.heap = np.zeros(257, dtype=np.int32)
        self.frequency = np.zeros(512, dtype=np.int32)
        self.code = np.zeros(256, dtype=np.uint32)
        self.code_length = np.zeros(256, dtype=np.uint8)
        self.decomp_tree = np.zeros(TREE_ENTRIES, dtype=np.int32)
        self.heap_length = 0

    # -- heap ---------------------------------------------------------------
    def _reheap(self, entry: int) -> None:
        flag = True
        value = int(self.heap[entry])
        while entry <= (self.heap_length >> 1) and flag:
            index = entry << 1
            if index < self.heap_length:
                if self.frequency[self.heap[index]] >= self.frequency[self.heap[index + 1]]:
                    index += 1
            if self.frequency[value] < self.frequency[self.heap[index]]:
                flag = False
            else:
                self.heap[entry] = self.heap[index]
                entry = index
        self.heap[entry] = value

    def _build_initial_heap(self) -> None:
        self.heap_length = 0
        for symbol in range(256):
            if self.frequency[symbol]:
                self.heap_length += 1
                self.heap[self.heap_length] = symbol
        for entry in range(self.heap_length, 0, -1):
            self._reheap(entry)

    def _build_code_tree(self) -> None:
        while self.heap_length != 1:
            value = int(self.heap[1])
            # the last heap slot moves to the root before the length shrinks
            self.heap[1] = self.heap[self.heap_length]
            self.heap_length -= 1
            self._reheap(1)
            findex = self.heap_length + 255
            self.frequency[findex] = (self.frequency[self.heap[1]]
                                      + self.frequency[value])
            self.father[value] = findex
            self.father[self.heap[1]] = -findex
            self.heap[1] = findex
            self._reheap(1)
        self.father[256] = 0

    def _generate_code_table(self) -> bool:
        for symbol in range(256):
            if not self.frequency[symbol]:
                self.code[symbol] = 0
                self.code_length[symbol] = 0
                continue
            length = 0
            bitcode = 0
            bit = 1
            parent = int(self.father[symbol])
            while parent:
                if parent < 0:
                    bitcode = (bitcode + bit) & 0xFFFFFFFF
                    parent = -parent
                parent = int(self.father[parent])
                bit = (bit * 2) & 0xFFFFFFFF
                length += 1
            self.code[symbol] = bitcode
            if length > 32:
                return False
            self.code_length[symbol] = length
        return True

    def _build_decomp_tree(self) -> None:
        self.decomp_tree[:] = 0
        self.decomp_tree[1] = 1
        node = 1
        for symbol in range(256):
            length = int(self.code_length[symbol])
            if not length:
                continue
            index = 1
            for shift in range(length - 1, 0, -1):
                index = int((self.decomp_tree[index] << 1)
                            + ((int(self.code[symbol]) >> shift) & 1))
                if index >= TREE_ENTRIES:
                    raise VmcError("decomp tree overflows 512 entries")
                if not self.decomp_tree[index]:
                    node += 1
                    self.decomp_tree[index] = node
            leaf = int((self.decomp_tree[index] << 1) + (int(self.code[symbol]) & 1))
            if leaf >= TREE_ENTRIES:
                raise VmcError("decomp tree overflows 512 entries")
            self.decomp_tree[leaf] = -symbol

    def build(self, freq: np.ndarray) -> None:
        """Builds a valid prefix code from a 256-entry frequency table."""
        self.frequency[:] = 0
        self.frequency[:256] = np.asarray(freq, dtype=np.int32)

        used = np.nonzero(self.frequency[:256])[0]
        if used.size == 0:
            self.frequency[0] = 1
            self.frequency[1] = 1
        elif used.size == 1:
            self.frequency[0 if used[0] != 0 else 1] = 1

        self._build_initial_heap()
        self._build_code_tree()
        if not self._generate_code_table():
            raise VmcError("Huffman code value out of range (cannot compress)")
        self._build_decomp_tree()

    # -- output -------------------------------------------------------------
    def tree_bytes(self) -> bytes:
        return np.asarray(self.decomp_tree, dtype="<i4").tobytes()

    def pack(self, values: np.ndarray) -> bytes:
        """Encodes one row's 2048 delta bytes into packed bits (MSB first)."""
        symbols = np.asarray(values, dtype=np.uint32)
        lengths = self.code_length[symbols].astype(np.int64)
        total = int(lengths.sum())
        if total == 0:
            return b""
        which = np.repeat(np.arange(symbols.size, dtype=np.int64), lengths)
        starts = np.empty(symbols.size, dtype=np.int64)
        starts[0] = 0
        np.cumsum(lengths[:-1], out=starts[1:])
        bit_pos = np.arange(total, dtype=np.int64) - starts[which]
        codes = self.code[symbols][which]
        bits = (codes >> (lengths[which] - 1 - bit_pos).astype(np.uint32)) & 1
        return np.packbits(bits.astype(np.uint8)).tobytes()


# ---------------------------------------------------------------------------
# whole-file entry points
# ---------------------------------------------------------------------------
def load_vmc(data, size_y: int, rows=None) -> tuple[np.ndarray, np.ndarray]:
    """Decodes a ``.vmc`` buffer into ``(height, meta)`` arrays.

    ``rows`` selects which rows of the file to decode (default: all). A row is
    self-contained once the shared trees are read, so decoding a window costs
    only the window -- importing a 512x512 window out of a 16384-row world is
    then instant instead of a full-map pass.
    """
    offsets, sizes = read_table(data, size_y)
    if rows is None:
        sel = np.arange(size_y, dtype=np.int64)
    else:
        sel = np.asarray(rows, dtype=np.int64).reshape(-1)
        if sel.size and (int(sel.min()) < 0 or int(sel.max()) >= size_y):
            raise VmcError(f"row out of range: {int(sel.min())}..{int(sel.max())} "
                           f"for a {size_y}-row file")
    decoder = VmcDecoder(data, size_y * RECORD_BYTES)
    height = np.empty((sel.size, H_SIZE), dtype=np.uint8)
    meta = np.empty((sel.size, H_SIZE), dtype=np.uint8)
    decoder.decode_all(data, offsets[sel], sizes[sel], height, meta)
    return height, meta


def save_vmc(height: np.ndarray, meta: np.ndarray) -> bytes:
    """Encodes ``(height, meta)`` arrays into a ``.vmc`` buffer."""
    if height.shape != meta.shape or height.ndim != 2 or height.shape[1] != H_SIZE:
        raise VmcError(f"bad surface shape {height.shape}")
    size_y = height.shape[0]

    # One frequency table for all heights, one for all flags (CompressMap).
    h = height.astype(np.int16)
    dh = np.empty_like(height)
    dh[:, 0] = h[:, 0]
    dh[:, 1:] = (h[:, 1:] - h[:, :-1]) & 0xFF
    m = meta.astype(np.uint8)
    dm = np.empty_like(meta)
    dm[:, 0] = m[:, 0]
    dm[:, 1:] = np.bitwise_xor(m[:, 1:], m[:, :-1])

    enc_h = HuffEncoder()
    enc_f = HuffEncoder()
    enc_h.build(np.bincount(dh.ravel(), minlength=256).astype(np.int32))
    enc_f.build(np.bincount(dm.ravel(), minlength=256).astype(np.int32))

    data_start = size_y * RECORD_BYTES + VmcDecoder.byte_size
    chunks: list[bytes] = []
    offsets = np.empty(size_y, dtype=np.int64)
    sizes = np.empty(size_y, dtype=np.int64)
    offset = data_start
    for y in range(size_y):
        a = enc_h.pack(dh[y])
        b = enc_f.pack(dm[y])
        chunks.append(a)
        chunks.append(b)
        offsets[y] = offset
        sizes[y] = len(a) + len(b)
        offset += sizes[y]

    parts = [write_table(offsets, sizes), enc_h.tree_bytes(), enc_f.tree_bytes()]
    parts.extend(chunks)
    return b"".join(parts)
