"""Uncompressed `.vmp` surface I/O — `load_vmp` from src/terra/vmap.cpp.

Row layout: ``H_SIZE`` height bytes followed by ``H_SIZE`` flag bytes, repeated
for every row (X wraps modulo ``H_SIZE``, Y is linear).

Heights and flags are exposed as 2-D ``uint8`` arrays of shape
``(size_y, H_SIZE)``.
"""

from __future__ import annotations

import os
from typing import BinaryIO

import numpy as np

from .constants import H_SIZE


class VmpError(ValueError):
    pass


def load_vmp_bytes(data: bytes | bytearray | memoryview, size_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Decodes a ``.vmp`` buffer into ``(height, meta)`` arrays."""
    total = H_SIZE * size_y
    need = total * 2
    if len(data) < need:
        raise VmpError(f"VMP too small: {len(data)} < {need}")
    arr = np.frombuffer(data, dtype=np.uint8, count=need)
    # (size_y, 2, H_SIZE) -> per-row [heights | flags]
    rows = arr.reshape(size_y, 2, H_SIZE)
    height = np.ascontiguousarray(rows[:, 0, :])
    meta = np.ascontiguousarray(rows[:, 1, :])
    return height, meta


def save_vmp_bytes(height: np.ndarray, meta: np.ndarray) -> bytes:
    """Encodes ``(height, meta)`` arrays into a ``.vmp`` buffer."""
    _check_shapes(height, meta)
    size_y = height.shape[0]
    out = np.empty((size_y, 2, H_SIZE), dtype=np.uint8)
    out[:, 0, :] = height
    out[:, 1, :] = meta
    return out.tobytes()


def load_vmp(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Loads ``.vmp`` from disk, inferring ``size_y`` from the file size."""
    size = os.path.getsize(path)
    stride = 2 * H_SIZE
    if size % stride:
        raise VmpError(f"VMP size {size} is not a multiple of the row stride {stride}")
    return load_vmp_bytes(_read_all(path), size // stride)


def save_vmp(path: str | os.PathLike, height: np.ndarray, meta: np.ndarray) -> None:
    with open(path, "wb") as fh:
        fh.write(save_vmp_bytes(height, meta))


def vmp_size_y(data: bytes | bytearray | memoryview) -> int:
    """``size_y`` implied by a ``.vmp`` buffer length."""
    stride = 2 * H_SIZE
    if len(data) % stride:
        raise VmpError(f"VMP size {len(data)} is not a multiple of the row stride {stride}")
    return len(data) // stride


def _read_all(path: str | os.PathLike) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _check_shapes(height: np.ndarray, meta: np.ndarray) -> None:
    if height.shape != meta.shape:
        raise VmpError(f"height {height.shape} and meta {meta.shape} shapes differ")
    if height.ndim != 2 or height.shape[1] != H_SIZE:
        raise VmpError(f"expected (size_y, {H_SIZE}) arrays, got {height.shape}")
    if height.dtype != np.uint8 or meta.dtype != np.uint8:
        raise VmpError("height/meta must be uint8")
