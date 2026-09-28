"""Palette handling — `PalettePrepare` / `read_palette` from src/road.cpp.

The renderers produce 8-bit *palette indices*; this module turns them into RGBA
for Blender vertex colours. Port of `loadPalette` / `buildPalette` in the
reference TypeScript viewer, which mirrors `PalettePrepare` in `src/road.cpp`.
"""

from __future__ import annotations

import numpy as np

from .constants import TERRAIN_MAX

PALETTE_BYTES = 768
PALETTE_ENTRIES = PALETTE_BYTES // 3


class PaletteError(ValueError):
    pass


def load_palette(data: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Reads a 768-byte 6-bit palette; values stay in 0..63."""
    raw = np.frombuffer(bytes(data), dtype=np.uint8) if not isinstance(data, np.ndarray) else data
    if raw.size < PALETTE_BYTES:
        raise PaletteError(f"palette must be at least {PALETTE_BYTES} bytes, got {raw.size}")
    return np.array(raw[:PALETTE_BYTES], dtype=np.uint8).reshape(PALETTE_ENTRIES, 3)


def build_palette(palette_file: np.ndarray, begin_colors: list[int],
                  end_colors: list[int]) -> np.ndarray:
    """Applies `PalettePrepare` and returns a ``(256, 3)`` 6-bit RGB table.

    * index 0 becomes black;
    * every ``Begin Colors`` entry is halved;
    * indices 224..239 are a fixed grayscale ramp.
    """
    rgb = np.zeros((PALETTE_ENTRIES, 3), dtype=np.uint8)
    rgb[:] = palette_file[:PALETTE_ENTRIES]
    rgb[0] = 0
    for i in range(TERRAIN_MAX):
        entry = begin_colors[i]
        if 0 <= entry < PALETTE_ENTRIES:
            rgb[entry] >>= 1
    for i in range(16):
        v = i * 4
        rgb[224 + i] = v
    return rgb


def to_rgba(rgb6: np.ndarray) -> np.ndarray:
    """6-bit -> 8-bit RGBA (`XGR_SetPal` scales by 4)."""
    rgba = np.empty((rgb6.shape[0], 4), dtype=np.uint8)
    rgba[:, :3] = (rgb6.astype(np.uint16) * 4).clip(0, 255).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def prepared_rgba(palette_file: np.ndarray, begin_colors: list[int],
                  end_colors: list[int]) -> np.ndarray:
    """Convenience: 6-bit file -> ready-to-use ``(256, 4)`` RGBA table."""
    return to_rgba(build_palette(palette_file, begin_colors, end_colors))
