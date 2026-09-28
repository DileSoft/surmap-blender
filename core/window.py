"""Windows into a surface: gathering slices and scattering values back.

A window is a rectangle in voxel space. X wraps (the map is a torus), Y does
not. Gathering a window that crosses the X seam simply repeats the tail of a
row at the start of the next, so Blender can show the seam like any other edge;
scattering maps back with ``& (H_SIZE - 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import H_SIZE
from .surface import PAIRS, Surface


class WindowError(ValueError):
    pass


@dataclass(frozen=True)
class Region:
    """A window of voxels: ``width`` columns starting at ``x0``, ``depth`` rows."""

    x0: int
    y0: int
    width: int
    depth: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.depth <= 0:
            raise WindowError(f"empty region {self}")

    @property
    def x1(self) -> int:
        return self.x0 + self.width - 1

    @property
    def y1(self) -> int:
        return self.y0 + self.depth - 1

    @property
    def columns(self) -> np.ndarray:
        """Absolute column indices, wrapped into ``0..H_SIZE-1``."""
        return (self.x0 + np.arange(self.width, dtype=np.int64)) % H_SIZE

    @property
    def pair_first(self) -> int:
        """Index of the first pair touched (the pair containing ``x0``)."""
        return self.x0 // 2

    @property
    def pair_count(self) -> int:
        """How many pairs the window covers, rounded up to whole pairs."""
        return (self.width + (self.x0 & 1) + 1) // 2

    def pair_region(self) -> "Region":
        """The same window expressed over pairs (even columns only)."""
        first = self.pair_first
        count = self.pair_count
        return Region(x0=first * 2, y0=self.y0, width=count * 2, depth=self.depth)


def check_region(region: Region, surface: Surface) -> None:
    if region.depth > surface.size_y:
        raise WindowError(
            f"region depth {region.depth} exceeds the world height {surface.size_y}"
        )


def gather(surface: Surface, region: Region) -> tuple[np.ndarray, np.ndarray]:
    """``(height, meta)`` slices of shape ``(depth, width)``."""
    check_region(region, surface)
    cols = region.columns
    rows = (region.y0 + np.arange(region.depth, dtype=np.int64)) % surface.size_y
    return surface.height[np.ix_(rows, cols)], surface.meta[np.ix_(rows, cols)]


def gather_pair_heights(surface: Surface, region: Region) -> tuple[np.ndarray, np.ndarray]:
    """``(deck, underside)`` of shape ``(depth, pair_count)`` for a window.

    Columns are read as whole pairs, so a window starting on an odd column still
    yields the pair that contains it.
    """
    check_region(region, surface)
    pairs = region.pair_region()
    first = pairs.x0 // 2
    idx = (first + np.arange(pairs.width // 2, dtype=np.int64)) % PAIRS
    rows = (region.y0 + np.arange(region.depth, dtype=np.int64)) % surface.size_y
    return surface.h_odd[np.ix_(rows, idx)], surface.h_even[np.ix_(rows, idx)]


def scatter_pair_heights(surface: Surface, region: Region, deck: np.ndarray,
                         underside: np.ndarray) -> None:
    """Writes pair heights back for every pair the window covers."""
    check_region(region, surface)
    pairs = region.pair_region()
    first = pairs.x0 // 2
    idx = (first + np.arange(pairs.width // 2, dtype=np.int64)) % PAIRS
    rows = (region.y0 + np.arange(region.depth, dtype=np.int64)) % surface.size_y
    surface.h_odd[np.ix_(rows, idx)] = np.uint8(deck)
    surface.h_even[np.ix_(rows, idx)] = np.uint8(underside)


def scatter_up_heights(surface: Surface, region: Region, values: np.ndarray) -> None:
    """Writes the upper surface of every voxel in the window."""
    check_region(region, surface)
    _heights, meta = gather(surface, region)
    from .surface import is_double

    dbl = is_double(meta)
    rows = (region.y0 + np.arange(region.depth, dtype=np.int64)) % surface.size_y
    write = ~dbl | (dbl & _X_ODD)
    idx = np.ix_(rows, region.columns)
    surface.height[idx] = np.where(write, np.uint8(values), surface.height[idx])


def scatter_plain_heights(surface: Surface, region: Region, values: np.ndarray) -> None:
    """Writes only the non-double voxels of the window (the lower mesh's peers)."""
    check_region(region, surface)
    _heights, meta = gather(surface, region)
    from .surface import is_double

    rows = (region.y0 + np.arange(region.depth, dtype=np.int64)) % surface.size_y
    idx = np.ix_(rows, region.columns)
    surface.height[idx] = np.where(is_double(meta), surface.height[idx], np.uint8(values))


def scatter_pair_terrain(surface: Surface, region: Region, values: np.ndarray,
                         mask: np.ndarray, *, upper: bool) -> None:
    """Writes the per-pair material of a window into the surface flags."""
    from .constants import TERRAIN_MASK, TERRAIN_OFFSET

    pairs = region.pair_region()
    rows = (region.y0 + np.arange(region.depth, dtype=np.int64)) % surface.size_y
    cols = (pairs.x0 // 2 + np.arange(mask.shape[1], dtype=np.int64)) % PAIRS
    idx = np.ix_(rows, cols)
    view = surface.m_odd if upper else surface.m_even
    cur = view[idx]
    keep = np.uint8(~TERRAIN_MASK & 0xFF)
    v = np.uint8(values) << np.uint8(TERRAIN_OFFSET)
    view[idx] = np.where(mask, (cur & keep) | (v & TERRAIN_MASK), cur).astype(np.uint8)


def scatter_single_terrain(surface: Surface, region: Region, values: np.ndarray) -> None:
    """Writes the per-voxel material of the non-double voxels of a window."""
    from .constants import TERRAIN_MASK, TERRAIN_OFFSET
    from .surface import is_double

    _heights, meta = gather(surface, region)
    plain = ~is_double(meta)
    rows = (region.y0 + np.arange(region.depth, dtype=np.int64)) % surface.size_y
    idx = np.ix_(rows, region.columns)
    cur = surface.meta[idx]
    keep = np.uint8(~TERRAIN_MASK & 0xFF)
    v = np.uint8(values) << np.uint8(TERRAIN_OFFSET)
    surface.meta[idx] = np.where(plain, (cur & keep) | (v & TERRAIN_MASK), cur).astype(np.uint8)


_X_ODD = (np.arange(H_SIZE) & 1) == 1
