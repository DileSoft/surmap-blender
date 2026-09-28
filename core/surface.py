"""The voxel surface: heights, flags and the double-level accessors.

Ported 1:1 from `src/terra/render.h` and `src/terra/land.cpp`.

Data model (verified against the retail worlds in the Vangers repo):

* A **pair** is an even column ``e`` and its odd neighbour ``e+1``; there are
  ``H_SIZE / 2`` pairs per row.
* ``DOUBLE_LEVEL`` is set on **both** flag bytes of a pair (verified: the counts
  of even/odd double flags are equal in every shipped world).
* ``height[e]`` is the lower (down) surface, ``height[e+1]`` the upper (up) one.
* The even flag byte carries the *down* material, the odd byte the *up* one.
* Both flag bytes carry 2 bits of the tunnel delta, packed as
  ``d4 = (delta_even << 2) | delta_odd``; thickness = ``(d4 + 1) << 3``.

X wraps modulo ``H_SIZE``; Y is linear.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    DELTA_MASK,
    DELTA_SHIFT,
    DOUBLE_LEVEL,
    H_SIZE,
    KEEP_TERRAIN,
    MIN_RDELTA,
    TERRAIN_MASK,
    TERRAIN_OFFSET,
    TUNNEL_PROOF,
)

PAIRS = H_SIZE // 2

_X = np.arange(H_SIZE, dtype=np.int64)
X_EVEN = (_X & 1) == 0
X_ODD = ~X_EVEN

#: Smallest ``up - down`` that still decodes as a tunnel (see solver).
#: ``t = (up - TUNNEL_PROOF - down) & ~7`` must stay above ``MIN_RDELTA``.
MIN_TUNNEL_GAP = TUNNEL_PROOF + 2 * MIN_RDELTA  # 32


# ---------------------------------------------------------------------------
# free functions on raw arrays
# ---------------------------------------------------------------------------
def is_double(meta: np.ndarray) -> np.ndarray:
    """``DOUBLE_LEVEL`` flag, per voxel."""
    return (meta & DOUBLE_LEVEL) != 0


def up_alt(height: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """``GET_UP_ALT`` — the surface drawn on top at every voxel."""
    dbl = is_double(meta)
    return np.where(dbl & X_EVEN, np.roll(height, -1, axis=1), height)


def down_alt(height: np.ndarray, meta: np.ndarray) -> np.ndarray:
    """``GET_DOWN_ALT`` — the surface underneath at every voxel."""
    dbl = is_double(meta)
    return np.where(dbl & X_ODD, np.roll(height, 1, axis=1), height)


def up_terrain(meta: np.ndarray) -> np.ndarray:
    """``GET_REAL_TERRAIN`` — material index (0..7) of the upper surface."""
    dbl = is_double(meta)
    src = np.where(X_ODD | ~dbl, meta, np.roll(meta, -1, axis=1))
    return ((src & TERRAIN_MASK) >> TERRAIN_OFFSET).astype(np.uint8)


def down_terrain(meta: np.ndarray) -> np.ndarray:
    """``GET_REAL_DOWNTERRAIN`` — material index (0..7) of the lower surface."""
    dbl = is_double(meta)
    prv = np.roll(meta, 1, axis=1)
    src = np.where(dbl, np.where(X_ODD, prv, meta), meta)
    return ((src & TERRAIN_MASK) >> TERRAIN_OFFSET).astype(np.uint8)


def delta_from_gap(gap: np.ndarray) -> np.ndarray:
    """``pixDownSet``'s quantisation: gap -> 4-bit delta value.

    ``t = (up - TUNNEL_PROOF - down) & ~(MIN_RDELTA - 1)`` clamped to
    ``MAX_RDELTA``, then ``d4 = (t >> DELTA_SHIFT) - 1``. Gaps too small to
    form a tunnel yield 0, which the caller treats as "collapse".
    """
    from .constants import MAX_RDELTA

    t = (gap.astype(np.int32) - TUNNEL_PROOF)
    t = np.maximum(t, 0)
    t = t & ~np.int32(MIN_RDELTA - 1)
    t = np.minimum(t, MAX_RDELTA)
    return np.clip((t >> DELTA_SHIFT) - 1, 0, 15).astype(np.uint8)


def gap_is_tunnel(gap: np.ndarray) -> np.ndarray:
    """True where the gap is wide enough to stay a double column."""
    t = np.maximum(gap.astype(np.int32) - TUNNEL_PROOF, 0) & ~np.int32(MIN_RDELTA - 1)
    return t > MIN_RDELTA


def thickness_from_delta(d4: np.ndarray) -> np.ndarray:
    """``(d4 + 1) << DELTA_SHIFT`` — physics clearance of a pair, in voxels."""
    return ((d4.astype(np.int32) + 1) << DELTA_SHIFT).astype(np.int32)


# ---------------------------------------------------------------------------
class Surface:
    """Heights + flags of one world, with the accessors above.

    All write methods take a **pair mask** of shape ``(size_y, PAIRS)``; entry
    ``i`` covers columns ``2i`` (lower) and ``2i + 1`` (upper).
    """

    __slots__ = ("height", "meta")

    def __init__(self, height: np.ndarray, meta: np.ndarray) -> None:
        if height.shape != meta.shape or height.ndim != 2 or height.shape[1] != H_SIZE:
            raise ValueError(f"bad surface shapes {height.shape} / {meta.shape}")
        if height.dtype != np.uint8 or meta.dtype != np.uint8:
            raise ValueError("height/meta must be uint8")
        self.height = height
        self.meta = meta

    # -- construction -------------------------------------------------------
    @classmethod
    def empty(cls, size_y: int, base_height: int = 0, terrain: int = 1) -> "Surface":
        """A flat world with no double columns, ready to be sculpted."""
        height = np.full((size_y, H_SIZE), base_height & 0xFF, dtype=np.uint8)
        meta = np.full((size_y, H_SIZE), (terrain << TERRAIN_OFFSET) & TERRAIN_MASK,
                       dtype=np.uint8)
        return cls(height, meta)

    def copy(self) -> "Surface":
        return Surface(self.height.copy(), self.meta.copy())

    # -- shape --------------------------------------------------------------
    @property
    def size_y(self) -> int:
        return self.height.shape[0]

    @property
    def size_x(self) -> int:
        return H_SIZE

    def _check_mask(self, mask: np.ndarray) -> None:
        if mask.shape != (self.size_y, PAIRS):
            raise ValueError(f"pair mask must be {(self.size_y, PAIRS)}, got {mask.shape}")

    # -- views (basic slicing => writes go through to the parent array) -----
    @property
    def h_even(self) -> np.ndarray:
        """Lower-surface heights of every pair — a view."""
        return self.height[:, 0::2]

    @property
    def h_odd(self) -> np.ndarray:
        """Upper-surface heights of every pair — a view."""
        return self.height[:, 1::2]

    @property
    def m_even(self) -> np.ndarray:
        """Flag bytes of the even columns — a view."""
        return self.meta[:, 0::2]

    @property
    def m_odd(self) -> np.ndarray:
        """Flag bytes of the odd columns — a view."""
        return self.meta[:, 1::2]

    # -- read ---------------------------------------------------------------
    def is_double(self) -> np.ndarray:
        """Per-voxel ``DOUBLE_LEVEL`` flag."""
        return is_double(self.meta)

    def pair_double(self) -> np.ndarray:
        """Canonical per-pair double flag, taken from the even column."""
        return (self.m_even & DOUBLE_LEVEL) != 0

    def pair_mismatches(self) -> int:
        """Pairs whose two flag bytes disagree (0 in every shipped world)."""
        return int(np.count_nonzero(self.pair_double() != ((self.m_odd & DOUBLE_LEVEL) != 0)))

    def up_alt(self) -> np.ndarray:
        return up_alt(self.height, self.meta)

    def down_alt(self) -> np.ndarray:
        return down_alt(self.height, self.meta)

    def up_terrain(self) -> np.ndarray:
        return up_terrain(self.meta)

    def down_terrain(self) -> np.ndarray:
        return down_terrain(self.meta)

    def pair_up(self) -> np.ndarray:
        """Deck height of every pair (the odd column, ``GET_UP_ALT``)."""
        return self.h_odd

    def pair_down(self) -> np.ndarray:
        """Underside height of every pair (the even column)."""
        return self.h_even

    def pair_gap(self) -> np.ndarray:
        """``up - down`` per pair, as int16; negative means an overhang."""
        return (self.h_odd.astype(np.int16) - self.h_even.astype(np.int16))

    def pair_delta(self) -> np.ndarray:
        """The 4-bit tunnel delta of every pair."""
        return (((self.m_even & DELTA_MASK) << 2) | (self.m_odd & DELTA_MASK)).astype(np.uint8)

    def pair_thickness(self) -> np.ndarray:
        return thickness_from_delta(self.pair_delta())

    def count_doubles(self) -> int:
        return int(np.count_nonzero(self.pair_double()))

    # -- write --------------------------------------------------------------
    def pair_set_double(self, mask: np.ndarray) -> None:
        """Sets/clears ``DOUBLE_LEVEL`` on both columns of every selected pair.

        Clearing also zeroes the delta fields, as ``MapCircleProcess`` does.
        """
        self._check_mask(mask)
        keep = np.uint8(~DOUBLE_LEVEL & 0xFF)
        keep_delta = np.uint8(~DELTA_MASK & 0xFF)
        self.m_even[mask] = (self.m_even[mask] & keep) | np.uint8(DOUBLE_LEVEL)
        self.m_odd[mask] = (self.m_odd[mask] & keep) | np.uint8(DOUBLE_LEVEL)
        cleared = mask & ~self.pair_double()
        self.m_even[cleared] &= keep_delta
        self.m_odd[cleared] &= keep_delta

    def pair_write_delta(self, d4: np.ndarray, mask: np.ndarray) -> None:
        """Writes the packed 4-bit delta into both flag bytes of masked pairs."""
        self._check_mask(mask)
        hi = (np.uint8(d4) >> np.uint8(2)) & np.uint8(0x03)
        lo = np.uint8(d4) & np.uint8(0x03)
        keep = np.uint8(~DELTA_MASK & 0xFF)
        self.m_even[mask] = (self.m_even[mask] & keep) | hi[mask]
        self.m_odd[mask] = (self.m_odd[mask] & keep) | lo[mask]

    def pair_write_down(self, values: np.ndarray, mask: np.ndarray) -> None:
        """Writes the underside height of masked pairs."""
        self._check_mask(mask)
        self.h_even[mask] = np.uint8(values)[mask]

    def pair_write_up(self, values: np.ndarray, mask: np.ndarray) -> None:
        """Writes the deck height of masked pairs."""
        self._check_mask(mask)
        self.h_odd[mask] = np.uint8(values)[mask]

    def pair_write_down_terrain(self, terrain: int, mask: np.ndarray) -> None:
        self._write_terrain(self.m_even, terrain, mask)

    def pair_write_up_terrain(self, terrain: int, mask: np.ndarray) -> None:
        self._write_terrain(self.m_odd, terrain, mask)

    def _write_terrain(self, view: np.ndarray, terrain: int, mask: np.ndarray) -> None:
        if terrain == KEEP_TERRAIN:
            return
        value = np.uint8((terrain << TERRAIN_OFFSET) & TERRAIN_MASK)
        keep = np.uint8(~TERRAIN_MASK & 0xFF)
        view[mask] = (view[mask] & keep) | value

    def write_up_heights(self, values: np.ndarray) -> None:
        """Writes the **upper** surface for every voxel.

        For a double pair only the odd column is written; the lower half keeps
        its own height.
        """
        dbl = is_double(self.meta)
        vals = np.uint8(values)
        np.copyto(self.height, vals, where=~dbl | (dbl & X_ODD))

    def write_single_heights(self, values: np.ndarray) -> None:
        """Writes plain (non-double) voxel heights, leaving pairs alone."""
        dbl = is_double(self.meta)
        np.copyto(self.height, np.uint8(values), where=~dbl)

    def write_single_terrain(self, terrain: int) -> None:
        """Rewrites the material of every non-double voxel."""
        if terrain == KEEP_TERRAIN:
            return
        dbl = is_double(self.meta)
        value = np.uint8((terrain << TERRAIN_OFFSET) & TERRAIN_MASK)
        keep = np.uint8(~TERRAIN_MASK & 0xFF)
        sel = ~dbl
        self.meta[sel] = (self.meta[sel] & keep) | value
