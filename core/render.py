"""Surface shading — `RenderPrepare` (src/terra/land.cpp) and LINE_render.

The two lookup tables built here are the ones every renderer in the game uses:

* ``lightCLR[terrain][255 - (lx - rx)]`` — diffuse light term, 0..255 (511 wide);
* ``palCLR[terrain][256 + light - depth]`` — palette index ramp (512 wide).

`surface_colors` applies the same shading as `LINE_render`, vectorised over a
whole window. It is used for the Blender vertex colours.

Caveat, same as in the original: the game has no LINE_render for the *lower*
layer, so the down-surface colours here are the same formula applied to the
lower altitudes. Exact lower-layer rendering (`regDownRender`,
surmap/dsidernd.cpp) is a separate, slower routine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    CLR_MAX,
    CLR_MAX_SIDE,
    DOUBLE_LEVEL,
    H_CORRECTION,
    MATERIAL_MAX,
    SHADOWDEEP,
    SHADOW_MASK,
    TERRAIN_DXKOEF,
    TERRAIN_JJKOEF,
    TERRAIN_MASK,
    TERRAIN_MATERIAL,
    TERRAIN_MAX,
    TERRAIN_OFFSET,
    TERRAIN_SDKOEF,
)
from .surface import Surface, down_alt, down_terrain, up_alt, up_terrain

# Land.cpp material table (TERRAIN16 is not defined)
TERRAIN_DXKOEF_A = np.array([1.0, 5.0], dtype=np.float64)
TERRAIN_SDKOEF_A = np.array([1.0, 1.25], dtype=np.float64)
TERRAIN_JJKOEF_A = np.array([1.0, 0.5], dtype=np.float64)
TERRAIN_MATERIAL_A = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)


@dataclass
class Luts:
    """The two renderer lookup tables plus the flood level."""

    light_clr: np.ndarray  # (TERRAIN_MAX, 511) uint8
    pal_clr: np.ndarray  # (TERRAIN_MAX, 512) uint8
    flood_level: int


def _js_round(x: float) -> int:
    """`Math.round` — halves go towards +infinity, unlike Python's `round`."""
    return math.floor(x + 0.5)


def render_prepare(begin_colors: list[int], end_colors: list[int],
                   flood_level: int = 0) -> Luts:
    """Port of `RenderPrepare` (src/terra/land.cpp), game/road build."""
    flood = flood_level & 0xFF

    light_material = np.zeros((MATERIAL_MAX, CLR_MAX), dtype=np.uint8)
    dx = 8.0
    sd = 256.0 / SHADOWDEEP
    for ind in range(MATERIAL_MAX):
        dxk = TERRAIN_DXKOEF_A[ind] * dx
        sdk = TERRAIN_SDKOEF_A[ind] * sd
        for j in range(-CLR_MAX_SIDE, CLR_MAX_SIDE + 1):
            jj = TERRAIN_JJKOEF_A[ind] * j
            v = _js_round((255.0 * (dxk * sdk - jj))
                          / math.sqrt((1.0 + sdk * sdk) * (dxk * dxk + jj * jj)))
            v = max(v, 0)
            light_material[ind, CLR_MAX_SIDE + j] = v

    light_clr = np.zeros((TERRAIN_MAX, CLR_MAX), dtype=np.uint8)
    pal_clr = np.zeros((TERRAIN_MAX, 512), dtype=np.uint8)
    for ind in range(TERRAIN_MAX):
        light_clr[ind] = light_material[TERRAIN_MATERIAL_A[ind]]

        pal = np.full(512, begin_colors[ind], dtype=np.uint8)
        colnum = end_colors[ind] - begin_colors[ind]

        if ind == 0:
            # Water: the ramp clamps at the flood level.
            d = (255 - flood) >> 1
            if d > 0:
                pal[512 - d:] = end_colors[ind]
            denom = 255.0 - d
            for j in range(256 - d):
                v = _js_round((j * 1.25 * colnum) / denom - 0.25 * colnum)
                v = max(v, 0)
                pal[256 + j] = begin_colors[ind] + v
        else:
            for j in range(256):
                pal[256 + j] = begin_colors[ind] + _js_round(j * colnum / 255.0)
        pal_clr[ind] = pal

    return Luts(light_clr=light_clr, pal_clr=pal_clr, flood_level=flood)


def _shade_indices(alt: np.ndarray, terrain: np.ndarray, shadow: np.ndarray,
                   luts: Luts) -> np.ndarray:
    """Palette indices for one surface, per voxel."""
    lx = np.roll(alt, 1, axis=1).astype(np.int16)
    rx = np.roll(alt, -1, axis=1).astype(np.int16)
    h = alt.astype(np.int16)

    lum = 255 - (lx - rx)  # index into lightCLR, 0..510
    l = luts.light_clr[terrain, lum].astype(np.int16)
    depth = (255 - h) >> H_CORRECTION

    bright = 256 + l - depth
    dim = 256 + ((l - depth) >> 1)
    idx = np.where(shadow.astype(bool), dim, bright).astype(np.int16)
    return luts.pal_clr[terrain, np.clip(idx, 0, 511)]


def line_render_indices(height: np.ndarray, meta: np.ndarray, luts: Luts) -> np.ndarray:
    """Faithful port of `LINE_render` (src/terra/land.cpp), full surface.

    Note the original's quirk: a double pair is drawn using the **lower**
    surface — its height and its down material — and both columns of the pair
    get that same colour. (This differs from `regRender`, which draws the deck.
    It is preserved here because this function is the cross-check against the
    reference renderer's pinned hashes.)

    ``regRender`` is the accurate renderer and is not reimplemented here.
    """
    dbl = (meta[:, 0::2] & DOUBLE_LEVEL) != 0  # a pair is flagged on both columns

    h_even = height[:, 0::2]
    h_odd = height[:, 1::2]
    lx_even = np.roll(height, 1, axis=1)[:, 0::2]  # height[x-1]
    rx_even = h_odd  # height[x+1]
    lx_odd = h_even  # height[x]
    # The two branches really do use different right neighbours: the plain
    # second pixel reads height[x+2] (x still even), while the double branch
    # reads height[x+3] (the original had already done x++ there).
    rx_odd = np.roll(height, -1, axis=1)[:, 1::2]  # height[x+2]
    rx_dbl = np.roll(height, -3, axis=1)[:, 0::2]  # height[x+3]

    terr_even = ((meta[:, 0::2] & TERRAIN_MASK) >> TERRAIN_OFFSET).astype(np.uint8)
    terr_odd = ((meta[:, 1::2] & TERRAIN_MASK) >> TERRAIN_OFFSET).astype(np.uint8)
    shadow_even = (meta[:, 0::2] & SHADOW_MASK) != 0
    shadow_odd = (meta[:, 1::2] & SHADOW_MASK) != 0

    def shade(terr, lx, rx, h, shadow):
        l = luts.light_clr[terr, np.clip(255 - (lx - rx), 0, CLR_MAX - 1)].astype(np.int16)
        depth = (255 - h.astype(np.int16)) >> H_CORRECTION
        idx = np.where(shadow, 256 + ((l - depth) >> 1), 256 + l - depth).astype(np.int16)
        return luts.pal_clr[terr, np.clip(idx, 0, 511)]

    # single-surface branch
    idx_even = shade(terr_even, lx_even.astype(np.int16), rx_even.astype(np.int16), h_even, shadow_even)
    idx_odd = shade(terr_odd, lx_odd.astype(np.int16), rx_odd.astype(np.int16), h_odd, shadow_odd)
    # double branch: the floor of the pair, painted across both columns
    idx_dbl = shade(terr_odd, h_odd.astype(np.int16), rx_dbl.astype(np.int16), h_odd, shadow_odd)

    out = np.empty(height.shape, dtype=np.uint8)
    out[:, 0::2] = np.where(dbl, idx_dbl, idx_even)
    out[:, 1::2] = np.where(dbl, idx_dbl, idx_odd)
    return out


def upper_indices_arr(height: np.ndarray, meta: np.ndarray, luts: Luts,
                      phase: int = 0) -> np.ndarray:
    """Palette indices of the upper surface of an arbitrary array pair."""
    alt = up_alt(height, meta, phase)
    return _shade_indices(alt, up_terrain(meta, phase),
                          (meta & SHADOW_MASK) != 0, luts).astype(np.uint8)


def lower_indices_arr(height: np.ndarray, meta: np.ndarray, luts: Luts,
                      phase: int = 0) -> np.ndarray:
    """Palette indices of the lower surface (see the module caveat)."""
    alt = down_alt(height, meta, phase)
    return _shade_indices(alt, down_terrain(meta, phase),
                          (meta & SHADOW_MASK) != 0, luts).astype(np.uint8)


def upper_indices(surface: Surface, luts: Luts) -> np.ndarray:
    """Palette indices of the **upper** surface, shaded per voxel.

    Equivalent to what ``regRender`` would show, minus the shadow pass.
    """
    return upper_indices_arr(surface.height, surface.meta, luts)


def lower_indices(surface: Surface, luts: Luts) -> np.ndarray:
    """Palette indices of the **lower** surface (see the module caveat)."""
    return lower_indices_arr(surface.height, surface.meta, luts)


def indices_to_rgba(indices: np.ndarray, rgba_table: np.ndarray) -> np.ndarray:
    """Maps palette indices to ``(rows, cols, 4)`` RGBA bytes."""
    return rgba_table[indices]
