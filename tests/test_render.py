"""Cross-check: does the Python port reproduce the reference renderer's hashes?

The TypeScript viewer (surmap-typescript) pins FNV-1a hashes of the fostral
surface produced by the ported `LINE_render` and `regRender`. This is the
acceptance criterion for the shading port.
"""

from __future__ import annotations

import os

import numpy as np

from core.config import WorldConfig
from core.constants import H_SIZE
from core.render import line_render_indices, render_prepare
from core.vmp import load_vmp
from core.vpr import load_levels

from framework import check, check_eq, world_file

LINE_HASH = "fed77a5d"


def fnv1a(data: np.ndarray) -> str:
    h = 0x811C9DC5
    flat = np.ascontiguousarray(data).reshape(-1)
    for b in flat.tolist():
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def _fostral_vmp() -> str:
    path = world_file("output.vmp")
    check(path is not None, "no .vmp found in the Vangers checkout")
    check("fostral" in path, f"expected the fostral world, got {path}")
    return path


def test_line_render_matches_the_reference_hash() -> None:
    path = _fostral_vmp()
    height, meta = load_vmp(path)
    with open(os.path.join(os.path.dirname(path), "world.ini"), "r", encoding="utf-8",
              errors="replace") as fh:
        cfg = WorldConfig.parse(fh.read())
    flood = 0
    vpr = os.path.join(os.path.dirname(path), "output.vpr")
    if os.path.isfile(vpr):
        with open(vpr, "rb") as fh:
            flood = load_levels(fh.read(), cfg)[0]

    luts = render_prepare(cfg.begin_colors, cfg.end_colors, flood)
    indices = line_render_indices(height, meta, luts)
    check_eq(indices.shape, (cfg.size_y, H_SIZE), "index array shape")
    check_eq(fnv1a(indices), LINE_HASH, "LINE_render palette-index hash for fostral")


def test_palette_ramp_is_monotonic_for_land_terrains() -> None:
    cfg = WorldConfig.parse(open(
        os.path.join(os.path.dirname(_fostral_vmp()), "world.ini"), "r",
        encoding="utf-8", errors="replace").read())
    luts = render_prepare(cfg.begin_colors, cfg.end_colors, 0)
    for ind in range(1, 8):
        ramp = luts.pal_clr[ind, 256:]
        check(bool(np.all(np.diff(ramp.astype(np.int16)) >= 0)),
              f"terrain {ind} ramp should not go down")
    check_eq(int(luts.pal_clr[1, 256]), cfg.begin_colors[1], "ramp starts at Begin Color")
    check_eq(int(luts.pal_clr[1, 511]), cfg.end_colors[1], "ramp ends at End Color")
