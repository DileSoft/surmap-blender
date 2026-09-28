"""Cross-check: does the Python port reproduce the reference renderer's output?

The TypeScript viewer (`surmap-typescript`) pins an FNV-1a hash of the whole
fostral surface produced by the ported `LINE_render`, and the Python port was
verified against it: `LINE_HASH_FULL_MAP = "fed77a5d"`.

Hashing 33.5 million bytes one at a time in Python is far too slow to keep in
the loop, so the regression guard is pinned on a 256-row slice of the small
world instead. The full-map value stays recorded here as the one-time
cross-project verification. Regenerate the slice hash with `pin_render_hash.py`.
"""

from __future__ import annotations

import os

import numpy as np

from core.config import WorldConfig
from core.constants import H_SIZE
from core.render import line_render_indices, render_prepare
from core.vmp import load_vmp_rows
from core.vpr import load_levels

from framework import SAMPLE_ROWS, check, check_eq, sample_config, small_vmp

#: FNV-1a of the *whole* fostral surface, cross-checked against surmap-typescript.
LINE_HASH_FULL_MAP = "fed77a5d"

#: FNV-1a of the first SAMPLE_ROWS rows of the small world. Regenerate with
#: `blender --background --python tests/pin_render_hash.py`.
LINE_HASH = "b946cbf2"


def fnv1a(data: np.ndarray) -> str:
    h = 0x811C9DC5
    flat = np.ascontiguousarray(data).reshape(-1)
    for b in flat.tolist():
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def sample_indices() -> np.ndarray:
    """`LINE_render` palette indices for a slice of a real world.

    The single source of truth for the pinned hash -- `pin_render_hash.py`
    calls this, so the two can never drift apart.
    """
    path = small_vmp()
    cfg = sample_config(path)
    rows = min(SAMPLE_ROWS, cfg.size_y)
    height, meta = load_vmp_rows(path, 0, rows)
    flood = 0
    vpr = os.path.join(os.path.dirname(path), "output.vpr")
    if os.path.isfile(vpr):
        with open(vpr, "rb") as fh:
            flood = load_levels(fh.read(), cfg)[0]
    luts = render_prepare(cfg.begin_colors, cfg.end_colors, flood)
    return line_render_indices(height, meta, luts)


def test_line_render_matches_the_reference_hash() -> None:
    indices = sample_indices()
    check_eq(indices.shape, (indices.shape[0], H_SIZE), "index array shape")
    check_eq(fnv1a(indices), LINE_HASH,
             f"LINE_render palette-index hash over {indices.shape[0]} rows")


def test_palette_ramp_is_monotonic_for_land_terrains() -> None:
    path = small_vmp()
    with open(os.path.join(os.path.dirname(path), "world.ini"), "r",
              encoding="utf-8", errors="replace") as fh:
        cfg = WorldConfig.parse(fh.read())
    luts = render_prepare(cfg.begin_colors, cfg.end_colors, 0)
    for ind in range(1, 8):
        ramp = luts.pal_clr[ind, 256:]
        check(bool(np.all(np.diff(ramp.astype(np.int16)) >= 0)),
              f"terrain {ind} ramp should not go down")
    check_eq(int(luts.pal_clr[1, 256]), cfg.begin_colors[1], "ramp starts at Begin Color")
    check_eq(int(luts.pal_clr[1, 511]), cfg.end_colors[1], "ramp ends at End Color")
