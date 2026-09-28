"""`.vmc` codec tests.

The decisive one is `test_vmc_decodes_to_the_same_surface_as_vmp`: the Vangers
checkout ships both formats for one world, so the decoder is checked against the
uncompressed surface directly. Everything real-data here reads a *slice* of rows
-- the format is row-seekable, and a full pass over a 16384-row world takes
seconds per file for no extra coverage.
"""

from __future__ import annotations

import os

import numpy as np

from core.constants import H_SIZE
from core.surface import Surface
from core.vmc import VmcDecoder, VmcError, load_vmc, read_table, save_vmc
from core.vmp import load_vmp_rows, vmp_size_y_of_file

from framework import (
    SAMPLE_ROWS,
    check,
    check_eq,
    read_vmc_rows,
    sample_config,
    vmc_world,
)


def test_vmc_decodes_to_the_same_surface_as_vmp() -> None:
    """The whole point of the codec: same world, same bytes on both sides."""
    vmc_path = vmc_world()
    vmp_path = os.path.join(os.path.dirname(vmc_path), "output.vmp")
    check(os.path.isfile(vmp_path), f"{vmc_path} has no .vmp sibling to compare against")
    cfg = sample_config(vmc_path)
    rows = min(SAMPLE_ROWS, cfg.size_y)

    vmp_h, vmp_m = load_vmp_rows(vmp_path, 0, rows)
    vmc_h, vmc_m = read_vmc_rows(vmc_path, rows)

    check_eq(vmc_h.shape, (rows, H_SIZE), "decoded height shape")
    check(np.array_equal(vmp_h, vmc_h), "heights differ between .vmp and .vmc")
    check(np.array_equal(vmp_m, vmc_m), "flags differ between .vmp and .vmc")


def test_vmc_roundtrip() -> None:
    """encode -> decode must be a fixed point, and it must actually compress."""
    height, meta = read_vmc_rows(vmc_world())
    rows = height.shape[0]

    packed = save_vmc(height, meta)
    h2, m2 = load_vmc(packed, rows)
    check(np.array_equal(height, h2), "height round-trip")
    check(np.array_equal(meta, m2), "flag round-trip")

    # the codec must earn its keep: raw would be rows * 2 * H_SIZE bytes
    raw_size = rows * 2 * H_SIZE
    check(len(packed) < raw_size,
          f"compression should shrink {raw_size} bytes, got {len(packed)}")
    print(f"    ({raw_size} -> {len(packed)} bytes, "
          f"{100 * len(packed) / raw_size:.0f}%)")


def test_vmc_decoded_surface_is_well_formed() -> None:
    """Decoded retail data must satisfy the pair invariant like any other."""
    surface = Surface(*read_vmc_rows(vmc_world()))
    check_eq(surface.pair_mismatches(), 0, "DOUBLE symmetry")
    check(surface.count_doubles() > 0, "expected some double columns")


def test_scalar_and_vector_decoders_agree() -> None:
    """The fast path must equal the literal port of `ExpandBuffer`."""
    vmc_path = vmc_world()
    cfg = sample_config(vmc_path)
    with open(vmc_path, "rb") as fh:
        raw = fh.read()
    size_y = cfg.size_y

    offsets, _ = read_table(raw, size_y)
    dec = VmcDecoder(raw, size_y * 6)
    rows = [0, 1, 7, 100, min(size_y - 1, 1000)]
    flat_h = np.zeros(len(rows) * H_SIZE, np.uint8)
    flat_m = np.zeros(len(rows) * H_SIZE, np.uint8)
    for i, y in enumerate(rows):
        dec.expand_row(raw, int(offsets[y]), flat_h, flat_m, i * H_SIZE)
    ref_h = flat_h.reshape(len(rows), H_SIZE)
    ref_m = flat_m.reshape(len(rows), H_SIZE)
    fast_h, fast_m = load_vmc(raw, size_y, rows=rows)
    for i, y in enumerate(rows):
        check(np.array_equal(ref_h[i], fast_h[i]), f"row {y} heights")
        check(np.array_equal(ref_m[i], fast_m[i]), f"row {y} flags")


def test_encoder_handles_a_world_with_a_single_symbol() -> None:
    """A flat world has one distinct delta, so the code has no depth at all.

    This is exactly the shape a freshly created world has, so the tree must
    stay valid: `DHuff` gets a phantom symbol to keep the code well-formed.
    """
    height = np.full((4, H_SIZE), 90, np.uint8)
    meta = np.full((4, H_SIZE), 8, np.uint8)
    h, m = load_vmc(save_vmc(height, meta), 4)
    check(np.array_equal(h, height), "flat heights survive")
    check(np.array_equal(m, meta), "flat flags survive")

    # a small double-level island on an otherwise flat world
    meta2 = meta.copy()
    meta2[1, 20] = 0x40 | 8
    meta2[1, 21] = 0x40 | 8
    height2 = height.copy()
    height2[1, 20] = 40
    height2[1, 21] = 120
    h2, m2 = load_vmc(save_vmc(height2, meta2), 4)
    check(np.array_equal(h2, height2), "heights survive around a bridge")
    check(np.array_equal(m2, meta2), "flags survive around a bridge")


def test_bad_boundary_is_reported() -> None:
    """A corrupted size table must fail loudly, not decode garbage."""
    vmc_path = vmc_world()
    cfg = sample_config(vmc_path)
    with open(vmc_path, "rb") as fh:
        raw = bytearray(fh.read())
    # claim row 0 is one byte longer than it really is
    raw[4] = (raw[4] + 1) & 0xFF
    try:
        load_vmc(bytes(raw), cfg.size_y, rows=range(0, 4))
    except VmcError:
        return
    raise AssertionError("expected a boundary error for a corrupted size table")


def test_row_selection_is_bounds_checked() -> None:
    """A bad window is an error, not a silent clamp or an out-of-bounds read."""
    vmc_path = vmc_world()
    cfg = sample_config(vmc_path)
    with open(vmc_path, "rb") as fh:
        raw = fh.read()
    for rows in (range(cfg.size_y - 1, cfg.size_y + 1), range(-2, 2)):
        try:
            load_vmc(raw, cfg.size_y, rows=rows)
        except VmcError:
            continue
        raise AssertionError(f"expected a range error for rows {list(rows)}")
