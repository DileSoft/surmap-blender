"""`.vmc` codec tests.

The decisive one is `test_vmc_matches_vmp_of_the_same_world`: the Vangers
checkout ships both formats for fostral, so the decoder is checked against the
uncompressed surface directly.
"""

from __future__ import annotations

import os
import time

import numpy as np

from core.constants import H_SIZE
from core.surface import Surface
from core.vmc import VmcDecoder, VmcError, load_vmc, read_table, save_vmc
from core.vmp import load_vmp_bytes, vmp_size_y

from framework import all_world_files, check, check_eq, world_file


def _vmc_vmp_pair() -> tuple[str, str]:
    """Paths of a world that ships both formats (fostral)."""
    for vmc in all_world_files("output.vmc"):
        vmp = os.path.join(os.path.dirname(vmc), "output.vmp")
        if os.path.isfile(vmp):
            return vmc, vmp
    raise AssertionError("no world ships both .vmc and .vmp")


def test_vmc_decodes_to_the_same_surface_as_vmp() -> None:
    vmc_path, vmp_path = _vmc_vmp_pair()
    with open(vmp_path, "rb") as fh:
        vmp_raw = fh.read()
    size_y = vmp_size_y(vmp_raw)
    vmp_h, vmp_m = load_vmp_bytes(vmp_raw, size_y)

    t0 = time.perf_counter()
    with open(vmc_path, "rb") as fh:
        vmc_raw = fh.read()
    vmc_h, vmc_m = load_vmc(vmc_raw, size_y)
    elapsed = time.perf_counter() - t0

    check_eq(vmc_h.shape, (size_y, H_SIZE), "decoded height shape")
    check(np.array_equal(vmp_h, vmc_h),
          f"heights differ between .vmp and .vmc (decode took {elapsed:.1f}s)")
    check(np.array_equal(vmp_m, vmc_m), "flags differ between .vmp and .vmc")
    print(f"    (.vmc decoded in {elapsed:.1f}s)")


def test_every_shipped_vmc_decodes() -> None:
    files = all_world_files("output.vmc")
    check(len(files) >= 10, f"expected many worlds, found {len(files)}")
    for path in files:
        with open(path, "rb") as fh:
            raw = fh.read()
        # sizeY comes from world.ini, next to the file
        from core.config import WorldConfig

        with open(os.path.join(os.path.dirname(path), "world.ini"), "r",
                  encoding="utf-8", errors="replace") as fh:
            cfg = WorldConfig.parse(fh.read())
        h, m = load_vmc(raw, cfg.size_y)
        check_eq(h.shape, (cfg.size_y, H_SIZE), f"{os.path.basename(path)} height shape")
        s = Surface(h, m)
        check_eq(s.pair_mismatches(), 0, f"{os.path.basename(path)}: DOUBLE symmetry")


def test_vmc_roundtrip_on_every_world() -> None:
    files = all_world_files("output.vmc")
    for path in files:
        from core.config import WorldConfig

        with open(os.path.join(os.path.dirname(path), "world.ini"), "r",
                  encoding="utf-8", errors="replace") as fh:
            cfg = WorldConfig.parse(fh.read())
        with open(path, "rb") as fh:
            h, m = load_vmc(fh.read(), cfg.size_y)
        packed = save_vmc(h, m)
        h2, m2 = load_vmc(packed, cfg.size_y)
        check(np.array_equal(h, h2), f"{os.path.basename(path)}: height round-trip")
        check(np.array_equal(m, m2), f"{os.path.basename(path)}: flag round-trip")


def test_scalar_and_vector_decoders_agree() -> None:
    """The fast path must equal the literal port of `ExpandBuffer`."""
    vmc_path, _ = _vmc_vmp_pair()
    from core.config import WorldConfig

    with open(os.path.join(os.path.dirname(vmc_path), "world.ini"), "r",
              encoding="utf-8", errors="replace") as fh:
        cfg = WorldConfig.parse(fh.read())
    with open(vmc_path, "rb") as fh:
        raw = fh.read()
    size_y = cfg.size_y

    offsets, sizes = read_table(raw, size_y)
    dec = VmcDecoder(raw, size_y * 6)
    rows = [0, 1, 7, 100, size_y - 1]
    flat_h = np.zeros(len(rows) * H_SIZE, np.uint8)
    flat_m = np.zeros(len(rows) * H_SIZE, np.uint8)
    for i, y in enumerate(rows):
        dec.expand_row(raw, int(offsets[y]), flat_h, flat_m, i * H_SIZE)
    ref_h = flat_h.reshape(len(rows), H_SIZE)
    ref_m = flat_m.reshape(len(rows), H_SIZE)
    fast_h, fast_m = load_vmc(raw, size_y)
    for i, y in enumerate(rows):
        check(np.array_equal(ref_h[i], fast_h[y]), f"row {y} heights")
        check(np.array_equal(ref_m[i], fast_m[y]), f"row {y} flags")


def test_encoder_handles_a_world_with_a_single_symbol() -> None:
    """A flat world has one distinct delta, so the code has no depth at all.

    This is exactly the shape a freshly created world has, so the tree must
    stay valid: `DHuff` gets a phantom symbol to keep the code well-formed.
    """
    height = np.full((4, H_SIZE), 90, np.uint8)
    meta = np.full((4, H_SIZE), 8, np.uint8)
    packed = save_vmc(height, meta)
    h, m = load_vmc(packed, 4)
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


def test_vmc_is_smaller_than_vmp() -> None:
    vmc_path, vmp_path = _vmc_vmp_pair()
    vmv = os.path.getsize(vmc_path)
    vmp = os.path.getsize(vmp_path)
    check(vmv < vmp, f"compression should shrink the file: {vmv} vs {vmp}")
    print(f"    ({vmp} -> {vmv} bytes, {100 * vmv / vmp:.0f}%)")


def test_bad_boundary_is_reported() -> None:
    vmc_path, _ = _vmc_vmp_pair()
    from core.config import WorldConfig

    with open(os.path.join(os.path.dirname(vmc_path), "world.ini"), "r",
              encoding="utf-8", errors="replace") as fh:
        cfg = WorldConfig.parse(fh.read())
    with open(vmc_path, "rb") as fh:
        raw = bytearray(fh.read())
    # claim row 0 is one byte longer than it really is
    raw[4] = (raw[4] + 1) & 0xFF
    try:
        load_vmc(bytes(raw), cfg.size_y)
    except VmcError:
        return
    raise AssertionError("expected a boundary error for a corrupted size table")
