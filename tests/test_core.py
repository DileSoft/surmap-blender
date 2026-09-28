"""Stage 0 tests: config, .vmp I/O, surface accessors and the tunnel solver.

The real-data tests use the Vangers checkout and assert byte-exact round-trips.
"""

from __future__ import annotations

import os

import numpy as np

from core.constants import (
    DELTA_MASK,
    DOUBLE_LEVEL,
    H_SIZE,
    MAX_RDELTA,
    MIN_RDELTA,
    TERRAIN_OFFSET,
    TUNNEL_PROOF,
)
from core.config import WorldConfig, WorldConfigError
from core.solver import solve
from core.surface import (
    MIN_TUNNEL_GAP,
    Surface,
    delta_from_gap,
    gap_is_tunnel,
    thickness_from_delta,
)
from core.vmp import load_vmp, load_vmp_bytes, load_vmp_rows, save_vmp_bytes, vmp_size_y

from framework import check, check_eq, sample_config, small_vmp, vmc_world


def _sample_world_ini() -> str:
    path = small_vmp()
    with open(os.path.join(os.path.dirname(path), "world.ini"), "r",
              encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# world.ini
# ---------------------------------------------------------------------------
def test_ini_parses_retail_config() -> None:
    cfg = WorldConfig.parse(_sample_world_ini())
    check_eq(cfg.map_power_x, 11, "Map Power X")
    check(cfg.size_y in (2048, 16384), f"unexpected size_y {cfg.size_y}")
    check_eq(cfg.begin_colors, [1, 32, 64, 72, 88, 104, 112, 120], "Begin Colors")
    check_eq(cfg.end_colors, [31, 63, 71, 87, 103, 111, 119, 127], "End Colors")
    check_eq(cfg.file_name, "output", "File Name")


def test_ini_compression_flag_matches_the_files_on_disk() -> None:
    """`world.ini` must agree with which surface file its world actually ships.

    Worlds come in both flavours -- an uncompressed `.vmp` and a `.vmc` whose
    row count only exists in the ini -- so a mismatch is a real trap.
    """
    vmp_path = small_vmp()
    check_eq(sample_config(vmp_path).is_compressed, False,
             f"{vmp_path} ships a .vmp, so the ini must not claim compression")

    vmc_dir = os.path.dirname(vmc_world())
    check_eq(sample_config(vmc_world()).is_compressed, True,
             f"{vmc_dir} ships a .vmc, so the ini must claim compression")


def test_ini_rejects_bad_version() -> None:
    try:
        WorldConfig.parse("[Global Parameters]\nMap Power X=11\n[Storage]\nVersion=9.9\n")
    except WorldConfigError:
        return
    raise AssertionError("expected WorldConfigError for a bad Storage Version")


def test_ini_roundtrip() -> None:
    cfg = WorldConfig.parse(_sample_world_ini())
    again = WorldConfig.parse(cfg.dumps())
    check_eq(again.map_power_y, cfg.map_power_y, "Map Power Y survives a round-trip")
    check_eq(again.begin_colors, cfg.begin_colors, "Begin Colors survive")
    check_eq(again.end_colors, cfg.end_colors, "End Colors survive")
    check_eq(again.is_compressed, cfg.is_compressed, "compression flag survives")
    check_eq(again.dynamic_palette.wave_terrain, cfg.dynamic_palette.wave_terrain,
             "wave terrain survives")
    check_eq([c.terrain for c in again.dynamic_palette.cycles],
             [c.terrain for c in cfg.dynamic_palette.cycles], "cycles survive")


def test_new_world_config_is_valid() -> None:
    from core.config import new_world_config

    cfg = new_world_config(12, compressed=False)
    again = WorldConfig.parse(cfg.dumps())
    check_eq(again.map_power_y, 12, "fresh world keeps its Y power")
    check_eq(again.is_compressed, False, "fresh world keeps its storage format")
    check_eq(again.size_y, 4096, "fresh world size_y")


# ---------------------------------------------------------------------------
# .vmp
# ---------------------------------------------------------------------------
def test_vmp_roundtrip_is_byte_exact() -> None:
    """Reading and writing a real `.vmp` must not change a single byte.

    One small world (2048 rows, 8 MB) is enough: the row layout is fixed, and
    sweeping every shipped world would cost minutes for no extra coverage.
    """
    path = small_vmp()
    with open(path, "rb") as fh:
        raw = fh.read()
    size_y = vmp_size_y(raw)
    check_eq(size_y * 2 * H_SIZE, len(raw), "row stride")
    height, meta = load_vmp(path)
    check_eq(height.shape, (size_y, H_SIZE), "height shape")
    check_eq(height.dtype, np.uint8, "height dtype")
    check_eq(save_vmp_bytes(height, meta), raw, "byte-exact round-trip")


def test_vmp_row_window_reads_the_same_bytes() -> None:
    """`load_vmp_rows` is what the import window uses; it must seek, not guess."""
    path = small_vmp()
    full_h, full_m = load_vmp(path)
    y0 = 3
    win_h, win_m = load_vmp_rows(path, y0, 17)
    check(np.array_equal(win_h, full_h[y0:y0 + 17]), "window heights")
    check(np.array_equal(win_m, full_m[y0:y0 + 17]), "window flags")

    # out-of-range requests are refused rather than silently clamped
    from core.vmp import VmpError

    for y0_bad, rows_bad in ((-1, 4), (0, -4), (full_h.shape[0] - 1, 8)):
        try:
            load_vmp_rows(path, y0_bad, rows_bad)
        except VmpError:
            continue
        raise AssertionError(f"expected a range error for {y0_bad}+{rows_bad}")


def test_vmp_reads_known_layout() -> None:
    # A hand-built buffer must decode row by row, heights first.
    size_y = 2
    raw = bytearray()
    for y in range(size_y):
        raw += bytes([10 + y] * H_SIZE)
        raw += bytes([0x40 | 0x08] * H_SIZE)
    height, meta = load_vmp_bytes(bytes(raw), size_y)
    check_eq(height[0].tolist(), [10] * H_SIZE, "row 0 heights")
    check_eq(height[1].tolist(), [11] * H_SIZE, "row 1 heights")
    check(all(bool(m & DOUBLE_LEVEL) for m in meta[0]), "flags decoded from the second half")


# ---------------------------------------------------------------------------
# surface accessors
# ---------------------------------------------------------------------------
def _pair_surface(gap: int = 64, size_y: int = 4) -> Surface:
    """A 4-row world with one double pair at columns (10, 11)."""
    s = Surface.empty(size_y, base_height=50, terrain=1)
    mask = np.zeros((size_y, H_SIZE // 2), dtype=bool)
    mask[:, 5] = True
    s.pair_set_double(mask)
    s.pair_write_up(np.full(mask.shape, 120, np.uint8), mask)
    s.pair_write_down(np.full(mask.shape, 120 - gap, np.uint8), mask)
    s.pair_write_delta(np.full(mask.shape, 5, np.uint8), mask)
    return s


def test_double_flag_is_set_on_both_columns() -> None:
    s = _pair_surface()
    check_eq(s.count_doubles(), 4, "one pair per row")
    check_eq(s.pair_mismatches(), 0, "both flag bytes carry DOUBLE_LEVEL")


def test_up_and_down_alt() -> None:
    s = _pair_surface(gap=64)
    up = s.up_alt()
    down = s.down_alt()
    check_eq(int(up[0, 11]), 120, "deck height at the odd column")
    check_eq(int(up[0, 10]), 120, "deck height seen from the even column")
    check_eq(int(down[0, 10]), 56, "underside at the even column")
    check_eq(int(down[0, 11]), 56, "underside seen from the odd column")
    check_eq(int(up[0, 5]), 50, "plain column keeps a single height")


def test_terrain_split_between_layers() -> None:
    s = _pair_surface()
    mask = np.ones_like(s.pair_double())
    s.pair_write_down_terrain(3, mask)
    s.pair_write_up_terrain(6, mask)
    check_eq(int(s.down_terrain()[0, 10]), 3, "down material lives in the even byte")
    check_eq(int(s.up_terrain()[0, 11]), 6, "up material lives in the odd byte")
    # the write must not disturb the other layer or the flags
    check(s.is_double()[0, 10] and s.is_double()[0, 11], "flags survive a material write")


def test_wrapping_columns() -> None:
    s = Surface.empty(2, base_height=40, terrain=1)
    mask = np.zeros((2, H_SIZE // 2), dtype=bool)
    mask[:, 0] = True  # pair (0, 1)
    s.pair_set_double(mask)
    s.pair_write_up(np.full(mask.shape, 200, np.uint8), mask)
    s.pair_write_down(np.full(mask.shape, 100, np.uint8), mask)
    # the even column of a pair reads its deck from x+1 ...
    check_eq(int(s.up_alt()[0, 0]), 200, "even column 0 sees the deck of its pair")
    # ... and the odd column of the previous pair reads its floor from x-1,
    # which is the X-torus boundary at 0/2047.
    check_eq(int(s.down_alt()[0, 1]), 100, "odd column 1 sees the floor of its pair")
    check_eq(int(s.up_alt()[0, H_SIZE - 1]), 40, "column 2047 is a plain column")
    check_eq(int(s.down_alt()[0, 0]), 100, "even column 0 is the floor")


# ---------------------------------------------------------------------------
# delta arithmetic
# ---------------------------------------------------------------------------
def test_delta_from_gap_matches_the_original_formula() -> None:
    # Retail worlds carry delta 1/1 -> d4 = 5 -> thickness 48 -> gap 64..71.
    check_eq(int(delta_from_gap(np.int32(64))), 5, "gap 64 gives the retail delta")
    check_eq(int(thickness_from_delta(np.uint8(5))), 48, "d4=5 is a 48-voxel tunnel")
    check_eq(int(delta_from_gap(np.int32(32))), 1, "the minimum valid tunnel")
    check_eq(int(thickness_from_delta(np.uint8(1))), 16, "d4=1 is a 16-voxel tunnel")
    # anything narrower than MIN_TUNNEL_GAP is not a tunnel
    for gap in (0, 8, 16, 24, 31):
        check(not bool(gap_is_tunnel(np.int32(gap))), f"gap {gap} is not a tunnel")
    for gap in (32, 40, 64, 144, 255):
        check(bool(gap_is_tunnel(np.int32(gap))), f"gap {gap} is a tunnel")
    check_eq(MIN_TUNNEL_GAP, TUNNEL_PROOF + 2 * MIN_RDELTA, "min gap formula")


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
def test_solver_repairs_a_new_bridge_instead_of_killing_it() -> None:
    s = Surface.empty(4, base_height=100, terrain=1)
    region = np.zeros((4, H_SIZE // 2), dtype=bool)
    region[:, 100:104] = True
    s.pair_set_double(region)
    # the mask was painted but the underside was never sculpted: gap is 0
    check_eq(int(s.pair_gap()[0, 100]), 0, "gap before solving")
    report = solve(s, edited=region, new_pairs=region)
    check_eq(report.repaired, 16, "four rows x four pairs repaired")
    check_eq(int(s.pair_gap()[0, 100]), MIN_TUNNEL_GAP, "underside lowered to a valid gap")
    check_eq(s.count_doubles(), 16, "the bridge survived")
    check_eq(int(s.pair_delta()[0, 100]), 1, "delta derived from the gap")
    check_eq(report.collapsed, 0, "nothing collapsed")


def test_solver_collapses_a_deck_pushed_into_the_floor() -> None:
    s = _pair_surface(gap=64)
    region = s.pair_double()
    # the author drags the deck down until it meets the floor
    s.pair_write_up(np.full(region.shape, 60, np.uint8), region)
    report = solve(s, edited=region)
    check_eq(report.collapsed, 4, "all four pairs collapsed")
    check_eq(s.count_doubles(), 0, "no double columns left")
    check_eq(int(s.height[0, 10]), 56, "the floor survives the collapse")
    check_eq(int(s.height[0, 11]), 56, "both columns merge onto the floor")
    check_eq(int(s.down_terrain()[0, 10]), 1, "floor material survives")


def test_solver_preserves_untouched_pairs_byte_for_byte() -> None:
    s = _pair_surface(gap=64)
    # an overhang: the deck sits below the floor, as shipped worlds do
    s.pair_write_up(np.full(s.pair_double().shape, 20, np.uint8), s.pair_double())
    before_h, before_m = s.height.copy(), s.meta.copy()

    report = solve(s)
    check_eq(report.total(), 0, "nothing to do without an edited region")
    check(np.array_equal(before_h, s.height), "heights untouched")
    check(np.array_equal(before_m, s.meta), "flags untouched")


def test_solver_enforce_is_opt_in_and_reports() -> None:
    s = _pair_surface(gap=64)
    s.pair_write_up(np.full(s.pair_double().shape, 20, np.uint8), s.pair_double())
    before = s.height.copy()

    report = solve(s, enforce=True)
    check_eq(report.enforced, 4, "pre-existing violations collapsed on request")
    check(not np.array_equal(before, s.height), "heights really changed")
    check_eq(s.count_doubles(), 0, "violations are gone")


def test_solver_is_idempotent() -> None:
    s = Surface.empty(4, base_height=100, terrain=1)
    region = np.zeros((4, H_SIZE // 2), dtype=bool)
    region[:, 10:20] = True
    s.pair_set_double(region)
    solve(s, edited=region, new_pairs=region)

    first_h, first_m = s.height.copy(), s.meta.copy()
    second = solve(s, edited=region)
    check_eq(second.total(), 0, "a second pass has nothing to do")
    check(np.array_equal(first_h, s.height), "heights are stable")
    check(np.array_equal(first_m, s.meta), "flags are stable")


def test_solver_never_touches_a_pair_outside_the_edited_region() -> None:
    s = _pair_surface(gap=64)
    before_h, before_m = s.height.copy(), s.meta.copy()

    far = np.zeros(s.pair_double().shape, dtype=bool)
    far[:, 300:310] = True  # columns 600..620, far from the pair at column 10
    s.pair_set_double(far)
    solve(s, edited=far, new_pairs=far)

    # everything left of column 512 must be bit-identical
    check(np.array_equal(before_h[:, :512], s.height[:, :512]),
          "heights left of the edit are untouched")
    check(np.array_equal(before_m[:, :512], s.meta[:, :512]),
          "flags left of the edit are untouched")
    check_eq(int(s.pair_delta()[0, 5]), 5, "the untouched pair keeps its original delta")


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------
def test_real_world_double_flags_are_symmetric() -> None:
    """Both columns of a real double pair must carry the flag."""
    s = Surface(*load_vmp(small_vmp()))
    check_eq(s.pair_mismatches(), 0, "DOUBLE symmetry")
    check(s.count_doubles() > 0, "expected some double columns")


def test_real_world_delta_statistics_are_pinned() -> None:
    """Regression baseline for what the shipped data actually looks like.

    The retail deltas are *not* a function of the current geometry: every one of
    the 16 `d4` values occurs, and pairs carrying the *same* `d4` have wildly
    different gaps. That is exactly why the solver never recomputes a pair the
    author did not touch.
    """
    s = Surface(*load_vmp(small_vmp()))
    dbl = s.pair_double()
    d4 = s.pair_delta()[dbl]
    gap = s.pair_gap()[dbl]

    check(dbl.sum() > 0, "the sample should contain double pairs")
    check_eq(sorted(np.unique(d4).tolist()), list(range(16)),
             "all 16 d4 values should occur in the shipped data")
    check(int((gap < MIN_TUNNEL_GAP).sum()) > 0,
          "shipped data violates the tunnel invariant (expected)")
    # the same d4 must not imply the same gap -- that is the whole point
    wide = gap[d4 == int(d4.max())]
    check(int(wide.max() - wide.min()) > MIN_TUNNEL_GAP,
          "a single d4 value should cover a range of real gaps")


def test_real_world_export_is_unchanged_after_solve() -> None:
    """Exporting an untouched world must be a byte-for-byte no-op."""
    path = small_vmp()
    with open(path, "rb") as fh:
        raw = fh.read()
    surface = Surface(*load_vmp(path))
    report = solve(surface)
    check_eq(report.total(), 0, "solve must be a no-op")
    check_eq(save_vmp_bytes(surface.height, surface.meta), raw,
             "untouched export is byte-identical")
