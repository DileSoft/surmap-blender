"""Tunnel invariant solver — the batch counterpart of SURMAP's `pixSet`.

`pixSet` / `pixDownSet` maintain the tunnel *incrementally*, collapsing a pair the
moment a delta would make the deck intersect the floor. Blender edits a whole
surface at once, so this module derives the same end state in one pass instead,
and never mutates topology while the user is sculpting.

What is authored vs derived:

* **authored** — the ``double`` mask, the two height fields, the materials;
* **derived** — the tunnel delta (a pure function of the gap) and the collapse.

Important finding from the shipped worlds: the invariant is a *tooling*
convention, not a data invariant. ``fostral`` contains 16 811 pairs whose deck
sits **below** the floor (overhangs, not tunnels), and every double pair there
carries the engine's constant delta (thickness 48) rather than one derived from
its geometry. So the solver only ever touches pairs the author actually edited;
``enforce=True`` is opt-in and reports how many untouched pairs it had to fix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import DELTA_MASK, DOUBLE_LEVEL, TERRAIN_MASK
from .surface import MIN_TUNNEL_GAP, Surface, delta_from_gap, gap_is_tunnel


@dataclass
class SolveReport:
    """What the solver changed, for the export report shown to the user."""

    repaired: int = 0
    """New pairs whose underside was lowered to reach a valid tunnel."""

    collapsed: int = 0
    """Edited pairs that lost their tunnel and merged into the floor."""

    deltas: int = 0
    """Pairs whose delta field was recomputed from the gap."""

    enforced: int = 0
    """Untouched pairs collapsed because ``enforce=True`` was requested."""

    def total(self) -> int:
        return self.repaired + self.collapsed + self.deltas + self.enforced

    def __str__(self) -> str:
        parts = []
        if self.repaired:
            parts.append(f"{self.repaired} new pairs lowered to a valid gap")
        if self.deltas:
            parts.append(f"{self.deltas} deltas recomputed")
        if self.collapsed:
            parts.append(f"{self.collapsed} tunnels collapsed")
        if self.enforced:
            parts.append(f"{self.enforced} pre-existing pairs enforced")
        return "; ".join(parts) if parts else "nothing to do"


def solve(
    surface: Surface,
    *,
    edited: np.ndarray | None = None,
    new_pairs: np.ndarray | None = None,
    enforce: bool = False,
    min_gap: int = MIN_TUNNEL_GAP,
) -> SolveReport:
    """Reconciles deltas and collapses tunnels for the pairs the author touched.

    Args:
        surface: mutated in place.
        edited: pair mask of everything the author changed. Pairs outside it are
            left byte-identical.
        new_pairs: subset of ``edited`` that were *created* by this edit. They
            are repaired by lowering the underside, never collapsed.
        enforce: also collapse untouched pairs that violate the invariant.
        min_gap: gap a freshly created tunnel is given (``32`` voxels).

    Call ``Surface.pair_set_double`` for the edited region *before* this, so the
    mask already reflects the author's intent.
    """
    report = SolveReport()

    if edited is None and new_pairs is None and not enforce:
        return report

    created_region = new_pairs if new_pairs is not None else np.zeros(
        surface.pair_double().shape, dtype=bool
    )
    region = edited if edited is not None else created_region
    region = region | created_region

    # 1. Freshly created pairs: push the underside down until the gap is a
    #    real tunnel. Authoring a bridge must never destroy the bridge.
    if created_region.any():
        double = surface.pair_double()
        gap = surface.pair_gap()
        need = created_region & double & (gap < min_gap)
        if need.any():
            target = (surface.h_odd.astype(np.int32) - min_gap).astype(np.uint8)
            surface.pair_write_down(target, need)
            report.repaired = int(np.count_nonzero(need))

    # 2. Recompute the delta of every edited double pair, collapsing the ones
    #    whose deck no longer clears the floor.
    if region.any():
        double = surface.pair_double()
        target = region & double
        if target.any():
            gap = surface.pair_gap()
            valid = gap_is_tunnel(gap)
            bad = target & ~valid
            if bad.any():
                _collapse(surface, bad)
                report.collapsed = int(np.count_nonzero(bad))
            good = target & valid
            if good.any():
                want = delta_from_gap(surface.pair_gap())
                changed = good & (want != surface.pair_delta())
                if changed.any():
                    surface.pair_write_delta(want, changed)
                    report.deltas = int(np.count_nonzero(changed))

    # 3. Optional: sweep the rest of the world.
    if enforce:
        double = surface.pair_double()
        viol = double & ~gap_is_tunnel(surface.pair_gap())
        if viol.any():
            _collapse(surface, viol)
            report.enforced = int(np.count_nonzero(viol))

    return report


def _collapse(surface: Surface, mask: np.ndarray) -> None:
    """Merges a double pair into its floor, as `pixSet` does.

    The floor survives: both columns take the lower height and the lower
    material, the flag and delta are cleared. `pixSet` averages the two
    neighbouring floor heights here; we keep the floor height itself, which is
    deterministic and idempotent for a batch solver.
    """
    floor = surface.h_even.copy()
    floor_terrain = (surface.m_even & TERRAIN_MASK).copy()

    surface.h_even[mask] = floor[mask]
    surface.h_odd[mask] = floor[mask]

    keep_terrain = np.uint8(~TERRAIN_MASK & 0xFF)
    surface.m_even[mask] = (surface.m_even[mask] & keep_terrain) | floor_terrain[mask]
    surface.m_odd[mask] = (surface.m_odd[mask] & keep_terrain) | floor_terrain[mask]

    keep_flags = np.uint8(~(DOUBLE_LEVEL | DELTA_MASK) & 0xFF)
    surface.m_even[mask] &= keep_flags
    surface.m_odd[mask] &= keep_flags
