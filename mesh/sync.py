"""Reading the two Blender meshes back into a :class:`Surface`.

This is the export path. The upper mesh is authoritative for the deck of every
pair and for plain (non-double) columns; the lower mesh is authoritative for the
underside and for the ``double`` mask. Materials come from the ``terrain``
attributes. Nothing is reconciled until :func:`heights_to_surface` runs the
solver, so the meshes stay the source of truth while the user is editing.
"""

from __future__ import annotations

import bpy
import numpy as np

from core.solver import SolveReport, solve
from core.surface import PAIRS, Surface
from core.window import (
    Region,
    scatter_pair_heights,
    scatter_pair_terrain,
    scatter_plain_heights,
    scatter_single_terrain,
)

from .build import ATTR_DOUBLE, ATTR_TERRAIN, get_object_region


class SyncError(RuntimeError):
    pass


def _attr(obj: bpy.types.Object, name: str) -> bpy.types.Attribute | None:
    return obj.data.attributes.get(name)


def vertex_z(obj: bpy.types.Object) -> np.ndarray:
    """Z of every vertex, in mesh order."""
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)[:, 2]


def grid_of(obj: bpy.types.Object, width: int, depth: int) -> np.ndarray:
    z = vertex_z(obj)
    if z.size != width * depth:
        raise SyncError(
            f"{obj.name}: expected {width * depth} vertices, found {z.size}"
        )
    return z.reshape(depth, width)


def upper_deck(obj: bpy.types.Object, region: Region) -> np.ndarray:
    """The deck height of every pair in the window, as ``(depth, pair_count)``."""
    deck = grid_of(obj, region.width, region.depth)
    offset = region.x0 & 1  # pair columns sit at odd window indices when x0 is even
    return deck[:, offset::2][:, : region.pair_count]


def upper_terrain(obj: bpy.types.Object, region: Region) -> np.ndarray:
    attr = _attr(obj, ATTR_TERRAIN)
    if attr is None:
        raise SyncError(f"{obj.name} has no '{ATTR_TERRAIN}' attribute")
    out = np.empty(len(obj.data.vertices), dtype=np.int32)
    attr.data.foreach_get("value", out)
    return out.reshape(region.depth, region.width)


def lower_underside(obj: bpy.types.Object, region: Region) -> np.ndarray:
    pairs = region.pair_region()
    return grid_of(obj, pairs.width // 2, region.depth)


def lower_terrain(obj: bpy.types.Object, region: Region) -> np.ndarray:
    attr = _attr(obj, ATTR_TERRAIN)
    if attr is None:
        raise SyncError(f"{obj.name} has no '{ATTR_TERRAIN}' attribute")
    pairs = region.pair_region()
    out = np.empty(len(obj.data.vertices), dtype=np.int32)
    attr.data.foreach_get("value", out)
    return out.reshape(region.depth, pairs.width // 2)


def double_mask(obj: bpy.types.Object, region: Region) -> np.ndarray:
    """The ``double`` attribute of the lower mesh, per pair."""
    attr = _attr(obj, ATTR_DOUBLE)
    if attr is None:
        raise SyncError(f"{obj.name} has no '{ATTR_DOUBLE}' attribute")
    pairs = region.pair_region()
    out = np.empty(len(obj.data.vertices), dtype=bool)
    attr.data.foreach_get("value", out)
    return out.reshape(region.depth, pairs.width // 2)


def heights_to_surface(surface: Surface, upper: bpy.types.Object,
                       lower: bpy.types.Object | None, *,
                       enforce: bool = False) -> SolveReport:
    """Pulls both meshes into ``surface`` and reconciles the tunnels.

    ``new_pairs`` is derived by comparing the incoming mask with the loaded
    one, so a freshly painted bridge is repaired rather than collapsed.
    """
    region_upper = get_object_region(upper)
    if lower is None:
        deck = grid_of(upper, region_upper.width, region_upper.depth)
        scatter_plain_heights(surface, region_upper, deck)
        surface.write_single_terrain_arr(upper_terrain(upper, region_upper))
        return solve(surface)

    region = get_object_region(lower)
    if (region.x0, region.y0, region.depth) != (region_upper.x0, region_upper.y0,
                                                 region_upper.depth):
        raise SyncError(
            f"the two meshes cover different windows: {region_upper} vs {region}"
        )

    mask = double_mask(lower, region)
    rows = (region.y0 + np.arange(region.depth, dtype=np.int64)) % surface.size_y
    first = region.pair_region().x0 // 2
    cols = (first + np.arange(mask.shape[1], dtype=np.int64)) % PAIRS

    # the solver works on whole-world masks, so lift the window mask into one
    full = np.zeros_like(surface.pair_double())
    full[np.ix_(rows, cols)] = mask
    created = full & ~surface.pair_double()

    surface.pair_set_double(full)
    deck = upper_deck(upper, region)
    underside = lower_underside(lower, region)
    scatter_pair_heights(surface, region, deck, underside)

    up_terr = upper_terrain(upper, region)[:, (region.x0 & 1)::2][:, : mask.shape[1]]
    scatter_pair_terrain(surface, region, up_terr, mask, upper=True)
    scatter_pair_terrain(surface, region, lower_terrain(lower, region), mask, upper=False)

    # plain columns keep the upper mesh's own heights and materials
    plain_deck = grid_of(upper, region_upper.width, region_upper.depth)
    scatter_plain_heights(surface, region_upper, plain_deck)
    scatter_single_terrain(surface, region_upper, upper_terrain(upper, region_upper))

    return solve(surface, edited=full, new_pairs=created, enforce=enforce)
