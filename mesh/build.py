"""Turning a window of a :class:`Surface` into Blender meshes.

Two objects are produced, mirroring the two layers of the format:

* ``upper`` — one vertex per voxel of the window, Z = ``GET_UP_ALT``. For a
  double pair the deck lives on the odd column, so this mesh shows the bridge.
* ``lower`` — one vertex per **pair** (even column), Z = the underside. The
  ``double`` attribute says which of those pairs actually exist.

Both meshes keep a constant topology, so sculpting and Blender's undo work
normally; whether a column is a bridge is data, not topology.
"""

from __future__ import annotations

import bpy
import numpy as np

from core.constants import H_SIZE
from core.surface import down_alt, down_terrain, is_double, up_alt, up_terrain
from core.window import Region, gather, gather_pair_heights

UPPER = "upper"
LOWER = "lower"

ATTR_TERRAIN = "terrain"
ATTR_DOUBLE = "double"
ATTR_COLOR = "surfcol"


def _quad_grid(width: int, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and loop starts of a ``width x depth`` vertex grid of quads."""
    quads = (width - 1) * (depth - 1)
    if quads <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    row = np.arange(depth - 1, dtype=np.int32)[:, None] * width
    col = np.arange(width - 1, dtype=np.int32)[None, :]
    a = (row + col).reshape(-1)
    starts = np.arange(quads, dtype=np.int32) * 4
    loops = np.stack([a, a + 1, a + 1 + width, a + width], axis=1).reshape(-1).astype(np.int32)
    return loops, starts


def _coords(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """``(depth, width, 3)`` -> flat float32 coordinate array."""
    grid = np.empty(zs.shape + (3,), dtype=np.float32)
    grid[..., 0] = xs[None, :]
    grid[..., 1] = ys[:, None]
    grid[..., 2] = zs
    return grid.reshape(-1)


def _new_mesh_object(name: str, loops: np.ndarray, starts: np.ndarray,
                     coords: np.ndarray) -> bpy.types.Object:
    """``coords`` is a flat float32 XYZ array; ``loops``/``starts`` describe quads."""
    nv = coords.size // 3
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(nv)
    mesh.vertices.foreach_set("co", coords)
    if loops.size:
        mesh.loops.add(loops.size)
        mesh.loops.foreach_set("vertex_index", loops)
        mesh.polygons.add(starts.size)
        mesh.polygons.foreach_set("loop_start", starts)
    mesh.update(calc_edges=True)
    mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
    return bpy.data.objects.new(name, mesh)


def _set_int(mesh: bpy.types.Mesh, name: str, values: np.ndarray) -> None:
    attr = mesh.attributes.get(name) or mesh.attributes.new(name=name, type="INT", domain="POINT")
    attr.data.foreach_set("value", values.reshape(-1).astype(np.int32))


def _set_bool(mesh: bpy.types.Mesh, name: str, values: np.ndarray) -> None:
    attr = mesh.attributes.get(name) or mesh.attributes.new(name=name, type="BOOLEAN",
                                                              domain="POINT")
    attr.data.foreach_set("value", values.reshape(-1).astype(bool))


def _set_color(mesh: bpy.types.Mesh, name: str, rgba: np.ndarray) -> None:
    """Writes per-vertex colours; values are expected 0..1 floats."""
    attr = mesh.attributes.get(name) or mesh.attributes.new(name=name, type="FLOAT_COLOR",
                                                              domain="POINT")
    attr.data.foreach_set("color", rgba.reshape(-1).astype(np.float32))
    try:
        mesh.color_attributes.active_color = attr
        mesh.attributes.default_color_name = name
    except (AttributeError, TypeError):
        pass  # cosmetic only


def set_object_region(obj: bpy.types.Object, region: Region, kind: str) -> None:
    obj["vmap_x0"] = region.x0
    obj["vmap_y0"] = region.y0
    obj["vmap_width"] = region.width
    obj["vmap_depth"] = region.depth
    obj["vmap_layer"] = kind


def get_object_region(obj: bpy.types.Object) -> Region:
    return Region(
        x0=int(obj.get("vmap_x0", 0)),
        y0=int(obj.get("vmap_y0", 0)),
        width=int(obj.get("vmap_width", H_SIZE)),
        depth=int(obj.get("vmap_depth", 1)),
    )


def build_upper(name: str, surface, region: Region, colors: np.ndarray | None = None,
                origin_y: int | None = None) -> bpy.types.Object:
    """One vertex per voxel; Z is the upper surface.

    ``region.y0`` says where the window's rows live *inside* ``surface``;
    ``origin_y`` is where the window sits *in the world*, which is what the mesh
    coordinates and the export metadata use. They differ whenever the caller
    hands in a surface that is already a slice of a world.
    """
    height, meta = gather(surface, region)
    phase = region.x0 & 1
    z = up_alt(height, meta, phase).astype(np.float32)
    terrain = up_terrain(meta, phase)
    dbl = is_double(meta)

    depth, width = region.depth, region.width
    world_region = _placed(region, origin_y)
    loops, starts = _quad_grid(width, depth)
    coords = _coords(region.columns.astype(np.float32),
                     (world_region.y0 + np.arange(depth)).astype(np.float32), z)
    obj = _new_mesh_object(name, loops, starts, coords)

    mesh = obj.data
    _set_int(mesh, ATTR_TERRAIN, terrain)
    _set_bool(mesh, ATTR_DOUBLE, dbl)
    if colors is not None:
        _set_color(mesh, ATTR_COLOR, colors.astype(np.float32) / 255.0)
    set_object_region(obj, world_region, UPPER)
    return obj


def build_lower(name: str, surface, region: Region, colors: np.ndarray | None = None,
                origin_y: int | None = None) -> bpy.types.Object:
    """One vertex per pair (even column); Z is the underside."""
    deck, underside = gather_pair_heights(surface, region)
    pairs = region.pair_region()
    width = pairs.width // 2
    depth = region.depth
    world_region = _placed(region, origin_y)

    # the material and flag of a pair, taken from its even column
    cols = (pairs.x0 + np.arange(pairs.width, dtype=np.int64)) % H_SIZE
    rows = (region.y0 + np.arange(depth, dtype=np.int64)) % surface.size_y
    meta = surface.meta[np.ix_(rows, cols)][:, 0::2]
    terrain = ((meta & 0x38) >> 3).astype(np.uint8)
    dbl = is_double(meta)

    loops, starts = _quad_grid(width, depth)
    xs = (pairs.x0 + np.arange(pairs.width, dtype=np.int64))[0::2].astype(np.float32)
    ys = (world_region.y0 + np.arange(depth)).astype(np.float32)
    coords = _coords(xs, ys, underside.astype(np.float32))

    obj = _new_mesh_object(name, loops, starts, coords)
    mesh = obj.data
    _set_int(mesh, ATTR_TERRAIN, terrain)
    _set_bool(mesh, ATTR_DOUBLE, dbl)
    if colors is not None:
        _set_color(mesh, ATTR_COLOR, colors.astype(np.float32) / 255.0)
    set_object_region(obj, world_region, LOWER)
    return obj


def _placed(region: Region, origin_y: int | None) -> Region:
    """The same window, but positioned in world coordinates."""
    if origin_y is None or origin_y == region.y0:
        return region
    return Region(x0=region.x0, y0=origin_y, width=region.width, depth=region.depth)
