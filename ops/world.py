"""Blender operators: import, export and world creation."""

from __future__ import annotations

import os

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import Operator

from core.config import WorldConfig, new_world_config
from core.palette import load_palette, prepared_rgba
from core.render import lower_indices_arr, render_prepare, upper_indices_arr
from core.surface import Surface
from core.vmp import load_vmp, save_vmp, vmp_size_y
from core.vmc import load_vmc, save_vmc
from core.window import Region, gather

from ..mesh import build
from ..mesh import sync

COLL_NAME = "VangersWorld"


def _collection() -> bpy.types.Collection:
    coll = bpy.data.collections.get(COLL_NAME)
    if coll is None:
        coll = bpy.data.collections.new(COLL_NAME)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _clear() -> None:
    coll = bpy.data.collections.get(COLL_NAME)
    if coll is None:
        return
    for obj in list(coll.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if isinstance(data, bpy.types.Mesh) and data.users == 0:
            bpy.data.meshes.remove(data)


def _palette_for(cfg: WorldConfig, directory: str) -> np.ndarray | None:
    """RGBA table from the world's palette file, if we can find it."""
    name = cfg.palette_file
    if not name:
        return None
    for candidate in (
        os.path.join(directory, name),
        os.path.join(directory, "harmony.pal"),
    ):
        if os.path.isfile(candidate):
            with open(candidate, "rb") as fh:
                return prepared_rgba(load_palette(fh.read()), cfg.begin_colors, cfg.end_colors)
    return None


def _load_surface_vmc(path: str, cfg: WorldConfig) -> Surface:
    """`.vmc` carries no row count, so `world.ini` must agree with it."""
    with open(path, "rb") as fh:
        data = fh.read()
    height, meta = load_vmc(data, cfg.size_y)
    return Surface(height, meta)


def _view_state() -> dict:
    scene = bpy.context.scene
    return {
        "region": (int(scene.vmap_x0), int(scene.vmap_y0), int(scene.vmap_width),
                   int(scene.vmap_depth)),
        "step": int(scene.vmap_step),
        "lower": bool(scene.vmap_lower_visible),
    }


class VANGERS_OT_import_world(Operator):
    """Load a Vangers world (.vmp + world.ini) into two meshes"""

    bl_idname = "vangers.import_world"
    bl_label = "Import world"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    x0: IntProperty(name="X start", default=0, min=0)
    y0: IntProperty(name="Y start", default=0, min=0)
    width: IntProperty(name="Columns", default=512, min=2, max=2048)
    depth: IntProperty(name="Rows", default=512, min=2)
    step: IntProperty(name="Step", default=1, min=1, max=8)

    def execute(self, context):
        path = self.filepath
        if not path or not os.path.isfile(path):
            self.report({"ERROR"}, "no .vmp file selected")
            return {"CANCELLED"}
        directory = os.path.dirname(path)
        ini_path = os.path.join(directory, "world.ini")
        if not os.path.isfile(ini_path):
            self.report({"ERROR"}, "world.ini must sit next to the .vmp")
            return {"CANCELLED"}
        with open(ini_path, "r", encoding="utf-8", errors="replace") as fh:
            cfg = WorldConfig.parse(fh.read())

        try:
            if path.lower().endswith(".vmc"):
                surface = _load_surface_vmc(path, cfg)
            else:
                height, meta = load_vmp(path)
                surface = Surface(height, meta)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"cannot read surface: {exc}")
            return {"CANCELLED"}

        region = Region(self.x0, self.y0, self.width, self.depth)
        if region.depth > surface.size_y:
            self.report({"ERROR"}, f"the world only has {surface.size_y} rows")
            return {"CANCELLED"}

        luts = render_prepare(cfg.begin_colors, cfg.end_colors, 0)
        rgba = _palette_for(cfg, directory)
        height_w, meta_w = gather(surface, region)
        phase = region.x0 & 1
        offset = phase
        pair_count = region.pair_count
        if rgba is not None:
            up_c = rgba[upper_indices_arr(height_w, meta_w, luts, phase)]
            lo_c = rgba[lower_indices_arr(height_w, meta_w, luts, phase)][
                :, offset::2
            ][:, :pair_count]
        else:
            up_c = lo_c = None

        _clear()
        coll = _collection()
        upper = build.build_upper("Vangers_upper", surface, region, up_c)
        lower = build.build_lower("Vangers_lower", surface, region, lo_c)
        coll.objects.link(upper)
        coll.objects.link(lower)
        lower.hide_set(not bool(context.scene.vmap_lower_visible))

        scene = context.scene
        scene.vmap_source = path
        scene.vmap_x0, scene.vmap_y0 = region.x0, region.y0
        scene.vmap_width, scene.vmap_depth = region.width, region.depth
        scene.vmap_step = self.step
        self.report({"INFO"}, f"loaded {os.path.basename(path)}: "
                              f"{region.width}x{region.depth} at ({region.x0},{region.y0})")
        return {"FINISHED"}


class VANGERS_OT_export_world(Operator):
    """Write the edited surface back to .vmp"""

    bl_idname = "vangers.export_world"
    bl_label = "Export world"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(subtype="FILE_PATH")
    file_format: EnumProperty(
        name="Format",
        items=(("AUTO", "Same as loaded", "keep the format the world was loaded from"),
               ("VMP", ".vmp", "uncompressed"),
               ("VMC", ".vmc", "compressed with the Huffman codec")),
        default="AUTO",
    )
    enforce: BoolProperty(
        name="Enforce tunnel rule everywhere",
        description="Also collapse pairs the author did not touch "
                    "(the shipped worlds contain many that are not valid tunnels)",
        default=False,
    )

    def execute(self, context):
        scene = context.scene
        source = getattr(scene, "vmap_source", "")
        if not source or not os.path.isfile(source):
            self.report({"ERROR"}, "import a world first")
            return {"CANCELLED"}
        upper = bpy.data.objects.get("Vangers_upper")
        lower = bpy.data.objects.get("Vangers_lower")
        if upper is None:
            self.report({"ERROR"}, "no Vangers meshes in the scene")
            return {"CANCELLED"}

        with open(os.path.join(os.path.dirname(source), "world.ini"), "r",
                  encoding="utf-8", errors="replace") as fh:
            cfg = WorldConfig.parse(fh.read())
        try:
            if source.lower().endswith(".vmc"):
                with open(source, "rb") as fh:
                    height, meta = load_vmc(fh.read(), cfg.size_y)
            else:
                height, meta = load_vmp(source)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"cannot re-read the source world: {exc}")
            return {"CANCELLED"}

        surface = Surface(height, meta)
        report = sync.heights_to_surface(surface, upper, lower, enforce=self.enforce)

        choice = self.file_format
        if choice == "AUTO":
            choice = "VMC" if source.lower().endswith(".vmc") else "VMP"
        out = self.filepath
        if not out:
            out = os.path.splitext(source)[0] + "." + choice.lower()
        if choice == "VMC":
            with open(out, "wb") as fh:
                fh.write(save_vmc(surface.height, surface.meta))
        else:
            save_vmp(out, surface.height, surface.meta)
        self.report({"INFO"}, f"saved {os.path.basename(out)} ({report})")
        return {"FINISHED"}


class VANGERS_OT_new_world(Operator):
    """Create an empty world and build its meshes"""

    bl_idname = "vangers.new_world"
    bl_label = "New world"
    bl_options = {"REGISTER", "UNDO"}

    rows_log2: IntProperty(name="Rows 2^n", default=12, min=4, max=14)
    base_height: IntProperty(name="Base height", default=96, min=0, max=255)
    terrain: IntProperty(name="Terrain", default=1, min=0, max=7)
    width: IntProperty(name="Columns", default=512, min=2, max=2048)
    depth: IntProperty(name="Rows", default=512, min=2)

    def execute(self, context):
        cfg = new_world_config(self.rows_log2, compressed=False)
        surface = Surface.empty(1 << self.rows_log2, self.base_height, self.terrain)
        region = Region(0, 0, self.width, self.depth)
        luts = render_prepare(cfg.begin_colors, cfg.end_colors, 0)

        _clear()
        coll = _collection()
        coll.objects.link(build.build_upper("Vangers_upper", surface, region))
        coll.objects.link(build.build_lower("Vangers_lower", surface, region))

        scene = context.scene
        scene.vmap_source = ""
        scene.vmap_x0, scene.vmap_y0 = 0, 0
        scene.vmap_width, scene.vmap_depth = self.width, self.depth
        self.report({"INFO"}, f"new world: {1 << self.rows_log2} rows, "
                              f"terrain {self.terrain}, base {self.base_height}")
        return {"FINISHED"}


class VANGERS_OT_paint_bridge(Operator):
    """Paint the double-level mask on the lower mesh, around the 3D cursor"""

    bl_idname = "vangers.paint_bridge"
    bl_label = "Paint bridge"
    bl_options = {"REGISTER", "UNDO"}

    radius: IntProperty(name="Radius", default=8, min=1, max=1024)
    value: BoolProperty(name="Bridge", default=True)

    def execute(self, context):
        lower = bpy.data.objects.get("Vangers_lower")
        if lower is None:
            self.report({"ERROR"}, "no lower mesh in the scene")
            return {"CANCELLED"}

        region = build.get_object_region(lower)
        pairs = region.pair_region()
        width = pairs.width // 2
        depth = region.depth
        if width == 0 or depth == 0:
            return {"CANCELLED"}

        mask = sync.double_mask(lower, region).copy()
        cursor = context.scene.cursor.location
        cols = (pairs.x0 + np.arange(pairs.width, dtype=np.int64))[0::2]
        rows = region.y0 + np.arange(depth, dtype=np.int64)

        i0 = int(np.argmin(np.abs(cols - cursor.x)))
        j0 = int(np.argmin(np.abs(rows - cursor.y)))
        rad = self.radius
        for dj in range(max(0, j0 - rad), min(depth, j0 + rad + 1)):
            dy = int(rows[dj]) - int(cursor.y)
            for di in range(max(0, i0 - rad), min(width, i0 + rad + 1)):
                dx = int(cols[di]) - int(cursor.x)
                if dx * dx + dy * dy <= rad * rad:
                    mask[dj, di] = self.value

        attr = lower.data.attributes.get(build.ATTR_DOUBLE)
        if attr is None:
            self.report({"ERROR"}, "lower mesh has no double attribute")
            return {"CANCELLED"}
        attr.data.foreach_set("value", mask.reshape(-1))
        lower.data.update()
        self.report({"INFO"}, f"{'bridge' if self.value else 'removed'} under the cursor")
        return {"FINISHED"}


class VANGERS_OT_paint_material(Operator):
    """Paint the terrain material attribute around the 3D cursor"""

    bl_idname = "vangers.paint_material"
    bl_label = "Paint material"
    bl_options = {"REGISTER", "UNDO"}

    terrain: IntProperty(name="Terrain", default=1, min=0, max=7)
    radius: IntProperty(name="Radius", default=8, min=1, max=1024)
    layer: EnumProperty(
        name="Layer",
        items=(("UP", "Upper", "material of the visible terrain"),
               ("DOWN", "Lower", "material under a bridge")),
        default="UP",
    )

    _NAMES = {"UP": "Vangers_upper", "DOWN": "Vangers_lower"}

    def execute(self, context):
        obj = bpy.data.objects.get(self._NAMES[self.layer])
        if obj is None:
            self.report({"ERROR"}, "import a world first")
            return {"CANCELLED"}
        attr = obj.data.attributes.get(build.ATTR_TERRAIN)
        if attr is None:
            self.report({"ERROR"}, f"{obj.name} has no terrain attribute")
            return {"CANCELLED"}

        region = build.get_object_region(obj)
        if self.layer == "UP":
            first_x, step, width = region.x0, 1, region.width
        else:
            pairs = region.pair_region()
            first_x, step, width = pairs.x0, 2, pairs.width // 2

        values = np.empty(len(obj.data.vertices), dtype=np.int32)
        attr.data.foreach_get("value", values)
        values = values.reshape(region.depth, width)

        cursor = context.scene.cursor.location
        xs = first_x + step * np.arange(width, dtype=np.int64)
        ys = region.y0 + np.arange(region.depth, dtype=np.int64)
        i0 = int(np.argmin(np.abs(xs - cursor.x)))
        j0 = int(np.argmin(np.abs(ys - cursor.y)))
        rad = self.radius
        for dj in range(max(0, j0 - rad), min(region.depth, j0 + rad + 1)):
            dy = int(ys[dj]) - int(cursor.y)
            for di in range(max(0, i0 - rad), min(width, i0 + rad + 1)):
                dx = int(xs[di]) - int(cursor.x)
                if dx * dx + dy * dy <= rad * rad:
                    values[dj, di] = self.terrain

        attr.data.foreach_set("value", values.reshape(-1))
        obj.data.update()
        self.report({"INFO"}, f"terrain {self.terrain} on the {self.layer.lower()} layer")
        return {"FINISHED"}


CLASSES = (
    VANGERS_OT_import_world,
    VANGERS_OT_export_world,
    VANGERS_OT_new_world,
    VANGERS_OT_paint_bridge,
    VANGERS_OT_paint_material,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    _register_scene_props()


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    for name in ("vmap_source", "vmap_x0", "vmap_y0", "vmap_width", "vmap_depth",
                 "vmap_step", "vmap_lower_visible", "vmap_terrain", "vmap_paint_radius"):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)


def _register_scene_props() -> None:
    bpy.types.Scene.vmap_source = StringProperty(name="Source", default="")
    bpy.types.Scene.vmap_x0 = IntProperty(name="X0", default=0)
    bpy.types.Scene.vmap_y0 = IntProperty(name="Y0", default=0)
    bpy.types.Scene.vmap_width = IntProperty(name="Width", default=512)
    bpy.types.Scene.vmap_depth = IntProperty(name="Depth", default=512)
    bpy.types.Scene.vmap_step = IntProperty(name="Step", default=1)
    bpy.types.Scene.vmap_lower_visible = BoolProperty(name="Show underside", default=True)
    bpy.types.Scene.vmap_terrain = IntProperty(name="Terrain", default=1, min=0, max=7)
    bpy.types.Scene.vmap_paint_radius = IntProperty(name="Brush radius", default=8,
                                                     min=1, max=1024)
