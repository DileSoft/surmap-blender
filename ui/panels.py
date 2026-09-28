"""Sidebar panels: the window, the tools and the export report."""

from __future__ import annotations

import bpy
import numpy as np
from bpy.types import Panel

from ..mesh import build


class _Panel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Vangers"


class VANGERS_PT_window(_Panel, Panel):
    bl_idname = "VANGERS_PT_window"
    bl_label = "World"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        col = layout.column(align=True)
        col.label(text=f"Source: {scene.vmap_source or '(new world)'}")
        col.label(text=f"Window: {scene.vmap_width}x{scene.vmap_depth} at "
                        f"({scene.vmap_x0}, {scene.vmap_y0})")

        box = layout.box()
        box.label(text="Окно импорта")
        box.prop(scene, "vmap_x0")
        box.prop(scene, "vmap_y0")
        box.prop(scene, "vmap_width")
        box.prop(scene, "vmap_depth")
        # No execution context here: UILayout.operator() has no `context`
        # parameter, and passing one positionally is a TypeError that kills the
        # rest of draw(). The operator opens the file browser from both
        # invoke() and execute(), so a plain button is enough.
        box.operator("vangers.import_world", text="Import world…", icon="IMPORT")
        box.label(text="Рядом с .vmp/.vmc должен лежать world.ini")

        layout.prop(scene, "vmap_lower_visible")


class VANGERS_PT_authoring(_Panel, Panel):
    bl_idname = "VANGERS_PT_authoring"
    bl_label = "Authoring"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        row = layout.row(align=True)
        row.operator("vangers.new_world", icon="ADD")
        row.operator("vangers.export_world", icon="EXPORT")
        layout.label(text="Экспорт пишет файл рядом с исходным миром;")
        layout.label(text="формат и «enforce» — в параметрах оператора (F9).")

        box = layout.box()
        box.label(text="Bridge (double level)")
        box.label(text="Paints the double mask under the 3D cursor.")
        op = box.operator("vangers.paint_bridge", text="Add bridge", icon="MESH_CUBE")
        op.value = True
        op = box.operator("vangers.paint_bridge", text="Remove bridge", icon="MESH_CUBE")
        op.value = False

        mat = layout.box()
        mat.label(text="Material (terrain 0..7)")
        op = mat.operator("vangers.paint_material", icon="COLOR")
        op.terrain = scene.vmap_terrain
        op.radius = scene.vmap_paint_radius
        op.layer = "UP"
        mat.prop(scene, "vmap_terrain")
        mat.prop(scene, "vmap_paint_radius")
        op = mat.operator("vangers.paint_material", text="Paint under", icon="COLOR")
        op.terrain = scene.vmap_terrain
        op.radius = scene.vmap_paint_radius
        op.layer = "DOWN"

        lower = bpy.data.objects.get("Vangers_lower")
        if lower is not None:
            attr = lower.data.attributes.get(build.ATTR_DOUBLE)
            if attr is not None:
                n = len(lower.data.vertices)
                vals = np.empty(n, dtype=bool)
                attr.data.foreach_get("value", vals)
                layout.label(text=f"bridges: {int(vals.sum())} / {n} pairs")


CLASSES = (VANGERS_PT_window, VANGERS_PT_authoring)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
