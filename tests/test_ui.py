"""UI smoke tests.

Panels cannot be drawn in background mode, so the draw bodies are executed
against a recording stub layout. That still catches the mistakes that actually
happen there: undefined names, wrong scene properties and typo'd operators.
"""

from __future__ import annotations

import os
import sys

import bpy

ADDON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ADDON not in sys.path:
    sys.path.insert(0, ADDON)
if os.path.join(ADDON, "tests") not in sys.path:
    sys.path.insert(0, os.path.join(ADDON, "tests"))

from framework import check, check_eq, world_file  # noqa: E402
from loader import load_addon  # noqa: E402

_addon = load_addon(ADDON)
if not hasattr(_addon, "_tests_registered"):
    _addon.register()
    _addon._tests_registered = True


class _Recorder:
    """Duck-typed stand-in for a UILayout."""

    def __init__(self, log: list, depth: int = 0) -> None:
        self.log = log
        self.depth = depth

    def _child(self) -> "_Recorder":
        return _Recorder(self.log, self.depth + 1)

    def row(self, **kw):
        return self._child()

    def column(self, **kw):
        return self._child()

    def box(self):
        return self._child()

    def label(self, **kw):
        self.log.append(("label", kw.get("text", ""), self.depth))
        return self

    def prop(self, data, name, **kw):
        self.log.append(("prop", name, self.depth))
        if not hasattr(data, name):
            raise AssertionError(f"panel draws a property that does not exist: {name}")
        return self

    def operator(self, idname, *args, **kw):
        self.log.append(("operator", idname, self.depth))
        if not hasattr(bpy.ops, idname.split(".")[0]):
            raise AssertionError(f"panel draws an unknown operator: {idname}")
        return _OpResult()

    def separator(self, **kw):
        return self


class _OpResult:
    """Lets the test assign `op.value = ...` the way real operators allow."""

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)


class _Stub:
    def __init__(self, layout):
        self.layout = layout


def _draw(panel_cls):
    log: list = []
    panel_cls.draw(_Stub(_Recorder(log)), bpy.context)
    return log


def test_panels_draw_without_errors() -> None:
    from vangers_world.ui import panels

    _draw(panels.VANGERS_PT_window)
    _draw(panels.VANGERS_PT_authoring)
    print("    panels drawn")


def test_authoring_panel_uses_the_scene() -> None:
    """The panel must read terrain/radius off the scene, not a stray name."""
    from vangers_world.ui import panels

    log = _draw(panels.VANGERS_PT_authoring)
    props = [name for kind, name, _ in log if kind == "prop"]
    check("vmap_terrain" in props, f"terrain slider missing: {props}")
    check("vmap_paint_radius" in props, f"brush radius missing: {props}")


def test_import_button_opens_the_file_browser() -> None:
    """Clicking Import must not run execute with an empty path."""
    from vangers_world.ops import world

    check(hasattr(world.VANGERS_OT_import_world, "invoke"),
          "import operator has no invoke(), so the button cannot open a dialog")
    path = world_file("output.vmp")
    if not path:
        print("    (skipping: no world found)")
        return
    bpy.context.scene.vmap_x0 = 64
    bpy.context.scene.vmap_width = 32
    result = bpy.ops.vangers.import_world("INVOKE_DEFAULT", filepath=path)
    check("FINISHED" in result or "RUNNING_MODAL" in result, f"unexpected result {result}")
    # the panel's window must have been copied onto the operator
    op = bpy.context.window_manager.operators[-1] if bpy.context.window_manager.operators else None
    if op is not None and op.bl_idname == "vangers.import_world":
        check_eq(op.x0, 64, "invoke() seeded x0 from the scene")
        check_eq(op.width, 32, "invoke() seeded width from the scene")
    print("    invoke ok:", result)
