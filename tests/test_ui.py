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


def _check_operator_call(idname, args, kw) -> None:
    """Validates a panel's `layout.operator()` call against the real RNA.

    Blender's signature is the source of truth, so this catches new mistakes
    too: a parameter that does not exist, or a second positional argument
    (``text`` is the only one, there is no execution context).
    """
    params = [p.identifier for p in
              bpy.types.UILayout.bl_rna.functions["operator"].parameters][1:]
    if "context" in params:  # future Blender; allow it, do not require it
        params.remove("context")
    for name in kw:
        if name not in params:
            raise AssertionError(
                f"layout.operator({idname!r}, ...) got an unknown argument "
                f"{name!r}; UILayout.operator accepts {params}")
    if len(args) > 1:
        raise AssertionError(
            f"layout.operator({idname!r}, {args!r}) passes an execution context, "
            "but UILayout.operator has no such parameter -- its second "
            "positional is `text`. The exception would abort draw() and the "
            "button would never appear.")


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
        # Mirror the real UILayout.operator signature. A permissive *args stub
        # hides exactly the mistake this test exists for: that function has no
        # execution-context parameter, so passing "INVOKE_DEFAULT" positionally
        # raises inside draw() and silently eats the rest of the panel.
        _check_operator_call(idname, args, kw)
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


def test_window_panel_offers_an_import_button() -> None:
    """The button has to survive draw(): the user's first symptom was its absence."""
    from vangers_world.ui import panels

    log = _draw(panels.VANGERS_PT_window)
    ops = [name for kind, name, _ in log if kind == "operator"]
    check("vangers.import_world" in ops, f"no import button in the panel: {ops}")


def test_button_without_a_path_opens_the_browser() -> None:
    """A panel button calls execute(), so the dialog must be reachable there.

    ``fileselect_add`` crashes Blender in background mode, so the call is
    recorded instead of made -- the point is which branch of the operator runs.
    """
    from vangers_world.ops import world

    calls: list = []
    real_open = world.open_file_browser
    real_can = world.can_open_browser
    world.open_file_browser = lambda context, op: calls.append(op)
    world.can_open_browser = lambda: True
    try:
        result = bpy.ops.vangers.import_world(filepath="")
    finally:
        world.open_file_browser = real_open
        world.can_open_browser = real_can

    check_eq(len(calls), 1, "execute() with no path must ask for a file")
    check("RUNNING_MODAL" in result, f"expected a pending operator, got {result}")
    print("    button without a path ->", result)


def test_import_does_not_crash_headless() -> None:
    """No browser to show in background: report instead of opening one.

    Blender turns an operator that reports ERROR and cancels into a Python
    RuntimeError when called from a script, so that is the expected signal.
    """
    try:
        result = bpy.ops.vangers.import_world(filepath="")
    except RuntimeError as exc:
        check("headless" in str(exc), f"unexpected error: {exc}")
        return
    check("CANCELLED" in result, f"headless import must refuse politely, got {result}")
