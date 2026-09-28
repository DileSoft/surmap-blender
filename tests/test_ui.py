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

from framework import check, check_eq, sample_config, sample_rows_file  # noqa: E402
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


def test_import_builds_meshes_from_a_real_world() -> None:
    """The button must load a real world and build both meshes.

    This is the `execute()` path -- which is what a panel button uses, and what
    background Blender gets even for `INVOKE_DEFAULT`.
    """
    from vangers_world.ops import world

    check(hasattr(world.VANGERS_OT_import_world, "invoke"),
          "import operator has no invoke(), so the button cannot open a dialog")
    path = sample_rows_file("output.vmp")
    result = bpy.ops.vangers.import_world(filepath=path, x0=0, y0=0, width=64, depth=32)
    check("FINISHED" in result, f"import failed: {result}")

    upper = bpy.data.objects.get("Vangers_upper")
    lower = bpy.data.objects.get("Vangers_lower")
    check(upper is not None and lower is not None, "both meshes should exist")
    check_eq(len(upper.data.vertices), 64 * 32, "one upper vertex per voxel")
    check_eq(len(lower.data.vertices), (64 // 2) * 32, "one lower vertex per pair")

    # the panel is the source of truth for the window
    scene = bpy.context.scene
    check_eq((scene.vmap_width, scene.vmap_depth), (64, 32),
             "the scene should record the imported window")


def test_import_of_an_offset_window_matches_a_full_read() -> None:
    """The windowed read must be indistinguishable from reading everything.

    This is the seam that is easy to get wrong: the surface handed to the mesher
    holds only the window's rows, so every lookup inside it has to start at
    row 0 while the window still keeps its place in the world.
    """
    import numpy as np

    from core.surface import Surface
    from core.vmp import load_vmp
    from core.window import Region, gather
    from vangers_world.mesh import build
    from vangers_world.ops import world

    path = sample_rows_file("output.vmp")
    x0, y0, width, depth = 4, 5, 40, 24  # pair-aligned in X, offset in Y

    # reference: the whole world, gathered at the absolute position
    reference = Surface(*load_vmp(path))
    world_region = Region(x0, y0, width, depth)
    ref_h, ref_m = gather(reference, world_region)

    # what the addon does: read only the rows, then gather at row 0
    height, meta = world._read_window(path, sample_config(path), y0, depth)
    windowed = Surface(height, meta)
    local = Region(x0, 0, width, depth)
    got_h, got_m = gather(windowed, local)
    check(np.array_equal(ref_h, got_h), "windowed heights match the full read")
    check(np.array_equal(ref_m, got_m), "windowed flags match the full read")

    # and the built mesh must sit at the window's place in the world
    obj = build.build_upper("probe", windowed, local, None, origin_y=y0)
    coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", coords)
    ys = coords.reshape(-1, 3)[:, 1]
    check_eq((int(ys.min()), int(ys.max())), (y0, y0 + depth - 1),
             "the mesh Y coordinates must be absolute")
    check_eq(int(obj["vmap_y0"]), y0, "the object must remember where it came from")


def test_offset_window_round_trips_byte_for_byte() -> None:
    """Import an offset window, export it, and get the original file back.

    This is the whole loop in one test: the windowed read, the mesh round trip
    and the solver's promise that an untouched world exports unchanged. The
    window starts on an odd column with an odd width, which is the case where a
    row-local surface and a world-positioned mesh could disagree.
    """
    import tempfile

    import numpy as np

    path = sample_rows_file("output.vmp")
    result = bpy.ops.vangers.import_world(filepath=path, x0=3, y0=5, width=41, depth=24)
    check("FINISHED" in result, f"import failed: {result}")

    # the window must have been widened outwards to whole pairs, keeping every
    # column the caller asked for
    scene = bpy.context.scene
    check_eq((scene.vmap_x0, scene.vmap_width), (2, 42),
             "an odd window is widened to complete pairs")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.vmp")
        export = bpy.ops.vangers.export_world(filepath=out, file_format="VMP")
        check("FINISHED" in export, f"export failed: {export}")
        with open(path, "rb") as fh:
            original = fh.read()
        with open(out, "rb") as fh:
            written = fh.read()

    check_eq(len(written), len(original), "the exported world must be the same size")
    check(np.array_equal(np.frombuffer(original, np.uint8),
                         np.frombuffer(written, np.uint8)),
          "an untouched world must export byte-identically")


def test_import_seeds_the_window_from_the_panel() -> None:
    """The window the user typed in the panel must reach the operator.

    Blender refuses to construct an Operator from Python, so the helper is
    exercised with a stand-in that only has the four slots it writes to.
    """
    from vangers_world.ops import world

    class FakeSelf:
        x0 = y0 = width = depth = 0

    scene = bpy.context.scene
    scene.vmap_x0, scene.vmap_y0 = 7, 9
    scene.vmap_width, scene.vmap_depth = 48, 24
    fake = FakeSelf()
    world.VANGERS_OT_import_world._seed_from_scene(fake, bpy.context)
    check_eq((fake.x0, fake.y0, fake.width, fake.depth), (7, 9, 48, 24),
             "the panel's window should be copied onto the operator")


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
