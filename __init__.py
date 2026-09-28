"""Vangers world editor for Blender.

A Blender port of the SURMAP surface tools: load a Vangers world (``.vmp`` +
``world.ini``), edit its two terrain layers as meshes, and write it back.

``core/`` holds the whole format model and never imports ``bpy``, so it can be
unit-tested with any Python that has numpy — see ``tests/run_tests.py``.
"""

from __future__ import annotations

bl_info = {
    "name": "Vangers World",
    "author": "surmap-blender",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Vangers",
    "description": "Edit Vangers VMP worlds: two-layer terrain meshes, bridges, materials",
    "category": "Import-Export",
}

import os
import sys

# Make the bundled `core` package importable when installed as a plain folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from . import ops, ui  # noqa: E402


def register() -> None:
    ops.register()
    ui.register()


def unregister() -> None:
    ui.unregister()
    ops.unregister()


if __name__ == "__main__":
    register()
