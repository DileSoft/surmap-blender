"""Prints the sliced LINE_render hash that tests/test_render.py pins.

Run it after a *deliberate* change to the shading, then update
`LINE_HASH` in tests/test_render.py. It calls the same `sample_indices()` the
test uses, so the two can never drift apart.

    blender --background --python tests/pin_render_hash.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import test_render  # noqa: E402

indices = test_render.sample_indices()
print(f"rows={indices.shape[0]}")
print(f'LINE_HASH = "{test_render.fnv1a(indices)}"')
print(f"(full-map reference stays {test_render.LINE_HASH_FULL_MAP})")
