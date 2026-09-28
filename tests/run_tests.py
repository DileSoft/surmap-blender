"""Entry point: `blender --background --python tests/run_tests.py`.

Collects every ``test_*`` callable from the ``test_*`` modules in this folder and
runs them with the dependency-free harness. Real-data tests need a checkout of
KranX/Vangers; set ``VANGERS_REPO`` to it (a sibling ``../Vangers`` is found
automatically), otherwise they skip.
"""

from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> int:
    from framework import run_all

    modules = sorted(
        name for name in os.listdir(HERE)
        if name.startswith("test_") and name.endswith(".py")
    )
    if not modules:
        print("no test modules found")
        return 1
    tests = []
    for name in modules:
        mod = importlib.import_module(name[:-3])
        for attr in sorted(vars(mod)):
            if attr.startswith("test_"):
                tests.append((f"{name[:-3]}.{attr}", getattr(mod, attr)))

    print(f"running {len(tests)} tests from {len(modules)} modules\n")
    return run_all(tests)


if __name__ == "__main__":
    sys.exit(main())
