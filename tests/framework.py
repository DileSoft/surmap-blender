"""A tiny test harness, so the suite runs anywhere numpy does.

Deliberately dependency-free: the same file works under Blender's bundled
Python, where pytest is not installed.
"""

from __future__ import annotations

import os
import traceback
from typing import Callable

#: The Vangers checkout used for real-data tests. Override with VANGERS_REPO.
VANGERS_REPO = os.environ.get(
    "VANGERS_REPO", ""
)

WORLD_DIRS = [
    r"vangers\bin\thechain",
    r"vangers\bin\surmap\thechain",
]


def world_file(name: str) -> str | None:
    """Absolute path of ``name`` (e.g. ``output.vmp``) in any known world dir."""
    for sub in WORLD_DIRS:
        root = os.path.join(VANGERS_REPO, sub)
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            candidate = os.path.join(root, entry, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def all_world_files(name: str) -> list[str]:
    out: list[str] = []
    for sub in WORLD_DIRS:
        root = os.path.join(VANGERS_REPO, sub)
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            candidate = os.path.join(root, entry, name)
            if os.path.isfile(candidate):
                out.append(candidate)
    return out


class Failure(AssertionError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Failure(msg)


def check_eq(actual, expected, msg: str) -> None:
    if actual != expected:
        raise Failure(f"{msg}\n  expected: {expected!r}\n  actual:   {actual!r}")


def run_all(tests: list[tuple[str, Callable[[], None]]]) -> int:
    passed = 0
    failed: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001 - report everything
            failed.append((name, traceback.format_exc()))
            print(f"FAIL  {name}")
        else:
            passed += 1
            print(f"ok    {name}")
    print(f"\n{passed} passed, {len(failed)} failed")
    for name, tb in failed:
        print(f"\n=== {name} ===\n{tb}")
    return 1 if failed else 0
