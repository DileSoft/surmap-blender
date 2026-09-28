"""A tiny test harness, so the suite runs anywhere numpy does.

Deliberately dependency-free: the same file works under Blender's bundled
Python, where pytest is not installed.

Real-data tests need a checkout of KranX/Vangers. Point ``VANGERS_REPO`` at it;
without it those tests skip instead of failing. They deliberately read only a
*slice* of the smallest available world -- a full pass over every world costs
minutes and proves nothing extra about the format.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Callable

#: Rows of real data the tests look at. Enough to cross-check the format, small
#: enough that the whole suite stays interactive.
SAMPLE_ROWS = 256

#: The world that ships both `.vmp` and `.vmc`, so the compressed decoder can be
#: checked against the uncompressed surface. It is the big one, which is exactly
#: why every test reads a *window* of it rather than the whole map.
BOTH_FORMATS_WORLD = "fostral"

#: The small uncompressed world (2048 rows, 8 MB). Cheap enough to read whole.
SMALL_VMP_WORLD = "satadi"


def _find_repo() -> str | None:
    """Locates a Vangers checkout, or None."""
    env = os.environ.get("VANGERS_REPO")
    if env and os.path.isdir(env):
        return env
    # a checkout sitting next to this repo is a good guess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (
        os.path.join(os.path.dirname(here), "Vangers"),
        os.path.join(here, "..", "Vangers"),
    ):
        candidate = os.path.abspath(candidate)
        if os.path.isdir(os.path.join(candidate, "vangers", "bin")):
            return candidate
    return None


VANGERS_REPO = _find_repo()

WORLD_DIRS = [
    r"vangers\bin\thechain",
    r"vangers\bin\surmap\thechain",
]


class Skipped(Exception):
    """Raised by `skip()` to end a test as skipped."""


def skip(reason: str) -> None:
    raise Skipped(reason)


def have_worlds() -> bool:
    return VANGERS_REPO is not None


def world_dir(name: str) -> str | None:
    """Directory of the named world, e.g. ``world_dir("fostral")``."""
    if VANGERS_REPO is None:
        return None
    for sub in WORLD_DIRS:
        candidate = os.path.join(VANGERS_REPO, sub, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def world_file(name: str, world: str | None = None) -> str | None:
    """Absolute path of ``name`` (``output.vmp``, ``world.ini``, ...).

    With no ``world`` given, the smallest world that has the file is returned:
    the tests only need a sample, and the smallest one is the quick one.
    """
    if VANGERS_REPO is None:
        return None
    hits: list[tuple[int, str]] = []
    for sub in WORLD_DIRS:
        root = os.path.join(VANGERS_REPO, sub)
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            if world is not None and entry != world:
                continue
            candidate = os.path.join(root, entry, name)
            if os.path.isfile(candidate):
                hits.append((os.path.getsize(candidate), candidate))
    if not hits:
        return None
    return min(hits)[1]


def all_world_files(name: str) -> list[str]:
    """Every ``name`` in every world dir (only for the exhaustive sweeps)."""
    if VANGERS_REPO is None:
        return []
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


def sample_rows_file(name: str, world: str | None = None) -> str:
    """A ``name`` file of a world we can actually test against."""
    path = world_file(name, world)
    if path is None:
        skip(f"no {name} in the Vangers checkout (set VANGERS_REPO)")
    return path


def small_vmp() -> str:
    """The small uncompressed world: 2048 rows, cheap enough to read whole."""
    return sample_rows_file("output.vmp", SMALL_VMP_WORLD)


def vmc_world() -> str:
    """A world with both formats, so `.vmc` can be compared against `.vmp`."""
    directory = world_dir(BOTH_FORMATS_WORLD)
    if directory is not None:
        if os.path.isfile(os.path.join(directory, "output.vmc")) and \
           os.path.isfile(os.path.join(directory, "output.vmp")):
            return os.path.join(directory, "output.vmc")
    # fall back to any world that happens to ship both
    for path in all_world_files("output.vmc"):
        if os.path.isfile(os.path.join(os.path.dirname(path), "output.vmp")):
            return path
    return sample_rows_file("output.vmc")


def sample_config(path: str):
    """`WorldConfig` of the world that owns ``path``."""
    from core.config import WorldConfig

    ini = os.path.join(os.path.dirname(path), "world.ini")
    with open(ini, "r", encoding="utf-8", errors="replace") as fh:
        return WorldConfig.parse(fh.read())


def read_vmc_rows(path: str, rows: int = SAMPLE_ROWS):
    """Decodes the first ``rows`` rows of a `.vmc` into `(height, meta)`."""
    from core.vmc import load_vmc

    cfg = sample_config(path)
    rows = min(rows, cfg.size_y)
    with open(path, "rb") as fh:
        data = fh.read()
    return load_vmc(data, cfg.size_y, rows=range(0, rows))


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
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    slow: list[tuple[str, float]] = []
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
        except Skipped as exc:
            skipped.append(f"{name} ({exc})")
            print(f"skip  {name}")
        except Exception:  # noqa: BLE001 - report everything
            failed.append((name, traceback.format_exc()))
            print(f"FAIL  {name}")
        else:
            elapsed = time.perf_counter() - t0
            passed += 1
            print(f"ok    {name}  ({elapsed:.2f}s)")
            if elapsed > 2.0:
                slow.append((name, elapsed))
        total = time.perf_counter() - t0
        if total > 5.0:
            print(f"      ^ {total:.1f}s -- this test is doing too much")

    print(f"\n{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
    for name, tb in failed:
        print(f"\n=== {name} ===\n{tb}")
    return 1 if failed else 0
