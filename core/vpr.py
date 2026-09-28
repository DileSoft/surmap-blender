"""`.vpr` flood-level table — `LoadVPR` from src/terra/vmap.cpp.

The file is *not* a flat per-row table. It is a structured container:

    nets -> sections -> parts, plus per-part geometry tables

and only after all of that comes the flood level per part, as little-endian
``uint32`` values whose low byte is the level. ``flood[i]`` is the water level
for the rows of part ``i`` (rows are grouped by ``Section Size Power``).

The renderers only consult the *first* level (`FloodLEVEL`) for the water ramp,
but the game reads the whole table, so a world we create must keep it intact —
see :func:`write_levels`, which patches a template in place.
"""

from __future__ import annotations

import struct

from .constants import H_SIZE


class VprError(ValueError):
    pass


def _layout(size_y: int, geo_power: int, section_power: int) -> tuple[int, int]:
    part_max = size_y >> section_power
    net_size = (H_SIZE * size_y) >> (2 * geo_power)
    offset = (
        2 * 4
        + (1 + 4 + 4) * 4
        + 2 * net_size
        + 2 * geo_power * 4
        + 2 * part_max * geo_power * 4
    )
    return offset, part_max


def flood_count(config) -> int:
    return config.size_y >> config.section_power


def load_levels(data: bytes | bytearray | memoryview, config) -> list[int]:
    """Decodes the per-part water levels."""
    size_y = config.size_y
    offset, part_max = _layout(size_y, config.geo_power, config.section_power)
    if len(data) < offset + part_max * 4:
        raise VprError(f"VPR too small: {len(data)} < {offset + part_max * 4}")
    out = []
    for i in range(part_max):
        (raw,) = struct.unpack_from("<I", data, offset + i * 4)
        out.append(raw & 0xFF)
    return out


def level_for_row(levels: list[int], row: int, section_power: int) -> int:
    """The water level that applies to ``row``."""
    return levels[min(row >> section_power, len(levels) - 1)]


def write_levels(data: bytes, levels: list[int], config) -> bytes:
    """Patches the level table of a template file, keeping the header intact."""
    size_y = config.size_y
    offset, part_max = _layout(size_y, config.geo_power, config.section_power)
    if len(data) < offset + part_max * 4:
        raise VprError("VPR template too small")
    if len(levels) != part_max:
        raise VprError(f"expected {part_max} levels, got {len(levels)}")
    buf = bytearray(data)
    for i, v in enumerate(levels):
        struct.pack_into("<I", buf, offset + i * 4, v & 0xFF)
    return bytes(buf)
