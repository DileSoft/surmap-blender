"""Core, bpy-free surface model and file I/O for Vangers worlds.

Nothing in this package imports ``bpy``, so it can be unit-tested with any
Python that has numpy — including the interpreter bundled with Blender.
"""

from .config import DynamicPalette, PaletteCycle, WorldConfig, WorldConfigError, new_world_config
from .constants import (
    DELTA_MASK,
    DOUBLE_LEVEL,
    H_SIZE,
    KEEP_TERRAIN,
    MAX_RDELTA,
    MIN_RDELTA,
    OBJSHADOW,
    SHADOW_MASK,
    TERRAIN_MASK,
    TERRAIN_MAX,
    TERRAIN_OFFSET,
    TUNNEL_PROOF,
)
from .ini import Ini, parse_int_list
from .solver import SolveReport, solve
from .surface import (
    MIN_TUNNEL_GAP,
    PAIRS,
    Surface,
    delta_from_gap,
    gap_is_tunnel,
    is_double,
    thickness_from_delta,
    up_alt,
    up_terrain,
    down_alt,
    down_terrain,
)
from .vmp import VmpError, load_vmp, load_vmp_bytes, save_vmp, save_vmp_bytes, vmp_size_y

__all__ = [
    "DynamicPalette",
    "PaletteCycle",
    "WorldConfig",
    "WorldConfigError",
    "new_world_config",
    "Ini",
    "parse_int_list",
    "SolveReport",
    "solve",
    "Surface",
    "MIN_TUNNEL_GAP",
    "PAIRS",
    "is_double",
    "up_alt",
    "down_alt",
    "up_terrain",
    "down_terrain",
    "delta_from_gap",
    "gap_is_tunnel",
    "thickness_from_delta",
    "load_vmp",
    "load_vmp_bytes",
    "save_vmp",
    "save_vmp_bytes",
    "vmp_size_y",
    "VmpError",
    "H_SIZE",
    "TERRAIN_MAX",
    "TERRAIN_OFFSET",
    "TERRAIN_MASK",
    "DOUBLE_LEVEL",
    "SHADOW_MASK",
    "OBJSHADOW",
    "DELTA_MASK",
    "MIN_RDELTA",
    "MAX_RDELTA",
    "TUNNEL_PROOF",
    "KEEP_TERRAIN",
]
