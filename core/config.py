"""`world.ini` handling — `analyzeINI` from src/terra/vmap.cpp.

Port of `parseWorldConfig` / `WorldConfig` in the reference TypeScript viewer,
plus a writer so the addon can create worlds from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .constants import MAP_POWER_X, STORAGE_VERSION, TERRAIN_MAX
from .ini import Ini, format_int_list, parse_int_list

# Section / key names as they appear in world.ini.
SEC_GLOBAL = "Global Parameters"
SEC_STORAGE = "Storage"
SEC_RENDER = "Rendering Parameters"
SEC_DYNAMIC = "Dynamic Palette"
SEC_CREATION = "Creation Parameters"

# Defaults used by the retail worlds (and by SURMAP for fresh files).
DEFAULT_BEGIN_COLORS = [1, 32, 64, 72, 88, 104, 112, 120]
DEFAULT_END_COLORS = [31, 63, 71, 87, 103, 111, 119, 127]
DEFAULT_SHADOW_OFFSETS = [10, 12, 4, 9, 6, 4, 4, 4]
DEFAULT_HEIGHT_SHIFTS = [2, 4, 8, 8, 5, 6, 6, 6]

SECTION_GLOBAL = SEC_GLOBAL
SECTION_STORAGE = SEC_STORAGE
SECTION_RENDERING = SEC_RENDER
SECTION_DYNAMIC = SEC_DYNAMIC
SECTION_CREATION = SEC_CREATION


class WorldConfigError(ValueError):
    """Raised when world.ini does not describe a surface we can load."""


@dataclass
class PaletteCycle:
    terrain: int = 0
    speed: int = 0
    ampl: int = 0
    red: int = 0
    green: int = 0
    blue: int = 0


@dataclass
class DynamicPalette:
    wave_terrain: int = -1
    cycles: list[PaletteCycle] = field(default_factory=list)


@dataclass
class WorldConfig:
    map_power_x: int = MAP_POWER_X
    map_power_y: int = 14
    geo_power: int = 5
    section_power: int = 7
    min_square_power: int = 4
    terrain_max: int = TERRAIN_MAX
    is_compressed: bool = True
    file_name: str = "output"
    palette_file: str = "harmony.pal"
    begin_colors: list[int] = field(default_factory=lambda: list(DEFAULT_BEGIN_COLORS))
    end_colors: list[int] = field(default_factory=lambda: list(DEFAULT_END_COLORS))
    shadow_offsets: list[int] = field(default_factory=lambda: list(DEFAULT_SHADOW_OFFSETS))
    height_shifts: list[int] = field(default_factory=lambda: list(DEFAULT_HEIGHT_SHIFTS))
    dynamic_palette: DynamicPalette = field(default_factory=DynamicPalette)
    default_terrain: int = 1
    version: str = STORAGE_VERSION

    @property
    def size_y(self) -> int:
        return 1 << self.map_power_y

    @property
    def size_x(self) -> int:
        return 1 << self.map_power_x

    # -- reading ------------------------------------------------------------
    @classmethod
    def parse(cls, ini_text: str) -> "WorldConfig":
        ini = Ini.parse(ini_text)

        map_power_x = ini.get_int(SEC_GLOBAL, "Map Power X", 0)
        if map_power_x != MAP_POWER_X:
            raise WorldConfigError(
                f"Incorrect X-Size (Map Power X = {map_power_x}, expected {MAP_POWER_X})"
            )
        version = ini.get(SEC_STORAGE, "Version", "") or ""
        if version != STORAGE_VERSION:
            raise WorldConfigError(
                f'Incorrect Storage Version "{version}", expected {STORAGE_VERSION}'
            )
        terrain_max = ini.get_int(SEC_RENDER, "Terrain Max", 0)
        if terrain_max and terrain_max != TERRAIN_MAX:
            raise WorldConfigError(f"Incorrect Terrain Max {terrain_max}")

        cycles: list[PaletteCycle] = []
        count = ini.get_int(SEC_DYNAMIC, "Terrain Number", 0)
        if count > 0:
            terrains = ini.get_int_list(SEC_DYNAMIC, "Terrains", count)
            speeds = ini.get_int_list(SEC_DYNAMIC, "Speeds", count)
            ampls = ini.get_int_list(SEC_DYNAMIC, "Amplitudes", count)
            reds = ini.get_int_list(SEC_DYNAMIC, "Red", count)
            greens = ini.get_int_list(SEC_DYNAMIC, "Green", count)
            blues = ini.get_int_list(SEC_DYNAMIC, "Blue", count)
            cycles = [
                PaletteCycle(terrains[i], speeds[i], ampls[i], reds[i], greens[i], blues[i])
                for i in range(count)
            ]

        begin = parse_int_list(ini.get(SEC_RENDER, "Begin Colors"), TERRAIN_MAX, 0)
        end = parse_int_list(ini.get(SEC_RENDER, "End Colors"), TERRAIN_MAX, 0)
        if not any(begin):
            begin = list(DEFAULT_BEGIN_COLORS)
        if not any(end):
            end = list(DEFAULT_END_COLORS)

        shadows = parse_int_list(ini.get(SEC_RENDER, "Shadow Offsets"), TERRAIN_MAX, 0)
        shifts = parse_int_list(ini.get(SEC_RENDER, "Height Shifts"), TERRAIN_MAX, 0)
        if not any(shadows):
            shadows = list(DEFAULT_SHADOW_OFFSETS)
        if not any(shifts):
            shifts = list(DEFAULT_HEIGHT_SHIFTS)

        return cls(
            map_power_x=map_power_x,
            map_power_y=ini.get_int(SEC_GLOBAL, "Map Power Y", 14),
            geo_power=ini.get_int(SEC_GLOBAL, "GeoNet Power", 5),
            section_power=ini.get_int(SEC_GLOBAL, "Section Size Power", 7),
            min_square_power=ini.get_int(SEC_GLOBAL, "Minimal Square Power", 4),
            terrain_max=terrain_max or TERRAIN_MAX,
            is_compressed=ini.get_int(SEC_STORAGE, "Compressed Format Using", 0) != 0,
            file_name=ini.get(SEC_STORAGE, "File Name", "") or "",
            palette_file=ini.get(SEC_STORAGE, "Palette File", "") or "",
            begin_colors=begin,
            end_colors=end,
            shadow_offsets=shadows,
            height_shifts=shifts,
            dynamic_palette=DynamicPalette(
                wave_terrain=ini.get_int(SEC_DYNAMIC, "Wave Terrain", -1), cycles=cycles
            ),
            default_terrain=ini.get_int(SEC_CREATION, "Default Terrain Type", 1),
            version=version,
        )

    # -- writing ------------------------------------------------------------
    def to_ini(self) -> Ini:
        ini = Ini()
        ini.set(SEC_GLOBAL, "Map Power X", self.map_power_x)
        ini.set(SEC_GLOBAL, "Map Power Y", self.map_power_y)
        ini.set(SEC_GLOBAL, "GeoNet Power", self.geo_power)
        ini.set(SEC_GLOBAL, "Section Size Power", self.section_power)
        ini.set(SEC_GLOBAL, "Minimal Square Power", self.min_square_power)

        ini.set(SEC_STORAGE, "Version", self.version)
        ini.set(SEC_STORAGE, "Compressed Format Using", 1 if self.is_compressed else 0)
        ini.set(SEC_STORAGE, "File Name", self.file_name)
        ini.set(SEC_STORAGE, "Palette File", self.palette_file)

        ini.set(SEC_RENDER, "Shadow Offsets", format_int_list(self.shadow_offsets))
        ini.set(SEC_RENDER, "Height Shifts", format_int_list(self.height_shifts))
        ini.set(SEC_RENDER, "Begin Colors", format_int_list(self.begin_colors))
        ini.set(SEC_RENDER, "End Colors", format_int_list(self.end_colors))

        cycles = self.dynamic_palette.cycles
        ini.set(SEC_DYNAMIC, "Wave Terrain", self.dynamic_palette.wave_terrain)
        ini.set(SEC_DYNAMIC, "Terrain Number", len(cycles))
        if cycles:
            ini.set(SEC_DYNAMIC, "Terrains", format_int_list(c.terrain for c in cycles))
            ini.set(SEC_DYNAMIC, "Speeds", format_int_list(c.speed for c in cycles))
            ini.set(SEC_DYNAMIC, "Amplitudes", format_int_list(c.ampl for c in cycles))
            ini.set(SEC_DYNAMIC, "Red", format_int_list(c.red for c in cycles))
            ini.set(SEC_DYNAMIC, "Green", format_int_list(c.green for c in cycles))
            ini.set(SEC_DYNAMIC, "Blue", format_int_list(c.blue for c in cycles))

        ini.set(SEC_CREATION, "Mesh Value", 256)
        ini.set(SEC_CREATION, "Noise Amplitude", 0)
        ini.set(SEC_CREATION, "Default Terrain Type", self.default_terrain)
        return ini

    def dumps(self) -> str:
        return self.to_ini().dumps()


def new_world_config(size_y_log2: int, *, file_name: str = "output",
                     palette_file: str = "harmony.pal",
                     compressed: bool = True) -> WorldConfig:
    """A minimal, valid configuration for a world created from scratch."""
    return WorldConfig(
        map_power_y=size_y_log2,
        file_name=file_name,
        palette_file=palette_file,
        is_compressed=compressed,
    )
