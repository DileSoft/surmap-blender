"""Constants ported 1:1 from the Vangers sources.

    src/common.h, src/terra/world.h, src/terra/render.h

Only the non-TERRAIN16 configuration is reproduced, which is what the retail
game ships.
"""

# --- src/common.h ----------------------------------------------------------
MAP_POWER_X = 11
H_SIZE = 1 << MAP_POWER_X  # 2048, the map is always this wide
H2_SIZE = 2 * H_SIZE

# --- src/terra/world.h (TERRAIN16 is not defined) --------------------------
TERRAIN_MAX = 8
TERRAIN_OFFSET = 3
TERRAIN_MASK = 0x38  # bits 5..3

# flag byte layout: [7 SHADOW][6 DOUBLE_LEVEL][5..3 TERRAIN][2 OBJSHADOW][1..0 DELTA]
SHADOW_MASK = 0x80
DOUBLE_LEVEL = 0x40
OBJSHADOW = 0x04
DELTA_MASK = 0x03
DELTA_SHIFT = 3

WATER_TERRAIN_INDEX = 0
MAIN_TERRAIN_INDEX = 1
WATER_TERRAIN = WATER_TERRAIN_INDEX << TERRAIN_OFFSET
MAIN_TERRAIN = MAIN_TERRAIN_INDEX << TERRAIN_OFFSET

# --- src/terra/render.h ----------------------------------------------------
H_CORRECTION = 1
CLR_MAX_SIDE = 255
CLR_MAX = 2 * CLR_MAX_SIDE + 1  # 511

# --- double-level delta encoding ------------------------------------------
# src/terra/world.h + src/iscreen/iworld.h
MIN_RDELTA = 1 << DELTA_SHIFT  # 8
MAX_RDELTA = 16 << DELTA_SHIFT  # 128
# src/terra/land.cpp
TUNNEL_PROOF = 16

# --- src/terra/siderend.cpp ------------------------------------------------
SS_WIDTH = 16
SHADOWDEEP = 384
SHADOWHEIGHT = 32
POSPOWER = 8
MAX_ALT = 255

# --- land.cpp material table (TERRAIN16 is not defined) ---------------------
MATERIAL_MAX = 2
TERRAIN_MATERIAL = [1, 0, 0, 0, 0, 0, 0, 0]
TERRAIN_DXKOEF = [1.0, 5.0]
TERRAIN_SDKOEF = [1.0, 1.25]
TERRAIN_JJKOEF = [1.0, 0.5]

# --- flag/limit shorthands -------------------------------------------------
WATER_LEVEL = 0
KEEP_TERRAIN = -1  # `CurrentTerrain` value meaning "leave the material alone"

# Storage version the loader insists on (src/terra/vmap.cpp analyzeINI).
STORAGE_VERSION = "1.4"
