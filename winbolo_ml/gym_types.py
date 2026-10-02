# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""ctypes mirror of winbolo_gym.h: game types and the action and observation structs."""

from __future__ import annotations

import ctypes

from winbolo_ml.features import (
    WBGYM_MAX_BASES,
    WBGYM_MAX_ENTITIES,
    WBGYM_MAX_EVENTS,
    WBGYM_MAX_PILLBOXES,
    WBGYM_MAX_SOUNDS,
    WBGYM_NUM_REWARD_COMPONENTS,
    WBGYM_NUM_SCALARS,
    WBGYM_SPATIAL_SIZE,
)

GAME_TYPE_OPEN = 1
GAME_TYPE_TOURNAMENT = 2
GAME_TYPE_STRICT_TOURNAMENT = 3


# --- V3 ctypes struct definitions (must match winbolo_gym.h) ---

class WinBoloEntity(ctypes.Structure):
    _fields_ = [
        ("rx",         ctypes.c_float),
        ("ry",         ctypes.c_float),
        ("type",       ctypes.c_uint8),
        ("allegiance", ctypes.c_int8),
        # 2 bytes padding (alignment to next float)
        ("direction",  ctypes.c_float),
        ("speed",      ctypes.c_float),
        ("strength",   ctypes.c_float),
        ("flags",      ctypes.c_uint8),
        ("id",         ctypes.c_uint8),
        # 2 bytes padding at end (struct alignment to 4)
    ]


class WinBoloSoundEvent(ctypes.Structure):
    _fields_ = [
        ("rx",         ctypes.c_float),
        ("ry",         ctypes.c_float),
        ("type",       ctypes.c_uint8),
        ("allegiance", ctypes.c_int8),
        ("sound_id",   ctypes.c_uint8),
        # 1 byte padding at end
    ]


class WinBoloMsgObs(ctypes.Structure):
    _fields_ = [
        ("from_player", ctypes.c_uint8),
        ("type",        ctypes.c_uint8),
        # 2 bytes padding before float array
        ("data",        ctypes.c_float * 4),
    ]


class WinBoloAllianceObs(ctypes.Structure):
    _fields_ = [
        ("alliance_id",      ctypes.c_uint8),
        ("alliance_members", ctypes.c_uint8 * 16),
        ("alliance_count",   ctypes.c_uint8),
    ]


class WinBoloPillObs(ctypes.Structure):
    _fields_ = [
        ("tx",    ctypes.c_uint8),
        ("ty",    ctypes.c_uint8),
        ("owner", ctypes.c_uint8),
        ("armor", ctypes.c_uint8),
    ]


class WinBoloBaseObs(ctypes.Structure):
    _fields_ = [
        ("tx",     ctypes.c_uint8),
        ("ty",     ctypes.c_uint8),
        ("owner",  ctypes.c_uint8),
        ("shells", ctypes.c_uint8),
        ("mines",  ctypes.c_uint8),
        ("armour", ctypes.c_uint8),
    ]


class WinBoloAction(ctypes.Structure):
    _fields_ = [
        ("accel",            ctypes.c_int),
        ("turn",             ctypes.c_int),
        ("shoot",            ctypes.c_int),
        ("lay_mine",         ctypes.c_int),
        ("gun_range_adjust", ctypes.c_int),
        ("build_action",     ctypes.c_int),
        ("build_rx",         ctypes.c_int),
        ("build_ry",         ctypes.c_int),
    ]


class WinBoloObs(ctypes.Structure):
    _fields_ = [
        # Terrain (2 layers)
        ("terrain",       (ctypes.c_float * WBGYM_SPATIAL_SIZE) * WBGYM_SPATIAL_SIZE),
        ("mines_map",     (ctypes.c_float * WBGYM_SPATIAL_SIZE) * WBGYM_SPATIAL_SIZE),

        # Entity list
        ("entities",      WinBoloEntity * WBGYM_MAX_ENTITIES),
        ("num_entities",  ctypes.c_uint16),

        # Scalars
        ("scalar",        ctypes.c_float * WBGYM_NUM_SCALARS),

        # LGM state
        ("man_rx",        ctypes.c_float),
        ("man_ry",        ctypes.c_float),
        ("man_direction", ctypes.c_float),

        # Sound events
        ("sounds",        WinBoloSoundEvent * WBGYM_MAX_SOUNDS),
        ("num_sounds",    ctypes.c_uint8),

        # Discrete game events
        ("events",        ctypes.c_uint8 * WBGYM_MAX_EVENTS),
        ("num_events",    ctypes.c_uint8),

        # Assistant message
        ("assistant_msg", ctypes.c_uint8),

        # Alliance (placeholder)
        ("alliance",      WinBoloAllianceObs),

        # Messages (placeholder)
        ("messages",      WinBoloMsgObs * 4),
        ("num_messages",  ctypes.c_uint8),

        # Metadata
        ("dead",          ctypes.c_uint8),
        ("game_over",     ctypes.c_uint8),
        ("game_won",      ctypes.c_uint8),
        ("tick",          ctypes.c_uint32),
        ("tank_x",        ctypes.c_float),
        ("tank_y",        ctypes.c_float),

        # Pill/base lists
        ("num_pillboxes", ctypes.c_uint8),
        ("pillboxes",     WinBoloPillObs * WBGYM_MAX_PILLBOXES),
        ("num_bases",     ctypes.c_uint8),
        ("bases",         WinBoloBaseObs * WBGYM_MAX_BASES),

        # Reward output (computed in C when weights are set)
        ("reward",            ctypes.c_float),
        ("reward_components", ctypes.c_float * WBGYM_NUM_REWARD_COMPONENTS),
    ]


class WinBoloRewardBatchOut(ctypes.Structure):
    """Mirrors the C WinBoloRewardBatchOut struct — holds output pointers for reward extraction."""
    _fields_ = [
        ("scalars",       ctypes.POINTER(ctypes.c_float)),
        ("events",        ctypes.POINTER(ctypes.c_uint8)),
        ("event_count",   ctypes.POINTER(ctypes.c_int32)),
        ("pills",         ctypes.POINTER(ctypes.c_float)),
        ("pill_count",    ctypes.POINTER(ctypes.c_int32)),
        ("bases",         ctypes.POINTER(ctypes.c_float)),
        ("base_count",    ctypes.POINTER(ctypes.c_int32)),
        ("entities",      ctypes.POINTER(ctypes.c_float)),
        ("entity_count",  ctypes.POINTER(ctypes.c_int32)),
        ("sounds",        ctypes.POINTER(ctypes.c_float)),
        ("sound_count",   ctypes.POINTER(ctypes.c_int32)),
        ("terrain",       ctypes.POINTER(ctypes.c_float)),
        ("dead",          ctypes.POINTER(ctypes.c_uint8)),
        ("game_over",     ctypes.POINTER(ctypes.c_uint8)),
        ("game_won",      ctypes.POINTER(ctypes.c_uint8)),
        ("tick",          ctypes.POINTER(ctypes.c_int32)),
        ("tank_x",        ctypes.POINTER(ctypes.c_float)),
        ("tank_y",        ctypes.POINTER(ctypes.c_float)),
        ("assistant_msg", ctypes.POINTER(ctypes.c_uint8)),
    ]


def setup_batch_ctypes(lib) -> None:
    """Register ctypes signatures for the batch C functions on *lib*.

    Call this once after loading the shared library (in addition to the
    existing single-env ctypes setup).
    """
    lib.winbolo_fill_actions_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int32),   # actions_flat [N,8]
        ctypes.POINTER(WinBoloAction),     # actions_out
        ctypes.c_int,                      # count
    ]
    lib.winbolo_fill_actions_batch.restype = None

    lib.winbolo_obs_to_numpy_batch.argtypes = [
        ctypes.POINTER(WinBoloObs),        # obs_array
        ctypes.POINTER(ctypes.c_float),    # terrain_out
        ctypes.POINTER(ctypes.c_float),    # scalar_out
        ctypes.POINTER(ctypes.c_float),    # entities_out
        ctypes.POINTER(ctypes.c_float),    # entity_mask_out
        ctypes.POINTER(ctypes.c_float),    # sounds_out
        ctypes.POINTER(ctypes.c_float),    # sound_mask_out
        ctypes.c_int,                      # count
    ]
    lib.winbolo_obs_to_numpy_batch.restype = None

    lib.winbolo_obs_to_reward_batch.argtypes = [
        ctypes.POINTER(WinBoloObs),             # obs_array
        ctypes.POINTER(WinBoloRewardBatchOut),   # out
        ctypes.c_int,                            # count
    ]
    lib.winbolo_obs_to_reward_batch.restype = None
