# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Observation and action space definitions, and the dimensions they share with winbolo_gym.h."""

from __future__ import annotations

import gymnasium
import numpy as np

# V3 observation dimensions (must match winbolo_gym.h)
WBGYM_SPATIAL_SIZE = 29
WBGYM_NUM_SCALARS = 26
WBGYM_MAX_EVENTS = 16
WBGYM_MAX_PILLBOXES = 16
WBGYM_MAX_BASES = 16
WBGYM_MAX_ENTITIES = 256
WBGYM_MAX_SOUNDS = 32
WBGYM_NUM_REWARD_COMPONENTS = 59

# Entity feature count for reward extraction (rx, ry, type, allegiance, dir, speed, strength, flags, id)
REWARD_ENTITY_FEATURES = 9

# Base feature count for reward extraction (tx, ty, owner, shells, mines, armour)
REWARD_BASE_FEATURES = 6

# Pill feature count for reward extraction (tx, ty, owner, armor)
REWARD_PILL_FEATURES = 4

# Terrain layers
NUM_TERRAIN_LAYERS = 2

# Entity feature count (rx, ry, type, allegiance, direction, speed, strength)
ENTITY_FEATURES = 7

# Sound feature count (rx, ry, type, allegiance)
SOUND_FEATURES = 4

# Action space dimensions: [accel(3), turn(3), shoot(2), mine(2), gun_range(3), build_action(6), build_rx(29), build_ry(29)]
ACTION_DIMS = [3, 3, 2, 2, 3, 6, 29, 29]
TOTAL_LOGITS = sum(ACTION_DIMS)  # 77


def get_observation_space() -> gymnasium.spaces.Dict:
    """Return the Dict observation space for the V3 WinBolo environment."""
    return gymnasium.spaces.Dict({
        "terrain": gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(WBGYM_SPATIAL_SIZE, WBGYM_SPATIAL_SIZE, NUM_TERRAIN_LAYERS),
            dtype=np.float32,
        ),
        "scalar": gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(WBGYM_NUM_SCALARS,),
            dtype=np.float32,
        ),
        "entities": gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(WBGYM_MAX_ENTITIES, ENTITY_FEATURES),
            dtype=np.float32,
        ),
        "entity_mask": gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(WBGYM_MAX_ENTITIES,),
            dtype=np.float32,
        ),
        "sounds": gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(WBGYM_MAX_SOUNDS, SOUND_FEATURES),
            dtype=np.float32,
        ),
        "sound_mask": gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(WBGYM_MAX_SOUNDS,),
            dtype=np.float32,
        ),
    })


def get_action_space() -> gymnasium.spaces.MultiDiscrete:
    """Return the MultiDiscrete action space for V3.

    [accel(3), turn(3), shoot(2), mine(2), gun_range(3), build_action(6), build_rx(29), build_ry(29)]
    """
    return gymnasium.spaces.MultiDiscrete(ACTION_DIMS)
