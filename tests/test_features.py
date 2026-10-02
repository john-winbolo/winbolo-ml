# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for feature extraction."""

from winbolo_ml.features import (
    ACTION_DIMS,
    TOTAL_LOGITS,
    WBGYM_MAX_ENTITIES,
    WBGYM_MAX_SOUNDS,
    WBGYM_NUM_SCALARS,
    WBGYM_SPATIAL_SIZE,
    get_action_space,
    get_observation_space,
)


class TestSpaces:
    def test_observation_space_v3(self):
        space = get_observation_space()
        assert "terrain" in space.spaces
        assert "scalar" in space.spaces
        assert "entities" in space.spaces
        assert "entity_mask" in space.spaces
        assert "sounds" in space.spaces
        assert "sound_mask" in space.spaces
        assert space["terrain"].shape == (WBGYM_SPATIAL_SIZE, WBGYM_SPATIAL_SIZE, 2)
        assert space["scalar"].shape == (WBGYM_NUM_SCALARS,)
        assert space["entities"].shape == (WBGYM_MAX_ENTITIES, 7)
        assert space["entity_mask"].shape == (WBGYM_MAX_ENTITIES,)
        assert space["sounds"].shape == (WBGYM_MAX_SOUNDS, 4)
        assert space["sound_mask"].shape == (WBGYM_MAX_SOUNDS,)

    def test_action_space_v3(self):
        space = get_action_space()
        assert list(space.nvec) == ACTION_DIMS
        assert sum(ACTION_DIMS) == TOTAL_LOGITS == 77

