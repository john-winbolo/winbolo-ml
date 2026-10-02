# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the reward weights: their order against the C library, and the presets."""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from winbolo_ml.features import WBGYM_NUM_REWARD_COMPONENTS
from winbolo_ml.rewards import (
    RewardConfig,
    blend_configs,
    phase1_movement,
    phase2_combat,
    phase3_resources,
    phase4_builder,
    phase5_tactical,
    phase6_mining,
    phase7_strategic,
    phase8_full,
    _zero_config,
)

HEADER = Path(__file__).resolve().parent.parent / "winbolo_gym.h"

# Where RewardConfig spells a component differently from its RC_* name.
_RC_NAME_TO_FIELD = {"pill_placement_qual": "pill_placement_quality"}


class TestMatchesLibrary:
    """The weights go to C by position, so a field out of order sends its
    weight to another component without any error."""

    def test_field_order_matches_rc_indices(self):
        rc = re.findall(r"#define RC_(\w+)\s+(\d+)", HEADER.read_text())
        by_index = sorted((int(i), name.lower()) for name, i in rc)
        assert [i for i, _ in by_index] == list(range(len(by_index)))
        expected = [_RC_NAME_TO_FIELD.get(name, name) for _, name in by_index]
        assert [f.name for f in fields(RewardConfig)] == expected

    def test_count_matches_header(self):
        count = int(re.search(r"#define WBGYM_NUM_REWARD_COMPONENTS\s+(\d+)",
                              HEADER.read_text()).group(1))
        assert len(fields(RewardConfig)) == count == WBGYM_NUM_REWARD_COMPONENTS


class TestPhasePresets:
    def test_zero_config(self):
        cfg = _zero_config()
        for f in fields(cfg):
            assert getattr(cfg, f.name) == 0.0

    def test_phase1_only_expected(self):
        cfg = phase1_movement()
        assert cfg.death == -1.0
        assert cfg.survival_tick == 0.005
        assert cfg.idle_penalty == -0.005
        assert cfg.own_mine_hit == -0.1
        # Everything else should be 0
        assert cfg.hit_dealt == 0.0
        assert cfg.kill == 0.0
        assert cfg.pill_captured == 0.0

    def test_phase2_inherits_phase1(self):
        cfg = phase2_combat()
        assert cfg.death == -1.0  # from phase 1
        assert cfg.hit_dealt == 0.05  # new in phase 2
        assert cfg.survival_tick == 0.001  # changed from phase 1

    def test_phase_inheritance_chain(self):
        """Each phase should have all rewards from previous phases."""
        p3 = phase3_resources()
        assert p3.death == -1.0  # phase 1
        assert p3.hit_dealt == 0.05  # phase 2
        assert p3.base_captured == 0.5  # phase 3

        p4 = phase4_builder()
        assert p4.lgm_lost == -0.5  # phase 4
        assert p4.base_captured == 0.5  # phase 3

        p5 = phase5_tactical()
        assert p5.pill_destroyed == 0.1  # phase 5
        assert p5.lgm_lost == -0.5  # phase 4

        p6 = phase6_mining()
        assert p6.mine_placed == 0.005  # phase 6
        assert p6.pill_destroyed == 0.1  # phase 5

        p7 = phase7_strategic()
        assert p7.offensive_pressure == 0.02  # phase 7
        assert p7.mine_placed == 0.005  # phase 6

    def test_phase8_equals_defaults(self):
        p8 = phase8_full()
        default = RewardConfig()
        for f in fields(p8):
            assert getattr(p8, f.name) == getattr(default, f.name)

    def test_blend_configs(self):
        a = _zero_config()
        b = RewardConfig()
        b.death = -2.0
        b.kill = 1.0
        blended = blend_configs(a, b, 0.5)
        assert blended.death == pytest.approx(-1.0)
        assert blended.kill == pytest.approx(0.5)

    def test_blend_endpoints(self):
        a = phase1_movement()
        b = phase2_combat()
        assert blend_configs(a, b, 0.0).death == a.death
        assert blend_configs(a, b, 1.0).hit_dealt == b.hit_dealt

    def test_phase2_penalises_ally_kills(self):
        cfg = phase2_combat()
        assert cfg.ally_killed == -0.2
        assert phase1_movement().ally_killed == 0.0
