# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reward weights for WinBolo RL training.

Rewards are computed in C inside libwinbolo_gym.so. RewardConfig holds one
weight per reward component, in the same order as the RC_* indices in
winbolo_gym.h, and the batched env sends them to the library at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# --- Scalar indices (obs_batch['scalars'][:, i]) ---
S_ARMOR = 0
S_SHELLS = 1
S_MINES = 2
S_TREES = 3
S_SPEED = 4
S_DIR_SIN = 5
S_DIR_COS = 6
S_RELOAD = 7
S_IN_BOAT = 8
S_HAS_PILLS = 9
S_PILL_COUNT = 10
S_DEAD = 11
S_OWN_PILL_FRAC = 12
S_ENEMY_PILL_FRAC = 13
S_ALLY_PILL_FRAC = 14
S_OWN_BASE_FRAC = 15
S_ALLY_BASE_FRAC = 16
S_TANK_X = 17
S_TANK_Y = 18
S_GUN_RANGE = 19
S_HIDDEN = 20
S_NEW_TANK = 21
S_TANK_OBSTRUCTED = 22
S_LGM_STATUS = 23
S_LGM_OBSTRUCTED = 24
S_DEATH_WAIT = 25

# --- Owner constants (pills/bases) ---
OWNER_NEUTRAL = 0
OWNER_SELF = 1
OWNER_ENEMY = 2
OWNER_ALLY = 3


# ── RewardConfig ─────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """Weights for all reward components. Set weight to 0 to disable."""

    # Category 1: Survival
    death: float = -1.0
    death_with_pills: float = -0.5
    death_with_lgm_out: float = -0.3
    survival_tick: float = 0.001
    # Category 2: Combat
    hit_dealt: float = 0.05
    hit_received: float = -0.05
    kill: float = 0.2
    kill_carrier: float = 0.3
    shot_fired: float = -0.002
    shot_accuracy: float = 0.1
    # Category 3: Pillbox
    pill_captured: float = 0.3
    pill_lost: float = -0.3
    pill_destroyed: float = 0.1
    own_pill_frac_delta: float = 1.0
    pill_placement_quality: float = 0.15
    pill_heated_tactical: float = 0.02
    # Category 4: Base
    base_captured: float = 0.5
    base_lost: float = -0.5
    own_base_frac_delta: float = 2.0
    base_ratio: float = 0.1
    all_bases_owned: float = 10.0
    # Category 5: Builder
    lgm_lost: float = -0.5
    lgm_parachuting_tick: float = -0.005
    enemy_lgm_killed: float = 0.4
    successful_build: float = 0.05
    failed_build: float = -0.02
    lgm_sent_dangerous: float = -0.03
    # Category 6: Resources
    resupply_efficiency: float = 0.02
    resupply_camping: float = -0.01
    trees_farmed: float = 0.01
    ammo_conservation: float = 0.05
    idle_penalty: float = -0.003
    # Category 7: Mining
    mine_placed: float = 0.005
    mine_kill: float = 0.15
    mine_placed_defensive: float = 0.01
    mine_placed_on_road: float = 0.01
    own_mine_hit: float = -0.1
    # Category 8: Territory
    offensive_pressure: float = 0.02
    defensive_coverage: float = 0.02
    road_built: float = 0.01
    wall_built: float = 0.01
    flank_bonus: float = 0.1
    # Category 9: Strategic
    initiative_score: float = 0.05
    spike_quality: float = 0.2
    dont_carry_too_many: float = -0.05
    # Category 10: Multi-agent (disabled by default)
    decoy_assist: float = 0.0
    team_coordination: float = 0.0
    message_useful: float = 0.0
    # Category 11: Phase 1 shaping (disabled by default, enable for Phase 1 only)
    exploration_bonus: float = 0.0
    base_proximity: float = 0.0
    speed_bonus: float = 0.0
    boat_overstay: float = 0.0
    on_land_bonus: float = 0.0
    approach_pillbox: float = 0.0
    facing_pillbox: float = 0.0
    pill_hit: float = 0.0
    resupply_shells: float = 0.0
    shoot_at_pill: float = 0.0
    # Category 2 again, appended to match the C list (RC_ALLY_KILLED = 58)
    ally_killed: float = -0.2


# ── Phase presets ─────────────────────────────────────────────────────────────

def _zero_config() -> RewardConfig:
    """Create a config with all weights set to 0."""
    cfg = RewardConfig()
    for f in fields(cfg):
        setattr(cfg, f.name, 0.0)
    return cfg


def phase1_movement() -> RewardConfig:
    """Phase 1: Movement & survival only."""
    cfg = _zero_config()
    cfg.death = -1.0
    cfg.survival_tick = 0.005
    cfg.idle_penalty = -0.005
    cfg.own_mine_hit = -0.1
    # Phase 1 shaping rewards (remove in Phase 2+)
    cfg.exploration_bonus = 0.002
    cfg.base_proximity = 0.003
    cfg.speed_bonus = 0.001
    cfg.boat_overstay = -0.003
    return cfg


def phase2_combat() -> RewardConfig:
    """Phase 2: Combat fundamentals (inherits Phase 1)."""
    cfg = phase1_movement()
    cfg.hit_dealt = 0.05
    cfg.hit_received = -0.05
    cfg.kill = 0.2
    cfg.ally_killed = -0.2
    cfg.shot_fired = -0.002
    cfg.shot_accuracy = 0.1
    cfg.survival_tick = 0.001
    return cfg


def phase3_resources() -> RewardConfig:
    """Phase 3: Resource loop (inherits Phase 2)."""
    cfg = phase2_combat()
    cfg.base_captured = 0.5
    cfg.base_lost = -0.5
    cfg.resupply_efficiency = 0.02
    cfg.resupply_camping = -0.01
    cfg.trees_farmed = 0.01
    cfg.ammo_conservation = 0.05
    return cfg


def phase4_builder() -> RewardConfig:
    """Phase 4: Builder operations + pillbox placement (inherits Phase 3)."""
    cfg = phase3_resources()
    cfg.lgm_lost = -0.5
    cfg.lgm_parachuting_tick = -0.005
    cfg.successful_build = 0.05
    cfg.failed_build = -0.02
    cfg.lgm_sent_dangerous = -0.03
    cfg.pill_captured = 0.3
    cfg.pill_lost = -0.3
    cfg.pill_placement_quality = 0.15
    cfg.road_built = 0.01
    cfg.wall_built = 0.01
    return cfg


def phase5_tactical() -> RewardConfig:
    """Phase 5: Tactical combat / pill wars (inherits Phase 4)."""
    cfg = phase4_builder()
    cfg.pill_destroyed = 0.1
    cfg.own_pill_frac_delta = 1.0
    cfg.pill_heated_tactical = 0.02
    cfg.enemy_lgm_killed = 0.4
    cfg.own_base_frac_delta = 2.0
    return cfg


def phase6_mining() -> RewardConfig:
    """Phase 6: Mining & area denial (inherits Phase 5)."""
    cfg = phase5_tactical()
    cfg.mine_placed = 0.005
    cfg.mine_kill = 0.15
    cfg.mine_placed_defensive = 0.01
    cfg.mine_placed_on_road = 0.01
    return cfg


def phase7_strategic() -> RewardConfig:
    """Phase 7: Full strategic play (inherits Phase 6)."""
    cfg = phase6_mining()
    cfg.offensive_pressure = 0.02
    cfg.defensive_coverage = 0.02
    cfg.flank_bonus = 0.1
    cfg.initiative_score = 0.05
    cfg.spike_quality = 0.2
    cfg.dont_carry_too_many = -0.05
    cfg.base_ratio = 0.1
    cfg.all_bases_owned = 10.0
    cfg.death_with_pills = -0.5
    cfg.death_with_lgm_out = -0.3
    cfg.kill_carrier = 0.3
    return cfg


def phase8_full() -> RewardConfig:
    """Phase 8: All rewards at default weights."""
    return RewardConfig()


def blend_configs(
    cfg_a: RewardConfig, cfg_b: RewardConfig, alpha: float
) -> RewardConfig:
    """Blend two configs: result = (1 - alpha) * cfg_a + alpha * cfg_b."""
    blended = RewardConfig()
    for f in fields(RewardConfig):
        val_a = getattr(cfg_a, f.name)
        val_b = getattr(cfg_b, f.name)
        setattr(blended, f.name, (1.0 - alpha) * val_a + alpha * val_b)
    return blended
