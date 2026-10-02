# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batched vectorized environment using winbolo_step_batch for N games in one process."""

from __future__ import annotations

import ctypes
import logging
from dataclasses import fields
from typing import Any, Sequence

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from winbolo_ml.gym_types import (
    GAME_TYPE_STRICT_TOURNAMENT,
    WinBoloAction,
    WinBoloObs,
    WinBoloRewardBatchOut,
    setup_batch_ctypes,
)
from winbolo_ml.features import (
    ENTITY_FEATURES,
    REWARD_BASE_FEATURES,
    REWARD_ENTITY_FEATURES,
    REWARD_PILL_FEATURES,
    SOUND_FEATURES,
    WBGYM_MAX_BASES,
    WBGYM_MAX_ENTITIES,
    WBGYM_MAX_EVENTS,
    WBGYM_MAX_PILLBOXES,
    WBGYM_MAX_SOUNDS,
    WBGYM_NUM_REWARD_COMPONENTS,
    WBGYM_NUM_SCALARS,
    WBGYM_SPATIAL_SIZE,
    NUM_TERRAIN_LAYERS,
    get_action_space,
    get_observation_space,
)
from winbolo_ml.rewards import (
    OWNER_SELF,
    RewardConfig,
    S_IN_BOAT,
)

logger = logging.getLogger(__name__)


class WinBoloBatchEnv(VecEnv):
    """Vectorized env that steps N games with a single winbolo_step_batch C call.

    Rewards are computed in C from the RewardConfig weights sent to each game
    at startup. step_wait returns the per-component rewards as an [N, 59] array
    in RewardConfig field order where a VecEnv would return infos.
    """

    def __init__(
        self,
        lib_path: str,
        map_path: str,
        num_envs: int,
        max_ticks: int = 6000,
        game_type: int = GAME_TYPE_STRICT_TOURNAMENT,
        reward_config: RewardConfig | None = None,
        end_on_all_pillbox_captures: bool = False,
        end_on_land: bool = False,
    ) -> None:
        self._observation_space = get_observation_space()
        self._action_space = get_action_space()
        super().__init__(num_envs, self._observation_space, self._action_space)

        self.map_path = map_path
        self.game_type = game_type
        self.max_ticks = max_ticks
        self.end_on_all_pillbox_captures = end_on_all_pillbox_captures
        self.end_on_land = end_on_land

        # Load shared library
        self._lib = ctypes.CDLL(lib_path)
        self._setup_ctypes()
        self._verify_struct_sizes()

        # Create N game instances
        self._games = (ctypes.c_void_p * num_envs)()
        for i in range(num_envs):
            self._games[i] = self._lib.winbolo_create(
                map_path.encode("utf-8"), game_type
            )
            if not self._games[i]:
                raise RuntimeError(f"winbolo_create failed for game {i}")

        # Allocate batch arrays
        self._actions = (WinBoloAction * num_envs)()
        self._obs_buf = (WinBoloObs * num_envs)()

        # ── Pre-allocated numpy buffers for batch obs extraction ──
        N = num_envs
        # Model observation buffers (written by winbolo_obs_to_numpy_batch)
        self._buf_terrain     = np.zeros((N, WBGYM_SPATIAL_SIZE, WBGYM_SPATIAL_SIZE, NUM_TERRAIN_LAYERS), dtype=np.float32)
        self._buf_scalar      = np.zeros((N, WBGYM_NUM_SCALARS), dtype=np.float32)
        self._buf_entities    = np.zeros((N, WBGYM_MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
        self._buf_entity_mask = np.zeros((N, WBGYM_MAX_ENTITIES), dtype=np.float32)
        self._buf_sounds      = np.zeros((N, WBGYM_MAX_SOUNDS, SOUND_FEATURES), dtype=np.float32)
        self._buf_sound_mask  = np.zeros((N, WBGYM_MAX_SOUNDS), dtype=np.float32)

        # Reward observation buffers (written by winbolo_obs_to_reward_batch)
        self._rw_scalars       = np.zeros((N, WBGYM_NUM_SCALARS), dtype=np.float32)
        self._rw_events        = np.zeros((N, WBGYM_MAX_EVENTS), dtype=np.uint8)
        self._rw_event_count   = np.zeros(N, dtype=np.int32)
        self._rw_pills         = np.zeros((N, WBGYM_MAX_PILLBOXES, REWARD_PILL_FEATURES), dtype=np.float32)
        self._rw_pill_count    = np.zeros(N, dtype=np.int32)
        self._rw_bases         = np.zeros((N, WBGYM_MAX_BASES, REWARD_BASE_FEATURES), dtype=np.float32)
        self._rw_base_count    = np.zeros(N, dtype=np.int32)
        self._rw_entities      = np.zeros((N, WBGYM_MAX_ENTITIES, REWARD_ENTITY_FEATURES), dtype=np.float32)
        self._rw_entity_count  = np.zeros(N, dtype=np.int32)
        self._rw_sounds        = np.zeros((N, WBGYM_MAX_SOUNDS, SOUND_FEATURES), dtype=np.float32)
        self._rw_sound_count   = np.zeros(N, dtype=np.int32)
        self._rw_terrain       = np.zeros((N, WBGYM_SPATIAL_SIZE, WBGYM_SPATIAL_SIZE), dtype=np.float32)
        self._rw_dead          = np.zeros(N, dtype=np.uint8)
        self._rw_game_over     = np.zeros(N, dtype=np.uint8)
        self._rw_game_won      = np.zeros(N, dtype=np.uint8)
        self._rw_tick          = np.zeros(N, dtype=np.int32)
        self._rw_tank_x        = np.zeros(N, dtype=np.float32)
        self._rw_tank_y        = np.zeros(N, dtype=np.float32)
        self._rw_assistant_msg = np.zeros(N, dtype=np.uint8)

        # Persistent WinBoloRewardBatchOut struct (pointers updated once)
        self._reward_batch_out = WinBoloRewardBatchOut()
        self._reward_batch_out.scalars       = self._rw_scalars.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._reward_batch_out.events        = self._rw_events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._reward_batch_out.event_count   = self._rw_event_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._reward_batch_out.pills         = self._rw_pills.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._reward_batch_out.pill_count    = self._rw_pill_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._reward_batch_out.bases         = self._rw_bases.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._reward_batch_out.base_count    = self._rw_base_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._reward_batch_out.entities      = self._rw_entities.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._reward_batch_out.entity_count  = self._rw_entity_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._reward_batch_out.sounds        = self._rw_sounds.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._reward_batch_out.sound_count   = self._rw_sound_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._reward_batch_out.terrain       = self._rw_terrain.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._reward_batch_out.dead          = self._rw_dead.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._reward_batch_out.game_over     = self._rw_game_over.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._reward_batch_out.game_won      = self._rw_game_won.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._reward_batch_out.tick          = self._rw_tick.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._reward_batch_out.tank_x        = self._rw_tank_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._reward_batch_out.tank_y        = self._rw_tank_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._reward_batch_out.assistant_msg = self._rw_assistant_msg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        self._reward_config = reward_config or RewardConfig()
        self._ticks = np.zeros(num_envs, dtype=np.int32)

        # Send reward weights to C for C-side reward computation
        self._component_names = [f.name for f in fields(self._reward_config)]
        c_count = self._lib.winbolo_reward_component_count()
        assert c_count == len(self._component_names), (
            f"Reward component count mismatch: C={c_count}, Python={len(self._component_names)}"
        )
        weights_array = np.array(
            [getattr(self._reward_config, name) for name in self._component_names],
            dtype=np.float32,
        )
        for i in range(num_envs):
            self._lib.winbolo_set_reward_weights(
                self._games[i],
                weights_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c_count,
            )

        # ── Strided numpy views into _obs_buf for zero-copy batch extraction ──
        obs_stride = ctypes.sizeof(WinBoloObs)
        obs_base = ctypes.addressof(self._obs_buf)
        buf_bytes = obs_stride * num_envs

        # Create a single numpy byte view over the entire obs buffer
        self._obs_raw = np.frombuffer(
            (ctypes.c_char * buf_bytes).from_address(obs_base), dtype=np.uint8
        )

        reward_offset = WinBoloObs.reward.offset
        self._c_rewards = np.ndarray(
            shape=(num_envs,), dtype=np.float32,
            buffer=self._obs_raw, offset=reward_offset,
            strides=(obs_stride,),
        )

        components_offset = WinBoloObs.reward_components.offset
        self._c_components = np.ndarray(
            shape=(num_envs, WBGYM_NUM_REWARD_COMPONENTS), dtype=np.float32,
            buffer=self._obs_raw, offset=components_offset,
            strides=(obs_stride, 4),
        )

        dead_offset = WinBoloObs.dead.offset
        self._c_dead = np.ndarray(
            shape=(num_envs,), dtype=np.uint8,
            buffer=self._obs_raw, offset=dead_offset,
            strides=(obs_stride,),
        )

        game_over_offset = WinBoloObs.game_over.offset
        self._c_game_over = np.ndarray(
            shape=(num_envs,), dtype=np.uint8,
            buffer=self._obs_raw, offset=game_over_offset,
            strides=(obs_stride,),
        )

        # Pending async step results
        self._pending_actions: np.ndarray | None = None

    def _setup_ctypes(self) -> None:
        lib = self._lib
        lib.winbolo_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.winbolo_create.restype = ctypes.c_void_p
        lib.winbolo_destroy.argtypes = [ctypes.c_void_p]
        lib.winbolo_destroy.restype = None
        lib.winbolo_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(WinBoloAction),
            ctypes.POINTER(WinBoloObs),
        ]
        lib.winbolo_step.restype = None
        lib.winbolo_reset.argtypes = [ctypes.c_void_p, ctypes.POINTER(WinBoloObs)]
        lib.winbolo_reset.restype = None
        lib.winbolo_step_batch.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(WinBoloAction),
            ctypes.POINTER(WinBoloObs),
            ctypes.c_int,
        ]
        lib.winbolo_step_batch.restype = None
        lib.winbolo_obs_size.argtypes = []
        lib.winbolo_obs_size.restype = ctypes.c_int
        lib.winbolo_action_size.argtypes = []
        lib.winbolo_action_size.restype = ctypes.c_int
        lib.winbolo_set_reward_weights.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.winbolo_set_reward_weights.restype = None
        lib.winbolo_reward_component_count.argtypes = []
        lib.winbolo_reward_component_count.restype = ctypes.c_int
        setup_batch_ctypes(lib)

    def _verify_struct_sizes(self) -> None:
        expected_obs = self._lib.winbolo_obs_size()
        actual_obs = ctypes.sizeof(WinBoloObs)
        if expected_obs != actual_obs:
            raise RuntimeError(
                f"WinBoloObs size mismatch: C={expected_obs}, Python={actual_obs}"
            )
        expected_action = self._lib.winbolo_action_size()
        actual_action = ctypes.sizeof(WinBoloAction)
        if expected_action != actual_action:
            raise RuntimeError(
                f"WinBoloAction size mismatch: C={expected_action}, Python={actual_action}"
            )

    def _extract_obs_batch(self) -> dict[str, np.ndarray]:
        """Extract model obs for all envs via a single C call into pre-allocated buffers."""
        self._lib.winbolo_obs_to_numpy_batch(
            self._obs_buf,
            self._buf_terrain.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._buf_scalar.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._buf_entities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._buf_entity_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._buf_sounds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._buf_sound_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.num_envs,
        )
        return {
            "terrain": self._buf_terrain,
            "scalar": self._buf_scalar,
            "entities": self._buf_entities,
            "entity_mask": self._buf_entity_mask,
            "sounds": self._buf_sounds,
            "sound_mask": self._buf_sound_mask,
        }

    def _extract_reward_batch(self) -> dict[str, np.ndarray]:
        """Extract reward inputs for all envs via a single C call into pre-allocated buffers."""
        self._lib.winbolo_obs_to_reward_batch(
            self._obs_buf,
            ctypes.byref(self._reward_batch_out),
            self.num_envs,
        )
        return {
            "scalars": self._rw_scalars,
            "events": self._rw_events,
            "event_count": self._rw_event_count,
            "pills": self._rw_pills,
            "pill_count": self._rw_pill_count,
            "bases": self._rw_bases,
            "base_count": self._rw_base_count,
            "entities": self._rw_entities,
            "entity_count": self._rw_entity_count,
            "sounds": self._rw_sounds,
            "sound_count": self._rw_sound_count,
            "terrain": self._rw_terrain,
            "dead": self._rw_dead.view(bool),
            "game_over": self._rw_game_over.view(bool),
            "game_won": self._rw_game_won.view(bool),
            "tick": self._rw_tick,
            "tank_x": self._rw_tank_x,
            "tank_y": self._rw_tank_y,
            "assistant_msg": self._rw_assistant_msg,
        }

    def _reset_single(self, idx: int) -> None:
        self._lib.winbolo_reset(self._games[idx], ctypes.byref(self._obs_buf[idx]))
        self._ticks[idx] = 0

    def reset(self) -> dict[str, np.ndarray]:
        for i in range(self.num_envs):
            self._reset_single(i)
        return self._extract_obs_batch()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions)

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        actions = self._pending_actions
        assert actions is not None, "step_async must be called before step_wait"
        self._pending_actions = None

        # Fill action structs — single C call replaces Python loop
        actions_i32 = np.ascontiguousarray(actions, dtype=np.int32)
        self._lib.winbolo_fill_actions_batch(
            actions_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._actions,
            self.num_envs,
        )

        # Single C call to step all games
        self._lib.winbolo_step_batch(
            self._games,
            self._actions,
            self._obs_buf,
            self.num_envs,
        )

        self._ticks += 1

        # ── Extract rewards & flags via strided numpy views ──
        # Copy rewards/components now — resets below will overwrite _obs_buf
        rewards = self._c_rewards.copy()
        components_array = self._c_components.copy()

        # ── Termination checks via strided views ──
        dead_arr = self._c_dead.astype(bool)
        game_over_arr = self._c_game_over.astype(bool)
        terminated = dead_arr | game_over_arr

        # Only call _extract_reward_batch if we need pill/base/scalar data for termination
        if self.end_on_all_pillbox_captures or self.end_on_land:
            reward_obs = self._extract_reward_batch()
            if self.end_on_all_pillbox_captures:
                pill_counts = reward_obs["pill_count"]
                pill_owners = reward_obs["pills"][:, :, 2]  # PILL_OWNER
                has_pills = pill_counts > 0
                pill_mask = np.arange(16)[None, :] < pill_counts[:, None]
                all_self = ((pill_owners == OWNER_SELF) | ~pill_mask).all(axis=1) & has_pills
                terminated = terminated | all_self
            if self.end_on_land:
                on_land = reward_obs["scalars"][:, S_IN_BOAT] < 0.5
                terminated = terminated | on_land

        truncated = ~terminated & (self._ticks >= self.max_ticks)
        dones = terminated | truncated

        # ── Handle resets for done envs ──
        for i in range(self.num_envs):
            if dones[i]:
                self._reset_single(i)

        # Single C call for model obs extraction — writes into pre-allocated buffers
        obs_out = self._extract_obs_batch()

        # Return components_array as batch numpy — avoids per-env dict construction
        return obs_out, rewards, dones, components_array

    def close(self) -> None:
        for i in range(self.num_envs):
            if self._games[i]:
                self._lib.winbolo_destroy(self._games[i])
                self._games[i] = None

    def env_is_wrapped(self, wrapper_class: type, indices: Sequence[int] | None = None) -> list[bool]:
        if indices is None:
            return [False] * self.num_envs
        return [False] * len(indices)

    def env_method(self, method_name: str, *method_args, indices: Sequence[int] | None = None, **method_kwargs) -> list:
        raise AttributeError(f"WinBoloBatchEnv has no per-env method '{method_name}'")

    def get_attr(self, attr_name: str, indices: Sequence[int] | None = None) -> list:
        n = len(indices) if indices is not None else self.num_envs
        attr_defaults = {"render_mode": None, "spec": None}
        if attr_name in attr_defaults:
            return [attr_defaults[attr_name]] * n
        raise AttributeError(f"WinBoloBatchEnv has no per-env attribute '{attr_name}'")

    def set_attr(self, attr_name: str, value: Any, indices: Sequence[int] | None = None) -> None:
        raise AttributeError(f"WinBoloBatchEnv does not support set_attr('{attr_name}')")

    def seed(self, seed: int | None = None) -> Sequence[int | None]:
        return [None] * self.num_envs

