#!/usr/bin/env python3
# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Training script for WinBolo RL agent using custom PPO."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from winbolo_ml.env_batch import WinBoloBatchEnv
from winbolo_ml.gym_types import GAME_TYPE_OPEN, GAME_TYPE_STRICT_TOURNAMENT, GAME_TYPE_TOURNAMENT
from winbolo_ml.features import ACTION_DIMS, TOTAL_LOGITS
from winbolo_ml.network import MultiDiscreteDistribution, WinBoloModel, model_from_checkpoint
from winbolo_ml.rewards import (
    RewardConfig,
    S_ARMOR, S_SHELLS, S_MINES, S_TREES, S_SPEED, S_IN_BOAT,
    S_DEAD, S_HIDDEN, S_TANK_OBSTRUCTED, S_LGM_STATUS,
    S_OWN_PILL_FRAC, S_OWN_BASE_FRAC, S_TANK_X, S_TANK_Y,
    phase1_movement,
    phase2_combat,
    phase3_resources,
    phase4_builder,
    phase5_tactical,
    phase6_mining,
    phase7_strategic,
    phase8_full,
)
from winbolo_ml.utils import load_config

logger = logging.getLogger(__name__)

OBS_KEYS = ["terrain", "scalar", "entities", "entity_mask", "sounds", "sound_mask"]

PHASE_PRESETS = {
    "phase1_movement": phase1_movement,
    "phase2_combat": phase2_combat,
    "phase3_resources": phase3_resources,
    "phase4_builder": phase4_builder,
    "phase5_tactical": phase5_tactical,
    "phase6_mining": phase6_mining,
    "phase7_strategic": phase7_strategic,
    "phase8_full": phase8_full,
}


def make_reward_config(reward_cfg: dict) -> RewardConfig:
    """Build a RewardConfig from the YAML rewards section.

    Supports::

        rewards:
          phase: phase2_combat       # preset name (default: phase8_full)
          overrides:                  # optional per-weight tweaks
            survival_tick: 0.01
    """
    phase_name = reward_cfg.get("phase", "phase8_full")
    if phase_name not in PHASE_PRESETS:
        raise ValueError(
            f"Unknown reward phase '{phase_name}'. "
            f"Available: {', '.join(PHASE_PRESETS)}"
        )
    config = PHASE_PRESETS[phase_name]()
    for key, val in reward_cfg.get("overrides", {}).items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown reward component '{key}' in overrides")
        setattr(config, key, float(val))
    return config


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self, num_envs: int, n_steps: int, device: torch.device) -> None:
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.device = device
        self.pos = 0

        # Observations stored as numpy, converted to torch on compute
        self.obs: dict[str, np.ndarray] = {}
        self.actions = np.zeros((n_steps, num_envs, 8), dtype=np.int64)
        self.rewards = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.head_masks = np.zeros((n_steps, num_envs, 8), dtype=np.bool_)

        # Computed after rollout
        self.advantages = np.zeros((n_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, num_envs), dtype=np.float32)

    def reset(self) -> None:
        self.pos = 0
        self.obs = {}

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        head_mask: np.ndarray | None = None,
    ) -> None:
        t = self.pos
        if t == 0:
            # Initialize obs storage on first add
            for key in OBS_KEYS:
                shape = (self.n_steps, self.num_envs) + obs[key].shape[1:]
                self.obs[key] = np.zeros(shape, dtype=np.float32)

        for key in OBS_KEYS:
            self.obs[key][t] = obs[key]
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.log_probs[t] = log_probs
        if head_mask is not None:
            self.head_masks[t] = head_mask
        self.pos += 1

    def compute_gae(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and discounted returns."""
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns = self.advantages + self.values

    def free_flat_tensors(self) -> None:
        """Release GPU memory from flat tensors."""
        self._flat_gpu = {}

    def prepare_flat_tensors(self) -> None:
        """Flatten rollout and upload to GPU once. Call after compute_gae."""
        total = self.n_steps * self.num_envs
        self._flat_gpu = {}
        for key in OBS_KEYS:
            flat_np = self.obs[key].reshape((total,) + self.obs[key].shape[2:])
            self._flat_gpu[key] = torch.from_numpy(flat_np).to(self.device, non_blocking=True)
        self._flat_gpu["actions"] = torch.from_numpy(
            self.actions.reshape(total, 8).copy()
        ).long().to(self.device, non_blocking=True)
        self._flat_gpu["log_probs"] = torch.from_numpy(
            self.log_probs.reshape(total).copy()
        ).to(self.device, non_blocking=True)
        self._flat_gpu["advantages"] = torch.from_numpy(
            self.advantages.reshape(total).copy()
        ).to(self.device, non_blocking=True)
        self._flat_gpu["returns"] = torch.from_numpy(
            self.returns.reshape(total).copy()
        ).to(self.device, non_blocking=True)
        self._flat_gpu["head_masks"] = torch.from_numpy(
            self.head_masks.reshape(total, 8).copy()
        ).bool().to(self.device, non_blocking=True)

    def get_batches(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        """Yield random mini-batches from pre-uploaded GPU tensors."""
        total = self.n_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        batches = []
        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            batch = {}
            for key in OBS_KEYS:
                batch[key] = self._flat_gpu[key][idx]
            batch["actions"] = self._flat_gpu["actions"][idx]
            batch["old_log_probs"] = self._flat_gpu["log_probs"][idx]
            batch["advantages"] = self._flat_gpu["advantages"][idx]
            batch["returns"] = self._flat_gpu["returns"][idx]
            batch["head_masks"] = self._flat_gpu["head_masks"][idx]
            batches.append(batch)
        return batches


def compute_dynamic_mask(
    static_mask: list[bool] | None,
    obs: dict[str, np.ndarray],
    dynamic_rules: dict | None,
) -> np.ndarray:
    """Compute per-env head mask combining static config with dynamic observation rules.

    Args:
        static_mask: Base mask from config (8 bools), or None for all-active.
        obs: Current observation dict with "scalar" key [N, 26].
        dynamic_rules: Dict of rule_name → params from config, or None to skip.

    Returns:
        [N, 8] bool numpy array.
    """
    N = obs["scalar"].shape[0]
    # Start from static mask
    if static_mask is not None:
        mask = np.tile(np.array(static_mask, dtype=np.bool_), (N, 1))
    else:
        mask = np.ones((N, 8), dtype=np.bool_)

    if not dynamic_rules:
        return mask

    # Rule: mask shoot (head 2) while on boat
    if dynamic_rules.get("no_shoot_on_boat", False):
        in_boat = obs["scalar"][:, S_IN_BOAT] > 0.5
        mask[in_boat, 2] = False

    return mask


@torch.no_grad()
def collect_rollout(
    env: WinBoloBatchEnv,
    model: WinBoloModel,
    buffer: RolloutBuffer,
    obs: dict[str, np.ndarray],
    device: torch.device,
    component_accum: dict[str, list[float]] | None = None,
    head_mask: list[bool] | None = None,
    dynamic_mask_rules: dict | None = None,
) -> dict[str, np.ndarray]:
    """Collect n_steps of experience. Returns the last obs."""
    buffer.reset()

    # Pre-allocate CUDA tensors for obs (avoids per-step torch.tensor allocation)
    obs_tensors = {k: torch.empty_like(torch.from_numpy(obs[k]), device=device) for k in OBS_KEYS}

    for _ in range(buffer.n_steps):
        # Copy numpy obs into pre-allocated CUDA tensors
        for k in OBS_KEYS:
            obs_tensors[k].copy_(torch.from_numpy(obs[k]), non_blocking=True)
        logits, values = model(**obs_tensors)
        values_np = values.squeeze(-1).cpu().numpy()

        # Compute per-env dynamic mask
        step_mask_np = compute_dynamic_mask(head_mask, obs, dynamic_mask_rules)
        step_mask = torch.from_numpy(step_mask_np).to(device)

        # Sample actions
        dist = MultiDiscreteDistribution(logits, head_mask=step_mask)
        actions = dist.sample()
        log_probs = dist.total_log_prob(actions)

        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()

        # Step environment
        next_obs, rewards, dones, components = env.step(actions_np)

        # Accumulate reward components for logging (batch numpy, no per-env dicts)
        if component_accum is not None:
            if "_sum" not in component_accum:
                component_accum["_sum"] = np.zeros(components.shape[1], dtype=np.float64)
                component_accum["_count"] = 0
            component_accum["_sum"] += components.sum(axis=0)
            component_accum["_count"] += components.shape[0]

        buffer.add(obs, actions_np, rewards, dones.astype(np.float32), values_np, log_probs_np,
                   head_mask=step_mask_np)
        obs = next_obs

    # Bootstrap value for last observation
    for k in OBS_KEYS:
        obs_tensors[k].copy_(torch.from_numpy(obs[k]), non_blocking=True)
    _, last_values = model(**obs_tensors)
    last_values_np = last_values.squeeze(-1).cpu().numpy()
    buffer.compute_gae(last_values_np, gamma=0.99, gae_lambda=0.95)

    return obs


def ppo_update(
    model: WinBoloModel,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    n_epochs: int,
    batch_size: int,
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    head_mask: list[bool] | None = None,
) -> dict[str, float]:
    """Run PPO update epochs over the rollout buffer."""
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        batches = buffer.get_batches(batch_size)
        for batch in batches:
            logits, values = model(
                batch["terrain"], batch["scalar"],
                batch["entities"], batch["entity_mask"],
                batch["sounds"], batch["sound_mask"],
            )
            values = values.squeeze(-1)

            # Use per-sample masks stored during rollout
            dist = MultiDiscreteDistribution(logits, head_mask=batch["head_masks"])
            actions = batch["actions"]

            # Log probs with build-coord masking
            new_log_probs = dist.total_log_prob(actions)
            old_log_probs = batch["old_log_probs"]

            # Advantages
            advantages = batch["advantages"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy loss (clipped PPO)
            ratio = torch.exp(new_log_probs - old_log_probs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            value_loss = F.mse_loss(values, batch["returns"])

            # Entropy with build-coord masking
            entropy = dist.total_entropy(actions).mean()

            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Logging
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_approx_kl += approx_kl
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
        "approx_kl": total_approx_kl / max(n_updates, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WinBolo RL agent (custom PPO)")
    parser.add_argument("--config", type=str, default="configs/phase1_pillbox.yaml",
                        help="Path to config YAML")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--map", type=str, default=None,
                        help="Override map path")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--tensorboard-log", type=str, default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--lib-path", type=str, default="./libwinbolo_gym.so",
                        help="Path to libwinbolo_gym.so")
    parser.add_argument("--game-type", type=str, default=None,
                        choices=["open", "tournament", "strict"],
                        help="Game type (default from config or strict)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default auto)")
    parser.add_argument("--reward-phase", type=str, default=None,
                        choices=list(PHASE_PRESETS.keys()),
                        help="Override reward phase preset")
    return parser.parse_args()


def _sync_model_weights(src: WinBoloModel, dst: WinBoloModel) -> None:
    """Copy weights from src model to dst model (possibly across devices)."""
    src_sd = src.state_dict()
    dst_device = next(dst.parameters()).device
    dst.load_state_dict({k: v.to(dst_device) for k, v in src_sd.items()})


def _log_iteration(
    writer: SummaryWriter,
    global_step: int,
    update_stats: dict[str, float],
    component_accum: dict[str, Any],
    buffer: RolloutBuffer,
    reward_config: RewardConfig,
    n_steps: int,
    fps: float,
    head_mask: list[bool] | None,
    iteration: int,
    num_iterations: int,
) -> float:
    """Log all TensorBoard metrics for an iteration. Returns mean_reward."""
    writer.add_scalar("train/policy_loss", update_stats["policy_loss"], global_step)
    writer.add_scalar("train/value_loss", update_stats["value_loss"], global_step)
    writer.add_scalar("train/entropy", update_stats["entropy"], global_step)
    writer.add_scalar("train/approx_kl", update_stats["approx_kl"], global_step)
    writer.add_scalar("train/fps", fps, global_step)

    mean_reward = buffer.rewards.sum(axis=0).mean()
    writer.add_scalar("reward/total", buffer.rewards.mean(), global_step)
    writer.add_scalar("rollout/mean_episode_reward", mean_reward, global_step)
    if "_sum" in component_accum and component_accum["_count"] > 0:
        comp_means = component_accum["_sum"] / component_accum["_count"]
        component_names = [f.name for f in __import__("dataclasses").fields(reward_config)]
        for j, name in enumerate(component_names):
            raw_mean = comp_means[j]
            weight = getattr(reward_config, name, 0.0)
            writer.add_scalar(f"reward_raw/{name}", raw_mean, global_step)
            writer.add_scalar(f"reward_weighted/{name}", weight * raw_mean, global_step)

    if "scalar" in buffer.obs:
        scalars = buffer.obs["scalar"]
        s_flat = scalars.reshape(-1, scalars.shape[-1])
        writer.add_scalar("obs/speed", s_flat[:, S_SPEED].mean(), global_step)
        writer.add_scalar("obs/armour", s_flat[:, S_ARMOR].mean(), global_step)
        writer.add_scalar("obs/in_boat", s_flat[:, S_IN_BOAT].mean(), global_step)
        writer.add_scalar("obs/dead", s_flat[:, S_DEAD].mean(), global_step)
        writer.add_scalar("obs/hidden", s_flat[:, S_HIDDEN].mean(), global_step)
        writer.add_scalar("obs/obstructed", s_flat[:, S_TANK_OBSTRUCTED].mean(), global_step)
        writer.add_scalar("obs/tank_x", s_flat[:, S_TANK_X].mean(), global_step)
        writer.add_scalar("obs/tank_y", s_flat[:, S_TANK_Y].mean(), global_step)
        writer.add_scalar("obs/tank_x_std", s_flat[:, S_TANK_X].std(), global_step)
        writer.add_scalar("obs/tank_y_std", s_flat[:, S_TANK_Y].std(), global_step)
        writer.add_scalar("obs/shells", s_flat[:, S_SHELLS].mean(), global_step)
        writer.add_scalar("obs/mines", s_flat[:, S_MINES].mean(), global_step)
        writer.add_scalar("obs/trees", s_flat[:, S_TREES].mean(), global_step)
        writer.add_scalar("obs/own_base_frac", s_flat[:, S_OWN_BASE_FRAC].mean(), global_step)
        writer.add_scalar("obs/own_pill_frac", s_flat[:, S_OWN_PILL_FRAC].mean(), global_step)
        writer.add_scalar("obs/lgm_status", s_flat[:, S_LGM_STATUS].mean(), global_step)

    actions_all = buffer.actions
    a_flat = actions_all.reshape(-1, 8)
    writer.add_scalar("actions/accel_fwd", (a_flat[:, 0] == 2).mean(), global_step)
    writer.add_scalar("actions/accel_brake", (a_flat[:, 0] == 0).mean(), global_step)
    writer.add_scalar("actions/accel_coast", (a_flat[:, 0] == 1).mean(), global_step)
    writer.add_scalar("actions/turn_left", (a_flat[:, 1] == 0).mean(), global_step)
    writer.add_scalar("actions/turn_straight", (a_flat[:, 1] == 1).mean(), global_step)
    writer.add_scalar("actions/turn_right", (a_flat[:, 1] == 2).mean(), global_step)
    writer.add_scalar("actions/shoot_pct", a_flat[:, 2].mean(), global_step)
    writer.add_scalar("actions/mine_pct", a_flat[:, 3].mean(), global_step)
    writer.add_scalar("actions/build_pct", (a_flat[:, 5] > 0).mean(), global_step)

    ep_dones = buffer.dones
    ep_done_count = ep_dones.sum()
    if ep_done_count > 0:
        writer.add_scalar("episode/dones_per_rollout", ep_done_count, global_step)
    writer.add_scalar("episode/length_approx", n_steps / max(ep_dones.mean(axis=0).sum(), 0.01), global_step)

    if iteration % 10 == 0 or iteration == 1:
        print(
            f"Iter {iteration}/{num_iterations} | "
            f"Steps: {global_step} | "
            f"Reward: {mean_reward:.2f} | "
            f"PL: {update_stats['policy_loss']:.4f} | "
            f"VL: {update_stats['value_loss']:.4f} | "
            f"Ent: {update_stats['entropy']:.3f} | "
            f"FPS: {fps:.0f}"
        )

    return mean_reward


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    env_cfg = cfg["env"]
    train_cfg = cfg["training"]
    reward_cfg = cfg.get("rewards", {})
    network_cfg = cfg.get("network", {})

    # Parameters
    num_envs = args.num_envs or train_cfg.get("num_envs", 4)
    total_timesteps = args.total_timesteps or train_cfg["total_timesteps"]
    map_path = args.map or env_cfg["map"]
    max_ticks = env_cfg.get("max_ticks", 6000)
    n_steps = train_cfg.get("n_steps", 2048)
    batch_size = train_cfg.get("batch_size", 64)
    n_epochs = train_cfg.get("n_epochs", 10)
    learning_rate = train_cfg.get("learning_rate", 3e-4)
    gamma = train_cfg.get("gamma", 0.99)
    gae_lambda = train_cfg.get("gae_lambda", 0.95)
    clip_range = train_cfg.get("clip_range", 0.2)
    ent_coef = train_cfg.get("ent_coef", 0.01)
    vf_coef = train_cfg.get("vf_coef", 0.5)
    max_grad_norm = train_cfg.get("max_grad_norm", 0.5)

    # Network config
    encoder_dim = network_cfg.get("encoder_dim", 64)
    hidden_dim = network_cfg.get("hidden_dim", 256)

    # Action mask: list of 8 bools, one per head
    # [accel, turn, shoot, mine, gun_range, build_action, build_rx, build_ry]
    head_mask = train_cfg.get("action_mask", None)

    # Dynamic mask rules: observation-based per-step masking
    dynamic_mask_rules = train_cfg.get("dynamic_mask", None)

    # Game type
    _game_type_map = {"open": GAME_TYPE_OPEN, "tournament": GAME_TYPE_TOURNAMENT, "strict": GAME_TYPE_STRICT_TOURNAMENT}
    if args.game_type is not None:
        game_type = _game_type_map[args.game_type]
    else:
        game_type = env_cfg.get("game_type", GAME_TYPE_STRICT_TOURNAMENT)
    end_on_pills = env_cfg.get("end_on_all_pillbox_captures", False)
    end_on_land = env_cfg.get("end_on_land", False)

    # Reward config: CLI override > YAML
    if args.reward_phase:
        reward_config = PHASE_PRESETS[args.reward_phase]()
    else:
        reward_config = make_reward_config(reward_cfg)

    # Device setup — pipeline across 2 GPUs if available
    num_gpus = torch.cuda.device_count()
    pipeline_mode = num_gpus >= 2 and args.device is None
    if pipeline_mode:
        train_device = torch.device("cuda:0")
        rollout_device = torch.device("cuda:1")
        print(f"Pipeline mode: rollout on {rollout_device}, PPO on {train_device}")
    else:
        if args.device:
            train_device = torch.device(args.device)
        else:
            train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rollout_device = train_device
        print(f"Using device: {train_device}")

    # Create environment
    env = WinBoloBatchEnv(
        lib_path=args.lib_path,
        map_path=map_path,
        num_envs=num_envs,
        max_ticks=max_ticks,
        game_type=game_type,
        reward_config=reward_config,
        end_on_all_pillbox_captures=end_on_pills,
        end_on_land=end_on_land,
    )

    # Training model (PPO updates on train_device)
    train_model = WinBoloModel(
        encoder_dim=encoder_dim,
        hidden_dim=hidden_dim,
    ).to(train_device)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=train_device, weights_only=False)
        train_model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze layers if configured
    # freeze_layers: list of layer group names to freeze
    # Valid groups: "encoders" (terrain/scalar/entity/sound encoders),
    #              "fusion", "action_head", "value_head",
    #              "build_type_head", "build_coord_head"
    freeze_groups = train_cfg.get("freeze_layers", None)
    if freeze_groups:
        _group_map = {
            "encoders": ["terrain_enc", "scalar_enc", "entity_enc", "sound_enc"],
            "fusion": ["fusion"],
            "action_head": ["action_head"],
            "value_head": ["value_head"],
            "build_type_head": ["build_type_head"],
            "build_coord_head": ["build_coord_head"],
        }
        frozen_modules = []
        for group in freeze_groups:
            for module_name in _group_map[group]:
                module = getattr(train_model, module_name)
                for param in module.parameters():
                    param.requires_grad = False
                frozen_modules.append(module_name)
        frozen_count = sum(1 for p in train_model.parameters() if not p.requires_grad)
        total_count = sum(1 for p in train_model.parameters())
        print(f"Frozen layers: {', '.join(frozen_modules)} ({frozen_count}/{total_count} params frozen)")

    trainable_params = [p for p in train_model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    if args.resume and "optimizer_state_dict" in checkpoint and not freeze_groups:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Rollout model (inference on rollout_device)
    if pipeline_mode:
        rollout_model = WinBoloModel(
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
        ).to(rollout_device)
        _sync_model_weights(train_model, rollout_model)
    else:
        rollout_model = train_model

    # Checkpoint directory
    phase_name = Path(args.config).stem
    checkpoint_dir = Path(args.checkpoint_dir) / phase_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=f"{args.tensorboard_log}/{phase_name}_{int(time.time())}")

    # Double-buffered rollout storage (flat tensors upload to train_device)
    buffers = [
        RolloutBuffer(num_envs, n_steps, train_device),
        RolloutBuffer(num_envs, n_steps, train_device),
    ]
    buf_idx = 0

    # Training loop
    num_iterations = total_timesteps // (n_steps * num_envs)
    timesteps_per_iteration = n_steps * num_envs
    global_step = 0
    best_mean_reward = float("-inf")

    print(f"Training for {total_timesteps} timesteps ({num_iterations} iterations)")
    print(f"  {num_envs} envs x {n_steps} steps = {timesteps_per_iteration} timesteps/iter")
    print(f"  batch_size={batch_size} | n_epochs={n_epochs} | {timesteps_per_iteration // batch_size} minibatches/epoch")
    print(f"Map: {map_path} | Max ticks: {max_ticks}")
    print(f"Config: {args.config}")
    print(f"Reward phase: {reward_cfg.get('phase', args.reward_phase or 'phase8_full')}")
    if head_mask:
        head_names = ["accel", "turn", "shoot", "mine", "gun_range", "build_action", "build_rx", "build_ry"]
        active = [n for n, m in zip(head_names, head_mask) if m]
        print(f"Action mask: {', '.join(active)} active")
    if dynamic_mask_rules:
        print(f"Dynamic mask rules: {dynamic_mask_rules}")
    if freeze_groups:
        print(f"Frozen: {', '.join(freeze_groups)}")
    print(f"TensorBoard: tensorboard --logdir {args.tensorboard_log}")

    obs = env.reset()
    training_start = time.time()

    # Pipeline state
    ppo_thread: threading.Thread | None = None
    ppo_stats_box: list[dict[str, float] | None] = [None]
    ppo_error_box: list[BaseException | None] = [None]
    # Deferred logging state from previous iteration
    prev_component_accum: dict[str, Any] = {}
    prev_buffer: RolloutBuffer | None = None
    prev_global_step = 0
    prev_iteration = 0
    ppo_start_time = 0.0  # when current PPO was kicked off

    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()

        # Select buffer for this rollout (alternates between 0 and 1)
        buffer = buffers[buf_idx]
        buf_idx = 1 - buf_idx

        # Collect rollout on rollout_device (overlaps with previous PPO if pipelined)
        component_accum: dict[str, Any] = {}
        rollout_model.eval()
        obs = collect_rollout(env, rollout_model, buffer, obs, rollout_device, component_accum,
                              head_mask=head_mask, dynamic_mask_rules=dynamic_mask_rules)

        # Wait for previous PPO to finish before uploading new tensors
        if ppo_thread is not None:
            ppo_thread.join()
            if ppo_error_box[0] is not None:
                raise ppo_error_box[0]

            # Sync updated weights to rollout model
            if pipeline_mode:
                _sync_model_weights(train_model, rollout_model)

            # Log the previous iteration — FPS = time between PPO starts
            now = time.time()
            iter_time = now - ppo_start_time if ppo_start_time > 0 else now - training_start
            fps = timesteps_per_iteration / max(iter_time, 0.001)
            mean_reward = _log_iteration(
                writer, prev_global_step, ppo_stats_box[0],
                prev_component_accum, prev_buffer, reward_config,
                n_steps, fps, head_mask, prev_iteration, num_iterations,
            )

            # Checkpoint (for previous iteration)
            save_freq = max(num_iterations // 20, 1)
            if prev_iteration % save_freq == 0:
                ckpt_path = checkpoint_dir / f"{phase_name}_{prev_global_step}.pt"
                torch.save({
                    "model_state_dict": train_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": prev_global_step,
                    "iteration": prev_iteration,
                    "action_mask": head_mask,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_path = checkpoint_dir / "best" / f"{phase_name}_best.pt"
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": train_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": prev_global_step,
                        "mean_reward": mean_reward,
                        "action_mask": head_mask,
                    }, best_path)

            # Free previous buffer's GPU tensors before uploading new ones
            prev_buffer.free_flat_tensors()

        # Upload new rollout to train_device
        buffer.prepare_flat_tensors()

        # Save state for deferred logging
        global_step += timesteps_per_iteration
        prev_component_accum = component_accum
        prev_buffer = buffer
        prev_global_step = global_step
        prev_iteration = iteration

        # Start PPO in background thread (or inline if single GPU)
        ppo_stats_box = [None]
        ppo_error_box = [None]

        def _run_ppo(buf=buffer) -> None:
            try:
                train_model.train()
                ppo_stats_box[0] = ppo_update(
                    train_model, optimizer, buf,
                    n_epochs=n_epochs, batch_size=batch_size,
                    clip_range=clip_range, ent_coef=ent_coef,
                    vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                    head_mask=head_mask,
                )
            except BaseException as e:
                ppo_error_box[0] = e

        if pipeline_mode:
            ppo_start_time = time.time()
            ppo_thread = threading.Thread(target=_run_ppo)
            ppo_thread.start()
        else:
            _run_ppo()
            ppo_thread = None

            # In sequential mode, log immediately
            iter_time = time.time() - iter_start
            fps = timesteps_per_iteration / iter_time
            mean_reward = _log_iteration(
                writer, global_step, ppo_stats_box[0],
                component_accum, buffer, reward_config,
                n_steps, fps, head_mask, iteration, num_iterations,
            )

            save_freq = max(num_iterations // 20, 1)
            if iteration % save_freq == 0:
                ckpt_path = checkpoint_dir / f"{phase_name}_{global_step}.pt"
                torch.save({
                    "model_state_dict": train_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "iteration": iteration,
                    "action_mask": head_mask,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    best_path = checkpoint_dir / "best" / f"{phase_name}_best.pt"
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": train_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "mean_reward": mean_reward,
                        "action_mask": head_mask,
                    }, best_path)

    # Wait for final PPO
    if ppo_thread is not None:
        ppo_thread.join()
        if ppo_error_box[0] is not None:
            raise ppo_error_box[0]
        iter_time = time.time() - ppo_start_time if ppo_start_time > 0 else 1.0
        fps = timesteps_per_iteration / max(iter_time, 0.001)
        mean_reward = _log_iteration(
            writer, prev_global_step, ppo_stats_box[0],
            prev_component_accum, prev_buffer, reward_config,
            n_steps, fps, head_mask, prev_iteration, num_iterations,
        )
        update_stats = ppo_stats_box[0]
    else:
        update_stats = ppo_stats_box[0]

    # Print final summary
    total_time = time.time() - training_start
    print(f"\n{'='*60}")
    print(f"Training complete — {global_step:,} steps in {total_time:.1f}s")
    print(f"  Final reward: {mean_reward:.2f} | Best reward: {best_mean_reward:.2f}")
    print(f"  Final PL: {update_stats['policy_loss']:.4f} | VL: {update_stats['value_loss']:.4f} | Ent: {update_stats['entropy']:.3f}")
    print(f"  Avg FPS: {global_step / total_time:.0f}")
    game_ticks_per_env = global_step // num_envs
    game_seconds = game_ticks_per_env / 50.0  # 50 ticks/sec in real WinBolo
    total_agent_hours = (global_step / 50.0) / 3600.0
    print(f"  Simulated: {game_seconds/3600:.1f}h per agent, {total_agent_hours:.0f}h total across {num_envs} agents")
    print(f"{'='*60}\n")

    # Final save
    final_path = checkpoint_dir / f"{phase_name}_final.pt"
    torch.save({
        "model_state_dict": train_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "action_mask": head_mask,
    }, final_path)
    print(f"Final model saved: {final_path}")

    # Export ONNX for final model
    try:
        from scripts.export_onnx import export_onnx
        onnx_path = final_path.with_suffix(".onnx")
        export_onnx(train_model, str(onnx_path), device="cpu", action_mask=head_mask)
        print(f"ONNX exported: {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    # Export ONNX for best checkpoint
    best_pt = checkpoint_dir / "best" / f"{phase_name}_best.pt"
    if best_pt.exists():
        try:
            best_ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            best_model = model_from_checkpoint(best_ckpt)
            best_onnx = best_pt.with_suffix(".onnx")
            best_mask = best_ckpt.get("action_mask", head_mask)
            export_onnx(best_model, str(best_onnx), device="cpu", action_mask=best_mask)
            print(f"ONNX exported (best): {best_onnx}")
        except Exception as e:
            print(f"Best ONNX export failed: {e}")

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
