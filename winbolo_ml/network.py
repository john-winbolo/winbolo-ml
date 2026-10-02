# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Custom policy/value network for WinBolo V3 observation and action space.

Architecture:
  terrain [B,29,29,2]  → CNN encoder    → [B, D]
  scalar  [B,26]       → MLP encoder    → [B, D]
  entities [B,256,7]   → DeepSets       → [B, D]  (masked by entity_mask)
  sounds  [B,32,4]     → DeepSets       → [B, D]  (masked by sound_mask)

  concat → [B, 4*D] → fusion MLP → shared_embed [B, H]

  shared_embed → action_head      → 13 logits (accel/turn/shoot/mine/gun_range)
  shared_embed → build_type_head  → 6 logits  (build_action)
  shared_embed → build_coord_head → 58 logits (build_rx + build_ry)
  shared_embed → value_head       → 1 scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from winbolo_ml.features import (
    ACTION_DIMS,
    ENTITY_FEATURES,
    SOUND_FEATURES,
    TOTAL_LOGITS,
    WBGYM_MAX_ENTITIES,
    WBGYM_MAX_SOUNDS,
    WBGYM_NUM_SCALARS,
    WBGYM_SPATIAL_SIZE,
)


class DeepSetsEncoder(nn.Module):
    """DeepSets: per-element MLP → masked mean pool → post MLP."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        mask: [B, N] — 1.0 for valid elements, 0.0 for padding
        Returns: [B, output_dim]
        """
        h = self.phi(x)  # [B, N, hidden]
        # Masked mean pool
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        h = h * mask_expanded
        count = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
        pooled = h.sum(dim=1) / count  # [B, hidden]
        return self.rho(pooled)


class TerrainCNN(nn.Module):
    """Small CNN for 29x29x2 terrain grid."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 15x15
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64, output_dim)

    def forward(self, terrain: torch.Tensor) -> torch.Tensor:
        """terrain: [B, 29, 29, 2] → [B, output_dim]"""
        x = terrain.permute(0, 3, 1, 2)  # [B, 2, 29, 29]
        x = self.conv(x)
        return F.relu(self.proj(x))


class WinBoloModel(nn.Module):
    """Combined policy + value network for WinBolo V3.

    Outputs:
      logits: [B, 77] — concatenated action logits for all heads
      value:  [B, 1]  — state value estimate
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        hidden_dim: int = 256,
        entity_hidden: int = 64,
        sound_hidden: int = 32,
    ) -> None:
        super().__init__()

        # Encoders
        self.terrain_enc = TerrainCNN(encoder_dim)
        self.scalar_enc = nn.Sequential(
            nn.Linear(WBGYM_NUM_SCALARS, 64),
            nn.ReLU(),
            nn.Linear(64, encoder_dim),
            nn.ReLU(),
        )
        self.entity_enc = DeepSetsEncoder(ENTITY_FEATURES, entity_hidden, encoder_dim)
        self.sound_enc = DeepSetsEncoder(SOUND_FEATURES, sound_hidden, encoder_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(4 * encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action heads
        # Movement: accel(3) + turn(3) + shoot(2) + mine(2) + gun_range(3) = 13
        self.action_head = nn.Linear(hidden_dim, 13)
        # Build type: build_action(6)
        self.build_type_head = nn.Linear(hidden_dim, 6)
        # Build coords: build_rx(29) + build_ry(29) = 58
        self.build_coord_head = nn.Linear(hidden_dim, 58)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        terrain: torch.Tensor,
        scalar: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        sounds: torch.Tensor,
        sound_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits: [B, 77] concatenated
          value:  [B, 1]
        """
        t_feat = self.terrain_enc(terrain)
        s_feat = self.scalar_enc(scalar)
        e_feat = self.entity_enc(entities, entity_mask)
        snd_feat = self.sound_enc(sounds, sound_mask)

        fused = self.fusion(torch.cat([t_feat, s_feat, e_feat, snd_feat], dim=1))

        action_logits = self.action_head(fused)       # [B, 13]
        build_type_logits = self.build_type_head(fused)  # [B, 6]
        build_coord_logits = self.build_coord_head(fused)  # [B, 58]

        logits = torch.cat([action_logits, build_type_logits, build_coord_logits], dim=1)  # [B, 77]
        value = self.value_head(fused)  # [B, 1]

        return logits, value

    def forward_inference(
        self,
        terrain: torch.Tensor,
        scalar: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        sounds: torch.Tensor,
        sound_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-only forward: returns logits [B, 77] only (for ONNX export)."""
        logits, _ = self.forward(terrain, scalar, entities, entity_mask, sounds, sound_mask)
        return logits


class MultiDiscreteDistribution:
    """Handles sampling and log-prob computation for the multi-head action space.

    Logit layout: [accel(3), turn(3), shoot(2), mine(2), gun_range(3), build_action(6), build_rx(29), build_ry(29)]

    Uses raw tensor ops (no torch.distributions.Categorical) with pre-computed
    log_softmax to minimize overhead.

    Args:
        logits: [B, 77] concatenated logits.
        head_mask: which heads are active. Accepts either:
            - list of 8 bools (static, same mask for all envs), OR
            - [B, 8] bool tensor (per-env dynamic mask)
            Inactive heads are excluded from log_prob and entropy and sample()
            returns HEAD_DEFAULTS for those heads.
    """

    # Default forced values for each head when masked (action-space indices)
    # accel=1 (coast), turn=1 (straight), shoot=0, mine=0,
    # gun_range=1 (no adjust), build_action=0, build_rx=14 (center), build_ry=14 (center)
    HEAD_DEFAULTS = [1, 1, 0, 0, 1, 0, 14, 14]

    def __init__(self, logits: torch.Tensor, head_mask: list[bool] | torch.Tensor | None = None) -> None:
        self.logits = logits
        self.dims = ACTION_DIMS
        self.splits = torch.split(logits, self.dims, dim=-1)
        B = logits.shape[0]

        # Normalize head_mask to a [B, 8] bool tensor
        if head_mask is None:
            self.head_mask = torch.ones(B, 8, dtype=torch.bool, device=logits.device)
        elif isinstance(head_mask, torch.Tensor):
            if head_mask.dim() == 1:
                self.head_mask = head_mask.unsqueeze(0).expand(B, -1)
            else:
                self.head_mask = head_mask
        else:
            # list of bools
            self.head_mask = torch.tensor(head_mask, dtype=torch.bool, device=logits.device).unsqueeze(0).expand(B, -1)

        # Determine if all envs share the same mask (common case — avoid per-env work)
        self._uniform = bool((self.head_mask == self.head_mask[0:1]).all())

        # Pre-compute log_softmax for all 8 heads (cheap, avoids branching later)
        self._log_softmax: list[torch.Tensor] = []
        for i in range(8):
            self._log_softmax.append(F.log_softmax(self.splits[i], dim=-1))

    def sample(self) -> torch.Tensor:
        """Sample one action per head using Gumbel-max trick. Returns [B, 8] int64 tensor."""
        B = self.logits.shape[0]
        device = self.logits.device
        defaults = torch.tensor(self.HEAD_DEFAULTS, dtype=torch.long, device=device)
        actions = defaults.unsqueeze(0).expand(B, -1).clone()

        for head_idx in range(8):
            active = self.head_mask[:, head_idx]  # [B] bool
            if not active.any():
                continue
            # Gumbel-max trick: argmax(logits + gumbel_noise) ~ Categorical(softmax(logits))
            u = torch.rand_like(self.splits[head_idx]).clamp_(1e-10, 1.0)
            gumbel = -(-u.log()).log()
            sampled = (self.splits[head_idx] + gumbel).argmax(dim=-1)
            actions[:, head_idx] = torch.where(active, sampled, actions[:, head_idx])

        return actions

    def log_prob(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-probs split into movement, build_type, build_coord.

        Returns:
          movement_logprob: [B] — sum of logprobs for accel/turn/shoot/mine/gun_range
          build_type_logprob: [B] — logprob for build_action
          build_coord_logprob: [B] — sum of logprobs for build_rx + build_ry
        """
        B = actions.shape[0]
        device = actions.device
        movement_lp = torch.zeros(B, device=device)
        build_type_lp = torch.zeros(B, device=device)
        build_coord_lp = torch.zeros(B, device=device)

        for head_idx in range(8):
            active = self.head_mask[:, head_idx].float()  # [B], 1.0 or 0.0
            lp = self._log_softmax[head_idx].gather(-1, actions[:, head_idx:head_idx + 1]).squeeze(-1)
            lp = lp * active  # zero out masked envs
            if head_idx < 5:
                movement_lp = movement_lp + lp
            elif head_idx == 5:
                build_type_lp = build_type_lp + lp
            else:
                build_coord_lp = build_coord_lp + lp

        return movement_lp, build_type_lp, build_coord_lp

    def entropy(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute entropy split into movement, build_type, build_coord.

        Returns three [B] tensors.
        """
        B = self.logits.shape[0]
        device = self.logits.device
        movement_ent = torch.zeros(B, device=device)
        build_type_ent = torch.zeros(B, device=device)
        build_coord_ent = torch.zeros(B, device=device)

        for head_idx in range(8):
            active = self.head_mask[:, head_idx].float()  # [B], 1.0 or 0.0
            log_p = self._log_softmax[head_idx]
            p = log_p.exp()
            ent = -(p * log_p).sum(dim=-1) * active  # zero out masked envs
            if head_idx < 5:
                movement_ent = movement_ent + ent
            elif head_idx == 5:
                build_type_ent = ent
            else:
                build_coord_ent = build_coord_ent + ent

        return movement_ent, build_type_ent, build_coord_ent

    def total_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Total log-prob with build-coord masking. Returns [B]."""
        movement_lp, build_type_lp, build_coord_lp = self.log_prob(actions)
        build_mask = (actions[:, 5] > 0).float()
        return movement_lp + build_type_lp + build_coord_lp * build_mask

    def total_entropy(self, actions: torch.Tensor) -> torch.Tensor:
        """Total entropy with build-coord masking. Returns [B]."""
        movement_ent, build_type_ent, build_coord_ent = self.entropy()
        build_mask = (actions[:, 5] > 0).float()
        return movement_ent + build_type_ent + build_coord_ent * build_mask


def model_from_checkpoint(checkpoint: dict, device: str | torch.device = "cpu") -> WinBoloModel:
    """Build a WinBoloModel from a checkpoint dict written by scripts/train.py.

    The checkpoint holds only the weights, so the encoder and hidden sizes are
    read off them rather than assumed, and a model trained with a config's
    non-default sizes loads as it was trained.
    """
    state = checkpoint["model_state_dict"]
    model = WinBoloModel(
        encoder_dim=state["scalar_enc.2.weight"].shape[0],
        hidden_dim=state["fusion.0.weight"].shape[0],
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_model(checkpoint_path: str, device: str | torch.device = "cpu") -> WinBoloModel:
    """Load a WinBoloModel from a checkpoint file written by scripts/train.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return model_from_checkpoint(checkpoint, device)


def greedy_actions(logits: torch.Tensor, head_mask: list[bool] | None = None) -> torch.Tensor:
    """The most likely action for each head. Returns [B, 8] int tensor.

    Heads the training config masked off get the same fixed values
    MultiDiscreteDistribution.sample() gives them.
    """
    head_mask = head_mask or [True] * len(ACTION_DIMS)
    actions = []
    for i, split in enumerate(torch.split(logits, ACTION_DIMS, dim=-1)):
        if head_mask[i]:
            actions.append(split.argmax(dim=-1))
        else:
            actions.append(torch.full((logits.shape[0],),
                                      MultiDiscreteDistribution.HEAD_DEFAULTS[i],
                                      dtype=torch.long, device=logits.device))
    return torch.stack(actions, dim=-1)
