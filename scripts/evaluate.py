#!/usr/bin/env python3
# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluation script for trained WinBolo RL agent."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import OBS_KEYS, make_reward_config
from winbolo_ml.env_batch import WinBoloBatchEnv
from winbolo_ml.gym_types import (
    GAME_TYPE_OPEN,
    GAME_TYPE_STRICT_TOURNAMENT,
    GAME_TYPE_TOURNAMENT,
)
from winbolo_ml.network import greedy_actions, load_model
from winbolo_ml.rewards import RewardConfig
from winbolo_ml.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained WinBolo agent")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a .pt checkpoint written by train.py")
    parser.add_argument("--config", type=str, required=True,
                        help="The config YAML the model was trained with")
    parser.add_argument("--map", type=str, default=None,
                        help="Override the config's map")
    parser.add_argument("-n", "--num-episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="Override the config's max ticks per episode")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-tick details")
    parser.add_argument("--lib-path", type=str, default="./libwinbolo_gym.so",
                        help="Path to libwinbolo_gym.so")
    parser.add_argument("--game-type", type=str, default=None,
                        choices=["open", "tournament", "strict"],
                        help="Override the config's game type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    head_mask = cfg.get("training", {}).get("action_mask", None)

    _game_type_map = {"open": GAME_TYPE_OPEN, "tournament": GAME_TYPE_TOURNAMENT, "strict": GAME_TYPE_STRICT_TOURNAMENT}
    if args.game_type is not None:
        game_type = _game_type_map[args.game_type]
    else:
        game_type = env_cfg.get("game_type", GAME_TYPE_STRICT_TOURNAMENT)

    model = load_model(args.model, device="cpu")

    # The same env and reward config training uses, so the rewards printed
    # here are the ones the model was trained on.
    env = WinBoloBatchEnv(
        lib_path=args.lib_path,
        map_path=args.map or env_cfg["map"],
        num_envs=1,
        max_ticks=args.max_ticks or env_cfg.get("max_ticks", 6000),
        game_type=game_type,
        reward_config=make_reward_config(cfg.get("rewards", {})),
        end_on_all_pillbox_captures=env_cfg.get("end_on_all_pillbox_captures", False),
        end_on_land=env_cfg.get("end_on_land", False),
    )

    # step() returns each reward component as a column, in RewardConfig order.
    column = {f.name: i for i, f in enumerate(fields(RewardConfig))}
    results = []

    print(f"{'Ep':>4} {'Ticks':>6} {'Reward':>8} {'Kills':>6} {'Deaths':>6} {'Pills':>6} {'Bases':>6}")
    print("-" * 52)

    obs = env.reset()
    for ep in range(args.num_episodes):
        totals = {"reward": 0.0, "kill": 0.0, "death": 0.0, "pill_captured": 0.0, "base_captured": 0.0}
        ticks = 0
        done = False

        while not done:
            with torch.no_grad():
                logits, _ = model(**{k: torch.tensor(obs[k]) for k in OBS_KEYS})
            actions = greedy_actions(logits, head_mask).numpy()
            # The env resets itself when an episode ends, so obs is then the
            # first observation of the next one.
            obs, rewards, dones, components = env.step(actions)
            done = bool(dones[0])
            ticks += 1

            totals["reward"] += float(rewards[0])
            for name in ("kill", "death", "pill_captured", "base_captured"):
                totals[name] += float(components[0, column[name]])

            if args.verbose:
                print(f"  tick={ticks} reward={float(rewards[0]):.4f}")

        ep_result = {
            "episode": ep + 1,
            "ticks": ticks,
            "reward": totals["reward"],
            "kills": int(totals["kill"]),
            "deaths": int(totals["death"]),
            "pillboxes_captured": int(totals["pill_captured"]),
            "bases_captured": int(totals["base_captured"]),
        }
        results.append(ep_result)

        print(f"{ep_result['episode']:>4} {ep_result['ticks']:>6} "
              f"{ep_result['reward']:>8.2f} {ep_result['kills']:>6} "
              f"{ep_result['deaths']:>6} {ep_result['pillboxes_captured']:>6} "
              f"{ep_result['bases_captured']:>6}")

    # Summary
    print("\n--- Summary ---")
    rewards = [r["reward"] for r in results]
    ticks = [r["ticks"] for r in results]
    kills = sum(r["kills"] for r in results)
    deaths = sum(r["deaths"] for r in results)
    pills = sum(r["pillboxes_captured"] for r in results)
    bases = sum(r["bases_captured"] for r in results)

    print(f"Episodes:   {len(results)}")
    print(f"Avg reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Avg ticks:  {np.mean(ticks):.0f}")
    print(f"Total kills/deaths: {kills}/{deaths}")
    print(f"Total pillboxes captured: {pills}")
    print(f"Total bases captured: {bases}")

    env.close()


if __name__ == "__main__":
    main()
