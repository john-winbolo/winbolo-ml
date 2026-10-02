<!--
Copyright (c) 2026 John Morrison.
SPDX-License-Identifier: GPL-3.0-or-later
-->

# WinBolo ML

A starting point for training agents to play classic
[WinBolo](https://github.com/john-winbolo/winbolo), the multiplayer top-down
tank game, with reinforcement learning.

It trains a PPO agent against the real game rules through `libwinbolo_gym`, a
library built from the WinBolo source that runs many games in one process and
computes rewards in C. A short curriculum takes the agent from sitting in its
starting boat to driving onto land and capturing bases. Trained models export
to ONNX, and WinBolo can load them as bots.

Everything past base capture is left for you to build: shooting pillboxes,
using the builder, fighting other tanks. [Where to go next](#where-to-go-next)
has notes on what makes that hard.

## Getting started

These steps are for Linux. You need Python 3.10 or later. A GPU makes training
about six times faster but is not required; see [Performance](#performance).

### 1. Build the gym library

Clone WinBolo and build its `winbolo_gym` target. WinBolo's `docs/BUILDING.md`
lists the packages its build needs.

```bash
git clone https://github.com/john-winbolo/winbolo.git
cd winbolo
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --target winbolo_gym -j$(nproc)
```

Copy the library and SDL3 into the root of this repository:

```bash
cp build/libwinbolo_gym.so build/libSDL3.so.0 /path/to/winbolo-ml/
```

The library also needs the OpenMP runtime, `libgomp1`, which comes with GCC.
From this repository's root, make the libraries findable:

```bash
export LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH"
```

`winbolo_gym.h` in this repository is a copy of the library's header from
WinBolo's `src/gym/`. When you update the library, copy the header across too.
The Python side checks the struct sizes and the reward list against the
library when it starts.

### 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[export,dev]"
pytest tests/
```

### 3. Train the first phase

```bash
python scripts/train.py --config configs/phase0_boat.yaml
```

Progress prints as it goes, and `tensorboard --logdir runs/` shows the reward
curves and each reward component. Checkpoints go to
`checkpoints/phase0_boat/`, with the best one in `best/`. Both the final and
the best model are exported to ONNX when training ends.

### 4. Evaluate it

```bash
python scripts/evaluate.py --model checkpoints/phase0_boat/phase0_boat_final.pt --config configs/phase0_boat.yaml -n 10
```

This plays the model's most likely action at every step, in the same
environment and with the same rewards it was trained with.

### 5. Watch it in the game

Training reward is not the same as playing well, so test every model in the
game itself. WinBolo must be built with ONNX Runtime, which is on by default.

Run it headless on a training map:

```bash
WINBOLO_ML_LOG=ml_brain_log.jsonl \
/path/to/winbolo/build/WinBoloHeadless --fast --map maps/phase0_boat.map \
  --brain checkpoints/phase0_boat/phase0_boat_final.onnx --ticks 2000
python scripts/view_ml_log.py ml_brain_log.jsonl
```

The game writes the file named by `WINBOLO_ML_LOG` as the model plays, one
line per tick, and `view_ml_log.py` draws the tank's path over the map along
with its actions and captures. The variable is read by the game and the
headless runner alike. Leave it unset and nothing is written: the log grows
by about half a kilobyte a tick for as long as the brain runs.

To play alongside it, copy the `.onnx` file into an `onnx` folder inside
the game's user brains directory, which the game creates on first run:

- macOS: `~/Library/Application Support/WinBolo/WinBolo/Brains/onnx/`
- Linux: `~/.local/share/WinBolo/WinBolo/Brains/onnx/`
- Windows: `%APPDATA%\WinBolo\WinBolo\Brains\onnx\`

It appears in the game's brain menu as `[ML] <name>`.

## The curriculum

Each phase resumes from the one before it. Long runs on small maps can
overtrain, so compare a phase's `best/` checkpoint with its final one and
resume from whichever plays better in the game.

| Phase | Config | Goal |
|---|---|---|
| 0 | `phase0_boat.yaml` | Drive off the boat onto land. The episode ends on landing. |
| 1.1 | `phase1-1_base.yaml` | Reach and capture one base from any of 16 water spawn points. |
| 1.2 | `phase1-2_base.yaml` | Capture as many of six bases on a larger island as possible. |
| 1.3 | `phase1-3_base.yaml` | As 1.2, with forest and walls in the way. |

```bash
python scripts/train.py --config configs/phase1-1_base.yaml \
  --resume checkpoints/phase0_boat/phase0_boat_final.pt
python scripts/train.py --config configs/phase1-2_base.yaml \
  --resume checkpoints/phase1-1_base/phase1-1_base_final.pt
python scripts/train.py --config configs/phase1-3_base.yaml \
  --resume checkpoints/phase1-2_base/best/phase1-2_base_best.pt
```

Each config sets its own number of parallel games and training steps. Phase
1.3 runs for 500 million steps and its final checkpoint plays worse than its
best one, so use `best/phase1-3_base_best.pt` when you go on from it.

The spawn points on the base maps are spread around the island in the water,
so the agent has to learn to navigate rather than memorise one path. All four
phases use only the accelerate and steer actions; the others are masked off.

### What to expect

From earlier runs of this curriculum, on a version of the gym from before its
event and timing fixes. Your numbers will differ, but these are the right
ballpark:

| Phase | Steps | Result |
|---|---|---|
| 0 | 5M | Reliably drives off the boat onto land. |
| 1.1 | 20M | Captures the base from most spawn points (5 of 8 tested). |
| 1.2 | 20M | Captures 3 to 4 of the 6 bases from most spawn points. |
| 1.3 | 500M | The best checkpoint captures 4 to 5 of the 6 bases; the final one had overtrained and did worse. |

## Performance

Measured on an Intel Core i9-14900KF (32 threads) with an RTX 4090, using a
Release build of `libwinbolo_gym`.

A step is one game tick. WinBolo runs 50 ticks a second, so 50 steps are one
second of play. The tables below also give how much game time passes for each
second of real time: in total across all the games running, and in each game.

The game itself is fast. Stepping games with random actions on the phase 1.2
map:

| Parallel games | Game steps per second | Game time per real second, in total | In each game |
|---|---|---|---|
| 1 | 21,000 | 7 minutes | 7 minutes (420x real time) |
| 8 | 157,000 | 52 minutes | 6.5 minutes (390x) |
| 64 | 437,000 | 2.4 hours | 2.3 minutes (137x) |

The library steps the games in parallel with OpenMP. A Debug build runs about
60% as fast.

Training is slower, because each step also runs the network, samples actions
and stores the results. Training phase 1.1 end to end:

| Setup | Training steps per second | Game time per real second, in total | In each game |
|---|---|---|---|
| 64 games, GPU (the config's default) | 8,900 | 3 minutes | 2.8 seconds (2.8x real time) |
| 256 games, GPU (`n_steps` cut to 1024) | 16,900 | 5.6 minutes | 1.3 seconds (1.3x) |
| 64 games, CPU only | 1,400 | 28 seconds | 0.4 seconds (0.4x) |

At each config's own settings:

| Phase | Steps | Game time played, in total | Training time |
|---|---|---|---|
| 0 (32 games, about 5,800 steps per second) | 5M | 28 hours | 15 minutes |
| 1.1 | 20M | 4.6 days | 40 minutes |
| 1.2 | 20M | 4.6 days | 40 minutes |
| 1.3 | 500M | 4 months | 16 hours |

The GPU will look mostly idle. The network is small, and with 64 games each
step is dominated by fixed per-step costs: launching small GPU operations,
copying actions back to the CPU and filling the rollout buffer. The game step
itself is under 10% of the time. Running more games per step spreads those
costs and scales almost linearly, so `--num-envs` is the main speed setting.
The rollout buffer holds `n_steps` times the number of games observations at
about 14 KB each, so lower `n_steps` as you raise the game count: 256 games at
the default 4096 steps would need around 15 GB.

## Configs

A config has four sections:

- `env`: the map, `max_ticks` (the episode length in game ticks, 50 to the
  second, so phase 1.1's `1200` is 24 seconds of play), the game
  type, and when an episode ends early (`end_on_land`,
  `end_on_all_pillbox_captures`).
- `training`: PPO settings, plus `action_mask` (which of the eight action
  heads are on) and optionally `dynamic_mask` and `freeze_layers`.
- `rewards`: a preset from `winbolo_ml/rewards.py` and per-weight overrides.
- `network`: encoder and hidden layer sizes.

### Rewards

Rewards are computed in C inside the library. `RewardConfig` in
`winbolo_ml/rewards.py` holds a weight for each of the 59 reward components,
covering survival, combat, pillboxes, bases, the builder, resources, mining
and territory. A config picks a preset, such as `phase1_movement`, and
overrides individual weights:

```yaml
rewards:
  phase: phase1_movement
  overrides:
    base_captured: 50.0
    base_proximity: 0.05
```

The weights are passed to C by position, so `RewardConfig`'s fields have to
stay in the order of the `RC_*` indices in `winbolo_gym.h`. A test checks
this against the header. To add a reward, add it to the library first, then
add its weight at the same position here.

TensorBoard shows each component under `reward_raw/`, which is the quickest
way to see what the agent is actually being paid for.

### Training flags

| Flag | Description |
|---|---|
| `--config` | Path to config YAML |
| `--num-envs` | Override the config's number of parallel games |
| `--total-timesteps` | Override the config's training steps |
| `--map` | Override the config's map |
| `--resume` | Resume from a saved checkpoint |
| `--lib-path` | Path to `libwinbolo_gym.so` (default: `./libwinbolo_gym.so`) |
| `--game-type` | `open`, `tournament` or `strict` (default from config, else strict) |
| `--reward-phase` | Override the config's reward preset |
| `--device` | `cuda` or `cpu` (default: auto) |
| `--tensorboard-log` | TensorBoard log directory (default: `runs/`) |
| `--checkpoint-dir` | Checkpoint directory (default: `checkpoints/`) |

## Exporting to ONNX

Training exports the final and best models itself. To export another
checkpoint:

```bash
python scripts/export_onnx.py --model checkpoints/phase1-2_base/best/phase1-2_base_best.pt --output my_bot.onnx
```

The config's `action_mask` is saved in each checkpoint and baked into the
exported weights, so heads that were off in training stay off in the game.

## How it works

The agent observes the game through:

- **Terrain** (29x29x2): the terrain around the tank and a mine overlay
- **Scalars** (26): armour, shells, mines, trees, speed, direction, boat and
  builder state, and the share of pillboxes and bases each side owns
- **Entities** (up to 256x7): tanks, shells, builders, explosions, pillboxes
  and bases, relative to the tank
- **Sounds** (up to 32x4): sounds the tank can hear and where they came from

Actions are `MultiDiscrete([3, 3, 2, 2, 3, 6, 29, 29])`: accelerate, steer,
shoot, lay mine, gun range, build action, and the build target's x and y.

The policy network encodes terrain with a CNN, scalars with an MLP, and
entities and sounds with DeepSets encoders, fuses them, and feeds PPO's actor
and critic heads.

One training step is one game tick. The library refuses to run with anything
but the classic game rules, so models always train on the classic game.

## Where to go next

The obvious next step is pillbox combat: shoot a pillbox down, drive over it
to capture it, and carry it. It is much harder than base capture. Some things
earlier attempts ran into:

- **Shaping can drown the objective.** Small per-tick rewards (approach,
  facing, staying on land) add up over thousands of ticks and can outweigh
  the reward for actually capturing something. Keep per-tick shaping small,
  and watch the `reward_raw/` components rather than the total.
- **Training reward is not game performance.** A high reward can come from
  shaping alone. Check every model in the game, and check whether an episode
  ended early with a capture or simply ran out of ticks.
- **Ammunition runs out.** A tank has 40 shells, and agents that fire while
  still on the boat arrive with none. `dynamic_mask: {no_shoot_on_boat: true}`
  blocks shooting from the boat during training, but the game does not apply
  it, so a model can still fire from the boat when it plays for real.
- **Small maps wreck navigation.** Training on a tiny map overwrites what the
  agent learned about getting around, and the checkpoint does not recover on
  bigger maps.
- **Unmasking an action head needs a low learning rate.** Turning on shooting
  at the usual 0.0003 tends to destroy the existing navigation; 0.00003 to
  0.0001 keeps it. An entropy coefficient around 0.02 worked best.
- **`base_proximity` rewards being near any base**, including ones the agent
  already owns, so agents camp on captured bases. Turn it off once there is
  more than one base to take.

## Docker

```bash
# Build (libwinbolo_gym.so and libSDL3.so.0 must be in the project root)
docker build -t winbolo-ml .

# Train, keeping checkpoints and logs in this directory
docker run --gpus all -v "$PWD:/app" winbolo-ml scripts/train.py --config configs/phase0_boat.yaml
```

## Project structure

```
configs/          Curriculum configs (env, training, rewards, network)
maps/             Training maps
scripts/
  train.py        PPO training with checkpoints and ONNX export
  evaluate.py     Run a trained model and print stats
  export_onnx.py  ONNX export, with the action mask baked in
  view_ml_log.py  Draw the log the game writes when it runs a model
tests/            Space definitions, and reward weights against the C library
winbolo_ml/
  env_batch.py    Batched environment over libwinbolo_gym
  gym_types.py    ctypes mirror of winbolo_gym.h
  features.py     Observation and action spaces
  rewards.py      Reward weights and presets
  network.py      Policy and value network
  utils.py        Config loading
winbolo_gym.h     The library's C header, copied from WinBolo
```

## Licence

WinBolo ML is released under the GNU General Public License, version 3 or (at
your option) any later version. See [LICENSE](LICENSE).
