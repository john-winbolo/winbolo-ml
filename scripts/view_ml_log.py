#!/usr/bin/env python3
# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Visualize ml_brain_log.jsonl — renders ASCII map with tank path, bases, pills."""

import json
import sys
from pathlib import Path

# Terrain type names (index matches BBUILDING=0, BRIVER=1, ... from brain.h)
TERRAIN_NAMES = [
    "building", "river", "swamp", "crater", "road", "forest",
    "rubble", "grass", "halfbuild", "boat", "deepsea", "base", "pillbox",
    "unknown",
]

# Single-char display for terrain
TERRAIN_CHAR = {
    0: "#",   # building/wall
    1: "~",   # river
    2: ",",   # swamp
    3: ".",   # crater
    4: "+",   # road
    5: "f",   # forest
    6: ".",   # rubble
    7: " ",   # grass
    8: "#",   # half building
    9: "B",   # boat
    10: "~",  # deep sea
    11: "R",  # refbase
    12: "P",  # pillbox
    13: "?",  # unknown
}


def load_log(path):
    map_data = None
    world_data = None
    ticks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["type"] == "map":
                map_data = rec
            elif rec["type"] == "world":
                world_data = rec
            elif rec["type"] == "tick":
                ticks.append(rec)
    return map_data, world_data, ticks


def render_map(map_data, world_data, ticks):
    """Render ASCII map with tank path overlay."""
    x1, y1, x2, y2 = map_data["x1"], map_data["y1"], map_data["x2"], map_data["y2"]
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    # Parse terrain into grid
    grid = []
    for row_hex in map_data["rows"]:
        row = []
        for i in range(0, len(row_hex), 2):
            raw = int(row_hex[i:i+2], 16)
            terrain = raw & 0x0F
            has_mine = (raw & 0x80) != 0
            row.append((terrain, has_mine))
        grid.append(row)

    # Build display grid
    display = []
    for r in range(h):
        row = []
        for c in range(w):
            t, mine = grid[r][c]
            ch = TERRAIN_CHAR.get(t, "?")
            row.append(ch)
        display.append(row)

    # Mark bases
    if world_data:
        owner_char = {0: "R", 1: "r", 2: "E", 3: "A"}  # neutral, self, enemy, ally
        for b in world_data.get("bases", []):
            bx, by = b["x"] - x1, b["y"] - y1
            if 0 <= bx < w and 0 <= by < h:
                display[by][bx] = owner_char.get(b["owner"], "R")

        # Mark pillboxes
        pill_char = {0: "P", 1: "p", 2: "Q", 3: "a"}
        for p in world_data.get("pills", []):
            px, py = p["x"] - x1, p["y"] - y1
            if 0 <= px < w and 0 <= py < h:
                display[py][px] = pill_char.get(p["owner"], "P")

    # Overlay tank path
    if ticks:
        # Show path with dots, start with S, end with E
        path_positions = []
        for tick in ticks:
            mx, my = tick["mx"] - x1, tick["my"] - y1
            if 0 <= mx < w and 0 <= my < h:
                path_positions.append((mx, my))

        # Mark path (use density — more visits = brighter)
        visit_count = {}
        for mx, my in path_positions:
            visit_count[(mx, my)] = visit_count.get((mx, my), 0) + 1

        density_chars = " .·:*#@"
        max_visits = max(visit_count.values()) if visit_count else 1
        for (mx, my), count in visit_count.items():
            idx = min(len(density_chars) - 1, int(count / max_visits * (len(density_chars) - 1)) + 1)
            display[my][mx] = density_chars[idx]

        # Mark start and end
        if path_positions:
            sx, sy = path_positions[0]
            display[sy][sx] = "S"
            ex, ey = path_positions[-1]
            display[ey][ex] = "E"

    # Print with coordinates
    print(f"\nMap: ({x1},{y1}) to ({x2},{y2})  [{w}x{h}]")
    print(f"Legend: S=start E=end R=base(neutral) r=base(own) P=pill(neutral) p=pill(own)")
    print(f"        #=wall ~=water f=forest +=road ' '=grass .·:*@=path density\n")

    # Column headers
    header = "    "
    for c in range(w):
        header += str((x1 + c) % 10)
    print(header)

    for r in range(h):
        row_str = f"{y1 + r:3d} " + "".join(display[r])
        print(row_str)
    print()


def print_summary(ticks):
    """Print movement summary stats."""
    if not ticks:
        print("No ticks recorded.")
        return

    print(f"=== Tick Summary ({len(ticks)} ticks) ===")
    first, last = ticks[0], ticks[-1]
    print(f"Start: ({first['mx']},{first['my']}) dir={first['dir']} spd={first['spd']} boat={first['boat']}")
    print(f"End:   ({last['mx']},{last['my']}) dir={last['dir']} spd={last['spd']} boat={last['boat']}")
    print(f"Dead:  {last.get('dead', 0)}")

    # Action distribution
    accel_counts = {0: 0, 1: 0, 2: 0}
    steer_counts = {0: 0, 1: 0, 2: 0}
    for t in ticks:
        accel_counts[t["accel"]] = accel_counts.get(t["accel"], 0) + 1
        steer_counts[t["steer"]] = steer_counts.get(t["steer"], 0) + 1

    print(f"\nAccel:  decel={accel_counts[0]}  coast={accel_counts[1]}  accel={accel_counts[2]}")
    print(f"Steer:  left={steer_counts[0]}  straight={steer_counts[1]}  right={steer_counts[2]}")

    # Base ownership changes
    base_changes = []
    prev_bases = None
    for t in ticks:
        bases = t.get("bases", [])
        if prev_bases is not None and bases != prev_bases:
            base_changes.append((t["t"], prev_bases, bases))
        prev_bases = bases
    if base_changes:
        print(f"\nBase ownership changes:")
        for tick, old, new in base_changes:
            print(f"  tick {tick}: {old} -> {new}")

    # Direction over time (sampled)
    print(f"\nDirection over time (sampled every {max(1, len(ticks)//20)} ticks):")
    step = max(1, len(ticks) // 20)
    dir_line = ""
    for i in range(0, len(ticks), step):
        t = ticks[i]
        # Direction 0-255 mapped to compass
        d = t["dir"]
        arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"
        compass = arrows.split()
        idx = int((d + 16) / 32) % 8
        dir_line += f"  t{t['t']:>5d}: dir={d:>3d}({compass[idx]}) spd={t['spd']:>2d} @({t['mx']},{t['my']})\n"
    print(dir_line)


def print_entities(ticks, tick_num=None):
    """Show entities at a specific tick or last tick."""
    if tick_num is not None:
        tick = next((t for t in ticks if t["t"] == tick_num), None)
        if not tick:
            print(f"Tick {tick_num} not found.")
            return
    else:
        tick = ticks[-1] if ticks else None

    if not tick:
        return

    ent_types = {0: "tank", 1: "shell", 2: "pill", 3: "base", 4: "lgm", 5: "para", 6: "expl"}
    alleg_names = {-1: "enemy", 0: "neutral", 1: "self", 2: "ally"}

    print(f"\n=== Entities at tick {tick['t']} ===")
    for e in tick.get("ents", []):
        etype = ent_types.get(e["t"], f"unk({e['t']})")
        alleg = alleg_names.get(e["a"], f"unk({e['a']})")
        print(f"  {etype:>6s} ({alleg:>7s}) at rel ({e['rx']:+.1f}, {e['ry']:+.1f}) str={e['str']:.2f}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ml_brain_log.jsonl> [--tick N]")
        sys.exit(1)

    log_path = sys.argv[1]
    tick_num = None
    if "--tick" in sys.argv:
        idx = sys.argv.index("--tick")
        tick_num = int(sys.argv[idx + 1])

    map_data, world_data, ticks = load_log(log_path)

    if map_data:
        render_map(map_data, world_data, ticks)
    else:
        print("No map data in log.")

    print_summary(ticks)

    if tick_num is not None:
        print_entities(ticks, tick_num)
    elif ticks:
        print_entities(ticks)


if __name__ == "__main__":
    main()
