#!/usr/bin/env python3
# Copyright (c) 2026 John Morrison.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Export trained WinBolo V3 model to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from winbolo_ml.network import MultiDiscreteDistribution, WinBoloModel, model_from_checkpoint


class InferenceWrapper(nn.Module):
    """Wraps WinBoloModel to return only logits (no value head) for ONNX export."""

    def __init__(self, model: WinBoloModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        terrain: torch.Tensor,
        scalar: torch.Tensor,
        entities: torch.Tensor,
        entity_mask: torch.Tensor,
        sounds: torch.Tensor,
        sound_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward_inference(
            terrain, scalar, entities, entity_mask, sounds, sound_mask
        )


def bake_action_mask(model: WinBoloModel, action_mask: list[bool]) -> None:
    """Modify model weights in-place so masked heads always output HEAD_DEFAULTS via argmax.

    For each masked head, zeros the corresponding rows in the linear layer weights
    and sets the bias so only the default action index has a positive value.
    This is baked into the weights — no runtime logic needed.
    """
    defaults = MultiDiscreteDistribution.HEAD_DEFAULTS
    head_names = ["accel", "turn", "shoot", "mine", "gun_range", "build_act", "build_rx", "build_ry"]

    # Head index → (layer, row_start, row_count)
    # action_head(13): accel[0:3], turn[3:6], shoot[6:8], mine[8:10], gun_range[10:13]
    # build_type_head(6): build_action[0:6]
    # build_coord_head(58): build_rx[0:29], build_ry[29:58]
    head_map = [
        (model.action_head, 0, 3),     # 0: accel
        (model.action_head, 3, 3),     # 1: turn
        (model.action_head, 6, 2),     # 2: shoot
        (model.action_head, 8, 2),     # 3: mine
        (model.action_head, 10, 3),    # 4: gun_range
        (model.build_type_head, 0, 6), # 5: build_action
        (model.build_coord_head, 0, 29),  # 6: build_rx
        (model.build_coord_head, 29, 29), # 7: build_ry
    ]

    with torch.no_grad():
        for i, (layer, row_start, row_count) in enumerate(head_map):
            if action_mask[i]:
                continue
            # Zero all weights for this head's rows so input doesn't matter
            layer.weight.data[row_start:row_start + row_count, :] = 0.0
            # Set bias: -10 for all, +10 for the default index
            layer.bias.data[row_start:row_start + row_count] = -10.0
            layer.bias.data[row_start + defaults[i]] = 10.0
            print(f"  Baked {head_names[i]}: default={defaults[i]}, "
                  f"bias={layer.bias.data[row_start:row_start + row_count].tolist()}")


def export_onnx(model: WinBoloModel, output_path: str, device: str = "cpu",
                action_mask: list[bool] | None = None) -> None:
    """Export a WinBoloModel to ONNX format."""
    model = model.to(device)
    model.eval()

    if action_mask is not None:
        print("Baking action mask into model weights:")
        bake_action_mask(model, action_mask)

        # Verify: run model on zeros and check argmax per head
        with torch.no_grad():
            dummy_t = torch.zeros(1, WBGYM_SPATIAL_SIZE, WBGYM_SPATIAL_SIZE, 2, device=device)
            dummy_s = torch.zeros(1, WBGYM_NUM_SCALARS, device=device)
            dummy_e = torch.zeros(1, WBGYM_MAX_ENTITIES, ENTITY_FEATURES, device=device)
            dummy_em = torch.zeros(1, WBGYM_MAX_ENTITIES, device=device)
            dummy_snd = torch.zeros(1, WBGYM_MAX_SOUNDS, SOUND_FEATURES, device=device)
            dummy_sm = torch.zeros(1, WBGYM_MAX_SOUNDS, device=device)
            logits = model.forward_inference(dummy_t, dummy_s, dummy_e, dummy_em, dummy_snd, dummy_sm)
            logits = logits[0].cpu().numpy()
            head_names = ["accel", "turn", "shoot", "mine", "gun_range", "build_act", "build_rx", "build_ry"]
            offset = 0
            print("Verification (argmax per head on zero input):")
            for i, size in enumerate(ACTION_DIMS):
                idx = int(np.argmax(logits[offset:offset + size]))
                tag = " (masked)" if not action_mask[i] else ""
                print(f"  {head_names[i]}: {idx}{tag}")
                offset += size

    wrapper = InferenceWrapper(model)
    wrapper = wrapper.to(device)
    wrapper.eval()

    dummy_terrain = torch.zeros(1, WBGYM_SPATIAL_SIZE, WBGYM_SPATIAL_SIZE, 2, device=device)
    dummy_scalar = torch.zeros(1, WBGYM_NUM_SCALARS, device=device)
    dummy_entities = torch.zeros(1, WBGYM_MAX_ENTITIES, ENTITY_FEATURES, device=device)
    dummy_entity_mask = torch.zeros(1, WBGYM_MAX_ENTITIES, device=device)
    dummy_sounds = torch.zeros(1, WBGYM_MAX_SOUNDS, SOUND_FEATURES, device=device)
    dummy_sound_mask = torch.zeros(1, WBGYM_MAX_SOUNDS, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_terrain, dummy_scalar, dummy_entities, dummy_entity_mask,
         dummy_sounds, dummy_sound_mask),
        output_path,
        input_names=["terrain", "scalar", "entities", "entity_mask", "sounds", "sound_mask"],
        output_names=["logits"],
        dynamic_axes=None,  # fixed batch size 1 for inference
        opset_version=18,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export WinBolo V3 model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX file path (default: same name with .onnx)")
    parser.add_argument("--action-mask", type=str, default=None,
                        help="Comma-separated 8 bools (e.g. true,true,false,false,false,false,false,false). "
                             "Auto-detected from checkpoint if saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is None:
        args.output = str(Path(args.model).with_suffix(".onnx"))

    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint)

    # Determine action mask
    action_mask = None
    if args.action_mask is not None:
        action_mask = [v.strip().lower() == "true" for v in args.action_mask.split(",")]
        assert len(action_mask) == 8, f"action_mask must have 8 values, got {len(action_mask)}"
    elif "action_mask" in checkpoint and checkpoint["action_mask"] is not None:
        action_mask = checkpoint["action_mask"]

    if action_mask is not None:
        active = [i for i, v in enumerate(action_mask) if v]
        masked = [i for i, v in enumerate(action_mask) if not v]
        head_names = ["accel", "turn", "shoot", "mine", "gun_range", "build_act", "build_rx", "build_ry"]
        print(f"Action mask: active={[head_names[i] for i in active]}, "
              f"defaults={[head_names[i] for i in masked]}")
    else:
        print("No action mask — all heads active")

    print(f"Exporting to {args.output}")
    export_onnx(model, args.output, device="cpu", action_mask=action_mask)

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output)
        result = sess.run(None, {
            "terrain": np.zeros((1, WBGYM_SPATIAL_SIZE, WBGYM_SPATIAL_SIZE, 2), dtype=np.float32),
            "scalar": np.zeros((1, WBGYM_NUM_SCALARS), dtype=np.float32),
            "entities": np.zeros((1, WBGYM_MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32),
            "entity_mask": np.zeros((1, WBGYM_MAX_ENTITIES), dtype=np.float32),
            "sounds": np.zeros((1, WBGYM_MAX_SOUNDS, SOUND_FEATURES), dtype=np.float32),
            "sound_mask": np.zeros((1, WBGYM_MAX_SOUNDS), dtype=np.float32),
        })
        logits = result[0][0]
        assert logits.shape == (TOTAL_LOGITS,), f"Expected ({TOTAL_LOGITS},), got {logits.shape}"

        # Decode sample action via argmax at each head
        actions = []
        offset = 0
        head_names = ["accel", "turn", "shoot", "mine", "gun_range", "build_act", "build_rx", "build_ry"]
        for i, size in enumerate(ACTION_DIMS):
            idx = int(np.argmax(logits[offset:offset + size]))
            actions.append(idx)
            offset += size
        print(f"Verification OK — logits shape: {result[0].shape}")
        for i, (name, val) in enumerate(zip(head_names, actions)):
            masked_tag = " (masked→default)" if action_mask and not action_mask[i] else ""
            print(f"  {name}: {val}{masked_tag}")
    except ImportError:
        print("onnxruntime not installed — skipping verification")

    print("Done.")


if __name__ == "__main__":
    main()
