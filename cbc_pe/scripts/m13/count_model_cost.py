"""
Parameter and MAC count of a model built from a training config (CPU, no data).

One forward pass on zeros of shape (1, n_detectors, 16384) in eval mode:
- Conv1d / Linear MACs through forward hooks.
- nn.MultiheadAttention (its projections are applied functionally, so Linear hooks do not
  see them): 4*T*d^2 for the q/k/v/out projections plus 2*T^2*d for scores and weighted sum.
- Output shape of each top-level module (and of each element of a top-level ModuleList).

Usage: python scripts/m13/count_model_cost.py --config configs/experiments/m13/<cfg>.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models import network  # noqa: E402

INPUT_LENGTH = 16384


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text())
    mcfg = cfg["model"]
    model = getattr(network, mcfg["class_name"])(**mcfg["kwargs"]).eval()
    n_detectors = int(mcfg["kwargs"].get("n_detectors", 3))

    macs = {"conv1d": 0, "linear": 0, "attn_proj": 0, "attn_scores": 0}
    hooks = []

    def conv_hook(m, inp, out):
        macs["conv1d"] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def linear_hook(m, inp, out):
        macs["linear"] += out.numel() * m.in_features

    def mha_hook(m, inp, out):
        # Query layout (T, B, d), batch 1.
        t, b, d = inp[0].shape
        macs["attn_proj"] += b * 4 * t * d * d
        macs["attn_scores"] += b * 2 * t * t * d

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear) and not isinstance(m, nn.modules.linear.NonDynamicallyQuantizableLinear):
            # MHA out_proj is a NonDynamicallyQuantizableLinear applied functionally; counted in mha_hook.
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(mha_hook))

    shapes = []

    def shape_hook(name):
        def hook(m, inp, out):
            if torch.is_tensor(out):
                shapes.append((name, tuple(out.shape)))
        return hook

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList):
            for i, sub in enumerate(child):
                hooks.append(sub.register_forward_hook(shape_hook(f"{name}[{i}]")))
        else:
            hooks.append(child.register_forward_hook(shape_hook(name)))

    with torch.no_grad():
        y = model(torch.zeros(1, n_detectors, INPUT_LENGTH))

    for h in hooks:
        h.remove()

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(macs.values())

    print(f"config: {args.config}")
    print(f"class: {mcfg['class_name']}  tag: {cfg.get('outputs', {}).get('checkpoint_tag')}")
    print(f"params: {n_params:,} (trainable {n_train:,})")
    print(f"output shape: {tuple(y.shape)}")
    print("MACs per sample (input 1x{}x{}):".format(n_detectors, INPUT_LENGTH))
    for k, v in macs.items():
        print(f"  {k:12s} {v / 1e6:12.2f} M")
    print(f"  {'total':12s} {total / 1e6:12.2f} M")
    print("module output shapes (forward order):")
    for name, shape in shapes:
        print(f"  {name:24s} {shape}")


if __name__ == "__main__":
    main()
