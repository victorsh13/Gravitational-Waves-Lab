#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PAIRS = [
    ("G1", "G1"),
    ("G1", "R"),
    ("R", "G1"),
    ("R", "R"),
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run M11.9 no-z G1/R 2x2 test predictions."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/experiments/m11_9_2x2"),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    predictor = project_root / "scripts" / "predict_cnn_hdf5.py"

    completed = 0
    failures = []

    for train_dom, test_dom in PAIRS:
        cfg = (
            project_root
            / args.config_dir
            / f"predict_m11_9_{train_dom}_to_{test_dom}_noz.json"
        )

        print()
        print("=" * 80)
        print(f"M11.9: {train_dom}_noz -> {test_dom}_noz")
        print("=" * 80)
        print("config:", cfg)

        if not cfg.exists():
            msg = f"Missing config: {cfg}"
            failures.append(msg)
            if not args.continue_on_error:
                raise FileNotFoundError(msg)
            print("ERROR:", msg)
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        result = subprocess.run(
            [
                str(args.python),
                str(predictor),
                "--config",
                str(cfg),
            ],
            cwd=project_root,
            env=env,
        )

        if result.returncode != 0:
            msg = (
                f"{train_dom}->{test_dom} failed "
                f"(return code {result.returncode})"
            )
            failures.append(msg)
            print("ERROR:", msg)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
        else:
            completed += 1

    print()
    print("=" * 80)
    print("M11.9 2x2 prediction summary")
    print("=" * 80)
    print("completed:", completed)
    print("failures:", len(failures))

    if failures:
        for msg in failures:
            print(" -", msg)
        raise SystemExit(1)

    print("All 4 prediction artifacts were generated.")
    print("M11.9 2x2 predictions: PASS")

if __name__ == "__main__":
    main()
