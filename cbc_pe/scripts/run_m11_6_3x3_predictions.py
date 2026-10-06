#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

DOMAINS = ("G0", "G1", "R")

DATASET_FILES = {
    "G0": "m11_6_G0_26k.h5",
    "G1": "m11_6_G1_26k.h5",
    "R": "m11_6_R_26k.h5",
}

CHECKPOINT_FILES = {
    "G0": "m11_6_domains_SimpleCNN_ResidualDilated_M11_6_G0_MSELoss_seed123_checkpoint.pt",
    "G1": "m11_6_domains_SimpleCNN_ResidualDilated_M11_6_G1_MSELoss_seed123_checkpoint.pt",
    "R": "m11_6_domains_SimpleCNN_ResidualDilated_M11_6_R_MSELoss_seed123_checkpoint.pt",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the complete M11.6 3x3 cross-domain prediction matrix."
    )
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/vserrano/cbc_pe_data"),
    )
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def expected_output(data_root: Path, train_domain: str, test_domain: str) -> Path:
    return (
        data_root
        / "results"
        / "m11_6_domains"
        / f"m11_6_{train_domain}_to_{test_domain}_test_predictions_embeddings.npz"
    )


def build_config(
    data_root: Path,
    train_domain: str,
    test_domain: str,
    overwrite: bool,
) -> dict:
    return {
        "data_root": str(data_root),
        "dataset": {
            "dataset_id": "m11_6_domains",
            "dataset_file": DATASET_FILES[test_domain],
            "split_file": "m11_6_splits.npz",
            "label_stats_file": "m11_6_label_stats_train_only.npz",
        },
        "prediction": {
            "checkpoint_file": CHECKPOINT_FILES[train_domain],
            "splits": ["test"],
            "seed": 123,
            "batch_size": 256,
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 2,
            "max_slice_overread": 4.0,
            "data_loading_mode": "hdf5_batch_slices",
        },
        "output": {
            "file_name": (
                f"m11_6_{train_domain}_to_{test_domain}_"
                "test_predictions_embeddings.npz"
            ),
            "overwrite": bool(overwrite),
        },
    }


def validate_inputs(project_root: Path, data_root: Path) -> Path:
    predictor = project_root / "scripts" / "predict_cnn_hdf5.py"
    if not predictor.exists():
        raise FileNotFoundError(f"Missing predictor: {predictor}")

    processed_dir = data_root / "processed" / "m11_6_domains"
    checkpoint_dir = data_root / "models" / "checkpoints" / "m11_6_domains"

    required = [
        processed_dir / "m11_6_splits.npz",
        processed_dir / "m11_6_label_stats_train_only.npz",
    ]
    required += [processed_dir / DATASET_FILES[d] for d in DOMAINS]
    required += [checkpoint_dir / CHECKPOINT_FILES[d] for d in DOMAINS]

    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError("Missing required M11.6 inputs:\n" + msg)

    return predictor


def main():
    args = parse_args()

    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else infer_project_root()
    )
    data_root = args.data_root.expanduser().resolve()

    predictor = validate_inputs(project_root, data_root)

    config_dir = (
        project_root
        / "configs"
        / "experiments"
        / "m11_6_3x3_generated"
    )
    config_dir.mkdir(parents=True, exist_ok=True)

    (data_root / "results" / "m11_6_domains").mkdir(
        parents=True, exist_ok=True
    )

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    matrix = [
        (train_domain, test_domain)
        for train_domain in DOMAINS
        for test_domain in DOMAINS
    ]

    print("=" * 80)
    print("M11.6 3x3 cross-domain predictions")
    print("=" * 80)
    print("project_root:", project_root)
    print("data_root:", data_root)
    print("python:", sys.executable)
    print("CUDA_VISIBLE_DEVICES:", env.get("CUDA_VISIBLE_DEVICES", "<unchanged>"))
    print()

    completed, skipped, failures = [], [], []

    for pos, (train_domain, test_domain) in enumerate(matrix, start=1):
        tag = f"{train_domain}->{test_domain}"
        out = expected_output(data_root, train_domain, test_domain)

        cfg = build_config(
            data_root=data_root,
            train_domain=train_domain,
            test_domain=test_domain,
            overwrite=args.overwrite,
        )

        cfg_path = (
            config_dir
            / f"predict_m11_6_{train_domain}_to_{test_domain}.json"
        )
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

        print("-" * 80)
        print(f"[{pos}/9] {tag}")
        print("dataset:", DATASET_FILES[test_domain])
        print("checkpoint:", CHECKPOINT_FILES[train_domain])
        print("output:", out)

        if out.exists() and not args.overwrite:
            print("status: SKIP (already exists)")
            skipped.append(tag)
            continue

        cmd = [
            sys.executable,
            str(predictor),
            "--config",
            str(cfg_path),
        ]
        print("command:", " ".join(cmd))

        if args.dry_run:
            print("status: DRY-RUN")
            continue

        proc = subprocess.run(cmd, cwd=project_root, env=env)

        if proc.returncode != 0:
            print(f"status: FAIL ({proc.returncode})")
            failures.append(tag)
            if not args.continue_on_error:
                raise SystemExit(proc.returncode)
        else:
            if not out.exists():
                raise RuntimeError(
                    f"{tag}: predictor returned success but output is missing: {out}"
                )
            print("status: PASS")
            completed.append(tag)

    print()
    print("=" * 80)
    print("M11.6 3x3 prediction summary")
    print("=" * 80)
    print("completed:", len(completed))
    print("skipped:", len(skipped))
    print("failures:", len(failures))

    if failures:
        raise SystemExit(1)

    if not args.dry_run:
        missing_outputs = [
            expected_output(data_root, a, b)
            for a, b in matrix
            if not expected_output(data_root, a, b).exists()
        ]
        if missing_outputs:
            msg = "\n".join(f"  - {p}" for p in missing_outputs)
            raise RuntimeError("Missing outputs after run:\n" + msg)

        print("All 9 prediction artifacts are present.")
        print("M11.6 3x3 predictions: PASS")


if __name__ == "__main__":
    main()
