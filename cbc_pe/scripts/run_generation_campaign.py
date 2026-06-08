#!/usr/bin/env python3

import argparse
import json
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_run_config(base_cfg, output_dir, m1, m2, file_index, seed):
    cfg = deepcopy(base_cfg)

    suffix = f"{file_index:02d}"

    cfg["output"]["file_name"] = (
        f"{output_dir}/"
        f"bbh_gw_only_32s_m1-{m1}_m2-{m2}_n100_{suffix}.h5"
    )

    cfg["generation"]["seed"] = int(seed)

    cfg["parameter_sampler"]["fixed"]["mass_1"] = float(m1)
    cfg["parameter_sampler"]["fixed"]["mass_2"] = float(m2)
    cfg["parameter_sampler"]["fixed"]["spin_1z"] = 0.0
    cfg["parameter_sampler"]["fixed"]["spin_2z"] = 0.0

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("configs/generation/tmp"),
        help="Temporary directory for generated one-run configs.",
    )

    args = parser.parse_args()

    campaign = load_json(args.campaign)

    base_config_path = Path(campaign["base_config"])
    base_cfg = load_json(base_config_path)

    output_dir = campaign["output_dir"]
    file_indices = campaign["file_indices"]
    mass_pairs = campaign["mass_pairs"]

    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    commands = []

    for pair in mass_pairs:
        m1 = int(pair["m1"])
        m2 = int(pair["m2"])
        base_seed = int(pair["base_seed"])

        for file_index in file_indices:
            seed = base_seed + (file_index - 1)

            cfg = build_run_config(
                base_cfg=base_cfg,
                output_dir=output_dir,
                m1=m1,
                m2=m2,
                file_index=file_index,
                seed=seed,
            )

            tmp_config_name = (
                f"tmp_gw_only_32s_m1-{m1}_m2-{m2}_"
                f"n100_{file_index:02d}.json"
            )
            tmp_config_path = args.tmp_dir / tmp_config_name

            save_json(cfg, tmp_config_path)

            cmd = [
                "python",
                "scripts/generate_bbh_dataset_hdf5.py",
                "--config",
                str(tmp_config_path),
            ]

            if args.overwrite:
                cmd.append("--overwrite")

            commands.append(cmd)

    print(f"Prepared {len(commands)} generation jobs.")

    for cmd in commands:
        print(" ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()