#!/usr/bin/env python3
"""Finalize a G3 trainio whose merge wrote all data but aborted in the closing G1 check.

The original merge_m13_G3_shards.py run called asstr() on a non-text file_group_id and stopped
before setting merge_complete. This script re-runs only the closing checks (manifest order, y and
file_group_id identical to the G1 trainio) and then writes the completion attributes. X and all
other datasets are left untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir, resolve_data_root
from scripts.m13.merge_m13_G3_shards import (DEFAULT_CONFIG, G1_TRAINIO, G1_NUM_ROWS, load_m13_config,
                                              require, same_col, sha256_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    m13 = load_m13_config(args.config)
    root = resolve_data_root(cli_data_root=args.data_root)
    directory = dataset_processed_dir(root, m13["dataset_id"])
    output = directory / f"{m13['dataset_id'].replace('M13_', 'm13_', 1)}_trainio.h5"
    g1_trainio = root / "processed" / G1_TRAINIO
    require(output.exists(), f"G3 trainio missing: {output}")
    require(g1_trainio.exists(), f"G1 trainio missing: {g1_trainio}")

    with h5py.File(output, "r+") as out:
        require("merge_complete" not in out.attrs, f"{output.name} already has merge_complete")
        require("initialization_complete" in out.attrs and bool(out.attrs["initialization_complete"]),
                "initialization_complete is not True")
        require(int(out.attrs.get("selected_rows", -1)) == G1_NUM_ROWS,
                f"selected_rows != {G1_NUM_ROWS}")
        require(out["X"].shape == (G1_NUM_ROWS, 3, 16384), f"Unexpected X shape {out['X'].shape}")
        require(np.array_equal(out["manifest_index"][:], np.arange(G1_NUM_ROWS)),
                "manifest_index is not manifest order")
        with h5py.File(g1_trainio, "r") as ref:
            require(np.array_equal(out["y"][:], ref["y"][:]), "y differs from G1 trainio")
            require(same_col(out["file_group_id"], ref["file_group_id"]),
                    "file_group_id differs from G1 trainio")
        out.attrs["verified_against_g1_trainio"] = str(g1_trainio.resolve())
        out.attrs["merge_finalized_by"] = "finalize_m13_G3_merge.py"
        out.attrs["merge_finalize_script_sha256"] = sha256_file(__file__)
        out.attrs["merge_finalize_note"] = (
            "merge_m13_G3_shards.py wrote all datasets and passed the y check against the G1 trainio, "
            "but aborted in the file_group_id check (asstr() on a non-text dataset, a type error). "
            "This script re-ran manifest_index, y and file_group_id checks without modifying X or any "
            "dataset, and then set merge_complete.")
        out.attrs["merge_complete"] = True
        out.flush()
    print(f"Finalized and verified: {output}", flush=True)


if __name__ == "__main__":
    main()
