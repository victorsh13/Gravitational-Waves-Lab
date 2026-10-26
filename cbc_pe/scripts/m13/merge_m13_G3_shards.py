#!/usr/bin/env python3
"""Merge M13 G3 shards into one trainio-layout HDF5 in manifest order.

Minimal copy of merge_m13_G2_shards.py (which is bound to the G2 config,
shard names and output name); only those three bindings change.
Layout follows scripts/trainio.py: X float32 chunks (64, 3, 16384) uncompressed,
y chunks (1024, 3); other datasets contiguous. Row alignment is verified
against the G1 trainio file (y and file_group_id); labels and sources are
those of G1, while X, noise and final distances differ by design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir, resolve_data_root
from scripts.generate_m11_final_G1 import ALIASES, require, sha256_file
from scripts.m13.generate_m13_G2_band import G1_NUM_ROWS
from scripts.m13.generate_m13_G3_whitening import load_m13_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs/generation/generate_m13_G3_band1024_w16.json"

G1_TRAINIO = "M11_final_G1_500k/m11_final_G1_500k_trainio.h5"
BLOCK = 256
# Attributes that legitimately differ between shards; recorded per shard instead.
SHARD_ATTRS = {"shard_index", "selected_rows", "shard_block_count", "raw_psd_files",
               "initialization_complete"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    m13 = load_m13_config(args.config)
    root = resolve_data_root(cli_data_root=args.data_root)
    directory = dataset_processed_dir(root, m13["dataset_id"])
    n_shards = args.num_shards
    paths = [directory / "shards" / f"g3_shard{i:02d}of{n_shards:02d}.h5" for i in range(n_shards)]
    found = sorted((directory / "shards").glob(f"g3_shard*of{n_shards:02d}.h5"))
    require(found == sorted(paths), f"Shard files differ from expected set: {found}")
    output = directory / f"{m13['dataset_id'].replace('M13_', 'm13_', 1)}_trainio.h5"
    if output.exists():
        raise FileExistsError(f"Merge output exists: {output}")
    g1_trainio = root / "processed" / G1_TRAINIO
    require(g1_trainio.exists(), f"G1 trainio missing: {g1_trainio}")

    shards = [h5py.File(p, "r") for p in paths]
    try:
        common = {k: v for k, v in shards[0].attrs.items() if k not in SHARD_ATTRS}
        names = set(shards[0].keys())
        for i, h in enumerate(shards):
            require(int(h.attrs["shard_index"]) == i and int(h.attrs["num_shards"]) == n_shards,
                    f"Shard attrs mismatch: {paths[i].name}")
            require(h.attrs["experiment_id"] == m13["experiment_id"], f"Wrong experiment: {paths[i].name}")
            require(set(h.keys()) == names, f"Shard schema mismatch: {paths[i].name}")
            for k, v in common.items():
                require(k in h.attrs and h.attrs[k] == v, f"Shard attr {k} differs: {paths[i].name}")
            require(bool((h["status"][:] == 1).all()), f"Incomplete rows in {paths[i].name}")

        # Coverage: every manifest row exactly once.
        row_index = [h["row_index"][:] for h in shards]
        all_rows = np.concatenate(row_index)
        require(len(all_rows) == G1_NUM_ROWS and
                np.array_equal(np.sort(all_rows), np.arange(G1_NUM_ROWS)),
                "Shards do not cover 0..499999 exactly once")
        shard_of = np.empty(G1_NUM_ROWS, dtype=np.int64)
        local_of = np.empty(G1_NUM_ROWS, dtype=np.int64)
        for i, index in enumerate(row_index):
            require(np.all(np.diff(index) > 0), f"row_index not increasing in {paths[i].name}")
            shard_of[index] = i
            local_of[index] = np.arange(len(index))

        print("Hashing shards ...", flush=True)
        shard_hashes = {p.name: sha256_file(p) for p in paths}
        per_shard = {p.name: {k: (h.attrs[k].item() if hasattr(h.attrs[k], "item") else h.attrs[k])
                              for k in SHARD_ATTRS - {"initialization_complete"}}
                     for p, h in zip(paths, shards)}

        data_names = sorted(names - {"X", "y", "row_index"} - set(ALIASES))
        with h5py.File(output, "x") as out:
            X = out.create_dataset("X", shape=(G1_NUM_ROWS, 3, 16384), dtype=np.float32,
                                   chunks=(64, 3, 16384), compression=None)
            y = out.create_dataset("y", shape=(G1_NUM_ROWS, 3), dtype=shards[0]["y"].dtype,
                                   chunks=(1024, 3), compression=None)
            # Gather each output block from the shards; local indices are increasing
            # within a shard, as required by h5py fancy indexing.
            for start in range(0, G1_NUM_ROWS, BLOCK):
                stop = min(start + BLOCK, G1_NUM_ROWS)
                xb = np.empty((stop - start, 3, 16384), dtype=np.float32)
                yb = np.empty((stop - start, 3), dtype=y.dtype)
                owners = shard_of[start:stop]
                for i in np.unique(owners):
                    mask = owners == i
                    local = local_of[start:stop][mask]
                    xb[mask] = shards[i]["X"][local]
                    yb[mask] = shards[i]["y"][local]
                X[start:stop] = xb
                y[start:stop] = yb
                if start % 10000 == 0:
                    print(f"{start}/{G1_NUM_ROWS}", flush=True)
            for name in data_names:
                src = shards[0][name]
                is_str = h5py.check_string_dtype(src.dtype)
                values = np.empty(G1_NUM_ROWS, dtype=object if is_str else src.dtype)
                for h, index in zip(shards, row_index):
                    values[index] = h[name].asstr()[:] if is_str else h[name][:]
                out.create_dataset(name, data=values, dtype=src.dtype)
            require(np.array_equal(out["manifest_index"][:], np.arange(G1_NUM_ROWS)),
                    "Merged manifest_index is not manifest order")
            for alias, target in ALIASES.items():
                out[alias] = out[target]
            out.attrs.update(common)
            out.attrs.update({
                "selected_rows": G1_NUM_ROWS,
                "selection_policy": "all_manifest_rows_merged_from_shards",
                "storage_layout_variant": "trainio",
                "storage_layout_note": "Shards merged in manifest order with training-optimized "
                                       "HDF5 chunking (scripts/trainio.py layout).",
                "shard_sha256": json.dumps(shard_hashes, sort_keys=True),
                "shard_attrs": json.dumps(per_shard, sort_keys=True),
                "merge_script_sha256": sha256_file(__file__),
                "initialization_complete": True,
            })
            out.flush()

            # Paired-row check against G1 trainio.
            with h5py.File(g1_trainio, "r") as ref:
                require(np.array_equal(out["y"][:], ref["y"][:]), "y differs from G1 trainio")
                require(np.array_equal(out["file_group_id"].asstr()[:],
                                       ref["file_group_id"].asstr()[:]),
                        "file_group_id differs from G1 trainio")
            out.attrs["verified_against_g1_trainio"] = str(g1_trainio.resolve())
            out.attrs["merge_complete"] = True
    finally:
        for h in shards:
            h.close()
    print(f"Merged and verified: {output}", flush=True)


if __name__ == "__main__":
    main()
