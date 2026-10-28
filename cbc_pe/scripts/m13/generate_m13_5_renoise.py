#!/usr/bin/env python3
"""M13.5: regenerate selected G3 rows changing ONLY the Gaussian noise realisation.

Physics, row metadata, PSD blocks, processing window and HDF5 layout are those of G3
(scripts/m13/generate_m13_G3_whitening.py and the G2/G1 helpers it imports; all unchanged).

Noise seed: build_g1_row draws gaussian_rng = default_rng(row["gaussian_seed"]) and uses it only
to derive the three per-detector noise_from_psd seeds; placement uses the independent
row["placement_seed"]. Replacing row["gaussian_seed"] therefore changes the noise and nothing
else (source, distance rescaling, placement, PSD, processing, window). Rule:
    noise_seed_new = SeedSequence([gaussian_seed, 135001]).generate_state(1, uint64)[0] >> 1
(63 bits, fits int64). New seeds must differ from every G1 manifest gaussian_seed, so no row
reuses another row's white-noise draw. --reproduce-original keeps gaussian_seed (control).

Output (exclusive unless --resume):
    <data_root>/processed/<dataset_id>/m13_5_renoise/renoise_shard{i:02d}of{N:02d}.h5
    <data_root>/processed/<dataset_id>/m13_5_renoise/control_reproduce.h5   (--reproduce-original)
Same schema as a G3 shard (X, y, row_index, status, manifest columns incl. split,
file_group_id, target_network_snr, gaussian_seed, metadata incl. final_distance), plus
noise_seed_original / noise_seed_new; checkpoint every 32 rows as in G3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir
import scripts.m13.generate_m13_G3_whitening as g3
from scripts.generate_m11_final_G1 import require, resolve_cached_psd_files, sha256_file
from scripts.m13.generate_m13_G2_band import (apply_overrides, check_versions, create_output,
                                              generate, read_g1_contract)

NOISE_SEED_SALT = 135001
NOISE_SEED_RULE = ("noise_seed_new = int(np.random.SeedSequence([gaussian_seed, 135001])"
                   ".generate_state(1, dtype=np.uint64)[0] >> 1); replaces row['gaussian_seed'] "
                   "in build_g1_row (seeds only gaussian_rng -> per-detector noise_from_psd seeds)")
OUTPUT_SUBDIR = "m13_5_renoise"


def new_noise_seed(seed):
    state = np.random.SeedSequence([int(seed), NOISE_SEED_SALT]).generate_state(1, dtype=np.uint64)[0]
    return int(state >> np.uint64(1))


def load_rows_file(path, n_manifest):
    rows = np.load(path)
    require(rows.ndim == 1 and rows.size > 0 and np.issubdtype(rows.dtype, np.integer),
            "rows-file must be a non-empty 1-D integer array")
    rows = rows.astype(np.int64)
    require(np.unique(rows).size == rows.size, "Duplicate row ids in rows-file")
    require(rows.min() >= 0 and rows.max() < n_manifest, "Row id out of manifest range")
    return np.sort(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="G3 generation config")
    parser.add_argument("--rows-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Override G1 manifest_path; still verified against G1 manifest_sha256.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reproduce-original", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()
    require(args.num_shards >= 1 and 0 <= args.shard < args.num_shards, "Invalid shard index")
    require(not (args.reproduce_original and args.num_shards != 1),
            "--reproduce-original writes a single control file; do not shard it")

    m13 = g3.load_m13_config(args.config)
    root, g1_path, g1_attrs, spec_g1, manifest_path, df = read_g1_contract(args, m13)
    spec_g3 = apply_overrides(spec_g1, m13["overrides"])
    directory = dataset_processed_dir(root, m13["dataset_id"]) / OUTPUT_SUBDIR

    # Row ids are G1 manifest positions (= G3 row_index = G3 trainio row).
    row_ids = load_rows_file(args.rows_file, len(df))
    subset = df.loc[row_ids]
    blocks = sorted(subset.block_id.astype(str).unique())
    shard_blocks = blocks[args.shard::args.num_shards]
    selected_df = subset.loc[subset.block_id.astype(str).isin(shard_blocks)].copy()
    require(not selected_df.empty, "No rows selected")
    if args.reproduce_original:
        output = directory / "control_reproduce.h5"
    else:
        output = directory / f"renoise_shard{args.shard:02d}of{args.num_shards:02d}.h5"
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume: {output}")
    if args.resume and not output.exists():
        raise FileNotFoundError(f"Resume output missing: {output}")

    original = selected_df["gaussian_seed"].to_numpy(dtype=np.int64)
    if args.reproduce_original:
        new = original.copy()
    else:
        new = np.array([new_noise_seed(s) for s in original], dtype=np.int64)
        require(not np.isin(new, df["gaussian_seed"].to_numpy(dtype=np.int64)).any(),
                "A new noise seed collides with a G1 manifest gaussian_seed")
    selected_df["noise_seed_original"] = original
    selected_df["noise_seed_new"] = new
    selected_df["gaussian_seed"] = new  # the only input build_g1_row reads for the noise

    g3.initialize_g3_physics(spec_g3, spec_g1)
    margin = g3.check_physics(spec_g3)
    rows = g3.adjust_processing_windows(selected_df, spec_g1)

    current_versions, mismatch = check_versions(g1_attrs, args.allow_version_mismatch)
    selection = {"selection_policy": "m13_5_rows_file_sorted_block_id_k_mod_num_shards",
                 "shard_index": args.shard, "num_shards": args.num_shards,
                 "shard_block_count": len(shard_blocks)}
    attrs = g3.build_attrs(m13, spec_g3, spec_g1, g1_path, g1_attrs, manifest_path, len(rows),
                           selection, margin)
    attrs["software_versions"] = json.dumps(current_versions, sort_keys=True)
    attrs["version_mismatch_allowed"] = bool(args.allow_version_mismatch)
    attrs["version_mismatch"] = json.dumps(mismatch, sort_keys=True)
    # build_attrs records the G3 module hash as generator_sha256; this script is recorded apart.
    attrs.update({
        "m13_5_mode": "reproduce_original" if args.reproduce_original else "renoise",
        "m13_5_generator_sha256": sha256_file(__file__),
        "g3_generator_sha256": sha256_file(g3.__file__),
        "m13_config_path": str(args.config.resolve()),
        "m13_config_file_sha256": sha256_file(args.config),
        "rows_file": str(args.rows_file.resolve()),
        "rows_file_sha256": sha256_file(args.rows_file),
        "rows_file_count": int(row_ids.size),
        "noise_seed_rule": "original gaussian_seed (control)" if args.reproduce_original
                           else NOISE_SEED_RULE,
        "noise_seed_salt": NOISE_SEED_SALT,
        "noise_paired_with_g3": bool(args.reproduce_original),
    })
    selected, attrs["raw_psd_files"] = resolve_cached_psd_files(root, rows)

    print(f"M13.5 {attrs['m13_5_mode']}: {len(rows)} rows in {len(shard_blocks)} blocks "
          f"(shard {args.shard}/{args.num_shards}) -> {output}", flush=True)
    if not output.exists():
        create_output(output, rows, attrs)
    generated, seconds = generate(output, rows, attrs, selected)
    if generated:
        print(f"Mean generation time: {seconds / generated:.3f} s/row ({generated} rows)", flush=True)
    # ru_maxrss is in KiB on Linux.
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak RSS of this process: {peak_kib / 1024 ** 2:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
