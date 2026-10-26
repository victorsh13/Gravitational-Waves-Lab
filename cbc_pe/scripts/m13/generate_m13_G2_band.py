#!/usr/bin/env python3
"""Generate M13 G2: frozen M11-final G1 with a single processing override.

G2 is paired with G1 row by row. The manifest, seeds, Gaussian noise,
placement, 512 Hz SNR band (hence final distances) are those recorded in the
canonical G1 HDF5; only signal_processor.lowpass_frequency changes. Physics is
reused by importing scripts.generate_m11_final_G1, never copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir, resolve_data_root
from scripts.prepare_m11_final_manifest import load_bank
import scripts.generate_m11_final_G1 as g1
from scripts.generate_m11_final_G1 import (DETECTORS, LABELS, read_psd_intervals,
                                           require, resolve_cached_psd_files,
                                           schema, sha256_file)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/generation/generate_m13_G2_band1024.json"
ALLOWED_OVERRIDES = {"signal_processor.lowpass_frequency": 1024.0}
G1_EXPERIMENT_ID = "M11_final_G1_500k"
G1_NUM_ROWS = 500000
VERSION_KEYS = ("numpy", "pycbc", "lalsuite", "scipy")
ALL_VERSION_KEYS = ("numpy", "pandas", "h5py", "pycbc", "lalsuite", "scipy", "torch")
PILOT_EXACT_FIELDS = (
    "final_distance", "final_network_snr", "actual_geocentric_time", "placement_offset_s",
    "snr_H1", "snr_L1", "snr_V1",
    "gaussian_seed_H1", "gaussian_seed_L1", "gaussian_seed_V1",
)
# Attributes rewritten for G2; every other G1 attribute is inherited unchanged.
G1_REPLACED_ATTRS = {
    "experiment_id", "config_json", "config_sha256", "manifest_path", "generator_sha256",
    "source_module_sha256", "software_versions", "raw_psd_files", "selected_rows",
    "selection_policy", "initialization_complete",
}


def config_sha256(spec):
    # Same canonicalisation as read_inputs in generate_m11_final_G1.py.
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_m13_config(path):
    m13 = json.loads(Path(path).read_text())
    require(set(m13) == {"experiment_id", "dataset_id", "base_g1_h5", "overrides"},
            f"Unexpected M13 config keys: {sorted(m13)}")
    require(m13["overrides"] == ALLOWED_OVERRIDES,
            f"Only {ALLOWED_OVERRIDES} is allowed; got {m13['overrides']}")
    require(m13["experiment_id"] == m13["dataset_id"] == "M13_G2_band1024_500k",
            "Wrong M13 experiment/dataset id")
    require(m13["base_g1_h5"] == "M11_final_G1_500k/m11_final_G1_500k.h5",
            "Wrong base G1 HDF5")
    return m13


def apply_overrides(spec_g1, overrides):
    spec = json.loads(json.dumps(spec_g1))
    for dotted, value in overrides.items():
        section, key = dotted.split(".")
        require(key in spec[section], f"Override target missing in G1 spec: {dotted}")
        spec[section][key] = value
    require(spec_g1["signal_processor"]["lowpass_frequency"] == 512.0, "G1 lowpass is not 512 Hz")
    return spec


def check_manifest_content(spec, root, df):
    """Content checks of read_inputs (G1); experiment/output path checks are not applicable."""
    require(len(df) == spec["generation"]["num_samples"] == G1_NUM_ROWS, "Wrong manifest size")
    require(df.config_sha256.eq(config_sha256(spec)).all(), "Manifest/config hash mismatch")
    require(df.experiment_id.eq(G1_EXPERIMENT_ID).all(), "Wrong experiment")
    require(spec["experiment_id"] == G1_EXPERIMENT_ID, "Wrong config experiment")
    require(tuple(spec["detectors"]) == DETECTORS, "Wrong detector order")
    require(spec["generation"]["domain"] == "G1", "G1 only")
    require(spec["generation"]["injection_mode"] == "full_projection", "Full projection required")
    require(spec["generation"]["placement_policy"] == "random_contained", "Wrong placement policy")
    require(spec["final_input_normalization"] == {
        "mode": "per_sample_per_detector_zscore", "epsilon": 1e-6,
    }, "Wrong input normalization")
    require(spec["empirical_psd"]["reference_duration_s"] == 128.0 and
            spec["empirical_psd"]["welch_segment_duration_s"] == 8.0, "Wrong PSD contract")
    require(spec["simulation"]["snr_high_frequency_cutoff"] == 512.0 and
            spec["simulation"]["low_frequency_cutoff"] == 30.0, "Wrong SNR band")
    require(df.source_id.is_unique and df.source_index.is_unique, "Duplicate sources")
    require(np.array_equal(df.manifest_index, np.arange(len(df))) and
            np.array_equal(df.source_index, np.arange(len(df))), "Manifest order changed")
    require(df.source_id.str.startswith("M11FINAL_G1_SRC_").all(), "Wrong source namespace")
    require(df["split"].value_counts().to_dict() == spec["generation"]["split_counts"],
            "Wrong split counts")
    require(df.reference_distance_mpc.eq(1000).all() and
            df.target_network_snr.between(10, 25).all(), "Invalid distance/SNR")
    require(df.geocentric_time_policy.eq("m11_6_random_contained_reproject").all(),
            "Wrong geocentric-time policy")
    require(np.isfinite(df.select_dtypes(include=np.number)).all().all(),
            "Non-finite manifest metadata")
    seeds = spec["generation"]["seeds"]
    require(df.placement_seed.eq(seeds["placement_base"] + df.source_index).all() and
            df.gaussian_seed.eq(seeds["gaussian_base"] + df.source_index).all(),
            "Manifest seed mismatch")
    crops = load_bank(spec, root).set_index("noise_crop_id")
    require(df.noise_crop_id.isin(crops.index).all(), "Unknown assigned crop")
    for name in ("file_group_id", "block_id", "split", "center_gps", "processing_start",
                 "processing_end", "psd_start", "psd_end"):
        require(df.noise_crop_id.map(crops[name]).eq(df[name]).all(),
                f"Manifest/bank mismatch: {name}")
    require(df.reference_geocentric_time.eq(df.center_gps).all(), "Wrong reference GPS")


def read_g1_contract(args, m13):
    root = resolve_data_root(cli_data_root=args.data_root)
    g1_path = root / "processed" / m13["base_g1_h5"]
    require(g1_path.exists(), f"Canonical G1 HDF5 missing: {g1_path}")
    with h5py.File(g1_path, "r") as h:
        g1_attrs = {key: h.attrs[key] for key in h.attrs}
        require(int(g1_attrs["selected_rows"]) == G1_NUM_ROWS, "G1 HDF5 is not the full dataset")
        require(bool((h["status"][:] == 1).all()), "Canonical G1 HDF5 is incomplete")
    spec_g1 = json.loads(g1_attrs["config_json"])
    require(config_sha256(spec_g1) == g1_attrs["config_sha256"], "G1 config_json/config_sha256 mismatch")
    # The manifest actually used by G1 (after the 4-source repair) is identified by hash.
    manifest_path = args.manifest or Path(g1_attrs["manifest_path"])
    manifest_hash = sha256_file(manifest_path)
    require(manifest_hash == g1_attrs["manifest_sha256"],
            f"Manifest hash differs from G1 manifest_sha256: {manifest_path}")
    df = pd.read_csv(manifest_path, float_precision="round_trip")
    check_manifest_content(spec_g1, root, df)
    return root, g1_path, g1_attrs, spec_g1, Path(manifest_path), df


def check_versions(g1_attrs, allow_mismatch):
    from importlib.metadata import version
    current = {name: version(name) for name in ALL_VERSION_KEYS}
    recorded = json.loads(g1_attrs["software_versions"])
    mismatch = {name: {"G1": recorded.get(name), "current": current[name]}
                for name in VERSION_KEYS if recorded.get(name) != current[name]}
    if mismatch:
        message = f"Software versions differ from G1: {mismatch}"
        require(allow_mismatch, message + " (use --allow-version-mismatch to override)")
        print("WARNING: " + message, flush=True)
    return current, mismatch


def build_attrs(m13, spec_g2, g1_path, g1_attrs, manifest_path, n_rows, selection):
    attrs = {key: value for key, value in g1_attrs.items() if key not in G1_REPLACED_ATTRS}
    attrs.update({
        "experiment_id": m13["experiment_id"], "dataset_id": m13["dataset_id"],
        "m13_variant": "G2_band1024",
        "lowpass_frequency_hz": float(spec_g2["signal_processor"]["lowpass_frequency"]),
        "signal_processor_overrides": json.dumps(m13["overrides"], sort_keys=True),
        "m13_config_json": json.dumps(m13, sort_keys=True),
        "m13_config_sha256": config_sha256(m13),
        "config_json": json.dumps(spec_g2, sort_keys=True),
        "config_sha256": config_sha256(spec_g2),
        "manifest_path": str(manifest_path.resolve()),
        "base_g1_h5": str(g1_path.resolve()),
        "base_g1_h5_size": int(g1_path.stat().st_size),
        "base_g1_config_sha256": g1_attrs["config_sha256"],
        "base_g1_manifest_sha256": g1_attrs["manifest_sha256"],
        "base_g1_generator_sha256": g1_attrs["generator_sha256"],
        "base_g1_source_module_sha256": g1_attrs["source_module_sha256"],
        "base_g1_software_versions": g1_attrs["software_versions"],
        "generator_sha256": sha256_file(__file__),
        "g1_physics_module_sha256": sha256_file(g1.__file__),
        "source_module_sha256": json.dumps({str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
            for p in sorted((PROJECT_ROOT / "src").rglob("*.py"))}, sort_keys=True),
        "selected_rows": n_rows, "manifest_rows": G1_NUM_ROWS,
    })
    attrs.update(selection)
    for key, label in (("g1_physics_module_sha256", "generator_sha256"),
                       ("source_module_sha256", "source_module_sha256")):
        if attrs[key] != g1_attrs[label]:
            print(f"WARNING: current {key} differs from G1 {label}; "
                  "the pilot comparison is the guard for paired physics.", flush=True)
    return attrs


def output_fields(rows):
    fields = schema(rows)
    fields["row_index"] = ((len(rows),), np.dtype("int64"))
    return fields


def create_output(output, rows, attrs):
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as h:
        for name, (shape, dtype) in output_fields(rows).items():
            options = {"chunks": (1, 3, 16384), "compression": "lzf"} if name == "X" else {}
            d = h.create_dataset(name, shape=shape, dtype=dtype, **options)
            if name in rows:
                d[:] = rows[name].to_numpy()
        h["row_index"][:] = rows.index.to_numpy(dtype=np.int64)
        h["y"][:] = rows[list(LABELS)].to_numpy(dtype=np.float32)
        h.attrs.update(attrs)
        h.attrs["initialization_complete"] = True
        h.flush()


def verify_output(h, rows, attrs):
    require(bool(h.attrs.get("initialization_complete", False)), "Incomplete HDF5 initialization")
    for key, value in attrs.items():
        require(key in h.attrs and h.attrs[key] == value, f"Resume attribute mismatch: {key}")
    fields = output_fields(rows)
    require(set(h.keys()) == set(fields), "Resume schema mismatch")
    for name, (shape, dtype) in fields.items():
        require(h[name].shape == shape and h[name].dtype == dtype, f"Resume schema mismatch: {name}")
    require(np.array_equal(h["row_index"][:], rows.index.to_numpy()), "Resume row_index mismatch")
    for name in rows:
        values = h[name].asstr()[:] if h5py.check_string_dtype(h[name].dtype) else h[name][:]
        require(np.array_equal(values, rows[name].to_numpy()), f"Resume manifest mismatch: {name}")
    require(np.array_equal(h["y"][:], rows[list(LABELS)].to_numpy(dtype=np.float32)),
            "Resume labels mismatch")
    require(np.isin(h["status"][:], [0, 1]).all(), "Invalid completion status")


def generate(output, rows, attrs, selected):
    """Fill output in G1 order (blocks sorted, rows in manifest order); return (rows, seconds)."""
    with h5py.File(output, "r+") as h:
        verify_output(h, rows, attrs)
        local = pd.Series(np.arange(len(rows)), index=rows.index)
        pending = rows.loc[h["status"][:] == 0]
        if pending.empty:
            print("Already complete; no samples rewritten.", flush=True)
            return 0, 0.0
        completed = len(rows) - len(pending)
        generated = 0
        checkpoint_rows = []
        t0 = time.monotonic()

        def checkpoint():
            if checkpoint_rows:
                h.flush()
                h["status"][sorted(checkpoint_rows)] = 1
                h.flush()
                checkpoint_rows.clear()

        for block_id, block_rows in pending.groupby("block_id", sort=True):
            long_strains = read_psd_intervals(block_rows.iloc[0], selected)
            psd_cache = {}
            for index, row in block_rows.iterrows():
                position = int(local[index])
                try:
                    X, metadata = g1.build_g1_row(row, long_strains=long_strains, psd_cache=psd_cache)
                    h["X"][position] = X
                    for name, value in metadata.items():
                        h[name][position] = value
                    # As in G1: only checkpointed rows are marked complete.
                    checkpoint_rows.append(position)
                    if len(checkpoint_rows) == 32:
                        checkpoint()
                except Exception as exc:
                    raise RuntimeError(f"Generation failed at {row.source_id} (row {index})") from exc
                completed += 1
                generated += 1
                if completed % 1000 == 0 or len(rows) <= 64:
                    elapsed = time.monotonic() - t0
                    rate = generated / elapsed
                    eta_h = (len(rows) - completed) / rate / 3600.0
                    print(f"Completed {completed}/{len(rows)}; {elapsed / generated:.2f} s/row; "
                          f"ETA {eta_h:.2f} h", flush=True)
        checkpoint()
        print(f"Complete: {output}", flush=True)
        return generated, time.monotonic() - t0


def compare_pilot(output, g1_path, generated, seconds):
    with h5py.File(output, "r") as p, h5py.File(g1_path, "r") as g:
        require(bool((p["status"][:] == 1).all()), "Pilot incomplete")
        index = p["row_index"][:]
        require(np.array_equal(index, np.arange(len(index))), "Pilot must be the first N rows")
        n = len(index)
        for name in PILOT_EXACT_FIELDS:
            # float64 is exact for uint32 seeds and avoids unsigned wrap in the diff.
            a, b = p[name][:].astype(np.float64), g[name][:n].astype(np.float64)
            require(np.allclose(a, b, rtol=1e-12, atol=0.0),
                    f"Pilot mismatch with G1 in {name}: max |diff| = {np.max(np.abs(a - b))}")
        print(f"Pilot metadata matches G1 (rtol 1e-12) for {n} rows: "
              + ", ".join(PILOT_EXACT_FIELDS), flush=True)
        x2 = p["X"][:].astype(np.float64)
        x1 = g["X"][:n].astype(np.float64)
    x1 -= x1.mean(axis=-1, keepdims=True)
    x2 -= x2.mean(axis=-1, keepdims=True)
    r = (x1 * x2).sum(-1) / np.sqrt((x1 ** 2).sum(-1) * (x2 ** 2).sum(-1))
    for k, ifo in enumerate(DETECTORS):
        print(f"Pearson(X_G1, X_G2) {ifo}: mean {r[:, k].mean():.4f}, "
              f"min {r[:, k].min():.4f}, max {r[:, k].max():.4f}", flush=True)
    if generated:
        print(f"Mean generation time: {seconds / generated:.3f} s/row ({generated} rows)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Override G1 manifest_path; still verified against G1 manifest_sha256.")
    parser.add_argument("--pilot", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()
    is_shard = args.shard is not None or args.num_shards is not None
    require((args.pilot is not None) != is_shard, "Use exactly one of --pilot N or --shard/--num-shards")

    m13 = load_m13_config(args.config)
    root, g1_path, g1_attrs, spec_g1, manifest_path, df = read_g1_contract(args, m13)
    spec_g2 = apply_overrides(spec_g1, m13["overrides"])
    directory = dataset_processed_dir(root, m13["dataset_id"])

    if args.pilot is not None:
        require(0 < args.pilot <= G1_NUM_ROWS, "--pilot must be in [1, 500000]")
        rows = df.iloc[:args.pilot].copy()
        output = directory / f"pilot_{args.pilot}.h5"
        selection = {"selection_policy": "first_N_manifest_rows", "pilot_rows": args.pilot}
    else:
        require(args.shard is not None and args.num_shards is not None, "--shard needs --num-shards")
        require(args.num_shards >= 1 and 0 <= args.shard < args.num_shards, "Invalid shard index")
        blocks = sorted(df.block_id.astype(str).unique())
        shard_blocks = blocks[args.shard::args.num_shards]
        rows = df.loc[df.block_id.astype(str).isin(shard_blocks)].copy()
        output = directory / "shards" / f"g2_shard{args.shard:02d}of{args.num_shards:02d}.h5"
        selection = {"selection_policy": "sorted_block_id_k_mod_num_shards",
                     "shard_index": args.shard, "num_shards": args.num_shards,
                     "shard_block_count": len(shard_blocks)}
    require(not rows.empty, "No rows selected")

    current_versions, mismatch = check_versions(g1_attrs, args.allow_version_mismatch)
    attrs = build_attrs(m13, spec_g2, g1_path, g1_attrs, manifest_path, len(rows), selection)
    attrs["software_versions"] = json.dumps(current_versions, sort_keys=True)
    attrs["version_mismatch_allowed"] = bool(args.allow_version_mismatch)
    attrs["version_mismatch"] = json.dumps(mismatch, sort_keys=True)
    selected, attrs["raw_psd_files"] = resolve_cached_psd_files(root, rows)

    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume: {output}")
    if args.resume and not output.exists():
        raise FileNotFoundError(f"Resume output missing: {output}")
    print(f"Initializing G2 physics (lowpass {attrs['lowpass_frequency_hz']} Hz) for "
          f"{len(rows)} rows -> {output}", flush=True)
    g1.initialize_physics(spec_g2)
    require(g1.PROCESSOR.lowpass_frequency == 1024.0, "Processor lowpass override not applied")
    if not output.exists():
        create_output(output, rows, attrs)
    generated, seconds = generate(output, rows, attrs, selected)
    if args.pilot is not None:
        compare_pilot(output, g1_path, generated, seconds)


if __name__ == "__main__":
    main()
