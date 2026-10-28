#!/usr/bin/env python3
"""M13.6 step 1: split G3 inputs into processed signal L(s) and processed noise L(n).

G3 builds X = zscore(L(s + n)) with L = SignalProcessor.process_network configured as
whitening_method="psd" (fixed block PSD: Welch over the block's 128 s off-source reference,
cached per block_id by get_block_psds), FIR high/low-pass and crop_to_config, with
apply_standardization=False; the signal is added to the noise (SignalInjector.inject, +=) before
L. L is therefore linear and X = zscore(L(s) + L(n)); the per-sample per-detector z-score
(stack_and_normalize_network) is the only non-linear step. These conditions are re-checked here.

Rows are built by scripts.generate_m11_final_G1.build_g1_row with the G3 physics
(scripts/m13/generate_m13_G3_whitening.py), unchanged. During one row, RowCapture swaps two g1
module helpers (looked up at call time): build_gaussian_processing_noise returns zero strains
(optionally also processing the original noise) and stack_and_normalize_network records the
pre-z-score network. Everything else (source, distance rescaling, placement, PSD, window) is G3.

psd_key = str(block_id): one PSD per block (requires a single psd_start/psd_end per block).

Modes (outputs exclusive, under <data_root>/processed/<dataset_id>/m13_6_aug/):
  signal      train rows (manifest split == "train"), noise = 0, L(s) before z-score:
              signal/signal_shard{i:02d}of{N:02d}.h5 (X_clean f16 (n,3,16384), row_index, y,
              psd_key, status); blocks sorted, block k -> shard k mod N; checkpoint every 32 rows.
  noise_bank  per train psd_key, K = --bank-k (default 64) noise-only L(n) before z-score, G3 noise
              procedure and processing length; keys sorted, key j -> shard j mod N;
              seeds default_rng(SeedSequence([key_hash, 136101, k])) drawn per
              detector as build_g1_row draws them from gaussian_seed:
              noise_bank/noise_bank_shard{i:02d}of{N:02d}.h5 (N_bank f16, psd_key, k, status).
  control     64 train rows (default_rng(136102)): zscore(L(s) + L(n_original)) against the G3
              trainio X, in float64 and with L(s), L(n) rounded to float16:
              control/control_report.json. Aborts if max|dX| > 1e-3 (float64 path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir
import scripts.generate_m11_final_G1 as g1
from scripts.generate_m11_final_G1 import (DETECTORS, LABELS, M10_INPUT_ZSCORE_EPS, read_psd_intervals,
                                           require, resolve_cached_psd_files, sha256_file)
import scripts.m13.generate_m13_G3_whitening as g3
from scripts.m13.generate_m13_G2_band import apply_overrides, check_versions, read_g1_contract

OUTPUT_SUBDIR = "m13_6_aug"
FP16_ABS_LIMIT = 6e4  # float16 max is 65504
CHECKPOINT_ROWS = 32
BANK_K = 64
BANK_SALT = 136101
CONTROL_SEED = 136102
CONTROL_ROWS = 64
CONTROL_EXPECTED_MAX = 1e-5
CONTROL_ABORT_MAX = 1e-3
BANK_SEED_RULE = ("key_hash = int.from_bytes(sha256(psd_key.utf8)[:8], 'big'); "
                  "rng = default_rng(SeedSequence([key_hash, 136101, k])); "
                  "seed_ifo = int(rng.integers(0, 2**32 - 1)) for ifo in (H1, L1, V1)")
X_OPTIONS = {"chunks": (1, 3, 16384), "compression": "lzf"}
STR = h5py.string_dtype()


def stack_raw(network):
    X = np.stack([np.asarray(network[ifo], dtype=np.float64) for ifo in DETECTORS], axis=0)
    require(X.shape == (len(DETECTORS), g1.CONFIG.length), f"Unexpected network shape {X.shape}")
    return X


def to_fp16(x, what):
    require(np.isfinite(x).all(), f"Non-finite values in {what}")
    peak = float(np.abs(x).max())
    require(peak <= FP16_ABS_LIMIT, f"|x| = {peak:.4g} > {FP16_ABS_LIMIT:g} in {what}")
    return x.astype(np.float16)


class RowCapture:
    """Swap two g1 helpers during one build_g1_row call; record L(s) and optionally L(n)."""

    def __init__(self, keep_noise):
        self.keep_noise = keep_noise
        self.L_s = None
        self.L_n = None

    def __enter__(self):
        self._noise_fn = g1.build_gaussian_processing_noise
        self._stack_fn = g1.stack_and_normalize_network
        g1.build_gaussian_processing_noise = self._noise
        g1.stack_and_normalize_network = self._stack
        return self

    def __exit__(self, *exc):
        g1.build_gaussian_processing_noise = self._noise_fn
        g1.stack_and_normalize_network = self._stack_fn
        return False

    def _noise(self, *, psds_proc, processing_start, detector_seeds, injector):
        if self.keep_noise:
            noise = self._noise_fn(psds_proc=psds_proc, processing_start=processing_start,
                                   detector_seeds=detector_seeds, injector=injector)
            self.L_n = stack_raw(g1.PROCESSOR.process_network(strains=noise, psds=psds_proc))
        # Same float64 dtype, length and epoch as the set_strain_start_time noise it replaces.
        return {ifo: injector.build_zero_strain(start_time=processing_start,
                                                length=g1.CONFIG.processing_length)
                for ifo in DETECTORS}

    def _stack(self, network):
        require(self.L_s is None, "stack_and_normalize_network called twice in one row")
        X_raw, X = self._stack_fn(network)
        self.L_s = np.asarray(X_raw, dtype=np.float64)
        return X_raw, X


def key_hash(psd_key):
    return int.from_bytes(hashlib.sha256(psd_key.encode("utf-8")).digest()[:8], "big")


def bank_seeds(psd_key, k):
    rng = np.random.default_rng(np.random.SeedSequence([key_hash(psd_key), BANK_SALT, int(k)]))
    seeds = {ifo: int(rng.integers(0, 2**32 - 1)) for ifo in DETECTORS}
    require(len(set(seeds.values())) == len(DETECTORS), f"Detector seed collision ({psd_key}, {k})")
    return seeds


def check_linear_processing(spec_g3):
    """Runtime guard of the step-0 conditions on the processor actually built."""
    sp = spec_g3["signal_processor"]
    require(sp["whitening_method"] == "psd" and g1.PROCESSOR.whitening_method == "psd",
            "L is linear only with whitening_method='psd' (fixed PSD)")
    require(not g1.PROCESSOR.apply_standardization,
            "apply_standardization=True adds a non-linear step inside L")
    require(g1.PROCESSOR.output_mode == "crop_to_config", "Expected output_mode='crop_to_config'")


def check_psd_key_per_block(df):
    # get_block_psds caches by block_id from the first row it sees; a unique PSD interval per
    # block makes the PSD (hence psd_key = block_id) independent of the generation order.
    n = df.groupby(df.block_id.astype(str))[["psd_start", "psd_end"]].nunique()
    require(bool((n == 1).all().all()), "psd_start/psd_end vary within a block_id; psd_key ambiguous")


def g3_trainio_path(root, m13):
    directory = dataset_processed_dir(root, m13["dataset_id"])
    return directory / f"{m13['dataset_id'].replace('M13_', 'm13_', 1)}_trainio.h5"


def setup(args):
    m13 = g3.load_m13_config(args.config)
    root, g1_path, g1_attrs, spec_g1, manifest_path, df = read_g1_contract(args, m13)
    spec_g3 = apply_overrides(spec_g1, m13["overrides"])
    require("split" in df.columns, "Manifest has no split column")
    check_psd_key_per_block(df)
    trainio = g3_trainio_path(root, m13)
    require(trainio.exists(), f"G3 trainio missing: {trainio}")
    g3.initialize_g3_physics(spec_g3, spec_g1)
    margin = g3.check_physics(spec_g3)
    check_linear_processing(spec_g3)
    current_versions, mismatch = check_versions(g1_attrs, args.allow_version_mismatch)
    stat = trainio.stat()

    def attrs_for(n_rows, selection):
        attrs = g3.build_attrs(m13, spec_g3, spec_g1, g1_path, g1_attrs, manifest_path, n_rows,
                               selection, margin)
        attrs["software_versions"] = json.dumps(current_versions, sort_keys=True)
        attrs["version_mismatch_allowed"] = bool(args.allow_version_mismatch)
        attrs["version_mismatch"] = json.dumps(mismatch, sort_keys=True)
        # build_attrs records the G3 module hash as generator_sha256; this script is recorded apart.
        attrs.update({
            "m13_6_mode": args.mode,
            "m13_6_generator_sha256": sha256_file(__file__),
            "g3_generator_sha256": sha256_file(g3.__file__),
            "g1_generator_sha256": sha256_file(g1.__file__),
            "m13_config_path": str(args.config.resolve()),
            "m13_config_file_sha256": sha256_file(args.config),
            "g3_trainio_path": str(trainio.resolve()),
            "g3_trainio_size": int(stat.st_size),
            "g3_trainio_mtime_ns": int(stat.st_mtime_ns),
            "psd_key_definition": "str(block_id); one fixed processing PSD per block",
            "stored_before_zscore": True,
            "x_dtype": "float16",
        })
        return attrs

    directory = dataset_processed_dir(root, m13["dataset_id"]) / OUTPUT_SUBDIR
    return root, spec_g1, df, trainio, directory, attrs_for


def shard_blocks(train_df, args):
    blocks = sorted(train_df.block_id.astype(str).unique())
    chosen = blocks[args.shard::args.num_shards]
    require(chosen, "No blocks selected for this shard")
    return blocks, chosen


def open_output(path, fields, fixed, attrs, resume):
    """Create (exclusive) or resume an HDF5; verify attrs, schema and fixed datasets."""
    if path.exists() and not resume:
        raise FileExistsError(f"Output exists; use --resume: {path}")
    if resume and not path.exists():
        raise FileNotFoundError(f"Resume output missing: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "x") as h:
            for name, (shape, dtype) in fields.items():
                h.create_dataset(name, shape=shape, dtype=dtype, **(X_OPTIONS if len(shape) == 3 else {}))
            for name, values in fixed.items():
                h[name][:] = values
            h.attrs.update(attrs)
            h.attrs["initialization_complete"] = True
            h.flush()
    h = h5py.File(path, "r+")
    require(bool(h.attrs.get("initialization_complete", False)), "Incomplete HDF5 initialization")
    for key, value in attrs.items():
        require(key in h.attrs and bool(np.all(h.attrs[key] == value)), f"Resume attribute mismatch: {key}")
    require(set(h.keys()) == set(fields), "Resume schema mismatch")
    for name, (shape, dtype) in fields.items():
        require(h[name].shape == shape and h[name].dtype == dtype, f"Resume schema mismatch: {name}")
    for name, values in fixed.items():
        stored = h[name].asstr()[:] if h5py.check_string_dtype(h[name].dtype) else h[name][:]
        require(np.array_equal(stored, values), f"Resume content mismatch: {name}")
    require(np.isin(h["status"][:], [0, 1]).all(), "Invalid completion status")
    return h


def run_signal(args, root, spec_g1, df, directory, attrs_for):
    train_df = df.loc[df.split == "train"]
    _, blocks = shard_blocks(train_df, args)
    selected_df = train_df.loc[train_df.block_id.astype(str).isin(blocks)]
    rows = g3.adjust_processing_windows(selected_df, spec_g1)
    n = len(rows)
    selection = {"selection_policy": "m13_6_train_rows_sorted_block_id_k_mod_num_shards",
                 "shard_index": args.shard, "num_shards": args.num_shards,
                 "shard_block_count": len(blocks)}
    attrs = attrs_for(n, selection)
    selected, attrs["raw_psd_files"] = resolve_cached_psd_files(root, rows)
    output = directory / "signal" / f"signal_shard{args.shard:02d}of{args.num_shards:02d}.h5"
    fields = {"X_clean": ((n, 3, g1.CONFIG.length), np.dtype("float16")),
              "row_index": ((n,), np.dtype("int64")),
              "y": ((n, len(LABELS)), np.dtype("float32")),
              "psd_key": ((n,), STR),
              "status": ((n,), np.dtype("int8"))}
    fixed = {"row_index": rows.index.to_numpy(dtype=np.int64),
             "y": rows[list(LABELS)].to_numpy(dtype=np.float32),
             "psd_key": rows.block_id.astype(str).to_numpy(dtype=object)}
    print(f"M13.6 signal: {n} train rows in {len(blocks)} blocks "
          f"(shard {args.shard}/{args.num_shards}) -> {output}", flush=True)

    h = open_output(output, fields, fixed, attrs, args.resume)
    try:
        local = dict(zip(rows.index, range(n)))
        pending = rows.loc[h["status"][:] == 0]
        if pending.empty:
            print("Already complete; no samples rewritten.", flush=True)
            return
        completed, generated, checkpoint_rows = n - len(pending), 0, []
        t0 = time.monotonic()

        def checkpoint():
            if checkpoint_rows:
                h.flush()
                h["status"][sorted(checkpoint_rows)] = 1
                h.flush()
                checkpoint_rows.clear()

        for _, block_rows in pending.groupby("block_id", sort=True):
            long_strains = read_psd_intervals(block_rows.iloc[0], selected)
            psd_cache = {}
            for index, row in block_rows.iterrows():
                try:
                    with RowCapture(keep_noise=False) as cap:
                        g1.build_g1_row(row, long_strains=long_strains, psd_cache=psd_cache)
                    h["X_clean"][local[index]] = to_fp16(cap.L_s, f"L(s) of row {index}")
                except Exception as exc:
                    raise RuntimeError(f"Signal failed at {row.source_id} (row {index})") from exc
                checkpoint_rows.append(local[index])
                if len(checkpoint_rows) == CHECKPOINT_ROWS:
                    checkpoint()
                completed += 1
                generated += 1
                if completed % 1000 == 0:
                    elapsed = time.monotonic() - t0
                    eta_h = (n - completed) / (generated / elapsed) / 3600.0
                    print(f"Completed {completed}/{n}; {elapsed / generated:.2f} s/row; ETA {eta_h:.2f} h",
                          flush=True)
        checkpoint()
        print(f"Complete: {output}; {(time.monotonic() - t0) / generated:.3f} s/row ({generated} rows)",
              flush=True)
    finally:
        h.close()


def run_noise_bank(args, root, spec_g1, df, directory, attrs_for):
    train_df = df.loc[df.split == "train"]
    all_keys, keys = shard_blocks(train_df, args)
    # One reference row per key: gives the PSD interval and the (G3-adjusted) processing_start.
    ref = train_df.loc[train_df.block_id.astype(str).isin(keys)].drop_duplicates("block_id")
    ref = g3.adjust_processing_windows(ref, spec_g1)
    ref = ref.set_index(ref.block_id.astype(str), drop=False).loc[keys]
    K = args.bank_k
    n = len(keys) * K
    selection = {"selection_policy": "m13_6_train_psd_keys_sorted_k_mod_num_shards",
                 "shard_index": args.shard, "num_shards": args.num_shards,
                 "shard_key_count": len(keys), "train_key_count": len(all_keys)}
    attrs = attrs_for(n, selection)
    attrs.update({"bank_k": K, "bank_seed_salt": BANK_SALT, "bank_seed_rule": BANK_SEED_RULE})
    selected, attrs["raw_psd_files"] = resolve_cached_psd_files(root, ref)
    output = directory / "noise_bank" / f"noise_bank_shard{args.shard:02d}of{args.num_shards:02d}.h5"
    fields = {"N_bank": ((n, 3, g1.CONFIG.length), np.dtype("float16")),
              "psd_key": ((n,), STR),
              "k": ((n,), np.dtype("int16")),
              "status": ((n,), np.dtype("int8"))}
    fixed = {"psd_key": np.repeat(np.array(keys, dtype=object), K),
             "k": np.tile(np.arange(K, dtype=np.int16), len(keys))}
    print(f"M13.6 noise_bank: {len(keys)}/{len(all_keys)} train keys x {K} "
          f"(shard {args.shard}/{args.num_shards}) -> {output}", flush=True)

    h = open_output(output, fields, fixed, attrs, args.resume)
    try:
        status = h["status"][:]
        injector = g1.SignalInjector(config=g1.CONFIG)  # only set_strain_start_time is used
        t0, done = time.monotonic(), 0
        for j, key in enumerate(keys):
            lo = j * K
            if (status[lo:lo + K] == 1).all():
                continue
            row = ref.loc[key]
            psds = g1.get_block_psds(row, long_strains=read_psd_intervals(row, selected), psd_cache={})
            block = np.empty((K, 3, g1.CONFIG.length), dtype=np.float16)
            for k in range(K):
                noise = g1.build_gaussian_processing_noise(
                    psds_proc=psds["proc"], processing_start=float(row["processing_start"]),
                    detector_seeds=bank_seeds(key, k), injector=injector)
                L_n = stack_raw(g1.PROCESSOR.process_network(strains=noise, psds=psds["proc"]))
                block[k] = to_fp16(L_n, f"L(n) of key {key}, k={k}")
            h["N_bank"][lo:lo + K] = block
            h.flush()
            h["status"][lo:lo + K] = 1  # checkpoint per key
            h.flush()
            done += 1
            if done % 10 == 0:
                print(f"Keys done {j + 1}/{len(keys)}; {(time.monotonic() - t0) / done:.1f} s/key",
                      flush=True)
        print(f"Complete: {output} ({done} keys generated)", flush=True)
    finally:
        h.close()


def run_control(args, root, spec_g1, df, trainio, directory, attrs_for):
    output = directory / "control" / "control_report.json"
    if output.exists():
        raise FileExistsError(f"Control report exists: {output}")
    train_idx = df.index[df.split == "train"].to_numpy(dtype=np.int64)
    pick = np.sort(np.random.default_rng(CONTROL_SEED).choice(train_idx, CONTROL_ROWS, replace=False))
    rows = g3.adjust_processing_windows(df.loc[pick], spec_g1)
    selected, raw_psd_files = resolve_cached_psd_files(root, rows)
    normalize = g1.normalize_input_per_sample_per_detector_zscore
    per_row = []
    with h5py.File(trainio, "r") as t:
        require(t["X"].shape[0] == len(df), "G3 trainio row count differs from the manifest")
        for _, block_rows in rows.groupby("block_id", sort=True):
            long_strains = read_psd_intervals(block_rows.iloc[0], selected)
            psd_cache = {}
            for index, row in block_rows.iterrows():
                with RowCapture(keep_noise=True) as cap:
                    g1.build_g1_row(row, long_strains=long_strains, psd_cache=psd_cache)
                require(np.array_equal(t["y"][index], row[list(LABELS)].to_numpy(dtype=np.float32)),
                        f"Trainio y differs at row {index}")
                X_g3 = np.asarray(t["X"][index], dtype=np.float64)
                X_rec = normalize(cap.L_s + cap.L_n, eps=M10_INPUT_ZSCORE_EPS)
                s16 = to_fp16(cap.L_s, f"L(s) of row {index}").astype(np.float64)
                n16 = to_fp16(cap.L_n, f"L(n) of row {index}").astype(np.float64)
                X_rec16 = normalize(s16 + n16, eps=M10_INPUT_ZSCORE_EPS)
                per_row.append({
                    "row_index": int(index),
                    "psd_key": str(row.block_id),
                    "max_abs_dX": float(np.abs(X_rec - X_g3).max()),
                    "max_abs_dX_fp16": float(np.abs(X_rec16 - X_g3).max()),
                    "max_abs_L_s_per_detector": np.abs(cap.L_s).max(axis=1).tolist(),
                    "max_abs_L_n_per_detector": np.abs(cap.L_n).max(axis=1).tolist(),
                    "std_L_n_per_detector": cap.L_n.std(axis=1).tolist(),
                })
                print(f"row {index}: max|dX| {per_row[-1]['max_abs_dX']:.3e}, "
                      f"fp16 {per_row[-1]['max_abs_dX_fp16']:.3e}", flush=True)
    worst = max(r["max_abs_dX"] for r in per_row)
    worst16 = max(r["max_abs_dX_fp16"] for r in per_row)
    verdict = ("PASS" if worst <= CONTROL_EXPECTED_MAX else
               "WARN" if worst <= CONTROL_ABORT_MAX else "FAIL")
    report = {
        "verdict": verdict,
        "max_abs_dX": worst,
        "max_abs_dX_fp16": worst16,
        "expected_max": CONTROL_EXPECTED_MAX,
        "abort_max": CONTROL_ABORT_MAX,
        "comparison": "zscore(L(s) + L(n_original)) vs G3 trainio X (float32), per-detector "
                      "z-score eps 1e-6; fp16 variant rounds L(s) and L(n) to float16 first",
        "control_seed": CONTROL_SEED,
        "n_rows": len(per_row),
        "provenance": {**{k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                          for k, v in attrs_for(len(rows), {"selection_policy": "m13_6_control",
                                                            "control_seed": CONTROL_SEED}).items()},
                       "raw_psd_files": raw_psd_files},
        "rows": per_row,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    print(f"Control {verdict}: max|dX| {worst:.3e} (fp16 {worst16:.3e}) -> {output}", flush=True)
    if verdict == "FAIL":
        raise SystemExit(f"Control failed: max|dX| {worst:.3e} > {CONTROL_ABORT_MAX:g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="G3 generation config")
    parser.add_argument("--mode", choices=["signal", "noise_bank", "control"], required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Override G1 manifest_path; still verified against G1 manifest_sha256.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bank-k", type=int, default=BANK_K, help="noise_bank realizations per psd_key")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()
    require(args.num_shards >= 1 and 0 <= args.shard < args.num_shards, "Invalid shard index")
    require(1 <= args.bank_k <= np.iinfo(np.int16).max + 1, "Invalid --bank-k")
    require(args.mode != "control" or (args.num_shards == 1 and not args.resume),
            "control writes a single report; do not shard or resume it")

    root, spec_g1, df, trainio, directory, attrs_for = setup(args)
    if args.mode == "signal":
        run_signal(args, root, spec_g1, df, directory, attrs_for)
    elif args.mode == "noise_bank":
        run_noise_bank(args, root, spec_g1, df, directory, attrs_for)
    else:
        run_control(args, root, spec_g1, df, trainio, directory, attrs_for)
    # ru_maxrss is in KiB on Linux.
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak RSS of this process: {peak_kib / 1024 ** 2:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
