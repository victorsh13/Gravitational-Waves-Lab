#!/usr/bin/env python3
"""Generate M13 G3 (pilot or shards): frozen M11-final G1 with processing overrides.

G3 changes the whitening/filtering stage (lowpass frequency, whitening filter
duration, processing context) and, optionally, the SNR band upper cutoff.
Manifest, sources, seeds, placement, target SNR, 4 s final window and
reference_geocentric_time are those of G1. Because the processing segment
length changes, the Gaussian noise is a different realisation (noise_from_psd
depends on the length): G3 is NOT noise-paired with G1. With a 1024 Hz SNR
cutoff the final distance is recomputed to reach the same target SNR in the
wider band, so final_distance (and per-detector SNRs) differ from G1. Physics
is reused by importing scripts.generate_m11_final_G1 and the G2 orchestration
helpers; both stay unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import resource
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir
import scripts.generate_m11_final_G1 as g1
from scripts.generate_m11_final_G1 import require, resolve_cached_psd_files, sha256_file
from scripts.m13.generate_m13_G2_band import (G1_NUM_ROWS, G1_REPLACED_ATTRS, PILOT_EXACT_FIELDS,
                                              apply_overrides, check_versions, config_sha256,
                                              create_output, generate, read_g1_contract)

CTX_START = "simulation.processing_context_start_samples"
CTX_END = "simulation.processing_context_end_samples"
LOWPASS = "signal_processor.lowpass_frequency"
WHITENING = "signal_processor.whitening_max_filter_duration"
SNR_CUTOFF = "simulation.snr_high_frequency_cutoff"
ALLOWED_OVERRIDE_KEYS = {LOWPASS, WHITENING, CTX_START, CTX_END, SNR_CUTOFF}
# Only the M13.0d band is authorised for the SNR cutoff override.
ALLOWED_SNR_CUTOFF_HZ = 1024.0
BASE_G1_H5 = "M11_final_G1_500k/m11_final_G1_500k.h5"
# X storage of create_output (imported from G2); the optional config "storage" block must match it.
STORAGE = {"X_chunks": [1, 3, 16384], "X_compression": "lzf"}
M13_0C_REFERENCE = ("M13.0c whitening sweep (docs/experiments/M13_0c_whitening.md): "
                    "16 s filter selected; residual violin modes accepted")
# Final-window times are recomputed in float64 GPS; 1 us is far below dt = 1/4096 s.
TIME_TOL_S = 1e-6
# Pilot distance-ratio guard (G3/G1). A wider SNR band can only add SNR at fixed
# distance, so the ratio is >= 1; ~1.02 is expected, > 1.05 indicates a bug.
DISTANCE_RATIO_MIN = 1.0 - 1e-9
DISTANCE_RATIO_EXPECTED_MAX = 1.02
DISTANCE_RATIO_MAX = 1.05


def load_m13_config(path):
    m13 = json.loads(Path(path).read_text())
    required = {"experiment_id", "dataset_id", "base_g1_h5", "overrides"}
    require(set(m13) in (required, required | {"storage"}), f"Unexpected M13 config keys: {sorted(m13)}")
    require(m13.get("storage", STORAGE) == STORAGE,
            f"storage must match the HDF5 layout written by create_output: {STORAGE}")
    overrides = m13["overrides"]
    unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
    require(not unknown, f"Only {sorted(ALLOWED_OVERRIDE_KEYS)} may be overridden; got {sorted(unknown)}")
    for key in (LOWPASS, WHITENING):
        if key in overrides:
            value = overrides[key]
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0,
                    f"{key} must be a positive number")
    for key in (CTX_START, CTX_END):
        if key in overrides:
            value = overrides[key]
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    f"{key} must be a non-negative integer number of samples")
    if SNR_CUTOFF in overrides:
        require(overrides[SNR_CUTOFF] == ALLOWED_SNR_CUTOFF_HZ,
                f"{SNR_CUTOFF} may only be overridden to {ALLOWED_SNR_CUTOFF_HZ}")
    require(m13["experiment_id"] == m13["dataset_id"] and m13["dataset_id"].startswith("M13_G3_"),
            "experiment_id must equal dataset_id and start with M13_G3_")
    require(m13["base_g1_h5"] == BASE_G1_H5, "Wrong base G1 HDF5")
    return m13


def initialize_g3_physics(spec_g3, spec_g1):
    """Initialise G1 physics with the G3 spec, allowing a non-512 Hz SNR cutoff.

    g1.initialize_physics guards the frozen G1 band (512 Hz). It is called with
    the G1 cutoff; CONFIG is then rebuilt with dataclasses.replace (re-runs the
    SimulationConfig validation, changes only the cutoff) and the three objects
    that capture CONFIG at construction are rebuilt as initialize_physics does.
    Everything else in build_g1_row reads the module global CONFIG at call time,
    including compute_network_optimal_snr and hence the distance rescaling.
    """
    cutoff_g1 = spec_g1["simulation"]["snr_high_frequency_cutoff"]
    cutoff_g3 = spec_g3["simulation"]["snr_high_frequency_cutoff"]
    spec_init = json.loads(json.dumps(spec_g3))
    spec_init["simulation"]["snr_high_frequency_cutoff"] = cutoff_g1
    g1.initialize_physics(spec_init)
    if cutoff_g3 == cutoff_g1:
        return
    from src.processing import SignalProcessor
    from src.waveform import WaveformGenerator
    from src.windowing import ProjectedNetworkWindowSelector
    g1.CONFIG = dataclasses.replace(g1.CONFIG, snr_high_frequency_cutoff=cutoff_g3)
    g1.WAVEFORM_GENERATOR = WaveformGenerator(g1.CONFIG)
    g1.WINDOW_SELECTOR = ProjectedNetworkWindowSelector(g1.CONFIG)
    g1.PROCESSOR = SignalProcessor(config=g1.CONFIG, **spec_g3["signal_processor"])


def check_physics(spec_g3):
    """Confirm the overrides reached CONFIG/PROCESSOR and the context covers edge corruption."""
    sim, sp = spec_g3["simulation"], spec_g3["signal_processor"]
    require(g1.CONFIG.snr_high_frequency_cutoff == sim["snr_high_frequency_cutoff"],
            "SNR cutoff override not applied")
    for obj in (g1.WAVEFORM_GENERATOR, g1.WINDOW_SELECTOR, g1.PROCESSOR):
        require(obj.config is g1.CONFIG, f"{type(obj).__name__} holds a stale CONFIG")
    require(g1.CONFIG.sampling_frequency == 4096, "Expected 4096 Hz sampling")
    require(g1.PROCESSOR.output_mode == "crop_to_config",
            "Processing-window adjustment assumes output_mode='crop_to_config'")
    require(g1.PROCESSOR.lowpass_frequency == sp["lowpass_frequency"], "Lowpass override not applied")
    require(g1.PROCESSOR.whitening_max_filter_duration == sp["whitening_max_filter_duration"],
            "Whitening override not applied")
    ctx_start = g1.CONFIG.processing_context_start_samples
    ctx_end = g1.CONFIG.processing_context_end_samples
    require(ctx_start == sim["processing_context_start_samples"] and
            ctx_end == sim["processing_context_end_samples"], "Context override not applied")
    margin = g1.PROCESSOR.corrupted_margin_samples_per_side()
    require(ctx_start >= margin and ctx_end >= margin,
            f"Context per side ({ctx_start}, {ctx_end}) < corrupted margin {margin} samples")
    return margin


def adjust_processing_windows(rows, spec_g1):
    """Widen [processing_start, processing_end] to the G3 context, keeping the final 4 s window.

    build_g1_row requires processing_end - processing_start == CONFIG.processing_duration
    and derives final_start = processing_start + context_start; shifting the
    edges by the context change in samples keeps final_start, final_center and
    reference_geocentric_time identical to G1. The original G1 edges are kept
    as g1_processing_start/g1_processing_end for provenance.
    """
    dt = g1.CONFIG.delta_t
    duration = g1.CONFIG.duration
    sim1 = spec_g1["simulation"]
    ctx1_start = sim1["processing_context_start_samples"]
    ctx1_end = sim1["processing_context_end_samples"]
    ctx3_start = g1.CONFIG.processing_context_start_samples
    ctx3_end = g1.CONFIG.processing_context_end_samples
    g1_processing_duration = duration + (ctx1_start + ctx1_end) * dt

    start1 = rows["processing_start"].to_numpy(dtype=np.float64)
    end1 = rows["processing_end"].to_numpy(dtype=np.float64)
    reference = rows["reference_geocentric_time"].to_numpy(dtype=np.float64)
    require(np.isclose(end1 - start1, g1_processing_duration).all(), "G1 processing duration mismatch")
    start3 = start1 - (ctx3_start - ctx1_start) * dt
    end3 = end1 + (ctx3_end - ctx1_end) * dt
    require(np.isclose(end3 - start3, g1.CONFIG.processing_duration).all(),
            "Adjusted processing duration differs from CONFIG.processing_duration")

    final_start1 = start1 + ctx1_start * dt
    final_start3 = start3 + ctx3_start * dt
    final_center1 = final_start1 + 0.5 * duration
    final_center3 = final_start3 + 0.5 * duration
    require(np.max(np.abs(final_start3 - final_start1)) < TIME_TOL_S, "final_start changed")
    require(np.max(np.abs(final_center3 - final_center1)) < TIME_TOL_S, "final_center changed")
    # Same tolerance as build_g1_row, before and after the adjustment.
    require(np.max(np.abs(final_center1 - reference)) < dt, "G1 final_center != reference GPS")
    require(np.max(np.abs(final_center3 - reference)) < dt, "G3 final_center != reference GPS")

    adjusted = rows.copy()
    adjusted["g1_processing_start"] = start1
    adjusted["g1_processing_end"] = end1
    adjusted["processing_start"] = start3
    adjusted["processing_end"] = end3
    print(f"Processing window: context {ctx1_start}/{ctx1_end} -> {ctx3_start}/{ctx3_end} samples; "
          f"final_start and final_center unchanged (max |diff| "
          f"{np.max(np.abs(final_center3 - final_center1)):.2e} s)", flush=True)
    return adjusted


def build_attrs(m13, spec_g3, spec_g1, g1_path, g1_attrs, manifest_path, n_rows, selection, margin):
    attrs = {key: value for key, value in g1_attrs.items() if key not in G1_REPLACED_ATTRS}
    sim, sp = spec_g3["simulation"], spec_g3["signal_processor"]
    attrs.update({
        "experiment_id": m13["experiment_id"], "dataset_id": m13["dataset_id"],
        "m13_variant": "G3_whitening",
        "noise_paired_with_g1": False,
        "lowpass_frequency_hz": float(sp["lowpass_frequency"]),
        "whitening_max_filter_duration_s": float(sp["whitening_max_filter_duration"]),
        "processing_context_start_samples": int(sim["processing_context_start_samples"]),
        "processing_context_end_samples": int(sim["processing_context_end_samples"]),
        "snr_low_frequency_cutoff_hz": float(sim["low_frequency_cutoff"]),
        "snr_high_frequency_cutoff_hz": float(sim["snr_high_frequency_cutoff"]),
        "distance_recomputed_for_snr_band": bool(
            sim["snr_high_frequency_cutoff"] != spec_g1["simulation"]["snr_high_frequency_cutoff"]),
        "processing_configuration": json.dumps({
            "lowpass_frequency_hz": float(sp["lowpass_frequency"]),
            "whitening_max_filter_duration_s": float(sp["whitening_max_filter_duration"]),
            "processing_context_start_samples": int(sim["processing_context_start_samples"]),
            "processing_context_end_samples": int(sim["processing_context_end_samples"]),
            "snr_band_hz": [float(sim["low_frequency_cutoff"]), float(sim["snr_high_frequency_cutoff"])],
            "signal_processor": sp,
        }, sort_keys=True),
        "m13_reference": M13_0C_REFERENCE,
        "storage_X": json.dumps(STORAGE, sort_keys=True),
        "corrupted_margin_samples_per_side": int(margin),
        "processing_window_adjustment": "processing_start/end widened by context change; "
                                        "G1 edges stored as g1_processing_start/end",
        "signal_processor_overrides": json.dumps(m13["overrides"], sort_keys=True),
        "m13_config_json": json.dumps(m13, sort_keys=True),
        "m13_config_sha256": config_sha256(m13),
        "config_json": json.dumps(spec_g3, sort_keys=True),
        "config_sha256": config_sha256(spec_g3),
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
    # Single digest of the per-file src hashes above, for quick provenance comparison.
    attrs["source_tree_sha256"] = hashlib.sha256(attrs["source_module_sha256"].encode()).hexdigest()
    attrs.update(selection)
    return attrs


def check_distance_ratio(final_g3, final_g1):
    ratio = final_g3 / final_g1
    q = np.quantile(ratio, [0.0, 0.05, 0.5, 0.95, 1.0])
    print("final_distance_G3 / final_distance_G1: min {:.6f}, q05 {:.6f}, median {:.6f}, "
          "q95 {:.6f}, max {:.6f}".format(*q), flush=True)
    require(np.isfinite(ratio).all(), "Non-finite distance ratio")
    require(q[0] >= DISTANCE_RATIO_MIN, f"Distance ratio below 1: min {q[0]:.12f}")
    require(q[-1] <= DISTANCE_RATIO_MAX, f"Distance ratio above {DISTANCE_RATIO_MAX}: max {q[-1]:.6f}")
    if q[-1] > DISTANCE_RATIO_EXPECTED_MAX:
        print(f"WARNING: max distance ratio {q[-1]:.6f} > expected ~{DISTANCE_RATIO_EXPECTED_MAX}",
              flush=True)


def report_pilot(output, g1_path):
    """Compare metadata with G1. Only the distance ratio is fatal (noise differs by design)."""
    with h5py.File(output, "r") as p, h5py.File(g1_path, "r") as g:
        require(bool((p["status"][:] == 1).all()), "Pilot incomplete")
        index = p["row_index"][:]
        require(np.array_equal(index, np.arange(len(index))), "Pilot must be the first N rows")
        n = len(index)
        require(np.array_equal(p["source_id"].asstr()[:], g["source_id"].asstr()[:n]),
                "Pilot source_id differs from G1")
        require(np.array_equal(p["g1_processing_start"][:], g["processing_start"][:n]) and
                np.array_equal(p["g1_processing_end"][:], g["processing_end"][:n]),
                "Stored G1 processing edges differ from G1 HDF5")
        final_g3 = p["final_distance"][:].astype(np.float64)
        final_g1 = g["final_distance"][:n].astype(np.float64)
        if bool(p.attrs["distance_recomputed_for_snr_band"]):
            check_distance_ratio(final_g3, final_g1)
        print(f"Pilot metadata vs G1 ({n} rows, rtol 1e-12; informative only):", flush=True)
        for name in PILOT_EXACT_FIELDS:
            a, b = p[name][:].astype(np.float64), g[name][:n].astype(np.float64)
            differ = ~np.isclose(a, b, rtol=1e-12, atol=0.0)
            print(f"  {name}: {int(differ.sum())}/{n} rows differ; "
                  f"max |diff| = {np.max(np.abs(a - b)):.3e}", flush=True)
    print("Note: the Gaussian noise is a different realisation from G1 by design "
          "(processing length changed); X is not compared.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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
    spec_g3 = apply_overrides(spec_g1, m13["overrides"])
    directory = dataset_processed_dir(root, m13["dataset_id"])

    # Row selection identical to G2 (scripts/m13/generate_m13_G2_band.py).
    if args.pilot is not None:
        require(0 < args.pilot <= G1_NUM_ROWS, "--pilot must be in [1, 500000]")
        selected_df = df.iloc[:args.pilot]
        output = directory / f"pilot_{args.pilot}.h5"
        selection = {"selection_policy": "first_N_manifest_rows", "pilot_rows": args.pilot}
    else:
        require(args.shard is not None and args.num_shards is not None, "--shard needs --num-shards")
        require(args.num_shards >= 1 and 0 <= args.shard < args.num_shards, "Invalid shard index")
        blocks = sorted(df.block_id.astype(str).unique())
        shard_blocks = blocks[args.shard::args.num_shards]
        selected_df = df.loc[df.block_id.astype(str).isin(shard_blocks)]
        output = directory / "shards" / f"g3_shard{args.shard:02d}of{args.num_shards:02d}.h5"
        selection = {"selection_policy": "sorted_block_id_k_mod_num_shards",
                     "shard_index": args.shard, "num_shards": args.num_shards,
                     "shard_block_count": len(shard_blocks)}
    require(not selected_df.empty, "No rows selected")
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume: {output}")
    if args.resume and not output.exists():
        raise FileNotFoundError(f"Resume output missing: {output}")

    initialize_g3_physics(spec_g3, spec_g1)
    margin = check_physics(spec_g3)
    rows = adjust_processing_windows(selected_df, spec_g1)

    current_versions, mismatch = check_versions(g1_attrs, args.allow_version_mismatch)
    attrs = build_attrs(m13, spec_g3, spec_g1, g1_path, g1_attrs, manifest_path, len(rows),
                        selection, margin)
    attrs["software_versions"] = json.dumps(current_versions, sort_keys=True)
    attrs["version_mismatch_allowed"] = bool(args.allow_version_mismatch)
    attrs["version_mismatch"] = json.dumps(mismatch, sort_keys=True)
    selected, attrs["raw_psd_files"] = resolve_cached_psd_files(root, rows)

    print(f"G3: lowpass {attrs['lowpass_frequency_hz']} Hz, whitening filter "
          f"{attrs['whitening_max_filter_duration_s']} s, context "
          f"{attrs['processing_context_start_samples']}/{attrs['processing_context_end_samples']} "
          f"samples (corrupted margin {margin}), SNR band {attrs['snr_low_frequency_cutoff_hz']}-"
          f"{attrs['snr_high_frequency_cutoff_hz']} Hz; {len(rows)} rows -> {output}", flush=True)
    if not output.exists():
        create_output(output, rows, attrs)
    generated, seconds = generate(output, rows, attrs, selected)
    if generated:
        if args.pilot is not None:
            with h5py.File(output, "r+") as h:
                # Timing of the last run that generated rows (read by m13_0c_whitening_sweep.py).
                h.attrs["pilot_generated_rows"] = int(generated)
                h.attrs["pilot_generation_seconds"] = float(seconds)
        print(f"Mean generation time: {seconds / generated:.3f} s/row ({generated} rows)", flush=True)
    if args.pilot is not None:
        report_pilot(output, g1_path)
    # ru_maxrss is in KiB on Linux.
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak RSS of this process: {peak_kib / 1024 ** 2:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
