#!/usr/bin/env python3
"""
M11.9 generation of paired G1/R domains WITHOUT per-sample/per-detector z-score.

Scientific question
-------------------
M11.6 found G1 ~ R after the final M10 input z-score. M11.9 tests whether
that equivalence depends on the z-score by changing only the persisted input
representation:

    M11.6: X = zscore(processed network)
    M11.9: X = processed network  (pre-zscore)

Everything else is inherited from the frozen M11.6 generation contract:
source population, manifest/splits, source_id, noise assignment, empirical
off-source PSDs, signal geometry, target optimal SNR, processing, and seeds.

Domains
-------
G1 = h(theta, d_emp) + Gaussian noise generated from empirical off-source PSD
R  = h(theta, d_emp) + real off-source detector strain

Operational contract
--------------------
- Process one file_group_id at a time.
- Keep only one HLV real-strain environment in RAM.
- Cache empirical PSDs only locally by block_id.
- Write G1/R immediately to HDF5.
- Persist a paired status vector for restart/resume.
- Preserve manifest_index and source_id across domains.
- Fail fast on an unexpected source-generation error.

This script intentionally reuses the validated M11.6 physics/helpers instead
of modifying the closed M11.6 generator.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Make cbc_pe importable when executed as:
#   python scripts/generate_m11_9_noz_domains.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the validated M11.6 physics and data-access implementation.
import scripts.generate_m11_6_domains as m116


DOMAINS = ("G1", "R")
DETECTORS = m116.DETECTORS
CONFIG = m116.CONFIG

EXPERIMENT_NAME = "M11.9"
REPRESENTATION = "processed_pre_zscore"


# ---------------------------------------------------------------------
# Input representation
# ---------------------------------------------------------------------

def stack_processed_network_raw(network: dict) -> np.ndarray:
    """
    Stack the final processed H1/L1/V1 network WITHOUT input z-score.

    This is exactly the X_raw object that existed immediately before
    normalize_input_per_sample_per_detector_zscore() in M11.6.
    """
    X_raw = np.stack(
        [np.asarray(network[ifo]) for ifo in DETECTORS],
        axis=0,
    )

    expected_shape = (len(DETECTORS), CONFIG.length)

    if X_raw.shape != expected_shape:
        raise ValueError(
            f"Unexpected processed network shape: "
            f"{X_raw.shape}, expected={expected_shape}"
        )

    if not np.all(np.isfinite(X_raw)):
        raise ValueError(
            "Processed pre-zscore network contains non-finite values."
        )

    return X_raw.astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# Paired G1/R builder
# ---------------------------------------------------------------------

def build_paired_noz_domains_for_manifest_row(
    row: pd.Series,
    *,
    long_strains: dict,
    psd_cache: dict,
):
    """
    M11.9 counterpart of M11.6 build_paired_domains_for_manifest_row().

    G0 is intentionally omitted. G1 and R preserve the exact M11.6
    empirical-PSD signal distance, geometry, target SNR, processing, seeds,
    and noise assignment. The only scientific change is that the processed
    network is persisted before the final per-sample/per-detector z-score.
    """
    source_id = str(row["source_id"])
    source_index = int(row["source_index"])

    placement_rng = np.random.default_rng(
        m116.BUILD_SEED + m116.PLACEMENT_SEED_OFFSET + source_index
    )
    gaussian_rng = np.random.default_rng(
        m116.BUILD_SEED + m116.GAUSSIAN_SEED_OFFSET + source_index
    )

    injector = m116.SignalInjector(
        config=CONFIG,
        rng=placement_rng,
    )

    # Keep seed construction identical to M11.6.
    gaussian_seeds = {
        ifo: int(gaussian_rng.integers(0, 2**32 - 1))
        for ifo in DETECTORS
    }

    params_ref = m116.CBCParameters(
        mass_1=float(row["mass_1"]),
        mass_2=float(row["mass_2"]),
        distance=float(row["reference_distance_mpc"]),
        inclination=float(row["inclination"]),
        ra=float(row["ra"]),
        dec=float(row["dec"]),
        spin_1z=float(row["spin_1z"]),
        spin_2z=float(row["spin_2z"]),
        polarization_angle=float(row["polarization_angle"]),
    )

    target_snr = float(row["target_network_snr"])
    processing_start = float(row["processing_start"])
    processing_end = float(row["processing_end"])

    if not np.isclose(
        processing_end - processing_start,
        CONFIG.processing_duration,
    ):
        raise ValueError(
            f"{source_id}: invalid processing duration "
            f"{processing_end - processing_start}"
        )

    final_start = (
        processing_start
        + CONFIG.processing_context_start_seconds
    )
    final_end = final_start + CONFIG.duration
    final_center = 0.5 * (final_start + final_end)

    # Provisional placement geometry at reference distance.
    waveform_ref = m116.WAVEFORM_GENERATOR.generate(params_ref)

    projection_provisional = m116.DETECTOR_PROJECTOR.project(
        h_plus=waveform_ref.h_plus,
        h_cross=waveform_ref.h_cross,
        parameters=params_ref,
        geocentric_coalescence_time=final_center,
    )

    windowed_provisional = m116.WINDOW_SELECTOR.select(
        projected_strains=projection_provisional.strains,
        max_duration=CONFIG.duration,
    )

    abstract_placement = (
        injector.choose_segment_placement_containing_network(
            signals=windowed_provisional.strains,
            placement_policy="random_contained",
            safe_margin_start=float(CONFIG.safe_margin_start),
            safe_margin_end=float(CONFIG.safe_margin_end),
            enforce_safe_margins=True,
        )
    )

    placement_shift = (
        final_start
        - float(abstract_placement.segment_start_time)
    )
    geocentric_time = final_center + placement_shift
    placement_offset_s = geocentric_time - final_center

    # Final reference projection at actual placement time.
    projection_ref = m116.DETECTOR_PROJECTOR.project(
        h_plus=waveform_ref.h_plus,
        h_cross=waveform_ref.h_cross,
        parameters=params_ref,
        geocentric_coalescence_time=geocentric_time,
    )

    windowed_ref = m116.WINDOW_SELECTOR.select(
        projected_strains=projection_ref.strains,
        max_duration=CONFIG.duration,
    )

    tol = 2.0 * CONFIG.delta_t

    if (
        windowed_ref.metadata.used_window_start_time
        < final_start - tol
    ):
        raise ValueError(
            f"{source_id}: signal starts outside final segment."
        )

    if (
        windowed_ref.metadata.used_window_end_time
        > final_end + tol
    ):
        raise ValueError(
            f"{source_id}: signal ends outside final segment."
        )

    zero_final = {
        ifo: injector.build_zero_strain(
            start_time=final_start,
            length=CONFIG.length,
        )
        for ifo in DETECTORS
    }

    signal_only_ref_results = injector.inject_network(
        noises=zero_final,
        signals=windowed_ref.strains,
    )

    signal_segments_ref = {
        ifo: signal_only_ref_results[ifo].strain
        for ifo in DETECTORS
    }

    empirical_psds = m116.get_block_psds(
        row,
        long_strains=long_strains,
        psd_cache=psd_cache,
    )

    psds_emp_snr = empirical_psds["snr"]
    psds_emp_proc = empirical_psds["proc"]

    _, snr_emp_ref = m116.compute_network_optimal_snr(
        signal_segments=signal_segments_ref,
        psds=psds_emp_snr,
        config=CONFIG,
    )

    distance_emp = m116.rescale_distance_for_target_network_snr(
        current_distance=params_ref.distance,
        current_network_snr=snr_emp_ref,
        target_network_snr=target_snr,
    )

    if not np.isfinite(distance_emp):
        raise RuntimeError(
            f"{source_id}: invalid empirical distance."
        )

    params_emp = params_ref.with_distance(distance_emp)

    signal_emp = m116.build_signal_network_for_params(
        params_emp,
        geocentric_time=geocentric_time,
        final_start=final_start,
        injector=injector,
    )

    snrs_emp_final, snr_emp_final = m116.compute_network_optimal_snr(
        signal_segments=signal_emp["segments"],
        psds=psds_emp_snr,
        config=CONFIG,
    )

    m116.validate_snr_rescaling(
        final_network_snr=snr_emp_final,
        target_network_snr=target_snr,
        relative_tolerance=CONFIG.snr_relative_tolerance,
    )

    noise_G1 = m116.build_gaussian_processing_noise(
        psds_proc=psds_emp_proc,
        processing_start=processing_start,
        detector_seeds=gaussian_seeds,
        injector=injector,
    )

    noise_R = {
        ifo: m116.extract_exact_processing_context(
            long_strains[ifo],
            start_time=processing_start,
            expected_length=CONFIG.processing_length,
        )
        for ifo in DETECTORS
    }

    injected_G1_results = injector.inject_network(
        noises=noise_G1,
        signals=signal_emp["projection"].strains,
    )
    injected_R_results = injector.inject_network(
        noises=noise_R,
        signals=signal_emp["projection"].strains,
    )

    injected_G1 = {
        ifo: injected_G1_results[ifo].strain
        for ifo in DETECTORS
    }
    injected_R = {
        ifo: injected_R_results[ifo].strain
        for ifo in DETECTORS
    }

    processed_G1 = m116.PROCESSOR.process_network(
        strains=injected_G1,
        psds=psds_emp_proc,
    )
    processed_R = m116.PROCESSOR.process_network(
        strains=injected_R,
        psds=psds_emp_proc,
    )

    # Scientific intervention of M11.9:
    # persist processed pre-zscore inputs, not the normalized representation.
    X_G1 = stack_processed_network_raw(processed_G1)
    X_R = stack_processed_network_raw(processed_R)

    expected_shape = (len(DETECTORS), CONFIG.length)

    for name, X in (
        ("G1", X_G1),
        ("R", X_R),
    ):
        if X.shape != expected_shape:
            raise ValueError(
                f"{source_id} {name}: X shape={X.shape}, "
                f"expected={expected_shape}"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(
                f"{source_id} {name}: non-finite final input."
            )

    diagnostics = {
        "source_id": source_id,
        "source_index": source_index,
        "file_group_id": int(row["file_group_id"]),
        "block_id": str(row["block_id"]),
        "noise_crop_id": str(row["noise_crop_id"]),
        "chirp_mass": float(params_ref.chirp_mass),
        "total_mass": float(params_ref.total_mass),
        "chi_eff": float(params_ref.chi_eff),
        "target_network_snr": target_snr,
        "geocentric_time": float(geocentric_time),
        "placement_offset_s": float(placement_offset_s),
        "full_network_duration": float(
            windowed_ref.metadata.full_network_duration
        ),
        "required_final_duration": float(
            windowed_ref.metadata.required_available_final_duration
        ),
        "is_truncated": bool(
            windowed_ref.metadata.is_truncated
        ),
    }

    return {
        "source_id": source_id,
        "G1": {
            "X": X_G1,
            "distance_mpc": float(distance_emp),
            "network_snr": float(snr_emp_final),
            "detector_snrs": dict(snrs_emp_final),
        },
        "R": {
            "X": X_R,
            "distance_mpc": float(distance_emp),
            "network_snr": float(snr_emp_final),
            "detector_snrs": dict(snrs_emp_final),
        },
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------
# HDF5 schema / persistence
# ---------------------------------------------------------------------

def create_domain_hdf5(
    path: Path,
    *,
    n_samples: int,
    domain: str,
):
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "X",
            shape=(n_samples, len(DETECTORS), CONFIG.length),
            dtype=np.float32,
            chunks=(1, len(DETECTORS), CONFIG.length),
            compression="lzf",
        )
        h5.create_dataset(
            "y",
            shape=(n_samples, 3),
            dtype=np.float32,
        )

        h5.create_dataset(
            "source_id",
            shape=(n_samples,),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "manifest_index",
            shape=(n_samples,),
            dtype=np.int64,
        )
        h5.create_dataset(
            "split",
            shape=(n_samples,),
            dtype=string_dtype,
        )

        for name in (
            "target_network_snr",
            "final_network_snr",
            "distance_mpc",
            "snr_H1",
            "snr_L1",
            "snr_V1",
            "geocentric_time",
            "placement_offset_s",
            "full_network_duration",
            "required_final_duration",
        ):
            h5.create_dataset(
                name,
                shape=(n_samples,),
                dtype=np.float64,
            )

        h5.create_dataset(
            "file_group_id",
            shape=(n_samples,),
            dtype=np.int64,
        )
        h5.create_dataset(
            "block_id",
            shape=(n_samples,),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "noise_crop_id",
            shape=(n_samples,),
            dtype=string_dtype,
        )
        h5.create_dataset(
            "is_truncated",
            shape=(n_samples,),
            dtype=np.bool_,
        )

        h5.create_dataset(
            "status",
            data=np.zeros(n_samples, dtype=np.uint8),
        )

        h5.attrs["experiment"] = EXPERIMENT_NAME
        h5.attrs["source_experiment"] = "M11.6"
        h5.attrs["domain"] = domain
        h5.attrs["sampling_frequency_hz"] = CONFIG.sampling_frequency
        h5.attrs["duration_s"] = CONFIG.duration
        h5.attrs["detector_order"] = ",".join(DETECTORS)
        h5.attrs["input_normalization"] = "none"
        h5.attrs["representation"] = REPRESENTATION
        h5.attrs["labels"] = "chirp_mass,total_mass,chi_eff"
        h5.attrs["waveform_approximant"] = CONFIG.waveform_approximant
        h5.attrs[
            "low_frequency_cutoff_hz"
        ] = CONFIG.low_frequency_cutoff


def write_domain_sample(
    h5,
    *,
    output_index: int,
    manifest_row: pd.Series,
    result: dict,
    domain: str,
):
    domain_result = result[domain]
    diagnostics = result["diagnostics"]

    X = np.asarray(domain_result["X"], dtype=np.float32)

    if X.shape != (len(DETECTORS), CONFIG.length):
        raise ValueError(
            f"{domain}: unexpected X shape {X.shape}"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError(
            f"{domain}: X contains non-finite values."
        )

    h5["X"][output_index] = X

    h5["y"][output_index] = np.asarray(
        [
            diagnostics["chirp_mass"],
            diagnostics["total_mass"],
            diagnostics["chi_eff"],
        ],
        dtype=np.float32,
    )

    h5["source_id"][output_index] = str(
        diagnostics["source_id"]
    )
    h5["manifest_index"][output_index] = int(
        manifest_row["manifest_index"]
    )
    h5["split"][output_index] = str(
        manifest_row["split"]
    )

    h5["target_network_snr"][output_index] = float(
        diagnostics["target_network_snr"]
    )
    h5["final_network_snr"][output_index] = float(
        domain_result["network_snr"]
    )
    h5["distance_mpc"][output_index] = float(
        domain_result["distance_mpc"]
    )

    for ifo in DETECTORS:
        h5[f"snr_{ifo}"][output_index] = float(
            domain_result["detector_snrs"][ifo]
        )

    h5["geocentric_time"][output_index] = float(
        diagnostics["geocentric_time"]
    )
    h5["placement_offset_s"][output_index] = float(
        diagnostics["placement_offset_s"]
    )
    h5["full_network_duration"][output_index] = float(
        diagnostics["full_network_duration"]
    )
    h5["required_final_duration"][output_index] = float(
        diagnostics["required_final_duration"]
    )

    h5["file_group_id"][output_index] = int(
        diagnostics["file_group_id"]
    )
    h5["block_id"][output_index] = str(
        diagnostics["block_id"]
    )
    h5["noise_crop_id"][output_index] = str(
        diagnostics["noise_crop_id"]
    )
    h5["is_truncated"][output_index] = bool(
        diagnostics["is_truncated"]
    )

    h5["status"][output_index] = np.uint8(1)


# ---------------------------------------------------------------------
# Resume / integrity helpers
# ---------------------------------------------------------------------

def completed_mask_from_files(h5_by_domain: dict) -> np.ndarray:
    masks = [
        np.asarray(
            h5_by_domain[domain]["status"][:],
            dtype=np.uint8,
        )
        for domain in DOMAINS
    ]
    stacked = np.stack(masks, axis=0)
    return np.all(stacked == 1, axis=0)


def validate_existing_output_files(
    h5_by_domain: dict,
    *,
    n_samples: int,
):
    for domain in DOMAINS:
        h5 = h5_by_domain[domain]

        if h5["X"].shape != (
            n_samples,
            len(DETECTORS),
            CONFIG.length,
        ):
            raise ValueError(
                f"{domain}: output X shape mismatch: "
                f"{h5['X'].shape}"
            )

        if h5["status"].shape != (n_samples,):
            raise ValueError(
                f"{domain}: status shape mismatch: "
                f"{h5['status'].shape}"
            )

        if h5.attrs.get("input_normalization") != "none":
            raise ValueError(
                f"{domain}: existing output is not marked as no-zscore."
            )

        if h5.attrs.get("representation") != REPRESENTATION:
            raise ValueError(
                f"{domain}: unexpected representation attr: "
                f"{h5.attrs.get('representation')}"
            )


def write_run_metadata(
    output_dir: Path,
    *,
    args,
    manifest: pd.DataFrame,
):
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "source_experiment": "M11.6",
        "scientific_intervention": (
            "Persist processed network before final "
            "per-sample/per-detector z-score."
        ),
        "manifest": str(args.manifest),
        "file_groups": str(args.file_groups),
        "data_root": str(args.data_root),
        "output_dir": str(output_dir),
        "n_manifest_rows": int(len(manifest)),
        "domains": list(DOMAINS),
        "detectors": list(DETECTORS),
        "build_seed": m116.BUILD_SEED,
        "placement_seed_offset": m116.PLACEMENT_SEED_OFFSET,
        "gaussian_seed_offset": m116.GAUSSIAN_SEED_OFFSET,
        "sampling_frequency": CONFIG.sampling_frequency,
        "duration": CONFIG.duration,
        "processing_duration": CONFIG.processing_duration,
        "waveform_approximant": CONFIG.waveform_approximant,
        "low_frequency_cutoff": CONFIG.low_frequency_cutoff,
        "input_normalization": "none",
        "representation": REPRESENTATION,
        "flush_every": int(args.flush_every),
        "max_sources": (
            None if args.max_sources is None
            else int(args.max_sources)
        ),
    }

    with open(
        output_dir / "generation_run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------
# Production driver
# ---------------------------------------------------------------------

def run(args):
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)
    file_groups_path = Path(args.file_groups)
    output_dir = Path(args.output_dir)
    gwosc_cache_dir = Path(args.gwosc_cache)

    output_dir.mkdir(parents=True, exist_ok=True)
    gwosc_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        manifest_path
    ).reset_index(drop=True)

    if "manifest_index" not in manifest.columns:
        manifest["manifest_index"] = np.arange(
            len(manifest),
            dtype=np.int64,
        )

    if not manifest["source_id"].is_unique:
        raise ValueError(
            "source_id must be unique in the master manifest."
        )

    if not manifest["manifest_index"].is_unique:
        raise ValueError(
            "manifest_index must be unique."
        )

    required_cols = {
        "source_id",
        "source_index",
        "manifest_index",
        "split",
        "mass_1",
        "mass_2",
        "reference_distance_mpc",
        "inclination",
        "ra",
        "dec",
        "spin_1z",
        "spin_2z",
        "polarization_angle",
        "target_network_snr",
        "file_group_id",
        "block_id",
        "noise_crop_id",
        "processing_start",
        "processing_end",
        "psd_start",
        "psd_end",
    }

    missing = sorted(
        required_cols - set(manifest.columns)
    )
    if missing:
        raise KeyError(
            f"Master manifest missing required columns: {missing}"
        )

    n_total = len(manifest)

    if args.max_sources is not None:
        max_sources = int(args.max_sources)
        if max_sources <= 0:
            raise ValueError(
                "--max-sources must be positive."
            )
        active_indices = set(
            manifest["manifest_index"]
            .sort_values()
            .head(max_sources)
            .astype(int)
            .tolist()
        )
    else:
        active_indices = set(
            manifest["manifest_index"]
            .astype(int)
            .tolist()
        )

    file_groups_df = pd.read_csv(
        file_groups_path
    )

    required_fg_cols = {
        "file_group_id",
        "anchor_event",
    }
    missing_fg = sorted(
        required_fg_cols - set(file_groups_df.columns)
    )
    if missing_fg:
        raise KeyError(
            f"file_groups CSV missing columns: {missing_fg}"
        )

    urls_by_file_group = m116.resolve_file_group_urls(
        file_groups_df
    )

    output_paths = {
        "G1": output_dir / "m11_9_G1_noz_26k.h5",
        "R": output_dir / "m11_9_R_noz_26k.h5",
    }

    if args.overwrite:
        for path in output_paths.values():
            if path.exists():
                path.unlink()

    all_exist = all(
        path.exists()
        for path in output_paths.values()
    )
    any_exist = any(
        path.exists()
        for path in output_paths.values()
    )

    if any_exist and not all_exist:
        raise RuntimeError(
            "Partial M11.9 output set found. Restore both "
            "G1/R files or rerun with --overwrite."
        )

    if not all_exist:
        for domain, path in output_paths.items():
            print(
                f"Creating {domain}_noz: {path}"
            )
            create_domain_hdf5(
                path,
                n_samples=n_total,
                domain=domain,
            )
    else:
        print(
            "Reusing existing M11.9 output files for resume."
        )

    h5_by_domain = {
        domain: h5py.File(path, "r+")
        for domain, path in output_paths.items()
    }

    log_path = (
        output_dir
        / "generation_failures.csv"
    )

    try:
        validate_existing_output_files(
            h5_by_domain,
            n_samples=n_total,
        )

        complete_mask = completed_mask_from_files(
            h5_by_domain
        )

        write_run_metadata(
            output_dir,
            args=args,
            manifest=manifest,
        )

        done_active = sum(
            bool(complete_mask[i])
            for i in active_indices
        )
        print(
            f"Already complete in active set: "
            f"{done_active}/{len(active_indices)}"
        )

        pending_rows = manifest[
            manifest["manifest_index"]
            .astype(int)
            .isin(active_indices)
        ].copy()

        pending_rows = pending_rows[
            ~pending_rows["manifest_index"]
            .astype(int)
            .map(
                lambda i: bool(
                    complete_mask[i]
                )
            )
        ]

        pending_group_ids = (
            pending_rows["file_group_id"]
            .astype(int)
            .drop_duplicates()
            .tolist()
        )

        print(
            f"Pending file groups: "
            f"{len(pending_group_ids)}"
        )

        total_written_this_run = 0

        progress = tqdm(
            total=len(active_indices),
            initial=done_active,
            desc="M11.9 no-z generation",
            unit="source",
            dynamic_ncols=True,
            smoothing=0.1,
            mininterval=5.0,
        )

        for group_pos, file_group_id in enumerate(
            pending_group_ids,
            start=1,
        ):
            group_rows = manifest[
                (
                    manifest["file_group_id"]
                    .astype(int)
                    == int(file_group_id)
                )
                & manifest["manifest_index"]
                .astype(int)
                .isin(active_indices)
            ].copy()

            group_rows = group_rows.sort_values(
                "manifest_index"
            )

            group_rows = group_rows[
                ~group_rows["manifest_index"]
                .astype(int)
                .map(
                    lambda i: bool(
                        complete_mask[i]
                    )
                )
            ]

            if len(group_rows) == 0:
                continue

            urls = urls_by_file_group[
                int(file_group_id)
            ]

            long_strains = (
                m116.load_file_group_strains(
                    file_group_id=int(file_group_id),
                    urls=urls,
                    gwosc_cache_dir=gwosc_cache_dir,
                )
            )

            local_psd_cache = {}

            try:
                for _, row in group_rows.iterrows():
                    manifest_index = int(
                        row["manifest_index"]
                    )
                    source_id = str(
                        row["source_id"]
                    )

                    if bool(
                        complete_mask[
                            manifest_index
                        ]
                    ):
                        continue

                    try:
                        result = (
                            build_paired_noz_domains_for_manifest_row(
                                row,
                                long_strains=long_strains,
                                psd_cache=local_psd_cache,
                            )
                        )

                        if not np.isclose(
                            result["G1"]["distance_mpc"],
                            result["R"]["distance_mpc"],
                            rtol=0.0,
                            atol=1e-10,
                        ):
                            raise RuntimeError(
                                f"{source_id}: "
                                "G1/R distance pairing failed."
                            )

                        for domain in DOMAINS:
                            write_domain_sample(
                                h5_by_domain[domain],
                                output_index=manifest_index,
                                manifest_row=row,
                                result=result,
                                domain=domain,
                            )

                        total_written_this_run += 1

                        if (
                            total_written_this_run
                            % int(args.flush_every)
                            == 0
                        ):
                            for h5 in (
                                h5_by_domain.values()
                            ):
                                h5.flush()

                        complete_mask[
                            manifest_index
                        ] = True

                        progress.update(1)

                        progress.set_postfix(
                            {
                                "group": (
                                    f"{group_pos}/"
                                    f"{len(pending_group_ids)}"
                                ),
                                "file_group_id": int(
                                    file_group_id
                                ),
                                "psd_blocks": len(
                                    local_psd_cache
                                ),
                            },
                            refresh=False,
                        )

                        del result

                    except Exception as exc:
                        failure = pd.DataFrame(
                            [
                                {
                                    "source_id": source_id,
                                    "manifest_index": manifest_index,
                                    "file_group_id": int(
                                        file_group_id
                                    ),
                                    "block_id": str(
                                        row["block_id"]
                                    ),
                                    "exception_type": type(
                                        exc
                                    ).__name__,
                                    "message": str(exc),
                                }
                            ]
                        )

                        write_header = (
                            not log_path.exists()
                        )
                        failure.to_csv(
                            log_path,
                            mode="a",
                            header=write_header,
                            index=False,
                        )

                        for h5 in (
                            h5_by_domain.values()
                        ):
                            h5.flush()

                        progress.write(
                            f"FAIL source_id={source_id} "
                            f"file_group_id={file_group_id} "
                            f"{type(exc).__name__}: {exc}"
                        )

                        raise

                progress.write(
                    f"Completed file_group_id="
                    f"{file_group_id} "
                    f"| group {group_pos}/"
                    f"{len(pending_group_ids)} "
                    f"| sources={len(group_rows)} "
                    f"| PSD blocks="
                    f"{len(local_psd_cache)}"
                )

            finally:
                for h5 in (
                    h5_by_domain.values()
                ):
                    h5.flush()

                del local_psd_cache
                del long_strains

                gc.collect()

        final_complete_mask = (
            completed_mask_from_files(
                h5_by_domain
            )
        )

        active_complete = sum(
            bool(final_complete_mask[i])
            for i in active_indices
        )

        progress.close()

        print()
        print("=" * 78)
        print("M11.9 GENERATION SUMMARY")
        print("=" * 78)
        print(
            f"Active sources complete: "
            f"{active_complete}/"
            f"{len(active_indices)}"
        )
        print(
            f"Written this run: "
            f"{total_written_this_run}"
        )

        if (
            active_complete
            != len(active_indices)
        ):
            raise RuntimeError(
                "Run ended with incomplete "
                "active sources."
            )

        active_sorted = sorted(
            active_indices
        )

        ids = {}
        indices = {}

        for domain in DOMAINS:
            ids[domain] = (
                h5_by_domain[domain][
                    "source_id"
                ][active_sorted]
                .astype(str)
            )
            indices[domain] = (
                h5_by_domain[domain][
                    "manifest_index"
                ][active_sorted]
            )

        if not np.array_equal(
            ids["G1"],
            ids["R"],
        ):
            raise RuntimeError(
                "Cross-domain source_id mismatch: "
                "G1 vs R"
            )

        if not np.array_equal(
            indices["G1"],
            indices["R"],
        ):
            raise RuntimeError(
                "Cross-domain manifest_index "
                "mismatch: G1 vs R"
            )

        g1_dist = h5_by_domain["G1"][
            "distance_mpc"
        ][active_sorted]
        r_dist = h5_by_domain["R"][
            "distance_mpc"
        ][active_sorted]

        if not np.allclose(
            g1_dist,
            r_dist,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Persisted G1/R distance "
                "pairing failed."
            )

        print(
            "Cross-domain pairing integrity: PASS"
        )
        print(
            "M11.9 no-z streaming generation: PASS"
        )

    finally:
        for h5 in h5_by_domain.values():
            try:
                h5.close()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired M11.9 G1/R datasets "
            "without final per-sample/per-detector "
            "z-score."
        )
    )

    default_data_root = Path(
        "/data/vserrano/cbc_pe_data"
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            default_data_root
            / "processed"
            / "m11_6_manifests"
            / "m11_6_master_experiment_manifest.csv"
        ),
    )

    parser.add_argument(
        "--file-groups",
        type=Path,
        default=(
            default_data_root
            / "processed"
            / "m11_6_manifests"
            / "m11_6_file_groups.csv"
        ),
    )

    parser.add_argument(
        "--gwosc-cache",
        type=Path,
        default=(
            default_data_root
            / "gwosc_cache"
            / "m11"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            default_data_root
            / "processed"
            / "m11_9_noz_domains"
        ),
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help=(
            "Flush both HDF5 files after this "
            "many completed source pairs."
        ),
    )

    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help=(
            "Optional pilot limit. Uses the first N "
            "manifest_index values without changing "
            "dataset geometry."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing M11.9 G1/R output "
            "files and start from zero."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.flush_every <= 0:
        raise ValueError(
            "--flush-every must be positive."
        )

    run(args)
