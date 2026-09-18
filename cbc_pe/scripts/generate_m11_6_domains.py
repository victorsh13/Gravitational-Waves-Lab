#!/usr/bin/env python3
"""
M11.6 streaming generation of paired G0/G1/R domains.

Scientific contract
-------------------
For each source_id from the frozen M11.6 master manifest:

    G0 = h(theta, d_G0)  + Gaussian noise from analytical PSD
    G1 = h(theta, d_emp) + Gaussian noise from empirical off-source PSD
    R  = h(theta, d_emp) + real off-source detector strain

The three domains share source parameters, physical GPS environment,
random-contained placement, target network optimal SNR, processing,
and final M10 per-sample/per-detector z-score.

Operational contract
--------------------
- Process one file_group_id at a time.
- Keep only one HLV real-strain environment in RAM.
- Cache empirical PSDs only locally by block_id.
- Write G0/G1/R immediately to HDF5.
- Persist a paired status vector for restart/resume.
- Preserve manifest_index and source_id across all domains.
- Fail fast on an unexpected source-generation error.

This script is the production promotion of the validated M11.7 streaming pilot.
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

from pycbc.noise import gaussian
from pycbc.psd import interpolate

from src.config import SimulationConfig
from src.detectors import DetectorProjector
from src.injection import SignalInjector
from src.models.dataset import normalize_input_per_sample_per_detector_zscore
from src.noise import NoiseModel
from src.parameters import CBCParameters
from src.processing import SignalProcessor
from src.real_data.catalog import (
    build_gwosc_urls_for_events,
    fetch_gwosc_catalog_events,
    gwosc_events_to_parameter_df,
)
from src.real_data.gwosc_utils import (
    download_if_needed,
    read_gwosc_hdf5_as_pycbc_timeseries,
)
from src.snr import (
    compute_network_optimal_snr,
    rescale_distance_for_target_network_snr,
    validate_snr_rescaling,
)
from src.waveform import WaveformGenerator
from src.windowing import ProjectedNetworkWindowSelector


DOMAINS = ("G0", "G1", "R")
DETECTORS = ("H1", "L1", "V1")

CATALOG = "GWTC-3-confident"
SAMPLE_RATE = 4096

PSD_SEGMENT_DURATION_S = 8.0

BUILD_SEED = 24680
PLACEMENT_SEED_OFFSET = 100_000
GAUSSIAN_SEED_OFFSET = 200_000

M10_INPUT_ZSCORE_EPS = 1e-6


# ---------------------------------------------------------------------
# Frozen M11.6 physical configuration
# ---------------------------------------------------------------------

CONFIG = SimulationConfig(
    simulation_regime="BBH",
    waveform_family="IMR",
    sampling_frequency=4096,
    duration=4.0,
    low_frequency_cutoff=30.0,
    waveform_approximant="SEOBNRv4_opt",
    target_network_snr_range=None,
    snr_relative_tolerance=0.05,
    snr_on_truncated_signal=True,
    truncation_policy="keep_last_segment",
    required_final_duration=1.0,
    safe_margin_start=0.0,
    safe_margin_end=0.0,
    processing_context_start_samples=1664,
    processing_context_end_samples=1664,
)

WAVEFORM_GENERATOR = WaveformGenerator(CONFIG)
DETECTOR_PROJECTOR = DetectorProjector(list(DETECTORS))
WINDOW_SELECTOR = ProjectedNetworkWindowSelector(CONFIG)

PROCESSOR = SignalProcessor(
    config=CONFIG,
    whitening_method="psd",
    apply_highpass=True,
    apply_lowpass=True,
    apply_standardization=False,
    output_mode="crop_to_config",
    whitening_low_frequency_cutoff=30.0,
    whitening_max_filter_duration=0.5,
    whitening_trunc_method="hann",
    highpass_frequency=30.0,
    lowpass_frequency=512.0,
    fir_order=256,
    fir_beta=5.0,
    remove_corrupted=True,
    rng=np.random.default_rng(BUILD_SEED + 300),
)

G0_NOISE_MODEL = NoiseModel(CONFIG)
PSDS_G0_SNR = {
    ifo: G0_NOISE_MODEL.get_psd(ifo, length=CONFIG.length)
    for ifo in DETECTORS
}
PSDS_G0_PROC = {
    ifo: G0_NOISE_MODEL.get_psd(ifo, length=CONFIG.processing_length)
    for ifo in DETECTORS
}


# ---------------------------------------------------------------------
# Empirical PSD helpers
# ---------------------------------------------------------------------

def estimate_raw_offsource_psd(
    strain,
    *,
    psd_start: float,
    psd_end: float,
    delta_f: float,
    target_flength: int,
    psd_segment_duration: float = PSD_SEGMENT_DURATION_S,
):
    """
    Raw empirical PSD:
        strain -> Welch -> interpolation

    No inverse-spectrum truncation here.
    SignalProcessor applies whitening conditioning exactly once later.
    """
    psd_data = strain.time_slice(float(psd_start), float(psd_end))

    if len(psd_data) == 0:
        raise ValueError("PSD reference window is empty.")

    if not np.all(np.isfinite(psd_data.numpy())):
        raise ValueError("PSD reference contains non-finite values.")

    psd = psd_data.psd(float(psd_segment_duration))
    psd = interpolate(psd, float(delta_f))

    target_flength = int(target_flength)

    if len(psd) > target_flength:
        psd = psd[:target_flength]
    elif len(psd) < target_flength:
        raise ValueError(
            f"Interpolated PSD shorter than target_flength: "
            f"{len(psd)} < {target_flength}"
        )

    if not np.all(np.isfinite(psd.numpy())):
        raise ValueError("Raw empirical PSD contains non-finite values.")

    return psd


def estimate_block_psds_for_detector(
    *,
    strain,
    psd_start: float,
    psd_end: float,
):
    return {
        "snr": estimate_raw_offsource_psd(
            strain,
            psd_start=psd_start,
            psd_end=psd_end,
            delta_f=CONFIG.delta_f,
            target_flength=CONFIG.flength,
        ),
        "proc": estimate_raw_offsource_psd(
            strain,
            psd_start=psd_start,
            psd_end=psd_end,
            delta_f=CONFIG.processing_delta_f,
            target_flength=CONFIG.processing_flength,
        ),
    }


def get_block_psds(
    row: pd.Series,
    *,
    long_strains: dict,
    psd_cache: dict,
):
    block_id = str(row["block_id"])

    if block_id in psd_cache:
        return psd_cache[block_id]

    psds = {"snr": {}, "proc": {}}

    for ifo in DETECTORS:
        result = estimate_block_psds_for_detector(
            strain=long_strains[ifo],
            psd_start=float(row["psd_start"]),
            psd_end=float(row["psd_end"]),
        )
        psds["snr"][ifo] = result["snr"]
        psds["proc"][ifo] = result["proc"]

    psd_cache[block_id] = psds
    return psds


# ---------------------------------------------------------------------
# Real-strain helpers
# ---------------------------------------------------------------------

def resolve_file_group_urls(
    file_groups_df: pd.DataFrame,
) -> dict[int, dict[str, str]]:
    """
    Resolve H1/L1/V1 4096-s GWOSC URLs once from anchor_event metadata.
    This does not load strain into RAM.
    """
    if "anchor_event" not in file_groups_df.columns:
        raise KeyError(
            "file_groups CSV must contain an 'anchor_event' column."
        )

    anchor_events = (
        file_groups_df["anchor_event"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    events_raw = fetch_gwosc_catalog_events(
        catalog=CATALOG,
        include_default_parameters=True,
    )
    events_df = gwosc_events_to_parameter_df(
        events_raw,
        catalog_name=CATALOG,
    )

    selected = events_df[
        events_df["event"].astype(str).isin(anchor_events)
    ].copy()

    missing_events = sorted(
        set(anchor_events) - set(selected["event"].astype(str))
    )
    if missing_events:
        raise ValueError(
            f"Anchor events missing from {CATALOG}: {missing_events}"
        )

    urls_by_event, failed_df = build_gwosc_urls_for_events(
        selected,
        catalog=CATALOG,
        sample_rate=SAMPLE_RATE,
    )

    if len(failed_df) > 0:
        raise RuntimeError(
            "Failed to resolve some GWOSC URLs:\n"
            + failed_df.to_string(index=False)
        )

    out: dict[int, dict[str, str]] = {}

    for _, row in file_groups_df.iterrows():
        file_group_id = int(row["file_group_id"])
        event = str(row["anchor_event"])

        event_urls = urls_by_event[event]

        missing = [ifo for ifo in DETECTORS if ifo not in event_urls]
        if missing:
            raise KeyError(
                f"file_group_id={file_group_id}, event={event}: "
                f"missing detector URLs {missing}"
            )

        out[file_group_id] = {
            ifo: str(event_urls[ifo])
            for ifo in DETECTORS
        }

    return out


def load_file_group_strains(
    *,
    file_group_id,
    urls,
    gwosc_cache_dir,
):
    """
    Load one complete HLV GWOSC environment.

    The full 4096 s files are not required to be globally finite.
    Finiteness is enforced later only on the exact intervals used by
    the experiment:

    - the off-source PSD reference window;
    - the real processing/noise context.

    This matches the M11.6 bank-selection contract and avoids scanning
    the full long strain unnecessarily.
    """
    strains = {}

    for ifo in DETECTORS:

        local_path = download_if_needed(
            urls[ifo],
            cache_dir=gwosc_cache_dir,
            force=False,
        )

        strains[ifo] = (
            read_gwosc_hdf5_as_pycbc_timeseries(
                local_path
            )
        )

    return strains


def extract_exact_processing_context(
    strain,
    *,
    start_time: float,
    expected_length: int,
):
    """
    Extract an exact CONFIG.processing_length segment by integer sample index.
    """
    start_time = float(start_time)
    delta_t = float(strain.delta_t)

    start_index = int(
        round(
            (start_time - float(strain.start_time))
            / delta_t
        )
    )
    end_index = start_index + int(expected_length)

    if start_index < 0 or end_index > len(strain):
        raise ValueError(
            "Requested processing context outside available strain: "
            f"start_index={start_index}, end_index={end_index}, len={len(strain)}"
        )

    segment = strain[start_index:end_index]
    segment.start_time = start_time

    if len(segment) != int(expected_length):
        raise ValueError(
            f"Processing context length mismatch: "
            f"{len(segment)} != {expected_length}"
        )

    if not np.all(np.isfinite(segment.numpy())):
        raise ValueError("Processing context contains non-finite values.")

    return segment


# ---------------------------------------------------------------------
# Signal/noise/processing helpers
# ---------------------------------------------------------------------

def build_signal_network_for_params(
    params: CBCParameters,
    *,
    geocentric_time: float,
    final_start: float,
    injector: SignalInjector,
):
    waveform = WAVEFORM_GENERATOR.generate(params)

    projection = DETECTOR_PROJECTOR.project(
        h_plus=waveform.h_plus,
        h_cross=waveform.h_cross,
        parameters=params,
        geocentric_coalescence_time=geocentric_time,
    )

    windowed = WINDOW_SELECTOR.select(
        projected_strains=projection.strains,
        max_duration=CONFIG.duration,
    )

    zeros = {
        ifo: injector.build_zero_strain(
            start_time=final_start,
            length=CONFIG.length,
        )
        for ifo in DETECTORS
    }

    injected = injector.inject_network(
        noises=zeros,
        signals=windowed.strains,
    )

    segments = {
        ifo: injected[ifo].strain
        for ifo in DETECTORS
    }

    return {
        "waveform": waveform,
        "projection": projection,
        "windowed": windowed,
        "segments": segments,
    }


def build_gaussian_processing_noise(
    *,
    psds_proc: dict,
    processing_start: float,
    detector_seeds: dict[str, int],
    injector: SignalInjector,
):
    noises = {}

    for ifo in DETECTORS:
        noise = gaussian.noise_from_psd(
            psd=psds_proc[ifo],
            length=CONFIG.processing_length,
            delta_t=CONFIG.delta_t,
            seed=int(detector_seeds[ifo]),
        )

        noises[ifo] = injector.set_strain_start_time(
            strain=noise,
            start_time=processing_start,
            expected_length=CONFIG.processing_length,
        )

    return noises


def stack_and_normalize_network(network: dict):
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
        raise ValueError("Processed network contains non-finite values.")

    X = normalize_input_per_sample_per_detector_zscore(
        X_raw,
        eps=M10_INPUT_ZSCORE_EPS,
    )

    return X_raw, X


# ---------------------------------------------------------------------
# Paired G0/G1/R builder
# ---------------------------------------------------------------------

def build_paired_domains_for_manifest_row(
    row: pd.Series,
    *,
    long_strains: dict,
    psd_cache: dict,
):
    source_id = str(row["source_id"])
    source_index = int(row["source_index"])

    placement_rng = np.random.default_rng(
        BUILD_SEED + PLACEMENT_SEED_OFFSET + source_index
    )
    gaussian_rng = np.random.default_rng(
        BUILD_SEED + GAUSSIAN_SEED_OFFSET + source_index
    )

    injector = SignalInjector(
        config=CONFIG,
        rng=placement_rng,
    )

    gaussian_seeds = {
        ifo: int(gaussian_rng.integers(0, 2**32 - 1))
        for ifo in DETECTORS
    }

    params_ref = CBCParameters(
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
    waveform_ref = WAVEFORM_GENERATOR.generate(params_ref)

    projection_provisional = DETECTOR_PROJECTOR.project(
        h_plus=waveform_ref.h_plus,
        h_cross=waveform_ref.h_cross,
        parameters=params_ref,
        geocentric_coalescence_time=final_center,
    )

    windowed_provisional = WINDOW_SELECTOR.select(
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
    projection_ref = DETECTOR_PROJECTOR.project(
        h_plus=waveform_ref.h_plus,
        h_cross=waveform_ref.h_cross,
        parameters=params_ref,
        geocentric_coalescence_time=geocentric_time,
    )

    windowed_ref = WINDOW_SELECTOR.select(
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

    empirical_psds = get_block_psds(
        row,
        long_strains=long_strains,
        psd_cache=psd_cache,
    )

    psds_emp_snr = empirical_psds["snr"]
    psds_emp_proc = empirical_psds["proc"]

    _, snr_G0_ref = compute_network_optimal_snr(
        signal_segments=signal_segments_ref,
        psds=PSDS_G0_SNR,
        config=CONFIG,
    )

    _, snr_emp_ref = compute_network_optimal_snr(
        signal_segments=signal_segments_ref,
        psds=psds_emp_snr,
        config=CONFIG,
    )

    distance_G0 = rescale_distance_for_target_network_snr(
        current_distance=params_ref.distance,
        current_network_snr=snr_G0_ref,
        target_network_snr=target_snr,
    )

    distance_emp = rescale_distance_for_target_network_snr(
        current_distance=params_ref.distance,
        current_network_snr=snr_emp_ref,
        target_network_snr=target_snr,
    )

    params_G0 = params_ref.with_distance(distance_G0)
    params_emp = params_ref.with_distance(distance_emp)

    signal_G0 = build_signal_network_for_params(
        params_G0,
        geocentric_time=geocentric_time,
        final_start=final_start,
        injector=injector,
    )

    signal_emp = build_signal_network_for_params(
        params_emp,
        geocentric_time=geocentric_time,
        final_start=final_start,
        injector=injector,
    )

    snrs_G0_final, snr_G0_final = compute_network_optimal_snr(
        signal_segments=signal_G0["segments"],
        psds=PSDS_G0_SNR,
        config=CONFIG,
    )

    snrs_emp_final, snr_emp_final = compute_network_optimal_snr(
        signal_segments=signal_emp["segments"],
        psds=psds_emp_snr,
        config=CONFIG,
    )

    validate_snr_rescaling(
        final_network_snr=snr_G0_final,
        target_network_snr=target_snr,
        relative_tolerance=CONFIG.snr_relative_tolerance,
    )

    validate_snr_rescaling(
        final_network_snr=snr_emp_final,
        target_network_snr=target_snr,
        relative_tolerance=CONFIG.snr_relative_tolerance,
    )

    noise_G0 = build_gaussian_processing_noise(
        psds_proc=PSDS_G0_PROC,
        processing_start=processing_start,
        detector_seeds=gaussian_seeds,
        injector=injector,
    )

    noise_G1 = build_gaussian_processing_noise(
        psds_proc=psds_emp_proc,
        processing_start=processing_start,
        detector_seeds=gaussian_seeds,
        injector=injector,
    )

    noise_R = {
        ifo: extract_exact_processing_context(
            long_strains[ifo],
            start_time=processing_start,
            expected_length=CONFIG.processing_length,
        )
        for ifo in DETECTORS
    }

    injected_G0_results = injector.inject_network(
        noises=noise_G0,
        signals=signal_G0["projection"].strains,
    )
    injected_G1_results = injector.inject_network(
        noises=noise_G1,
        signals=signal_emp["projection"].strains,
    )
    injected_R_results = injector.inject_network(
        noises=noise_R,
        signals=signal_emp["projection"].strains,
    )

    injected_G0 = {
        ifo: injected_G0_results[ifo].strain
        for ifo in DETECTORS
    }
    injected_G1 = {
        ifo: injected_G1_results[ifo].strain
        for ifo in DETECTORS
    }
    injected_R = {
        ifo: injected_R_results[ifo].strain
        for ifo in DETECTORS
    }

    processed_G0 = PROCESSOR.process_network(
        strains=injected_G0,
        psds=PSDS_G0_PROC,
    )
    processed_G1 = PROCESSOR.process_network(
        strains=injected_G1,
        psds=psds_emp_proc,
    )
    processed_R = PROCESSOR.process_network(
        strains=injected_R,
        psds=psds_emp_proc,
    )

    _, X_G0 = stack_and_normalize_network(processed_G0)
    _, X_G1 = stack_and_normalize_network(processed_G1)
    _, X_R = stack_and_normalize_network(processed_R)

    expected_shape = (len(DETECTORS), CONFIG.length)

    for name, X in (
        ("G0", X_G0),
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

    if not np.isfinite(distance_emp):
        raise RuntimeError(
            f"{source_id}: invalid empirical distance."
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
        "G0": {
            "X": X_G0,
            "distance_mpc": float(distance_G0),
            "network_snr": float(snr_G0_final),
            "detector_snrs": dict(snrs_G0_final),
        },
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

        # Per-domain status is kept for integrity checking.
        h5.create_dataset(
            "status",
            data=np.zeros(n_samples, dtype=np.uint8),
        )

        h5.attrs["domain"] = domain
        h5.attrs["sampling_frequency_hz"] = CONFIG.sampling_frequency
        h5.attrs["duration_s"] = CONFIG.duration
        h5.attrs["detector_order"] = ",".join(DETECTORS)
        h5.attrs[
            "input_normalization"
        ] = "per_sample_per_detector_zscore"
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

    # Mark this domain row complete only after all fields are written.
    h5["status"][output_index] = np.uint8(1)


# ---------------------------------------------------------------------
# Resume / integrity helpers
# ---------------------------------------------------------------------

def completed_mask_from_files(h5_by_domain: dict) -> np.ndarray:
    """
    A source is considered complete only if all three domain files mark it 1.
    """
    masks = [
        np.asarray(h5_by_domain[domain]["status"][:], dtype=np.uint8)
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


def write_run_metadata(
    output_dir: Path,
    *,
    args,
    manifest: pd.DataFrame,
):
    metadata = {
        "manifest": str(args.manifest),
        "file_groups": str(args.file_groups),
        "data_root": str(args.data_root),
        "output_dir": str(output_dir),
        "n_manifest_rows": int(len(manifest)),
        "domains": list(DOMAINS),
        "detectors": list(DETECTORS),
        "build_seed": BUILD_SEED,
        "placement_seed_offset": PLACEMENT_SEED_OFFSET,
        "gaussian_seed_offset": GAUSSIAN_SEED_OFFSET,
        "sampling_frequency": CONFIG.sampling_frequency,
        "duration": CONFIG.duration,
        "processing_duration": CONFIG.processing_duration,
        "waveform_approximant": CONFIG.waveform_approximant,
        "low_frequency_cutoff": CONFIG.low_frequency_cutoff,
        "input_normalization": "per_sample_per_detector_zscore",
        "input_normalization_eps": M10_INPUT_ZSCORE_EPS,
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

    manifest = pd.read_csv(manifest_path).reset_index(drop=True)

    if "manifest_index" not in manifest.columns:
        manifest["manifest_index"] = np.arange(
            len(manifest),
            dtype=np.int64,
        )

    if not manifest["source_id"].is_unique:
        raise ValueError("source_id must be unique in the master manifest.")

    if not manifest["manifest_index"].is_unique:
        raise ValueError("manifest_index must be unique.")

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

    missing = sorted(required_cols - set(manifest.columns))
    if missing:
        raise KeyError(
            f"Master manifest missing required columns: {missing}"
        )

    n_total = len(manifest)

    if args.max_sources is not None:
        max_sources = int(args.max_sources)
        if max_sources <= 0:
            raise ValueError("--max-sources must be positive.")
        active_indices = set(
            manifest["manifest_index"]
            .sort_values()
            .head(max_sources)
            .astype(int)
            .tolist()
        )
    else:
        active_indices = set(
            manifest["manifest_index"].astype(int).tolist()
        )

    file_groups_df = pd.read_csv(file_groups_path)

    required_fg_cols = {"file_group_id", "anchor_event"}
    missing_fg = sorted(
        required_fg_cols - set(file_groups_df.columns)
    )
    if missing_fg:
        raise KeyError(
            f"file_groups CSV missing columns: {missing_fg}"
        )

    urls_by_file_group = resolve_file_group_urls(
        file_groups_df
    )

    output_paths = {
        domain: output_dir / f"m11_6_{domain}_26k.h5"
        for domain in DOMAINS
    }

    if args.overwrite:
        for path in output_paths.values():
            if path.exists():
                path.unlink()

    all_exist = all(path.exists() for path in output_paths.values())
    any_exist = any(path.exists() for path in output_paths.values())

    if any_exist and not all_exist:
        raise RuntimeError(
            "Partial output set found. Either restore all three domain files "
            "or rerun with --overwrite."
        )

    if not all_exist:
        for domain, path in output_paths.items():
            print(f"Creating {domain}: {path}")
            create_domain_hdf5(
                path,
                n_samples=n_total,
                domain=domain,
            )
    else:
        print("Reusing existing output files for resume.")

    h5_by_domain = {
        domain: h5py.File(path, "r+")
        for domain, path in output_paths.items()
    }

    log_path = output_dir / "generation_failures.csv"

    try:
        validate_existing_output_files(
            h5_by_domain,
            n_samples=n_total,
        )

        complete_mask = completed_mask_from_files(h5_by_domain)

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

        # Only file groups that still contain unfinished active sources.
        pending_rows = manifest[
            manifest["manifest_index"].astype(int).isin(active_indices)
        ].copy()

        pending_rows = pending_rows[
            ~pending_rows["manifest_index"]
            .astype(int)
            .map(lambda i: bool(complete_mask[i]))
        ]

        pending_group_ids = (
            pending_rows["file_group_id"]
            .astype(int)
            .drop_duplicates()
            .tolist()
        )

        print(
            f"Pending file groups: {len(pending_group_ids)}"
        )

        total_written_this_run = 0

        progress = tqdm(
            total=len(active_indices),
            initial=done_active,
            desc="M11.6 generation",
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
                (manifest["file_group_id"].astype(int) == int(file_group_id))
                & manifest["manifest_index"].astype(int).isin(active_indices)
            ].copy()

            group_rows = group_rows.sort_values(
                "manifest_index"
            )

            group_rows = group_rows[
                ~group_rows["manifest_index"]
                .astype(int)
                .map(lambda i: bool(complete_mask[i]))
            ]

            if len(group_rows) == 0:
                continue

            urls = urls_by_file_group[int(file_group_id)]

            long_strains = load_file_group_strains(
                file_group_id=int(file_group_id),
                urls=urls,
                gwosc_cache_dir=gwosc_cache_dir,
            )

            local_psd_cache = {}

            try:
                for row_pos, (_, row) in enumerate(
                    group_rows.iterrows(),
                    start=1,
                ):
                    manifest_index = int(row["manifest_index"])
                    source_id = str(row["source_id"])

                    # Re-check because status may have changed earlier in this run.
                    if bool(complete_mask[manifest_index]):
                        continue

                    try:
                        result = build_paired_domains_for_manifest_row(
                            row,
                            long_strains=long_strains,
                            psd_cache=local_psd_cache,
                        )

                        # Paired scientific contract.
                        if not np.isclose(
                            result["G1"]["distance_mpc"],
                            result["R"]["distance_mpc"],
                            rtol=0.0,
                            atol=1e-10,
                        ):
                            raise RuntimeError(
                                f"{source_id}: G1/R distance pairing failed."
                            )

                        # Write all domains first.
                        for domain in DOMAINS:
                            write_domain_sample(
                                h5_by_domain[domain],
                                output_index=manifest_index,
                                manifest_row=row,
                                result=result,
                                domain=domain,
                            )

                        # Flush periodically after complete triplets.
                        total_written_this_run += 1

                        if (
                            total_written_this_run % int(args.flush_every) == 0
                        ):
                            for h5 in h5_by_domain.values():
                                h5.flush()

                        complete_mask[manifest_index] = True

                        progress.update(1)

                        progress.set_postfix(
                            {
                                "group": f"{group_pos}/{len(pending_group_ids)}",
                                "file_group_id": int(file_group_id),
                                "psd_blocks": len(local_psd_cache),
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
                                    "file_group_id": int(file_group_id),
                                    "block_id": str(row["block_id"]),
                                    "exception_type": type(exc).__name__,
                                    "message": str(exc),
                                }
                            ]
                        )

                        write_header = not log_path.exists()
                        failure.to_csv(
                            log_path,
                            mode="a",
                            header=write_header,
                            index=False,
                        )

                        for h5 in h5_by_domain.values():
                            h5.flush()

                        progress.write(
                            f"FAIL source_id={source_id} "
                            f"file_group_id={file_group_id} "
                            f"{type(exc).__name__}: {exc}"
                        )

                        raise

                progress.write(
                    f"Completed file_group_id={file_group_id} "
                    f"| group {group_pos}/{len(pending_group_ids)} "
                    f"| sources={len(group_rows)} "
                    f"| PSD blocks={len(local_psd_cache)}"
                )
        
            finally:
                # Flush and release this real environment before next group.
                for h5 in h5_by_domain.values():
                    h5.flush()

                del local_psd_cache
                del long_strains

                gc.collect()

        # Final integrity state.
        final_complete_mask = completed_mask_from_files(
            h5_by_domain
        )

        active_complete = sum(
            bool(final_complete_mask[i])
            for i in active_indices
        )

        progress.close()

        print()
        print("=" * 78)
        print("GENERATION SUMMARY")
        print("=" * 78)
        print(
            f"Active sources complete: "
            f"{active_complete}/{len(active_indices)}"
        )
        print(
            f"Written this run: {total_written_this_run}"
        )

        if active_complete != len(active_indices):
            raise RuntimeError(
                "Run ended with incomplete active sources."
            )

        # Pairing integrity on active rows.
        active_sorted = sorted(active_indices)

        ids = {}
        indices = {}
        for domain in DOMAINS:
            ids[domain] = (
                h5_by_domain[domain]["source_id"][active_sorted]
                .astype(str)
            )
            indices[domain] = (
                h5_by_domain[domain]["manifest_index"][active_sorted]
            )

        for domain in ("G1", "R"):
            if not np.array_equal(ids["G0"], ids[domain]):
                raise RuntimeError(
                    f"Cross-domain source_id mismatch: G0 vs {domain}"
                )
            if not np.array_equal(
                indices["G0"],
                indices[domain],
            ):
                raise RuntimeError(
                    f"Cross-domain manifest_index mismatch: "
                    f"G0 vs {domain}"
                )

        g1_dist = h5_by_domain["G1"]["distance_mpc"][active_sorted]
        r_dist = h5_by_domain["R"]["distance_mpc"][active_sorted]

        if not np.allclose(
            g1_dist,
            r_dist,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Persisted G1/R distance pairing failed."
            )

        print("Cross-domain pairing integrity: PASS")
        print("M11.6 streaming generation: PASS")

    finally:
        for h5 in h5_by_domain.values():
            try:
                h5.close()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired M11.6 G0/G1/R datasets "
            "with one real HLV environment in RAM at a time."
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
            / "m11_6_domains"
        ),
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Flush all three HDF5 files after this many completed source triplets.",
    )

    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. Uses the first N manifest_index values "
            "without changing dataset geometry."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing G0/G1/R output files and start from zero.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive.")

    run(args)
