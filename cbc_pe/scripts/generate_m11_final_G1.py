#!/usr/bin/env python3
"""Generate only M11-final G1 from its frozen manifest; never sample sources.

Physics helpers below are copied from generate_m11_6_domains.py. Only the
configured SNR band, manifest seeds and G1-only orchestration differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir, resolve_data_root
from scripts.prepare_m11_final_manifest import load_bank

DETECTORS = ("H1", "L1", "V1")
LABELS = ("chirp_mass", "total_mass", "chi_eff")
PSD_SEGMENT_DURATION_S = 8.0
M10_INPUT_ZSCORE_EPS = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_physics(spec):
    # Lazy imports keep --help and --validate-only free of waveform initialization.
    global CONFIG, WAVEFORM_GENERATOR, DETECTOR_PROJECTOR, WINDOW_SELECTOR, PROCESSOR
    global SignalInjector, CBCParameters, gaussian, interpolate
    global compute_network_optimal_snr, rescale_distance_for_target_network_snr
    global validate_snr_rescaling, normalize_input_per_sample_per_detector_zscore
    from pycbc.noise import gaussian
    from pycbc.psd import interpolate
    from src.config import SimulationConfig
    from src.detectors import DetectorProjector
    from src.injection import SignalInjector
    from src.models.dataset import normalize_input_per_sample_per_detector_zscore
    from src.parameters import CBCParameters
    from src.processing import SignalProcessor
    from src.snr import (compute_network_optimal_snr,
                         rescale_distance_for_target_network_snr, validate_snr_rescaling)
    from src.waveform import WaveformGenerator
    from src.windowing import ProjectedNetworkWindowSelector

    CONFIG = SimulationConfig(**spec["simulation"])
    require(CONFIG.snr_high_frequency_cutoff == 512.0, "Expected 512 Hz SNR cutoff")
    WAVEFORM_GENERATOR = WaveformGenerator(CONFIG)
    DETECTOR_PROJECTOR = DetectorProjector(list(DETECTORS))
    WINDOW_SELECTOR = ProjectedNetworkWindowSelector(CONFIG)
    PROCESSOR = SignalProcessor(config=CONFIG, **spec["signal_processor"])


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


def build_g1_row(
    row: pd.Series,
    *,
    long_strains: dict,
    psd_cache: dict,
):
    source_id = str(row["source_id"])
    source_index = int(row["source_index"])

    placement_rng = np.random.default_rng(
        int(row["placement_seed"])
    )
    gaussian_rng = np.random.default_rng(
        int(row["gaussian_seed"])
    )

    injector = SignalInjector(
        config=CONFIG,
        rng=placement_rng,
    )

    gaussian_seeds = {
        ifo: int(gaussian_rng.integers(0, 2**32 - 1))
        for ifo in DETECTORS
    }

    require(len(set(gaussian_seeds.values())) == 3, "Detector Gaussian seed collision")

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

    require(abs(final_center - float(row["reference_geocentric_time"])) < CONFIG.delta_t,
            "Reference GPS does not match assigned final segment")

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

    _, snr_ref = compute_network_optimal_snr(signal_segments_ref, psds_emp_snr, CONFIG)
    distance = rescale_distance_for_target_network_snr(
        params_ref.distance, snr_ref, target_snr,
    )
    signal = build_signal_network_for_params(
        params_ref.with_distance(distance), geocentric_time=geocentric_time,
        final_start=final_start, injector=injector,
    )
    detector_snrs, final_snr = compute_network_optimal_snr(
        signal["segments"], psds_emp_snr, CONFIG,
    )
    validate_snr_rescaling(final_snr, target_snr, CONFIG.snr_relative_tolerance)
    noise = build_gaussian_processing_noise(
        psds_proc=psds_emp_proc, processing_start=processing_start,
        detector_seeds=gaussian_seeds, injector=injector,
    )
    injected = injector.inject_network(noises=noise, signals=signal["projection"].strains)
    processed = PROCESSOR.process_network(
        strains={ifo: injected[ifo].strain for ifo in DETECTORS}, psds=psds_emp_proc,
    )
    _, X = stack_and_normalize_network(processed)
    require(np.isfinite(X).all(), "Non-finite normalized input")
    metadata = {
        "reference_network_snr": snr_ref,
        "final_network_snr": final_snr,
        "final_distance": distance,
        "actual_geocentric_time": geocentric_time,
        "placement_offset_s": placement_offset_s,
        "full_network_duration": windowed_ref.metadata.full_network_duration,
        "required_final_duration": windowed_ref.metadata.required_available_final_duration,
        "is_truncated": windowed_ref.metadata.is_truncated,
    }
    for ifo in DETECTORS:
        metadata[f"snr_{ifo}"] = detector_snrs[ifo]
        metadata[f"gaussian_seed_{ifo}"] = gaussian_seeds[ifo]
    return np.asarray(X, dtype=np.float32), metadata


FLOAT_OUTPUTS = (
    "reference_network_snr", "final_network_snr", "final_distance",
    "actual_geocentric_time", "placement_offset_s", "full_network_duration",
    "required_final_duration", "snr_H1", "snr_L1", "snr_V1",
)
ALIASES = {"distance_mpc": "final_distance", "geocentric_time": "actual_geocentric_time",
           "reference_distance": "reference_distance_mpc"}


def read_inputs(args):
    spec = json.loads(args.config.read_text())
    config_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = resolve_data_root(cli_data_root=args.data_root, config_data_root=spec["data_root"])
    directory = dataset_processed_dir(root, spec["output"]["dataset_id"])
    path = args.manifest or directory / spec["output"]["manifest_file"]
    manifest_hash = sha256_file(path)
    df = pd.read_csv(path, float_precision="round_trip")
    require(len(df) == spec["generation"]["num_samples"] == 500000, "Wrong manifest size")
    require(df.config_sha256.eq(config_hash).all(), "Manifest/config hash mismatch")
    require(df.experiment_id.eq("M11_final_G1_500k").all(), "Wrong experiment")
    require(spec["experiment_id"] == "M11_final_G1_500k", "Wrong config experiment")
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
    n = args.limit if args.limit is not None else len(df)
    require(0 < n <= len(df), "--limit must be in [1, 500000]")
    canonical = directory / "m11_final_G1_500k.h5"
    if args.output_suffix:
        require(all(c.isalnum() or c in "_-" for c in args.output_suffix), "Invalid suffix")
    output = args.output or directory / (
        "m11_final_G1_500k" + (f"_{args.output_suffix}" if args.output_suffix else "") + ".h5"
    )
    require(not (args.output and args.output_suffix), "Use --output OR --output-suffix")
    if args.limit is not None:
        require(output.resolve() != canonical.resolve(),
                "Pilot requires a separate --output or --output-suffix")
    require(output.resolve() not in (path.resolve(), args.config.resolve()), "Input/output collision")
    attrs = {
        "schema_version": 1, "experiment_id": spec["experiment_id"], "domain": "G1",
        "signal_context_mode": "full_projection",
        "input_normalization": "per_sample_per_detector_zscore",
        "input_zscore_epsilon": 1e-6,
        "snr_low_frequency_cutoff": 30.0, "snr_high_frequency_cutoff": 512.0,
        "target_snr_policy": "uniform_for_every_source",
        "empirical_psd_contract": "128s off-source; Welch 8s; raw interpolation; no preconditioning",
        "whitening_contract": "raw empirical PSD; single inverse-spectrum truncation, Hann 0.5s",
        "config_sha256": config_hash, "config_json": json.dumps(spec, sort_keys=True),
        "manifest_path": str(path.resolve()), "manifest_sha256": manifest_hash,
        "sampling_frequency_hz": 4096, "duration_s": 4.0,
        "detector_order": ",".join(DETECTORS), "labels": ",".join(LABELS),
        "waveform_approximant": "SEOBNRv4_opt", "selected_rows": n,
        "selection_policy": "first_N_manifest_rows", "manifest_rows": len(df),
        "generator_sha256": sha256_file(__file__),
        "source_module_sha256": json.dumps({str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
            for p in sorted((PROJECT_ROOT / "src").rglob("*.py"))}, sort_keys=True),
    }
    return spec, root, df.iloc[:n].copy(), output, attrs


def resolve_cached_psd_files(root, manifest):
    """Read cached file headers only; fail on missing/ambiguous assigned intervals.

    G1 needs only 128 s of raw strain per PSD block, not the real-noise crop.
    This replaces M11.6's catalogue lookup and loading entire 4096 s files.
    """
    files = {ifo: [] for ifo in DETECTORS}
    for ifo in DETECTORS:
        for path in sorted((root / "gwosc_cache/m11").glob(f"*-{ifo}_*.hdf5")):
            with h5py.File(path, "r") as h:
                start = float(h["meta/GPSstart"][()])
                duration = float(h["meta/Duration"][()])
                rate = len(h["strain/Strain"]) / duration
            if rate == 4096:
                files[ifo].append((path, start, start + duration))
    selected = {}
    provenance = {}
    for row in manifest.drop_duplicates("block_id").itertuples():
        for ifo in DETECTORS:
            matches = [(p, start) for p, start, end in files[ifo]
                       if start <= row.psd_start and row.psd_end <= end]
            require(len(matches) == 1, f"Expected one cached {ifo} file for {row.block_id}; got {len(matches)}")
            path, start = matches[0]
            selected[row.block_id, ifo] = (path, start)
            stat = path.stat()
            provenance[str(path.resolve())] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return selected, json.dumps(provenance, sort_keys=True)


def read_psd_intervals(row, selected):
    from pycbc.types import TimeSeries
    strains = {}
    for ifo in DETECTORS:
        path, start = selected[row.block_id, ifo]
        first = int(round((row.psd_start - start) * 4096))
        last = first + 128 * 4096
        require(abs(start + first / 4096 - row.psd_start) < 1e-7, "PSD sample misalignment")
        with h5py.File(path, "r") as h:
            values = h["strain/Strain"][first:last]
        require(len(values) == 128 * 4096, "Incomplete PSD reference")
        strains[ifo] = TimeSeries(values, delta_t=1 / 4096, epoch=float(row.psd_start))
    return strains


def schema(manifest):
    n = len(manifest)
    fields = {"X": ((n, 3, 16384), np.dtype("float32")),
              "y": ((n, 3), np.dtype("float32")),
              "status": ((n,), np.dtype("uint8")),
              "is_truncated": ((n,), np.dtype("bool"))}
    for name in manifest.columns:
        dtype = (np.dtype("int64") if pd.api.types.is_integer_dtype(manifest[name])
                 else np.dtype("float64") if pd.api.types.is_numeric_dtype(manifest[name])
                 else h5py.string_dtype("utf-8"))
        fields[name] = ((n,), dtype)
    for name in FLOAT_OUTPUTS:
        fields[name] = ((n,), np.dtype("float64"))
    for ifo in DETECTORS:
        fields[f"gaussian_seed_{ifo}"] = ((n,), np.dtype("uint32"))
    return fields


def create_output(output, manifest, attrs):
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as h:
        for name, (shape, dtype) in schema(manifest).items():
            options = {"chunks": (1, 3, 16384), "compression": "lzf"} if name == "X" else {}
            d = h.create_dataset(name, shape=shape, dtype=dtype, **options)
            if name in manifest:
                d[:] = manifest[name].to_numpy()
        h["y"][:] = manifest[list(LABELS)].to_numpy(dtype=np.float32)
        for alias, target in ALIASES.items():
            h[alias] = h[target]
        h.attrs.update(attrs)
        h.attrs["initialization_complete"] = True
        h.flush()


def verify_output(h, manifest, attrs):
    require(bool(h.attrs.get("initialization_complete", False)), "Incomplete HDF5 initialization")
    for key, value in attrs.items():
        require(key in h.attrs and h.attrs[key] == value, f"Resume attribute mismatch: {key}")
    fields = schema(manifest)
    require(set(h.keys()) == set(fields) | set(ALIASES), "Resume schema mismatch")
    for name, (shape, dtype) in fields.items():
        require(h[name].shape == shape and h[name].dtype == dtype, f"Resume schema mismatch: {name}")
    for alias, target in ALIASES.items():
        require(h[alias].id == h[target].id, f"Broken metadata alias: {alias}")
    for name in manifest:
        values = h[name].asstr()[:] if h5py.check_string_dtype(h[name].dtype) else h[name][:]
        require(np.array_equal(values, manifest[name].to_numpy()), f"Resume manifest mismatch: {name}")
    require(np.array_equal(h["y"][:], manifest[list(LABELS)].to_numpy(dtype=np.float32)),
            "Resume labels mismatch")
    require(np.isin(h["status"][:], [0, 1]).all(), "Invalid completion status")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT /
                        "configs/generation/generate_m11_final_G1_500k.json")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-suffix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate metadata and cached PSD availability; generate nothing.")
    args = parser.parse_args()
    spec, root, manifest, output, attrs = read_inputs(args)
    selected, attrs["raw_psd_files"] = resolve_cached_psd_files(root, manifest)
    if args.validate_only:
        print(f"Validated {len(manifest)} selected rows; output would be {output}", flush=True)
        return
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; use --resume: {output}")
    if args.resume and not output.exists():
        raise FileNotFoundError(f"Resume output missing: {output}")
    print(f"Initializing G1 physics for {len(manifest)} rows -> {output}", flush=True)
    initialize_physics(spec)
    # Software versions are part of resume compatibility, alongside source hashes.
    from importlib.metadata import version
    attrs["software_versions"] = json.dumps({name: version(name) for name in
        ("numpy", "pandas", "h5py", "pycbc", "lalsuite", "scipy", "torch")}, sort_keys=True)
    if not output.exists():
        create_output(output, manifest, attrs)
    with h5py.File(output, "r+") as h:
        verify_output(h, manifest, attrs)
        pending = manifest.loc[h["status"][:] == 0]
        if pending.empty:
            print("Already complete; no samples rewritten.", flush=True)
            return
        completed = len(manifest) - len(pending)
        checkpoint_rows = []

        def checkpoint():
            if checkpoint_rows:
                h.flush()
                h["status"][sorted(checkpoint_rows)] = 1
                h.flush()
                checkpoint_rows.clear()

        for block_id, rows in pending.groupby("block_id", sort=True):
            long_strains = read_psd_intervals(rows.iloc[0], selected)
            psd_cache = {}
            for index, row in rows.iterrows():
                try:
                    X, metadata = build_g1_row(row, long_strains=long_strains, psd_cache=psd_cache)
                    h["X"][index] = X
                    for name, value in metadata.items():
                        h[name][index] = value
                    # Uncheckpointed rows retain status=0, even if HDF5 writes
                    # their data early. Resume regenerates them deterministically.
                    checkpoint_rows.append(int(index))
                    if len(checkpoint_rows) == 32:
                        checkpoint()
                except Exception as exc:
                    raise RuntimeError(f"Generation failed at {row.source_id} (row {index})") from exc
                completed += 1
                if completed % 1000 == 0 or len(manifest) <= 64:
                    print(f"Completed {completed}/{len(manifest)}", flush=True)
        checkpoint()
        print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
