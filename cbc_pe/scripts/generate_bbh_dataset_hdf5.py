from __future__ import annotations

from pathlib import Path
import sys
import os
import argparse
import time
import json
from typing import Any

import numpy as np
import h5py


PARAMETER_KEYS = [
    "mass_1",
    "mass_2",
    "distance",
    "inclination",
    "ra",
    "dec",
    "spin_1z",
    "spin_2z",
    "polarization_angle",
    "total_mass",
    "chirp_mass",
    "chi_eff",
]

LABEL_NAMES = ["chirp_mass", "total_mass", "chi_eff"]

PLACEMENT_KEYS = {
    "segment_start_time": "float64",
    "segment_end_time": "float64",
    "earliest_signal_start_time": "float64",
    "latest_signal_end_time": "float64",
    "valid_start_min": "float64",
    "valid_start_max": "float64",
    "signal_network_duration": "float64",
    "safe_margin_start": "float64",
    "safe_margin_end": "float64",
    "margin_before_signal": "float64",
    "margin_after_signal": "float64",
    "margins_respected": "bool",
    "enforce_safe_margins": "bool",
}

WINDOWING_KEYS = {
    "is_truncated": "bool",
    "full_network_start_time": "float64",
    "full_network_end_time": "float64",
    "full_network_duration": "float64",
    "used_window_start_time": "float64",
    "used_window_end_time": "float64",
    "used_window_duration": "float64",
    "segment_duration": "float64",
    "max_window_duration": "float64",
    "required_final_duration": "float64",
    "required_available_final_duration": "float64",
    "seconds_before_network_end_in_window": "float64",
    "fraction_network_duration_used": "float64",
}

PROJECTION_NETWORK_KEYS = {
    "geocentric_coalescence_time": "float64",
}

PROJECTION_DETECTOR_KEYS = {
    "expected_detector_time_delays": "float64",
    "detector_arrival_times": "float64",
    "projected_start_times": "float64",
    "projected_end_times": "float64",
}

SNR_EXTRA_KEYS = {
    "initial_network": "float32",
    "target_network": "float32",
    "snr_rescaled": "bool",
    "distance_before_rescale": "float32",
    "distance_after_rescale": "float32",
}

INJECTION_KEYS = {
    "signal_start_time": "float64",
    "signal_end_time": "float64",
    "segment_start_time": "float64",
    "segment_end_time": "float64",
    "signal_start_index": "int64",
    "signal_end_index": "int64",
    "overlap_start_index_strain": "int64",
    "overlap_end_index_strain": "int64",
    "overlap_start_index_signal": "int64",
    "overlap_end_index_signal": "int64",
    "n_signal_samples": "int64",
    "n_injected_samples": "int64",
    "n_clipped_before": "int64",
    "n_clipped_after": "int64",
    "is_partially_clipped": "bool",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CBC BBH dataset directly into HDF5."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to generation JSON config.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Override config and overwrite existing HDF5 file.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Override config and resume existing HDF5 file.",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_required(d: dict[str, Any], key: str):
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return base / path


def build_simulation_config(sim_cfg: dict[str, Any]):
    from src.config import SimulationConfig

    return SimulationConfig(
        duration=float(sim_cfg.get("duration", 4.0)),
        safe_margin_start=float(sim_cfg.get("safe_margin_start", 0.0)),
        safe_margin_end=float(sim_cfg.get("safe_margin_end", 0.0)),
        processing_context_start_samples=int(
            sim_cfg.get("processing_context_start_samples", 1664)
        ),
        processing_context_end_samples=int(
            sim_cfg.get("processing_context_end_samples", 1664)
        ),
    )


def build_builder(
    config,
    detector_names,
    signal_processor_kwargs,
    label_transformer_kwargs,
    parameter_sampler_kwargs,
    rng,
    ):
    
    from src.dataset import DatasetBuilder

    return DatasetBuilder.from_config(
        config=config,
        detector_names=detector_names,
        signal_processor_kwargs=signal_processor_kwargs,
        label_transformer_kwargs=label_transformer_kwargs,
        parameter_sampler_kwargs=parameter_sampler_kwargs,
        rng=rng,
    )


def parameter_to_dict(p) -> dict[str, float]:
    return {
        "mass_1": float(p.mass_1),
        "mass_2": float(p.mass_2),
        "distance": float(p.distance),
        "inclination": float(p.inclination),
        "ra": float(p.ra),
        "dec": float(p.dec),
        "spin_1z": float(p.spin_1z),
        "spin_2z": float(p.spin_2z),
        "polarization_angle": float(p.polarization_angle),
        "total_mass": float(p.total_mass),
        "chirp_mass": float(p.chirp_mass),
        "chi_eff": float(p.chi_eff),
    }


def get_nested(d: dict, keys: list[str], default=np.nan):
    current = d

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def get_nested_required(d: dict, keys: list[str]):
    current = d

    for key in keys:
        if not isinstance(current, dict):
            raise KeyError(f"Expected dict while reading nested key path: {keys}")

        if key not in current:
            raise KeyError(f"Missing nested key path: {keys}")

        current = current[key]

    return current


def create_hdf5_file(
    path: Path,
    num_samples: int,
    config,
    detector_names: list[str],
    generation_cfg: dict[str, Any],
    signal_processor_cfg: dict[str, Any],
    overwrite: bool,
):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"File already exists: {path}. "
                "Use --overwrite or set output.overwrite=true, "
                "or use --resume to continue."
            )
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)

    f = h5py.File(path, "w")

    chunk_size = int(generation_cfg["chunk_size"])
    seed = int(generation_cfg["seed"])

    f.attrs["num_samples"] = int(num_samples)
    f.attrs["num_written"] = int(0)
    f.attrs["dataset_status"] = "in_progress"

    f.attrs["seed"] = int(seed)
    f.attrs["chunk_size"] = int(chunk_size)

    f.attrs["detector_names"] = json.dumps(detector_names)
    f.attrs["label_names"] = json.dumps(LABEL_NAMES)

    f.attrs["duration"] = float(config.duration)
    f.attrs["length"] = int(config.length)
    f.attrs["sampling_frequency"] = float(config.sampling_frequency)
    f.attrs["low_frequency_cutoff"] = float(config.low_frequency_cutoff)

    f.attrs["processing_context_start_samples"] = int(
        config.processing_context_start_samples
    )
    f.attrs["processing_context_end_samples"] = int(
        config.processing_context_end_samples
    )
    f.attrs["processing_length"] = int(config.processing_length)

    f.attrs["placement_policy"] = str(generation_cfg.get("placement_policy", "random_contained"))
    f.attrs["standardize_labels"] = bool(generation_cfg.get("standardize_labels", False))

    f.attrs["strain_mode"] = str(generation_cfg.get("strain_mode", "in_noise"))

    # Store processor config as JSON string in attrs.
    f.attrs["signal_processor_config"] = json.dumps(signal_processor_cfg)

    x_chunk_n = min(64, num_samples)
    scalar_chunk_n = min(4096, num_samples)

    f.create_dataset(
        "X",
        shape=(num_samples, len(detector_names), config.length),
        dtype="float32",
        chunks=(x_chunk_n, len(detector_names), config.length),
    )

    f.create_dataset(
        "y",
        shape=(num_samples, len(LABEL_NAMES)),
        dtype="float32",
        chunks=(min(1024, num_samples), len(LABEL_NAMES)),
    )

    param_group = f.create_group("parameters")

    for key in PARAMETER_KEYS:
        param_group.create_dataset(
            key,
            shape=(num_samples,),
            dtype="float32",
            chunks=(scalar_chunk_n,),
        )

    placement_group = f.create_group("placement")

    for key, dtype in PLACEMENT_KEYS.items():
        placement_group.create_dataset(
            key,
            shape=(num_samples,),
            dtype=dtype,
            chunks=(scalar_chunk_n,),
        )

    windowing_group = f.create_group("windowing")

    for key, dtype in WINDOWING_KEYS.items():
        windowing_group.create_dataset(
            key,
            shape=(num_samples,),
            dtype=dtype,
            chunks=(scalar_chunk_n,),
        )

    projection_group = f.create_group("projection")

    for key, dtype in PROJECTION_NETWORK_KEYS.items():
        projection_group.create_dataset(
            key,
            shape=(num_samples,),
            dtype=dtype,
            chunks=(scalar_chunk_n,),
        )

    for det in detector_names:
        det_group = projection_group.create_group(det)

        for key, dtype in PROJECTION_DETECTOR_KEYS.items():
            det_group.create_dataset(
                key,
                shape=(num_samples,),
                dtype=dtype,
                chunks=(scalar_chunk_n,),
            )

    inj_group = f.create_group("injection")

    for det in detector_names:
        det_group = inj_group.create_group(det)

        for key, dtype in INJECTION_KEYS.items():
            det_group.create_dataset(
                key,
                shape=(num_samples,),
                dtype=dtype,
                chunks=(scalar_chunk_n,),
            )

    snr_group = f.create_group("snr")

    snr_group.create_dataset(
        "network",
        shape=(num_samples,),
        dtype="float32",
        chunks=(scalar_chunk_n,),
    )

    for key, dtype in SNR_EXTRA_KEYS.items():
        snr_group.create_dataset(
            key,
            shape=(num_samples,),
            dtype=dtype,
            chunks=(scalar_chunk_n,),
        )

    for det in detector_names:
        snr_group.create_dataset(
            det,
            shape=(num_samples,),
            dtype="float32",
            chunks=(scalar_chunk_n,),
        )

    gen_group = f.create_group("generation")

    gen_group.create_dataset(
        "chunk_id",
        shape=(num_samples,),
        dtype="int32",
        chunks=(scalar_chunk_n,),
    )

    gen_group.create_dataset(
        "local_index",
        shape=(num_samples,),
        dtype="int32",
        chunks=(scalar_chunk_n,),
    )

    gen_group.create_dataset(
        "seed",
        shape=(num_samples,),
        dtype="int64",
        chunks=(scalar_chunk_n,),
    )


    return f


def open_or_create_hdf5(
    path: Path,
    num_samples: int,
    config,
    detector_names: list[str],
    generation_cfg: dict[str, Any],
    signal_processor_cfg: dict[str, Any],
    overwrite: bool,
    resume: bool,
):
    if overwrite and resume:
        raise ValueError("Use either overwrite or resume, not both.")

    if resume:
        if not path.exists():
            raise FileNotFoundError(f"Cannot resume. File does not exist: {path}")

        f = h5py.File(path, "r+")

        existing_num_samples = int(f.attrs["num_samples"])

        if existing_num_samples != num_samples:
            f.close()
            raise ValueError(
                f"Existing file has num_samples={existing_num_samples}, "
                f"but requested num_samples={num_samples}."
            )

        status = f.attrs.get("dataset_status", "unknown")
        num_written = int(f.attrs.get("num_written", 0))

        print(f"Resuming existing HDF5 file: {path}")
        print(f"dataset_status: {status}")
        print(f"num_written: {num_written}")

        return f

    return create_hdf5_file(
        path=path,
        num_samples=num_samples,
        config=config,
        detector_names=detector_names,
        generation_cfg=generation_cfg,
        signal_processor_cfg=signal_processor_cfg,
        overwrite=overwrite,
    )


def write_batch_to_hdf5(
    f: h5py.File,
    batch,
    start: int,
    end: int,
    chunk_id: int,
    chunk_seed: int,
    detector_names: list[str],
):
    expected_n = end - start

    X = batch.X.astype(np.float32)
    y = batch.y.astype(np.float32)

    if X.shape[0] != expected_n:
        raise ValueError(f"Batch X size mismatch: {X.shape[0]} vs {expected_n}")

    if y.shape[0] != expected_n:
        raise ValueError(f"Batch y size mismatch: {y.shape[0]} vs {expected_n}")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or Inf.")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or Inf.")

    f["X"][start:end] = X
    f["y"][start:end] = y

    params_as_dicts = [parameter_to_dict(p) for p in batch.parameters]

    for key in PARAMETER_KEYS:
        values = np.asarray([p[key] for p in params_as_dicts], dtype=np.float32)
        f[f"parameters/{key}"][start:end] = values

    metadata = batch.metadata

    for key in PLACEMENT_KEYS:
        values = np.asarray(
            [get_nested_required(m, ["placement", key]) for m in metadata],
            dtype=f[f"placement/{key}"].dtype,
        )
        f[f"placement/{key}"][start:end] = values

    for key in WINDOWING_KEYS:
        values = np.asarray(
            [get_nested_required(m, ["windowing", key]) for m in metadata],
            dtype=f[f"windowing/{key}"].dtype,
        )
        f[f"windowing/{key}"][start:end] = values

    for key in PROJECTION_NETWORK_KEYS:
        values = np.asarray(
            [get_nested_required(m, ["projection", key]) for m in metadata],
            dtype=f[f"projection/{key}"].dtype,
        )
        f[f"projection/{key}"][start:end] = values

    for det in detector_names:
        for key in PROJECTION_DETECTOR_KEYS:
            values = np.asarray(
                [get_nested_required(m, ["projection", key, det]) for m in metadata],
                dtype=f[f"projection/{det}/{key}"].dtype,
            )
            f[f"projection/{det}/{key}"][start:end] = values


    network_snr = np.asarray(
        [get_nested_required(m, ["snr", "final_network_snr"]) for m in metadata],
        dtype=np.float32,
    )
    f["snr/network"][start:end] = network_snr

    for det in detector_names:
        det_snr = np.asarray(
            [get_nested_required(m, ["snr", "final_detector_snrs", det]) for m in metadata],
            dtype=np.float32,
        )
        f[f"snr/{det}"][start:end] = det_snr

    snr_extra_map = {
        "initial_network": ["snr", "initial_network_snr"],
        "target_network": ["snr", "target_network_snr"],
        "snr_rescaled": ["snr", "snr_rescaled"],
        "distance_before_rescale": ["snr", "distance_before_rescale"],
        "distance_after_rescale": ["snr", "distance_after_rescale"],
    }

    for out_key, nested_keys in snr_extra_map.items():
        values = np.asarray(
            [get_nested_required(m, nested_keys) for m in metadata],
            dtype=f[f"snr/{out_key}"].dtype,
        )
        f[f"snr/{out_key}"][start:end] = values

    for det in detector_names:
        for key in INJECTION_KEYS:
            values = np.asarray(
                [get_nested_required(m, ["injection", det, key]) for m in metadata],
                dtype=f[f"injection/{det}/{key}"].dtype,
            )
            f[f"injection/{det}/{key}"][start:end] = values

    f["generation/chunk_id"][start:end] = np.full(expected_n, chunk_id, dtype=np.int32)
    f["generation/local_index"][start:end] = np.arange(expected_n, dtype=np.int32)
    f["generation/seed"][start:end] = np.full(expected_n, chunk_seed, dtype=np.int64)

    f.attrs["num_written"] = int(end)


def save_sidecar_metadata_json(
    output_file: Path,
    generation_config_path: Path,
    full_config: dict[str, Any],
    status: str,
):
    metadata_file = output_file.with_suffix(".metadata.json")

    payload = {
        "dataset_file": output_file.name,
        "format": "hdf5",
        "status": status,
        "generation_config_file": str(generation_config_path),
        "config": full_config,
        "notes": (
            "Large dataset stored in HDF5. "
            "Per-sample numerical metadata is stored as HDF5 datasets."
        ),
    }

    save_json(payload, metadata_file)

    print(f"Saved sidecar metadata to: {metadata_file}")


def main():
    args = parse_args()
    cfg = load_json(args.config)

    project_root = Path(get_required(cfg, "project_root"))
    data_root = Path(get_required(cfg, "data_root"))

    if not project_root.exists():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")

    os.chdir(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    output_cfg = get_required(cfg, "output")
    generation_cfg = get_required(cfg, "generation")
    simulation_cfg = get_required(cfg, "simulation")
    detector_names = get_required(cfg, "detectors")
    signal_processor_cfg = get_required(cfg, "signal_processor")
    label_transformer_cfg = cfg.get("label_transformer", {})
    parameter_sampler_cfg = cfg.get("parameter_sampler", {})

    output_file = resolve_path(data_root / "processed", get_required(output_cfg, "file_name"))

    # CLI overrides config.
    overwrite = bool(args.overwrite or output_cfg.get("overwrite", False))
    resume = bool(args.resume or output_cfg.get("resume", False))

    if overwrite and resume:
        raise ValueError("Cannot use overwrite and resume at the same time.")

    num_samples = int(get_required(generation_cfg, "num_samples"))
    chunk_size = int(generation_cfg.get("chunk_size", 2500))
    seed = int(generation_cfg.get("seed", 1234))
    progress_every = int(generation_cfg.get("progress_every", 100))
    placement_policy = str(generation_cfg.get("placement_policy", "random_contained"))
    standardize_labels = bool(generation_cfg.get("standardize_labels", False))
    strain_mode = str(generation_cfg.get("strain_mode", "noisy"))

    if strain_mode not in {"in_noise", "gw_only"}:
        raise ValueError(
            "generation.strain_mode must be one of {'in_noise', 'gw_only'}, "
            f"got {strain_mode}."
        )

    generation_cfg["chunk_size"] = chunk_size
    generation_cfg["seed"] = seed

    print("=" * 80)
    print("Generating directly to HDF5")
    print("=" * 80)
    print("config:", args.config)
    print("project_root:", project_root)
    print("data_root:", data_root)
    print("output_file:", output_file)
    print("num_samples:", num_samples)
    print("chunk_size:", chunk_size)
    print("seed:", seed)
    print("overwrite:", overwrite)
    print("resume:", resume)
    print("detectors:", detector_names)
    print("strain_mode:", strain_mode)

    config = build_simulation_config(simulation_cfg)

    print()
    print("Simulation")
    print("duration:", config.duration)
    print("length:", config.length)
    print("processing_length:", config.processing_length)
    print("context start:", config.processing_context_start_samples)
    print("context end:", config.processing_context_end_samples)

    total_chunks = int(np.ceil(num_samples / chunk_size))

    h5 = open_or_create_hdf5(
        path=output_file,
        num_samples=num_samples,
        config=config,
        detector_names=detector_names,
        generation_cfg=generation_cfg,
        signal_processor_cfg=signal_processor_cfg,
        overwrite=overwrite,
        resume=resume,
    )

    global_t0 = time.perf_counter()
    status = "failed_or_interrupted"

    try:
        num_written = int(h5.attrs["num_written"])

        if num_written % chunk_size != 0 and num_written != num_samples:
            raise ValueError(
                f"Cannot resume cleanly from num_written={num_written}. "
                "Expected it to be a multiple of chunk_size."
            )

        start_chunk = num_written // chunk_size

        print()
        print(f"Total chunks: {total_chunks}")
        print(f"Starting from chunk {start_chunk + 1}/{total_chunks}")

        for chunk_id in range(start_chunk, total_chunks):
            start = chunk_id * chunk_size
            end = min(start + chunk_size, num_samples)
            current_size = end - start
            chunk_seed = seed + chunk_id

            print()
            print("=" * 80)
            print(f"Chunk {chunk_id + 1}/{total_chunks}")
            print(f"Samples: [{start}:{end}] ({current_size})")
            print(f"Seed: {chunk_seed}")
            print("=" * 80)

            rng = np.random.default_rng(chunk_seed)
            builder = build_builder(
                config=config,
                detector_names=detector_names,
                signal_processor_kwargs=signal_processor_cfg,
                label_transformer_kwargs=label_transformer_cfg,
                parameter_sampler_kwargs=parameter_sampler_cfg,
                rng=rng,
            )

            t0 = time.perf_counter()

            batch = builder.build_dataset(
                num_samples=current_size,
                standardize_labels=standardize_labels,
                placement_policy=placement_policy,
                progress_every=progress_every,
                strain_mode=strain_mode,
            )

            gen_elapsed = time.perf_counter() - t0

            print(f"Writing chunk {chunk_id + 1} to HDF5...")

            write_batch_to_hdf5(
                f=h5,
                batch=batch,
                start=start,
                end=end,
                chunk_id=chunk_id,
                chunk_seed=chunk_seed,
                detector_names=detector_names,
            )

            h5.flush()

            elapsed = time.perf_counter() - t0

            print(f"Chunk generation time: {gen_elapsed:.1f} s")
            print(f"Chunk total time: {elapsed:.1f} s")
            print(f"Seconds/sample: {elapsed / current_size:.4f}")
            print(f"Samples/s: {current_size / elapsed:.2f}")
            print(f"num_written: {int(h5.attrs['num_written'])}")

            del batch
            del builder

        h5.attrs["dataset_status"] = "complete"
        h5.flush()
        status = "complete"

    finally:
        h5.close()

        save_sidecar_metadata_json(
            output_file=output_file,
            generation_config_path=args.config,
            full_config=cfg,
            status=status,
        )

    global_elapsed = time.perf_counter() - global_t0

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)
    print(f"status: {status}")
    print(f"Total elapsed: {global_elapsed:.1f} s")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()