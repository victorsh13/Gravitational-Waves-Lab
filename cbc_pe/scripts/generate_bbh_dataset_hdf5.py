from __future__ import annotations

from pathlib import Path
import sys
import argparse
import time
import json
import numpy as np
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.dataset import DatasetBuilder


DETECTOR_NAMES = ["H1", "L1", "V1"]
LABEL_NAMES = ["chirp_mass", "total_mass", "chi_eff"]

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-file", type=str, required=True)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--progress-every", type=int, default=100)

    return parser.parse_args()


def build_config() -> SimulationConfig:
    return SimulationConfig(
        duration=4.0,
        safe_margin_start=0.0,
        safe_margin_end=0.0,
        processing_context_start_samples=1664,
        processing_context_end_samples=1664,
    )


def build_builder(config: SimulationConfig, rng: np.random.Generator) -> DatasetBuilder:
    return DatasetBuilder.from_config(
        config=config,
        detector_names=DETECTOR_NAMES,
        signal_processor_kwargs={
            "whitening_method": "psd",
            "apply_highpass": True,
            "apply_lowpass": True,
            "apply_standardization": False,
            "output_mode": "crop_to_config",

            "whitening_low_frequency_cutoff": 30.0,
            "whitening_max_filter_duration": 0.5,
            "whitening_trunc_method": "hann",

            "highpass_frequency": 30.0,
            "lowpass_frequency": 512.0,

            "fir_order": 256,
            "fir_beta": 5.0,
            "remove_corrupted": True,
        },
        label_transformer_kwargs={},
        rng=rng,
    )


def parameter_to_dict(p) -> dict:
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


def create_hdf5_file(
    path: Path,
    num_samples: int,
    config: SimulationConfig,
    chunk_size: int,
    seed: int,
    overwrite: bool = False,
):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"File already exists: {path}. "
                "Use --overwrite to replace it or --resume to continue."
            )
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)

    f = h5py.File(path, "w")

    f.attrs["num_samples"] = int(num_samples)
    f.attrs["num_written"] = int(0)
    f.attrs["dataset_status"] = "in_progress"

    f.attrs["seed"] = int(seed)
    f.attrs["chunk_size"] = int(chunk_size)

    f.attrs["detector_names"] = json.dumps(DETECTOR_NAMES)
    f.attrs["label_names"] = json.dumps(LABEL_NAMES)

    f.attrs["duration"] = float(config.duration)
    f.attrs["length"] = int(config.length)
    f.attrs["sampling_frequency"] = float(config.sampling_frequency)
    f.attrs["low_frequency_cutoff"] = float(config.low_frequency_cutoff)

    f.attrs["processing_context_start_samples"] = int(config.processing_context_start_samples)
    f.attrs["processing_context_end_samples"] = int(config.processing_context_end_samples)
    f.attrs["processing_length"] = int(config.processing_length)

    x_chunk_n = min(64, num_samples)
    scalar_chunk_n = min(4096, num_samples)

    f.create_dataset(
        "X",
        shape=(num_samples, len(DETECTOR_NAMES), config.length),
        dtype="float32",
        chunks=(x_chunk_n, len(DETECTOR_NAMES), config.length),
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

    snr_group = f.create_group("snr")

    snr_group.create_dataset(
        "network",
        shape=(num_samples,),
        dtype="float32",
        chunks=(scalar_chunk_n,),
    )

    for det in DETECTOR_NAMES:
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
    config: SimulationConfig,
    chunk_size: int,
    seed: int,
    overwrite: bool,
    resume: bool,
):
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

        print(f"Resuming existing HDF5 file: {path}")
        print(f"num_written = {int(f.attrs['num_written'])}")

        return f

    return create_hdf5_file(
        path=path,
        num_samples=num_samples,
        config=config,
        chunk_size=chunk_size,
        seed=seed,
        overwrite=overwrite,
    )


def write_batch_to_hdf5(
    f: h5py.File,
    batch,
    start: int,
    end: int,
    chunk_id: int,
    chunk_seed: int,
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

    network_snr = np.asarray(
        [get_nested(m, ["snr", "final_network_snr"]) for m in metadata],
        dtype=np.float32,
    )
    f["snr/network"][start:end] = network_snr

    for det in DETECTOR_NAMES:
        det_snr = np.asarray(
            [get_nested(m, ["snr", "final_detector_snrs", det]) for m in metadata],
            dtype=np.float32,
        )
        f[f"snr/{det}"][start:end] = det_snr

    f["generation/chunk_id"][start:end] = np.full(expected_n, chunk_id, dtype=np.int32)
    f["generation/local_index"][start:end] = np.arange(expected_n, dtype=np.int32)
    f["generation/seed"][start:end] = np.full(expected_n, chunk_seed, dtype=np.int64)

    f.attrs["num_written"] = int(end)


def save_sidecar_metadata_json(
    output_file: Path,
    args,
    config: SimulationConfig,
    status: str,
):
    metadata_file = output_file.with_suffix(".metadata.json")

    payload = {
        "dataset_file": output_file.name,
        "format": "hdf5",
        "status": status,
        "num_samples": int(args.num_samples),
        "chunk_size": int(args.chunk_size),
        "seed": int(args.seed),
        "detector_names": DETECTOR_NAMES,
        "label_names": LABEL_NAMES,
        "simulation": {
            "duration": float(config.duration),
            "length": int(config.length),
            "sampling_frequency": float(config.sampling_frequency),
            "low_frequency_cutoff": float(config.low_frequency_cutoff),
            "processing_context_start_samples": int(config.processing_context_start_samples),
            "processing_context_end_samples": int(config.processing_context_end_samples),
            "processing_length": int(config.processing_length),
        },
        "notes": (
            "Large dataset stored in HDF5. "
            "Per-sample numerical metadata is stored as HDF5 datasets."
        ),
    }

    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved sidecar metadata to: {metadata_file}")


def main():
    args = parse_args()

    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both.")

    config = build_config()
    output_file = Path(args.output_file)

    print("Generating directly to HDF5")
    print(f"Output file: {output_file}")
    print(f"Total samples: {args.num_samples}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Seed: {args.seed}")

    total_chunks = int(np.ceil(args.num_samples / args.chunk_size))

    h5 = open_or_create_hdf5(
        path=output_file,
        num_samples=args.num_samples,
        config=config,
        chunk_size=args.chunk_size,
        seed=args.seed,
        overwrite=args.overwrite,
        resume=args.resume,
    )

    global_t0 = time.perf_counter()

    try:
        num_written = int(h5.attrs["num_written"])
        start_chunk = num_written // args.chunk_size

        if num_written % args.chunk_size != 0 and num_written != args.num_samples:
            raise ValueError(
                f"Cannot resume cleanly from num_written={num_written}. "
                "Expected it to be a multiple of chunk_size."
            )

        print(f"Starting from chunk {start_chunk + 1}/{total_chunks}")

        for chunk_id in range(start_chunk, total_chunks):
            start = chunk_id * args.chunk_size
            end = min(start + args.chunk_size, args.num_samples)
            current_size = end - start
            chunk_seed = args.seed + chunk_id

            print()
            print("=" * 80)
            print(f"Chunk {chunk_id + 1}/{total_chunks}")
            print(f"Samples: [{start}:{end}] ({current_size})")
            print(f"Seed: {chunk_seed}")
            print("=" * 80)

            rng = np.random.default_rng(chunk_seed)
            builder = build_builder(config=config, rng=rng)

            t0 = time.perf_counter()

            batch = builder.build_dataset(
                num_samples=current_size,
                standardize_labels=False,
                placement_policy="random_contained",
                progress_every=args.progress_every,
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
            )

            h5.flush()

            elapsed = time.perf_counter() - t0

            print(f"Chunk generation time: {gen_elapsed:.1f} s")
            print(f"Chunk total time: {elapsed:.1f} s")
            print(f"Samples/s: {current_size / elapsed:.2f}")
            print(f"num_written: {int(h5.attrs['num_written'])}")

            del batch
            del builder

        h5.attrs["dataset_status"] = "complete"
        h5.flush()

    except Exception:
        h5.attrs["dataset_status"] = "failed_or_interrupted"
        h5.flush()
        raise

    finally:
        h5.close()

    global_elapsed = time.perf_counter() - global_t0

    save_sidecar_metadata_json(
        output_file=output_file,
        args=args,
        config=config,
        status="complete",
    )

    print()
    print("Finished.")
    print(f"Total elapsed: {global_elapsed:.1f} s")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()