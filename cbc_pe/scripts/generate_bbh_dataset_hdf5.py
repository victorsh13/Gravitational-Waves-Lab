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
    parser.add_argument("--progress-every", type=int, default=100)

    return parser.parse_args()


def build_builder(config, rng):
    return DatasetBuilder.from_config(
        config=config,
        detector_names=["H1", "L1", "V1"],
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


def parameter_to_dict(p):
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


def get_nested(d, keys, default=np.nan):
    current = d

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def create_hdf5_file(path, num_samples, config, overwrite=False):
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    f = h5py.File(path, "w")

    f.attrs["num_samples"] = num_samples
    f.attrs["detector_names"] = json.dumps(["H1", "L1", "V1"])
    f.attrs["label_names"] = json.dumps(["chirp_mass", "total_mass", "chi_eff"])
    f.attrs["duration"] = float(config.duration)
    f.attrs["length"] = int(config.length)
    f.attrs["sampling_frequency"] = float(config.sampling_frequency)
    f.attrs["low_frequency_cutoff"] = float(config.low_frequency_cutoff)

    f.create_dataset(
        "X",
        shape=(num_samples, 3, config.length),
        dtype="float32",
        chunks=(min(64, num_samples), 3, config.length),
    )

    f.create_dataset(
        "y",
        shape=(num_samples, 3),
        dtype="float32",
        chunks=(min(1024, num_samples), 3),
    )

    param_group = f.create_group("parameters")

    for key in PARAMETER_KEYS:
        param_group.create_dataset(
            key,
            shape=(num_samples,),
            dtype="float32",
            chunks=(min(4096, num_samples),),
        )

    snr_group = f.create_group("snr")

    for key in ["network", "H1", "L1", "V1"]:
        snr_group.create_dataset(
            key,
            shape=(num_samples,),
            dtype="float32",
            chunks=(min(4096, num_samples),),
        )

    return f


def write_batch_to_hdf5(f, batch, start, end):
    X = batch.X.astype(np.float32)
    y = batch.y.astype(np.float32)

    if X.shape[0] != end - start:
        raise ValueError(f"Batch X size mismatch: {X.shape[0]} vs {end - start}")

    f["X"][start:end] = X
    f["y"][start:end] = y

    params_as_dicts = [parameter_to_dict(p) for p in batch.parameters]

    for key in PARAMETER_KEYS:
        f[f"parameters/{key}"][start:end] = np.array(
            [p[key] for p in params_as_dicts],
            dtype=np.float32,
        )

    metadata = batch.metadata

    f["snr/network"][start:end] = np.array(
        [get_nested(m, ["snr", "final_network_snr"]) for m in metadata],
        dtype=np.float32,
    )

    for det in ["H1", "L1", "V1"]:
        f[f"snr/{det}"][start:end] = np.array(
            [get_nested(m, ["snr", "final_detector_snrs", det]) for m in metadata],
            dtype=np.float32,
        )


def main():
    args = parse_args()

    config = SimulationConfig(
        duration=4.0,
        safe_margin_start=0.0,
        safe_margin_end=0.0,
        processing_context_start_samples=1664,
        processing_context_end_samples=1664,
    )

    output_file = Path(args.output_file)

    print("Generating directly to HDF5")
    print(f"Output file: {output_file}")
    print(f"Total samples: {args.num_samples}")
    print(f"Chunk size: {args.chunk_size}")

    total_chunks = int(np.ceil(args.num_samples / args.chunk_size))

    h5 = create_hdf5_file(
        path=output_file,
        num_samples=args.num_samples,
        config=config,
        overwrite=args.overwrite,
    )

    global_t0 = time.perf_counter()

    try:
        for chunk_id in range(total_chunks):
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
            write_batch_to_hdf5(h5, batch, start, end)
            h5.flush()

            elapsed = time.perf_counter() - t0

            print(f"Chunk generation time: {gen_elapsed:.1f} s")
            print(f"Chunk total time: {elapsed:.1f} s")
            print(f"Samples/s: {current_size / elapsed:.2f}")

            del batch
            del builder

    finally:
        h5.close()

    global_elapsed = time.perf_counter() - global_t0

    print()
    print("Finished.")
    print(f"Total elapsed: {global_elapsed:.1f} s")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()