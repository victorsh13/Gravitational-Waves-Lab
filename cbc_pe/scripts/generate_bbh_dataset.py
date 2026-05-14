# scripts/generate_bbh_dataset.py

from pathlib import Path
import sys
import argparse
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.dataset import DatasetBuilder
from src.io import save_dataset_npz


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument(
        "--file-name",
        type=str,
        default="bbh_4s_seobnrv4opt_snr10-25_nX.npz",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)

    

    return parser.parse_args()


def main():
    args = parse_args()
    

    

    config = SimulationConfig(
        duration=4.0,
        safe_margin_start=0.0,
        safe_margin_end=0.0,
        processing_context_start_samples=1664,
        processing_context_end_samples=1664,
    )
    rng = np.random.default_rng(args.seed)

    print("Output duration:", config.duration)
    print("Output length:", config.length)
    print("Processing duration:", config.processing_duration)
    print("Processing length:", config.processing_length)
    print("Context start samples:", config.processing_context_start_samples)
    print("Context end samples:", config.processing_context_end_samples)


    builder = DatasetBuilder.from_config(
        config=config,
        detector_names=["H1", "L1", "V1"],
        signal_processor_kwargs = {
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

    t0 = time.perf_counter()

    batch = builder.build_dataset(
        num_samples=args.num_samples,
        standardize_labels=False,
        placement_policy="random_contained",
        progress_every=args.progress_every,
    )

    elapsed = time.perf_counter() - t0

    print()
    print("Generation finished")
    print(f"Samples: {args.num_samples}")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Seconds/sample: {elapsed / args.num_samples:.3f}")
    print(f"Samples/s: {args.num_samples / elapsed:.2f}")


    dataset_path = save_dataset_npz(
        batch=batch,
        output_dir=Path(args.output_dir),
        file_name=args.file_name,
        detector_names=["H1", "L1", "V1"],
        overwrite=args.overwrite,
        ask_before_overwrite=not args.overwrite,
    )

    print(f"Saved to: {dataset_path}")


if __name__ == "__main__":
    main()