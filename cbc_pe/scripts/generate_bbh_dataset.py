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

    parser.add_argument("--num-samples", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--file-name", type=str, default="bbh_raw_4s_seobnrv4opt_snr10-25_n3500.npz")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = SimulationConfig()
    rng = np.random.default_rng(args.seed)

    builder = DatasetBuilder.from_config(
        config=config,
        detector_names=["H1", "L1", "V1"],
        signal_processor_kwargs={
            "apply_whitening": False,
            "apply_lowpass": False,
            "apply_highpass": False,
            "apply_standardization": False,
            "preserve_length": True,
        },
        label_transformer_kwargs={},
        rng=rng,
    )

    t0 = time.perf_counter()

    batch = builder.build_dataset(
        num_samples=args.num_samples,
        standardize_labels=False,
        placement_policy="random_contained",
        progress_every=100,
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