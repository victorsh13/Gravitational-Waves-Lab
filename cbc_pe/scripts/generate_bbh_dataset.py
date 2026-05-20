from __future__ import annotations
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

    # New chunk arguments
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")

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
    

    print("Output duration:", config.duration)
    print("Output length:", config.length)
    print("Processing duration:", config.processing_duration)
    print("Processing length:", config.processing_length)
    print("Context start samples:", config.processing_context_start_samples)
    print("Context end samples:", config.processing_context_end_samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Case 1: No chunks
    if args.chunk_size is None:
        rng = np.random.default_rng(args.seed)
        builder = build_builder(config=config, rng=rng)

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
            output_dir=output_dir,
            file_name=args.file_name,
            detector_names=["H1", "L1", "V1"],
            overwrite=args.overwrite,
            ask_before_overwrite=not args.overwrite,
        )

        print(f"Saved to: {dataset_path}")
        return
    

    # Case 2: chunked generation
    chunk_size = args.chunk_size
    total_samples = args.num_samples
    total_chunks = int(np.ceil(total_samples / chunk_size))

    if args.num_chunks is None:
        end_chunk = total_chunks
    else:
        end_chunk = min(args.start_chunk + args.num_chunks, total_chunks)

    print()
    print("Chunked generation")
    print(f"Total samples: {total_samples}")
    print(f"Chunk size: {chunk_size}")
    print(f"Total chunks: {total_chunks}")
    print(f"Generating chunks: {args.start_chunk + 1} to {end_chunk}")

    # Base output directory provided by the user
    base_output_dir = Path(args.output_dir)

    # Name of the full dataset, derived from file_name without extension
    base_file = Path(args.file_name)
    dataset_name = base_file.stem
    suffix = base_file.suffix

    # Directory where chunks will be saved
    chunk_output_dir = base_output_dir / dataset_name
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base output directory: {base_output_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"Chunk output directory: {chunk_output_dir}")


    global_t0 = time.perf_counter()

    for chunk_id in range(args.start_chunk, end_chunk):
        start = chunk_id * chunk_size
        remaining = total_samples - start
        current_chunk_size = min(chunk_size, remaining)

        chunk_seed = args.seed + chunk_id

        stem = Path(args.file_name).stem
        suffix = Path(args.file_name).suffix

        chunk_file_name = (
            f"{stem}_chunk{chunk_id + 1:02d}"
            f"_of{total_chunks:02d}"
            f"_seed{chunk_seed}"
            f"_n{current_chunk_size}"
            f"{suffix}"
        )

        chunk_path = output_dir / chunk_file_name

        if args.skip_existing and chunk_path.exists():
            print()
            print(f"[chunk {chunk_id}] Skipping existing file: {chunk_path}")
            continue

        print()
        print("=" * 80)
        print(f"[chunk {chunk_id + 1}/{total_chunks}]")
        print(f"Samples in chunk: {current_chunk_size}")
        print(f"Chunk seed: {chunk_seed}")
        print(f"Output file: {chunk_file_name}")
        print("=" * 80)

        rng = np.random.default_rng(chunk_seed)
        builder = build_builder(config=config, rng=rng)

        t0 = time.perf_counter()

        batch = builder.build_dataset(
            num_samples=current_chunk_size,
            standardize_labels=False,
            placement_policy="random_contained",
            progress_every=args.progress_every,
        )

        elapsed = time.perf_counter() - t0

        dataset_path = save_dataset_npz(
            batch=batch,
            output_dir=chunk_output_dir,
            file_name=chunk_file_name,
            detector_names=["H1", "L1", "V1"],
            overwrite=args.overwrite,
            ask_before_overwrite=not args.overwrite,
        )

        print()
        print(f"[chunk {chunk_id + 1}] Generation finished")
        print(f"Samples: {current_chunk_size}")
        print(f"Elapsed: {elapsed:.1f} s")
        print(f"Seconds/sample: {elapsed / current_chunk_size:.3f}")
        print(f"Samples/s: {current_chunk_size / elapsed:.2f}")
        print(f"Saved to: {dataset_path}")

        # Important: release memory before next chunk
        del batch
        del builder

    global_elapsed = time.perf_counter() - global_t0

    print()
    print("All requested chunks finished")
    print(f"Elapsed total: {global_elapsed:.1f} s")
    print(f"Seconds/chunk: {global_elapsed / total_chunks:.3f}")
    print(f"Chunks/s: {total_chunks / global_elapsed:.2f}")


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


if __name__ == "__main__":
    main()