from pathlib import Path
import argparse
import re
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks-dir",
        type=str,
        required=True,
        help="Directory containing the chunk .npz files.",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path of the merged output .npz file.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*_chunk*.npz",
        help="Glob pattern used to find chunk files.",
    )

    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed instead of np.savez.",
    )

    return parser.parse_args()


def extract_chunk_id(path: Path) -> int:
    """
    Extracts chunk id from names like:
    dataset_chunk000_of003_seed1234_n5000.npz
    """
    match = re.search(r"_chunk(\d+)_", path.name)

    if match is None:
        raise ValueError(f"Could not extract chunk id from file name: {path.name}")

    return int(match.group(1))


def main():
    args = parse_args()

    chunks_dir = Path(args.chunks_dir)
    output_file = Path(args.output_file)

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory does not exist: {chunks_dir}")

    chunk_paths = sorted(
        chunks_dir.glob(args.pattern),
        key=extract_chunk_id,
    )

    if len(chunk_paths) == 0:
        raise FileNotFoundError(
            f"No chunk files found in {chunks_dir} with pattern {args.pattern}"
        )

    print("Found chunks:")
    for path in chunk_paths:
        print("  ", path.name)

    print()
    print(f"Number of chunks: {len(chunk_paths)}")

    # Inspect first file
    first = np.load(chunk_paths[0], allow_pickle=True)
    keys = list(first.keys())

    print()
    print("Keys found in first chunk:")
    for key in keys:
        print(f"  {key}: shape={first[key].shape}, dtype={first[key].dtype}")

    # Main arrays expected in your dataset
    concat_keys = []
    static_keys = []

    n_first = None

    if "X" in keys:
        n_first = first["X"].shape[0]
    elif "y" in keys:
        n_first = first["y"].shape[0]
    else:
        raise KeyError("Could not find 'X' or 'y' in first chunk.")

    for key in keys:
        arr = first[key]

        # Concatenate arrays whose first dimension matches number of samples
        if arr.shape != () and arr.shape[0] == n_first:
            concat_keys.append(key)
        else:
            static_keys.append(key)

    print()
    print("Keys to concatenate:")
    for key in concat_keys:
        print("  ", key)

    print()
    print("Static keys copied from first chunk:")
    for key in static_keys:
        print("  ", key)

    merged = {}

    # Concatenate sample-wise arrays
    for key in concat_keys:
        print()
        print(f"Merging key: {key}")

        arrays = []

        for path in chunk_paths:
            with np.load(path, allow_pickle=True) as data:
                arr = data[key]

                print(f"  {path.name}: shape={arr.shape}, dtype={arr.dtype}")

                arrays.append(arr)

        merged[key] = np.concatenate(arrays, axis=0)

        print(f"  merged shape: {merged[key].shape}")

    # Copy static metadata from first chunk
    with np.load(chunk_paths[0], allow_pickle=True) as data:
        for key in static_keys:
            merged[key] = data[key]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Saving merged file to: {output_file}")

    if args.compressed:
        np.savez_compressed(output_file, **merged)
    else:
        np.savez(output_file, **merged)

    print("Done.")

    print()
    print("Final merged shapes:")
    with np.load(output_file, allow_pickle=True) as data:
        for key in data.keys():
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")


if __name__ == "__main__":
    main()