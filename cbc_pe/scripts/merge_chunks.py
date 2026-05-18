from pathlib import Path
import argparse
import re
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks-dir",
        type=str,
        required=True,
        help="Directory containing the chunk .npz and .metadata.json files.",
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

    parser.add_argument(
        "--merge-metadata",
        action="store_true",
        help="Also merge companion .metadata.json files.",
    )

    return parser.parse_args()


def extract_chunk_id(path: Path) -> int:
    """
    Extracts chunk id from names like:
    dataset_chunk001_of003_seed1234_n5000.npz
    """
    match = re.search(r"_chunk(\d+)_", path.name)

    if match is None:
        raise ValueError(f"Could not extract chunk id from file name: {path.name}")

    return int(match.group(1))


def metadata_path_for_chunk(chunk_path: Path) -> Path:
    """
    Converts:
        xxx.npz
    into:
        xxx.metadata.json
    """
    return chunk_path.with_suffix(".metadata.json")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def merge_npz_chunks(chunk_paths, output_file: Path, compressed: bool = False):
    """
    Merge .npz chunks into one .npz file.
    Concatenates arrays whose first dimension matches number of samples.
    Copies static arrays from the first chunk.
    """
    first = np.load(chunk_paths[0], allow_pickle=True)
    keys = list(first.keys())

    print()
    print("Keys found in first chunk:")
    for key in keys:
        print(f"  {key}: shape={first[key].shape}, dtype={first[key].dtype}")

    if "X" in keys:
        n_first = first["X"].shape[0]
    elif "y" in keys:
        n_first = first["y"].shape[0]
    else:
        first.close()
        raise KeyError("Could not find 'X' or 'y' in first chunk.")

    concat_keys = []
    static_keys = []

    for key in keys:
        arr = first[key]

        if arr.shape != () and arr.shape[0] == n_first:
            concat_keys.append(key)
        else:
            static_keys.append(key)

    first.close()

    print()
    print("Keys to concatenate:")
    for key in concat_keys:
        print("  ", key)

    print()
    print("Static keys copied from first chunk:")
    for key in static_keys:
        print("  ", key)

    merged = {}

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
        print(f"  merged shape: {merged[key].shape}, dtype={merged[key].dtype}")

    with np.load(chunk_paths[0], allow_pickle=True) as data:
        for key in static_keys:
            merged[key] = data[key]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Saving merged .npz to: {output_file}")

    if compressed:
        np.savez_compressed(output_file, **merged)
    else:
        np.savez(output_file, **merged)

    print("Merged .npz saved.")

    return merged


def merge_metadata_json(chunk_paths, output_file: Path):
    """
    Merge companion .metadata.json files.

    Expected structure per chunk:
    {
      "dataset_file": "...chunkXXX...npz",
      "num_samples": 5000,
      "X_shape": [5000, 3, 16384],
      "y_shape": [5000, 3],
      "parameters": [...],
      "metadata": [...]
    }

    Output structure is compatible with src.io.load_dataset_npz().
    """

    metadata_paths = [metadata_path_for_chunk(path) for path in chunk_paths]

    missing = [path for path in metadata_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing metadata files:\n"
            + "\n".join(str(path) for path in missing)
        )

    all_parameters = []
    all_metadata = []
    chunk_summaries = []

    total_samples = 0
    global_x_shape = None
    global_y_shape = None

    for i, (chunk_path, metadata_path) in enumerate(zip(chunk_paths, metadata_paths)):
        print()
        print(f"Merging metadata chunk {i + 1}/{len(chunk_paths)}")
        print(f"  npz:      {chunk_path.name}")
        print(f"  metadata: {metadata_path.name}")

        meta = load_json(metadata_path)

        required_keys = ["num_samples", "X_shape", "y_shape", "parameters", "metadata"]
        for key in required_keys:
            if key not in meta:
                raise KeyError(f"Missing key '{key}' in {metadata_path}")

        n_meta = int(meta["num_samples"])
        parameters = meta["parameters"]
        metadata = meta["metadata"]

        if not isinstance(parameters, list):
            raise TypeError(f"'parameters' must be a list in {metadata_path}")

        if not isinstance(metadata, list):
            raise TypeError(f"'metadata' must be a list in {metadata_path}")

        if len(parameters) != n_meta:
            raise ValueError(
                f"parameters length mismatch in {metadata_path.name}: "
                f"{len(parameters)} vs num_samples={n_meta}"
            )

        if len(metadata) != n_meta:
            raise ValueError(
                f"metadata length mismatch in {metadata_path.name}: "
                f"{len(metadata)} vs num_samples={n_meta}"
            )

        # Cross-check against actual NPZ
        with np.load(chunk_path, allow_pickle=True) as data:
            if "X" not in data.files:
                raise KeyError(f"'X' missing in {chunk_path}")

            if "y" not in data.files:
                raise KeyError(f"'y' missing in {chunk_path}")

            x_shape = data["X"].shape
            y_shape = data["y"].shape

        if x_shape[0] != n_meta:
            raise ValueError(
                f"X/metadata mismatch for {chunk_path.name}: "
                f"X.shape[0]={x_shape[0]}, metadata num_samples={n_meta}"
            )

        if y_shape[0] != n_meta:
            raise ValueError(
                f"y/metadata mismatch for {chunk_path.name}: "
                f"y.shape[0]={y_shape[0]}, metadata num_samples={n_meta}"
            )

        all_parameters.extend(parameters)
        all_metadata.extend(metadata)
        total_samples += n_meta

        if global_x_shape is None:
            global_x_shape = list(x_shape)
            global_y_shape = list(y_shape)
        else:
            global_x_shape[0] += x_shape[0]
            global_y_shape[0] += y_shape[0]

        chunk_summaries.append({
            "chunk_index_in_merge": i,
            "chunk_id_from_filename": extract_chunk_id(chunk_path),
            "npz_file": str(chunk_path),
            "metadata_file": str(metadata_path),
            "dataset_file_recorded_inside_metadata": meta.get("dataset_file"),
            "num_samples": n_meta,
            "X_shape": list(x_shape),
            "y_shape": list(y_shape),
        })

    merged_metadata = {
        "dataset_file": output_file.name,
        "num_samples": int(total_samples),
        "X_shape": global_x_shape,
        "y_shape": global_y_shape,
        "parameters": all_parameters,
        "metadata": all_metadata,

        # Extra merge provenance. load_dataset_npz ignores these safely.
        "merged": True,
        "num_chunks": len(chunk_paths),
        "source_npz_files": [str(path) for path in chunk_paths],
        "source_metadata_files": [str(path) for path in metadata_paths],
        "chunks": chunk_summaries,
    }

    output_metadata_file = output_file.with_suffix(".metadata.json")

    print()
    print(f"Saving merged metadata to: {output_metadata_file}")

    save_json(merged_metadata, output_metadata_file)

    print("Merged metadata saved.")
    print(f"Total samples: {total_samples}")
    print(f"len(parameters): {len(all_parameters)}")
    print(f"len(metadata): {len(all_metadata)}")

    return output_metadata_file


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

    merge_npz_chunks(
        chunk_paths=chunk_paths,
        output_file=output_file,
        compressed=args.compressed,
    )

    if args.merge_metadata:
        merge_metadata_json(
            chunk_paths=chunk_paths,
            output_file=output_file,
        )

    print()
    print("Final merged .npz shapes:")
    with np.load(output_file, allow_pickle=True) as data:
        for key in data.keys():
            print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

    if args.merge_metadata:
        metadata_file = output_file.with_suffix(".metadata.json")
        meta = load_json(metadata_file)

        print()
        print("Final merged metadata:")
        print(f"  file: {metadata_file}")
        print(f"  num_samples: {meta['num_samples']}")
        print(f"  X_shape: {meta['X_shape']}")
        print(f"  y_shape: {meta['y_shape']}")
        print(f"  len(parameters): {len(meta['parameters'])}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()