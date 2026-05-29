#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a generated CBC HDF5 dataset and its sidecar metadata JSON."
    )

    parser.add_argument(
        "h5_file",
        type=Path,
        help="Path to the generated HDF5 file.",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional path to the sidecar metadata JSON. "
            "If omitted, the script looks for <h5_file_stem>.metadata.json."
        ),
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=8,
        help="Maximum number of values to print from one-dimensional datasets.",
    )

    return parser.parse_args()


def infer_metadata_path(h5_file: Path) -> Path:
    return h5_file.with_suffix(".metadata.json")


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def preview_array(arr: np.ndarray, max_items: int = 8) -> str:
    arr = np.asarray(arr)

    if arr.ndim == 0:
        return repr(arr.item())

    flat = arr.reshape(-1)

    if flat.size <= max_items:
        return np.array2string(flat, precision=6, separator=", ")

    head = flat[:max_items]
    return np.array2string(head, precision=6, separator=", ") + f" ... ({flat.size} values)"


def print_h5_attrs(f: h5py.File) -> None:
    print_header("HDF5 attributes")

    if len(f.attrs) == 0:
        print("No root attributes found.")
        return

    for key, value in f.attrs.items():
        print(f"{key}: {value}")


def print_h5_tree(
    obj: h5py.Group | h5py.Dataset,
    indent: int = 0,
    max_items: int = 8,
) -> None:
    prefix = " " * indent

    if isinstance(obj, h5py.Dataset):
        print(
            f"{prefix}- {obj.name}: "
            f"shape={obj.shape}, dtype={obj.dtype}"
        )

        if obj.ndim <= 1 and obj.size > 0:
            try:
                data = obj[()]
                print(f"{prefix}  preview: {preview_array(data, max_items=max_items)}")
            except Exception as exc:
                print(f"{prefix}  preview failed: {exc}")

        return

    if isinstance(obj, h5py.Group):
        print(f"{prefix}+ {obj.name}/")

        for key in obj.keys():
            print_h5_tree(obj[key], indent=indent + 2, max_items=max_items)


def print_selected_dataset_summary(f: h5py.File, max_items: int = 8) -> None:
    print_header("Selected dataset summary")

    selected_paths = [
        "X",
        "y",
        "parameters/mass_1",
        "parameters/mass_2",
        "parameters/total_mass",
        "parameters/chirp_mass",
        "parameters/chi_eff",
        "parameters/distance",
        "snr/network",
        "snr/H1",
        "snr/L1",
        "snr/V1",
    ]

    for path in selected_paths:
        if path not in f:
            print(f"{path}: not found")
            continue

        dset = f[path]
        print(f"{path}: shape={dset.shape}, dtype={dset.dtype}")

        if dset.ndim <= 1:
            print(f"  values: {preview_array(dset[()], max_items=max_items)}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_json_tree(obj: Any, indent: int = 0, max_depth: int = 4) -> None:
    prefix = " " * indent

    if indent // 2 >= max_depth:
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}/")
                print_json_tree(value, indent=indent + 2, max_depth=max_depth)
            elif isinstance(value, list):
                print(f"{prefix}{key}: [list, len={len(value)}]")
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(obj, list):
        print(f"{prefix}[list, len={len(obj)}]")
    else:
        print(f"{prefix}{obj}")


def print_metadata_summary(metadata: dict[str, Any]) -> None:
    print_header("Metadata JSON summary")

    for key in ["dataset_file", "format", "status", "generation_config_file"]:
        if key in metadata:
            print(f"{key}: {metadata[key]}")

    print()
    print("Top-level metadata keys:")
    print(list(metadata.keys()))

    config = metadata.get("config", {})

    if config:
        print()
        print("Config sections:")
        print(list(config.keys()))

        for section in [
            "output",
            "generation",
            "simulation",
            "parameter_sampler",
            "detectors",
            "signal_processor",
            "label_transformer",
        ]:
            if section in config:
                print()
                print(f"config.{section}:")
                print(json.dumps(config[section], indent=2))


def main() -> None:
    args = parse_args()

    h5_file = args.h5_file
    metadata_file = args.metadata or infer_metadata_path(h5_file)

    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

    print_header("Input files")
    print("HDF5 file:", h5_file)
    print("Metadata JSON:", metadata_file)

    with h5py.File(h5_file, "r") as f:
        print_h5_attrs(f)

        print_header("HDF5 structure")
        print_h5_tree(f, max_items=args.max_items)

        print_selected_dataset_summary(f, max_items=args.max_items)

    if metadata_file.exists():
        metadata = load_json(metadata_file)
        print_metadata_summary(metadata)

        print_header("Metadata JSON tree")
        print_json_tree(metadata, max_depth=4)
    else:
        print_header("Metadata JSON")
        print(f"Metadata file not found: {metadata_file}")


if __name__ == "__main__":
    main()