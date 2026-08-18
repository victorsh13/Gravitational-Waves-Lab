from __future__ import annotations

from pathlib import Path
import os
import sys
import json
import argparse
from typing import Any

import numpy as np
import h5py


DEFAULT_LABEL_NAMES = ["chirp_mass", "total_mass", "chi_eff"]

SCRIPT_PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_PROJECT_ROOT),
    )

from src.paths import (
    dataset_processed_dir,
    resolve_data_root,
    resolve_processed_artifact,
    resolve_project_root,
)

def parse_args():

    parser = argparse.ArgumentParser(
        description="Create reproducible train/val/cal/test splits for an HDF5 CBC dataset."
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Optional repository-root override. "
            "Otherwise use the config value if valid, "
            "then auto-detect the repository root."
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Optional external data-root override. "
            "Precedence: CLI, CBC_PE_DATA_ROOT, config."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to split JSON config.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Override config and overwrite existing split/stat files.",
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


def handle_existing(path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {path}. "
            "Use output.overwrite=true in config or pass --overwrite."
        )


def decode_attr_json(value, default):
    if value is None:
        return default

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    return default


def make_split_name(train_size, val_size, cal_size, test_size, seed):
    name = f"train{train_size}_val{val_size}"

    if cal_size > 0:
        name += f"_cal{cal_size}"

    if test_size > 0:
        name += f"_test{test_size}"

    name += f"_seed{seed}"

    return name


def main():
    args = parse_args()
    cfg = load_json(args.config)

    project_root = resolve_project_root(
        cli_project_root=args.project_root,
        config_project_root=cfg.get(
            "project_root"
        ),
    )

    data_root = resolve_data_root(
        cli_data_root=args.data_root,
        config_data_root=cfg.get(
            "data_root"
        ),
    )

    if not project_root.exists():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")

    os.chdir(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    dataset_cfg = get_required(cfg, "dataset")
    splits_cfg = get_required(cfg, "splits")
    output_cfg = get_required(cfg, "output")

    dataset_id = get_required(dataset_cfg, "dataset_id")
    dataset_file = get_required(dataset_cfg, "dataset_file")

    dataset_path = resolve_processed_artifact(
        data_root=data_root,
        dataset_id=dataset_id,
        file_name=dataset_file,
    )

    if not dataset_path.exists():
        raise FileNotFoundError(f"HDF5 dataset file not found: {dataset_path}")

    train_size = int(get_required(splits_cfg, "train_size"))
    val_size = int(get_required(splits_cfg, "val_size"))
    cal_size = int(splits_cfg.get("cal_size", 0))
    test_size = int(splits_cfg.get("test_size", 0))

    seed = int(splits_cfg.get("seed", 123))
    shuffle = bool(splits_cfg.get("shuffle", True))

    overwrite = bool(args.overwrite or output_cfg.get("overwrite", False))

    output_dir_cfg = output_cfg.get("output_dir", "processed")
    output_dir = dataset_processed_dir(data_root,dataset_id,)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_cfg.get("prefix", dataset_id)

    split_name = make_split_name(
        train_size=train_size,
        val_size=val_size,
        cal_size=cal_size,
        test_size=test_size,
        seed=seed,
    )

    splits_path = output_dir / f"{prefix}_splits_{split_name}.npz"
    stats_path = output_dir / f"{prefix}_label_stats_train_only_{split_name}.npz"
    metadata_path = output_dir / f"{prefix}_splits_{split_name}.metadata.json"

    handle_existing(splits_path, overwrite=overwrite)
    handle_existing(stats_path, overwrite=overwrite)
    handle_existing(metadata_path, overwrite=overwrite)

    print("=" * 80)
    print("Creating HDF5 splits")
    print("=" * 80)
    print("config:", args.config)
    print("project_root:", project_root)
    print("data_root:", data_root)
    print("dataset_id:", dataset_id)
    print("dataset_path:", dataset_path)
    print("output_dir:", output_dir)
    print("prefix:", prefix)
    print("split_name:", split_name)
    print("overwrite:", overwrite)

    with h5py.File(dataset_path, "r") as f:
        if "X" not in f:
            raise KeyError("HDF5 file does not contain dataset 'X'.")

        if "y" not in f:
            raise KeyError("HDF5 file does not contain dataset 'y'.")

        X_shape = f["X"].shape
        y_shape = f["y"].shape

        n_samples = int(X_shape[0])

        if y_shape[0] != n_samples:
            raise ValueError(
                f"X and y sample counts differ: X={X_shape[0]}, y={y_shape[0]}"
            )

        num_written = int(f.attrs.get("num_written", n_samples))
        dataset_status = f.attrs.get("dataset_status", "unknown")

        print()
        print("Dataset info:")
        print("X shape:", X_shape)
        print("y shape:", y_shape)
        print("num_samples:", n_samples)
        print("num_written:", num_written)
        print("dataset_status:", dataset_status)

        if num_written != n_samples:
            raise ValueError(
                f"Dataset incomplete: num_written={num_written}, num_samples={n_samples}"
            )

        if dataset_status != "complete":
            raise ValueError(
                f"Dataset status is not complete: dataset_status={dataset_status}"
            )

        label_names = decode_attr_json(
            f.attrs.get("label_names", None),
            DEFAULT_LABEL_NAMES,
        )

        requested = train_size + val_size + cal_size + test_size

        if requested > n_samples:
            raise ValueError(
                f"Requested split size {requested} exceeds dataset size {n_samples}."
            )

        if requested < n_samples:
            print()
            print(
                f"Warning: requested split size {requested} is smaller than "
                f"dataset size {n_samples}. Unused samples: {n_samples - requested}"
            )

        rng = np.random.default_rng(seed)
        indices = np.arange(n_samples, dtype=np.int64)

        if shuffle:
            rng.shuffle(indices)

        start = 0

        train_idx = indices[start:start + train_size]
        start += train_size

        val_idx = indices[start:start + val_size]
        start += val_size

        cal_idx = indices[start:start + cal_size]
        start += cal_size

        test_idx = indices[start:start + test_size]
        start += test_size

        # Sanity checks
        split_arrays = [train_idx, val_idx]

        if cal_size > 0:
            split_arrays.append(cal_idx)

        if test_size > 0:
            split_arrays.append(test_idx)

        all_used = np.concatenate(split_arrays)

        if len(np.unique(all_used)) != len(all_used):
            raise ValueError("Overlapping indices detected between splits.")

        if np.any(all_used < 0) or np.any(all_used >= n_samples):
            raise ValueError("Some split indices are outside dataset range.")

        # HDF5 fancy indexing is safer with sorted indices.
        # For mean/std, order does not matter.
        y_train = f["y"][np.sort(train_idx)]

        y_mean = y_train.mean(axis=0).astype(np.float32)
        y_std = y_train.std(axis=0).astype(np.float32)

    if np.any(y_std <= 0):
        raise ValueError(f"Invalid train label std values: {y_std}")

    print()
    print("Split sizes:")
    print("train:", len(train_idx))
    print("val:  ", len(val_idx))
    print("cal:  ", len(cal_idx))
    print("test: ", len(test_idx))

    print()
    print("Train-only label stats:")
    print("label_names:", label_names)
    print("y_mean:", y_mean)
    print("y_std: ", y_std)

    split_payload = {
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "seed": np.array(seed),
        "shuffle": np.array(shuffle),
        "dataset_path": np.array(str(dataset_path)),
        "dataset_id": np.array(dataset_id),
        "label_names": np.array(label_names),
    }

    if cal_size > 0:
        split_payload["cal_idx"] = cal_idx.astype(np.int64)

    if test_size > 0:
        split_payload["test_idx"] = test_idx.astype(np.int64)

    np.savez(splits_path, **split_payload)

    np.savez(
        stats_path,
        y_mean=y_mean,
        y_std=y_std,
        label_names=np.array(label_names),
        train_idx=train_idx.astype(np.int64),
        seed=np.array(seed),
        shuffle=np.array(shuffle),
        dataset_path=np.array(str(dataset_path)),
        dataset_id=np.array(dataset_id),
    )

    metadata_payload = {
        "split_config_file": str(args.config),
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_num_samples": int(n_samples),
        "requested_samples": int(train_size + val_size + cal_size + test_size),
        "unused_samples": int(n_samples - (train_size + val_size + cal_size + test_size)),
        "seed": int(seed),
        "shuffle": bool(shuffle),
        "splits": {
            "train_size": int(train_size),
            "val_size": int(val_size),
            "cal_size": int(cal_size),
            "test_size": int(test_size),
        },
        "files": {
            "splits_file": str(splits_path),
            "label_stats_file": str(stats_path),
        },
        "label_names": label_names,
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }

    save_json(metadata_payload, metadata_path)

    print()
    print("=" * 80)
    print("Saved files")
    print("=" * 80)
    print("splits:", splits_path)
    print("label stats:", stats_path)
    print("metadata:", metadata_path)
    print()
    print("Done.")


if __name__ == "__main__":
    main()