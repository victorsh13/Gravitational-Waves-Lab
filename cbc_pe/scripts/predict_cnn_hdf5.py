from __future__ import annotations

from pathlib import Path
import argparse
import importlib
import json
import os
import sys
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract predictions and embeddings from a trained HDF5 CNN checkpoint."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to prediction JSON config.",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_required(d: dict[str, Any], key: str):
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]


def resolve_path(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value)

    if path.is_absolute():
        return path

    return base / path


def import_model_class(class_name: str):
    network_module = importlib.import_module("src.models.network")

    if not hasattr(network_module, class_name):
        available = [
            name for name in dir(network_module)
            if not name.startswith("_")
        ]
        raise AttributeError(
            f"Model class '{class_name}' not found in src.models.network. "
            f"Available names include: {available}"
        )

    return getattr(network_module, class_name)


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

    from src.models.dataset import HDF5RegressionDataset
    from src.models.hdf5_batch_dataset import HDF5BatchIterableDataset
    from src.models.evaluate import extract_predictions_and_embeddings
    from src.models.utils import set_seed

    dataset_cfg = get_required(cfg, "dataset")
    prediction_cfg = get_required(cfg, "prediction")
    output_cfg = cfg.get("output", {})

    data_processed = data_root / "processed"
    checkpoints_dir = data_root / "models" / "checkpoints"
    results_dir = data_root / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_path(
        data_processed,
        get_required(dataset_cfg, "dataset_file"),
    )
    split_path = resolve_path(
        data_processed,
        get_required(dataset_cfg, "split_file"),
    )
    label_stats_path = resolve_path(
        data_processed,
        get_required(dataset_cfg, "label_stats_file"),
    )
    checkpoint_path = resolve_path(
        checkpoints_dir,
        get_required(prediction_cfg, "checkpoint_file"),
    )

    for name, path in {
        "dataset": dataset_path,
        "split": split_path,
        "label stats": label_stats_path,
        "checkpoint": checkpoint_path,
    }.items():
        if path is None or not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    seed = int(prediction_cfg.get("seed", 123))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Prediction environment")
    print("=" * 80)
    print("device:", device)
    print("dataset_path:", dataset_path)
    print("split_path:", split_path)
    print("label_stats_path:", label_stats_path)
    print("checkpoint_path:", checkpoint_path)

    with h5py.File(dataset_path, "r") as f:
        X_shape = f["X"].shape
        y_shape = f["y"].shape

        n_samples = int(X_shape[0])
        n_detectors = int(X_shape[1])
        signal_length = int(X_shape[2])
        n_outputs = int(y_shape[1])

    splits_npz = np.load(split_path)

    split_dict = {}

    for split_name in ["train", "val", "cal", "test"]:
        key = f"{split_name}_idx"
        if key in splits_npz.files:
            split_dict[split_name] = splits_npz[key].astype(np.int64)

    requested_splits = prediction_cfg.get(
        "splits",
        list(split_dict.keys()),
    )

    requested_splits = [str(name) for name in requested_splits]

    for split_name in requested_splits:
        if split_name not in split_dict:
            raise KeyError(
                f"Requested split '{split_name}' is unavailable. "
                f"Available: {list(split_dict)}"
            )

    stats = np.load(label_stats_path)
    y_mean = stats["y_mean"].astype(np.float32)
    y_std = stats["y_std"].astype(np.float32)

    if "label_names" in stats.files:
        label_names = [str(x) for x in stats["label_names"].tolist()]
    else:
        label_names = ["chirp_mass", "total_mass", "chi_eff"]

    print()
    print("=" * 80)
    print("Loading checkpoint")
    print("=" * 80)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_config'.")

    model_config = checkpoint["model_config"]

    input_normalization_cfg = prediction_cfg.get(
        "input_normalization",
        model_config.get("input_normalization", {"enabled": False}),
    )

    print("input_normalization:", input_normalization_cfg)

    class_name = model_config["class_name"]
    model_kwargs = dict(model_config["model_kwargs"])

    model_kwargs.setdefault("n_detectors", n_detectors)
    model_kwargs.setdefault("n_outputs", n_outputs)

    model_class = import_model_class(class_name)
    model = model_class(**model_kwargs).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("class_name:", class_name)
    print("model_kwargs:", model_kwargs)
    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint best val:", checkpoint.get("best_val_loss"))

    batch_size = int(prediction_cfg.get("batch_size", 256))
    num_workers = int(prediction_cfg.get("num_workers", 4))
    pin_memory = bool(
        prediction_cfg.get("pin_memory", device.type == "cuda")
    )
    prefetch_factor = int(prediction_cfg.get("prefetch_factor", 2))
    max_slice_overread = float(
        prediction_cfg.get("max_slice_overread", 4.0)
    )
    data_loading_mode = str(
        prediction_cfg.get("data_loading_mode", "hdf5_batch_slices")
    )

    print()
    print("=" * 80)
    print("Prediction settings")
    print("=" * 80)
    print("requested_splits:", requested_splits)
    print("data_loading_mode:", data_loading_mode)
    print("batch_size:", batch_size)
    print("num_workers:", num_workers)
    print("pin_memory:", pin_memory)
    print("prefetch_factor:", prefetch_factor)
    print("max_slice_overread:", max_slice_overread)

    save_payload = {
        "y_mean": y_mean,
        "y_std": y_std,
        "label_names": np.array(label_names),
        "checkpoint_file": str(checkpoint_path),
        "dataset_path": str(dataset_path),
        "split_path": str(split_path),
        "label_stats_path": str(label_stats_path),
        "model_config": np.array(model_config, dtype=object),
        "input_normalization": np.array(input_normalization_cfg, dtype=object),
        "available_splits": np.array(requested_splits),
    }

    for split_name in requested_splits:
        indices = split_dict[split_name]

        print()
        print("=" * 80)
        print(f"Predicting split: {split_name}")
        print("=" * 80)
        print("n_samples:", len(indices))

        if data_loading_mode in {"hdf5_batch", "hdf5_batch_slices"}:
            dataset = HDF5BatchIterableDataset(
                h5_path=dataset_path,
                indices=indices,
                y_mean=y_mean,
                y_std=y_std,
                input_normalization=input_normalization_cfg,
                batch_size=batch_size,
                drop_last=False,
                seed=seed,
                shuffle_batches=False,
                shuffle_within_batch=False,
                max_slice_overread=max_slice_overread,
            )

            loader_kwargs = {
                "batch_size": None,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
            }

            if num_workers > 0:
                loader_kwargs.update(
                    persistent_workers=False,
                    prefetch_factor=prefetch_factor,
                )

            loader = DataLoader(
                dataset,
                **loader_kwargs,
            )

        else:
            dataset = HDF5RegressionDataset(
                h5_path=dataset_path,
                indices=indices,
                y_mean=y_mean,
                y_std=y_std,
                input_normalization=input_normalization_cfg,
            )

            loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": False,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
            }

            if num_workers > 0:
                loader_kwargs.update(
                    persistent_workers=False,
                    prefetch_factor=prefetch_factor,
                )

            loader = DataLoader(
                dataset,
                **loader_kwargs,
            )

        pred, emb, y_eval = extract_predictions_and_embeddings(
            model=model,
            loader=loader,
            device=device,
        )

        print("pred:", pred.shape)
        print("emb:", emb.shape)
        print("y:", y_eval.shape)

        if len(pred) != len(indices):
            raise ValueError(
                f"Prediction count mismatch for {split_name}: "
                f"pred={len(pred)}, indices={len(indices)}"
            )

        save_payload[f"pred_{split_name}"] = pred
        save_payload[f"emb_{split_name}"] = emb
        save_payload[f"y_{split_name}"] = y_eval

        # Important: batch-level prediction yields data in sorted physical-index
        # order, so store indices in the same order.
        if data_loading_mode in {"hdf5_batch", "hdf5_batch_slices"}:
            save_payload[f"idx_{split_name}"] = np.sort(indices)
        else:
            save_payload[f"idx_{split_name}"] = indices

        if hasattr(dataset, "close"):
            dataset.close()

    output_name = output_cfg.get("file_name")

    if output_name is None:
        checkpoint_stem = checkpoint_path.stem

        if checkpoint_stem.endswith("_checkpoint"):
            checkpoint_stem = checkpoint_stem[:-len("_checkpoint")]

        split_suffix = "_".join(requested_splits)

        output_name = (
            f"{checkpoint_stem}_"
            f"{split_suffix}_predictions_embeddings.npz"
        )

    output_path = resolve_path(results_dir, output_name)

    if output_path is None:
        raise ValueError("Could not resolve output path.")

    overwrite = bool(output_cfg.get("overwrite", False))

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Set output.overwrite=true to replace it."
        )

    print()
    print("=" * 80)
    print("Saving predictions and embeddings")
    print("=" * 80)
    print("output_path:", output_path)

    np.savez_compressed(
        output_path,
        **save_payload,
    )

    print()
    print("=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()