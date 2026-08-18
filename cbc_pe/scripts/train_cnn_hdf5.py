from __future__ import annotations

from pathlib import Path
import os
import sys
import json
import argparse
import importlib
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SCRIPT_PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_PROJECT_ROOT),
    )

from src.paths import (
    resolve_data_root,
    resolve_processed_artifact,
    resolve_project_root,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN baseline for CBC parameter regression from HDF5."
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
        help="Path to JSON config file.",
    )

    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None

    path = Path(value)

    if path.is_absolute():
        return path

    return base / path


def get_required(d: dict, key: str):
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]


def import_model_class(class_name: str):
    """
    Dynamically import model class from src.models.network.

    Example:
        class_name = "CNN_Pool"
        class_name = "SimpleCNN"
    """
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


def make_checkpoint_name(dataset_id: str, model_config: dict, training_config: dict, checkpoint_tag: str):
    class_name = model_config["class_name"]
    seed = training_config["seed"]
    loss_name = training_config.get("loss", "MSELoss")

    return (
        f"{dataset_id}"
        f"_{class_name}"
        f"_{checkpoint_tag}"
        f"_{loss_name}"
        f"_seed{seed}"
        f"_checkpoint.pt"
    )


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

    # Project imports after sys.path setup
    from src.models.dataset import HDF5RegressionDataset
    from src.models.hdf5_batch_dataset import HDF5BatchIterableDataset
    from src.models.train import train_model
    from src.models.utils import set_seed
    from src.models.samplers import SortedBlockBatchSampler
    

    dataset_cfg = get_required(cfg, "dataset")
    model_cfg = get_required(cfg, "model")
    training_cfg = get_required(cfg, "training")
    outputs_cfg = cfg.get("outputs", {})
    input_normalization_cfg = cfg.get(
        "input_normalization",
        {"enabled": False},
    )

    print()
    print("=" * 80)
    print("Input normalization")
    print("=" * 80)
    print(input_normalization_cfg)

    dataset_id = get_required(dataset_cfg, "dataset_id")

    models_dir = data_root / "models"

    results_root = data_root / "results"
    checkpoints_root = models_dir / "checkpoints"

    results_dir = (results_root / dataset_id)

    checkpoints_dir = (checkpoints_root / dataset_id)

    for path in [
        models_dir,
        results_root,
        checkpoints_root,
        results_dir,
        checkpoints_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_processed_artifact(
        data_root=data_root,
        dataset_id=dataset_id,
        file_name=get_required(
            dataset_cfg,
            "dataset_file",
        ),
    )

    split_path = resolve_processed_artifact(
        data_root=data_root,
        dataset_id=dataset_id,
        file_name=get_required(
            dataset_cfg,
            "split_file",
        ),
    )

    label_stats_path = resolve_processed_artifact(
        data_root=data_root,
        dataset_id=dataset_id,
        file_name=get_required(
            dataset_cfg,
            "label_stats_file",
        ),
    )

    if dataset_path is None or not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if split_path is None or not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    if label_stats_path is None or not label_stats_path.exists():
        raise FileNotFoundError(f"Label stats file not found: {label_stats_path}")

    seed = int(training_cfg.get("seed", 123))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Environment")
    print("=" * 80)
    print("hostname:", os.uname().nodename)
    print("project_root:", project_root)
    print("data_root:", data_root)
    print("cwd:", Path.cwd())
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("device:", device)

    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        print("n GPUs visible:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))

    print()
    print("=" * 80)
    print("Input files")
    print("=" * 80)
    print("dataset_path:", dataset_path)
    print("split_path:", split_path)
    print("label_stats_path:", label_stats_path)

    print()
    print("=" * 80)
    print("Inspecting HDF5 dataset")
    print("=" * 80)

    with h5py.File(dataset_path, "r") as f:
        if "X" not in f:
            raise KeyError("HDF5 dataset does not contain key 'X'.")
        if "y" not in f:
            raise KeyError("HDF5 dataset does not contain key 'y'.")

        X_shape = f["X"].shape
        y_shape = f["y"].shape

        n_samples = X_shape[0]
        n_detectors = X_shape[1]
        signal_length = X_shape[2]
        n_outputs = y_shape[1]

        print("X shape:", X_shape)
        print("X dtype:", f["X"].dtype)
        print("y shape:", y_shape)
        print("y dtype:", f["y"].dtype)

        print("attrs:")
        for key in f.attrs.keys():
            print(f"  {key}: {f.attrs[key]}")

        num_written = int(f.attrs.get("num_written", n_samples))
        if num_written != n_samples:
            raise ValueError(
                f"HDF5 dataset appears incomplete: "
                f"num_written={num_written}, n_samples={n_samples}"
            )

        status = f.attrs.get("dataset_status", None)
        if status is not None and status != "complete":
            raise ValueError(f"HDF5 dataset status is not complete: {status}")

    print()
    print("=" * 80)
    print("Loading splits and label stats")
    print("=" * 80)

    splits = np.load(split_path)

    train_idx = splits["train_idx"].astype(np.int64)
    val_idx = splits["val_idx"].astype(np.int64)

    cal_idx = splits["cal_idx"].astype(np.int64) if "cal_idx" in splits.files else None
    test_idx = splits["test_idx"].astype(np.int64) if "test_idx" in splits.files else None

    stats = np.load(label_stats_path)
    y_mean = stats["y_mean"].astype(np.float32)
    y_std = stats["y_std"].astype(np.float32)

    if "label_names" in stats.files:
        label_names = [str(x) for x in stats["label_names"].tolist()]
    else:
        label_names = ["chirp_mass", "total_mass", "chi_eff"]

    
    print("train size:", len(train_idx))
    print("val size:", len(val_idx))
    print("cal size:", 0 if cal_idx is None else len(cal_idx))
    print("test size:", 0 if test_idx is None else len(test_idx))
    print("label_names:", label_names)
    print("y_mean:", y_mean)
    print("y_std:", y_std)

    if np.any(y_std <= 0):
        raise ValueError(f"Invalid y_std values: {y_std}")

    split_dict = {
        "train": train_idx,
        "val": val_idx,
    }

    if cal_idx is not None:
        split_dict["cal"] = cal_idx

    if test_idx is not None:
        split_dict["test"] = test_idx

    split_names = list(split_dict.keys())

    for i, name_a in enumerate(split_names):
        for name_b in split_names[i + 1:]:
            overlap = np.intersect1d(split_dict[name_a], split_dict[name_b])
            if len(overlap) > 0:
                raise ValueError(
                    f"Overlap detected between {name_a} and {name_b}: "
                    f"{len(overlap)} samples"
                )

    print()
    print("=" * 80)
    print("Creating datasets/loaders")
    print("=" * 80)

    train_dataset = HDF5RegressionDataset(
        h5_path=dataset_path,
        indices=train_idx,
        y_mean=y_mean,
        y_std=y_std,
        input_normalization=input_normalization_cfg,
    )

    val_dataset = HDF5RegressionDataset(
        h5_path=dataset_path,
        indices=val_idx,
        y_mean=y_mean,
        y_std=y_std,
        input_normalization=input_normalization_cfg,
    )

    cal_dataset = None
    test_dataset = None

    if cal_idx is not None:
        cal_dataset = HDF5RegressionDataset(
            h5_path=dataset_path,
            indices=cal_idx,
            y_mean=y_mean,
            y_std=y_std,
            input_normalization=input_normalization_cfg,
        )

    if test_idx is not None:
        test_dataset = HDF5RegressionDataset(
            h5_path=dataset_path,
            indices=test_idx,
            y_mean=y_mean,
            y_std=y_std,
            input_normalization=input_normalization_cfg,
        )

    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    data_loading_mode = str(training_cfg.get("data_loading_mode", "sample"))

    print("data_loading_mode:", data_loading_mode)

    batch_sampler_mode = str(training_cfg.get("batch_sampler", "default"))
    drop_last = bool(training_cfg.get("drop_last", True))
    shuffle_batches = bool(training_cfg.get("shuffle_batches", True))
    shuffle_within_batch = bool(training_cfg.get("shuffle_within_batch", False))
    max_slice_overread = float(training_cfg.get("max_slice_overread", 4.0))
    persistent_workers = bool(training_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(training_cfg.get("prefetch_factor", 2))

    print("batch_sampler:", batch_sampler_mode)
    print("drop_last:", drop_last)
    print("shuffle_batches:", shuffle_batches)
    print("shuffle_within_batch:", shuffle_within_batch)
    print("max_slice_overread:", max_slice_overread)
    print("persistent_workers:", persistent_workers if num_workers > 0 else False)
    print("prefetch_factor:", prefetch_factor if num_workers > 0 else None)

    if data_loading_mode in {"hdf5_batch", "hdf5_batch_slices"}:
        print("Using HDF5BatchIterableDataset for train.")

        train_batch_dataset = HDF5BatchIterableDataset(
            h5_path=dataset_path,
            indices=train_idx,
            y_mean=y_mean,
            y_std=y_std,
            input_normalization=input_normalization_cfg,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
            shuffle_batches=shuffle_batches,
            shuffle_within_batch=shuffle_within_batch,
            max_slice_overread=max_slice_overread,
        )

        train_loader_kwargs = dict(
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if num_workers > 0:
            train_loader_kwargs.update(
                persistent_workers=False,
                prefetch_factor=prefetch_factor,
            )

        train_loader = DataLoader(
            train_batch_dataset,
            **train_loader_kwargs,
        )

    elif batch_sampler_mode in {"sorted_block", "sorted_block_batches"}:
        train_batch_sampler = SortedBlockBatchSampler(
            split_indices=train_idx,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
            shuffle_batches=shuffle_batches,
            shuffle_within_batch=shuffle_within_batch,
        )

        train_loader_kwargs = dict(
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if num_workers > 0:
            train_loader_kwargs.update(
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )

        train_loader = DataLoader(
            train_dataset,
            **train_loader_kwargs,
        )

    else:
        train_loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        if num_workers > 0:
            train_loader_kwargs.update(
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )

        train_loader = DataLoader(
            train_dataset,
            **train_loader_kwargs,
        )

    val_num_workers = int(training_cfg.get("val_num_workers", 0))
    val_pin_memory = bool(training_cfg.get("val_pin_memory", pin_memory))
    val_persistent_workers = bool(
        training_cfg.get("val_persistent_workers", val_num_workers > 0)
    )

    if data_loading_mode in {"hdf5_batch", "hdf5_batch_slices"}:
        print("Using HDF5BatchIterableDataset for val.")

        val_batch_dataset = HDF5BatchIterableDataset(
            h5_path=dataset_path,
            indices=val_idx,
            y_mean=y_mean,
            y_std=y_std,
            input_normalization=input_normalization_cfg,
            batch_size=batch_size,
            drop_last=False,
            seed=seed + 10_000,
            shuffle_batches=False,
            shuffle_within_batch=False,
            max_slice_overread=max_slice_overread,
        )

        val_loader_kwargs = dict(
            batch_size=None,
            num_workers=val_num_workers,
            pin_memory=val_pin_memory,
        )

        if val_num_workers > 0:
            val_loader_kwargs.update(
                persistent_workers=False,
                prefetch_factor=prefetch_factor,
            )

        val_loader = DataLoader(
            val_batch_dataset,
            **val_loader_kwargs,
        )

    else:
        val_loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=val_pin_memory,
        )

        if val_num_workers > 0:
            val_loader_kwargs.update(
                persistent_workers=val_persistent_workers,
                prefetch_factor=prefetch_factor,
            )

        val_loader = DataLoader(
            val_dataset,
            **val_loader_kwargs,
        )

    # Sanity check.
    # Use a deterministic, single-process loader to avoid expensive random HDF5
    # access before training, especially for large datasets.
    sanity_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    X_batch, y_batch = next(iter(sanity_loader))

    print("X_batch:", X_batch.shape, X_batch.dtype)
    print("y_batch:", y_batch.shape, y_batch.dtype)
    print("X finite:", torch.isfinite(X_batch).all().item())
    print("y finite:", torch.isfinite(y_batch).all().item())
    print("y batch mean:", y_batch.mean(dim=0))
    print("y batch std:", y_batch.std(dim=0))

    ## Sanity C
    X_channel_means = X_batch.mean(dim=2)
    X_channel_stds = X_batch.std(dim=2, unbiased=False)

    print("X per-sample/channel mean, first 5:")
    print(X_channel_means[:5])

    print("X per-sample/channel std, first 5:")
    print(X_channel_stds[:5])

    print("mean abs X channel mean:", X_channel_means.abs().mean().item())
    print("mean X channel std:", X_channel_stds.mean().item())

    if input_normalization_cfg.get("enabled", False):
        if not torch.allclose(
            X_channel_means,
            torch.zeros_like(X_channel_means),
            atol=1e-4,
            rtol=0.0,
        ):
            raise ValueError("Input z-score sanity check failed: channel means are not ~0.")

        if not torch.allclose(
            X_channel_stds,
            torch.ones_like(X_channel_stds),
            atol=1e-4,
            rtol=0.0,
        ):
            raise ValueError("Input z-score sanity check failed: channel stds are not ~1.")

    ###


    if X_batch.shape[1] != n_detectors:
        raise ValueError(
            f"Batch n_detectors mismatch: {X_batch.shape[1]} vs {n_detectors}"
        )

    if y_batch.shape[1] != n_outputs:
        raise ValueError(
            f"Batch n_outputs mismatch: {y_batch.shape[1]} vs {n_outputs}"
        )

    print()
    print("=" * 80)
    print("Building model")
    print("=" * 80)

    class_name = get_required(model_cfg, "class_name")
    architecture_name = model_cfg.get("architecture_name", class_name)
    model_kwargs = dict(model_cfg.get("kwargs", {}))

    # Inject dimensions from dataset unless explicitly provided.
    model_kwargs.setdefault("n_detectors", n_detectors)
    model_kwargs.setdefault("n_outputs", n_outputs)

    model_class = import_model_class(class_name)

    print("model class:", class_name)
    print("model kwargs:")
    for key, value in model_kwargs.items():
        print(f"  {key}: {value}")

    model = model_class(**model_kwargs).to(device)

    full_model_config = {
        "architecture": architecture_name,
        "class_name": class_name,
        "dataset_id": dataset_id,
        "dataset_format": "hdf5",
        "dataset_path": str(dataset_path),
        "split_path": str(split_path),
        "label_stats_path": str(label_stats_path),
        "n_samples": n_samples,
        "n_detectors": n_detectors,
        "signal_length": signal_length,
        "n_outputs": n_outputs,
        "label_names": label_names,
        "model_kwargs": model_kwargs,
        "normalization": "GroupNorm",
        "input_normalization": input_normalization_cfg,
        "loss": training_cfg.get("loss", "MSELoss"),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "cal_size": 0 if cal_idx is None else len(cal_idx),
        "test_size": 0 if test_idx is None else len(test_idx),
        "training_seed": seed,
        "split_seed": int(splits["seed"]) if "seed" in splits.files else None, 
    }

    print()
    print("Full model config:")
    for key, value in full_model_config.items():
        print(f"{key}: {value}")

    print()
    print("=" * 80)
    print("Training")
    print("=" * 80)

    best_checkpoint, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        model_config=full_model_config,
        seed=seed,
        batch_size=batch_size,
        max_epochs=int(training_cfg.get("max_epochs", 200)),
        patience=int(training_cfg.get("patience", 40)),
        learning_rate=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 3e-4)),
    )

    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    print()
    print("=" * 80)
    print("Best checkpoint")
    print("=" * 80)
    print("Best epoch:", best_checkpoint["epoch"])
    print("Best val loss:", best_checkpoint["best_val_loss"])

    checkpoint_tag = outputs_cfg.get("checkpoint_tag", "default")
    checkpoint_file_name = make_checkpoint_name(
        dataset_id=dataset_id,
        model_config={"class_name": class_name},
        training_config={"seed": seed, "loss": training_cfg.get("loss", "MSELoss")},
        checkpoint_tag=checkpoint_tag,
    )

    checkpoint_path = checkpoints_dir / checkpoint_file_name

    torch.save(best_checkpoint, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)

    # Save history separately. Useful for plotting without loading checkpoint.
    history_file_name = checkpoint_file_name.replace("_checkpoint.pt", "_history.npz")
    history_path = results_dir / history_file_name

    if isinstance(history, dict):
        np.savez_compressed(history_path, **history)
        print("Saved history:", history_path)
    else:
        # Fallback if your train_model returns a list or another object.
        np.savez_compressed(history_path, history=np.array(history, dtype=object))
        print("Saved history:", history_path)

    train_dataset.close()
    val_dataset.close()

    if cal_dataset is not None:
        cal_dataset.close()

    if test_dataset is not None:
        test_dataset.close()

    print()
    print("=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()