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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN baseline for CBC parameter regression from HDF5."
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

    project_root = Path(get_required(cfg, "project_root"))
    data_root = Path(get_required(cfg, "data_root"))

    if not project_root.exists():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")

    os.chdir(project_root)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Project imports after sys.path setup
    from src.models.dataset import HDF5RegressionDataset
    from src.models.train import train_model
    from src.models.evaluate import extract_predictions_and_embeddings
    from src.models.utils import set_seed

    dataset_cfg = get_required(cfg, "dataset")
    model_cfg = get_required(cfg, "model")
    training_cfg = get_required(cfg, "training")
    outputs_cfg = cfg.get("outputs", {})

    dataset_id = get_required(dataset_cfg, "dataset_id")

    data_processed = data_root / "processed"
    models_dir = data_root / "models"
    results_dir = data_root / "results"
    checkpoints_dir = models_dir / "checkpoints"

    for path in [data_processed, models_dir, results_dir, checkpoints_dir]:
        path.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_path(data_processed, get_required(dataset_cfg, "dataset_file"))
    split_path = resolve_path(data_processed, get_required(dataset_cfg, "split_file"))
    label_stats_path = resolve_path(data_processed, get_required(dataset_cfg, "label_stats_file"))

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

    stats = np.load(label_stats_path)
    y_mean = stats["y_mean"].astype(np.float32)
    y_std = stats["y_std"].astype(np.float32)

    if "label_names" in stats.files:
        label_names = [str(x) for x in stats["label_names"].tolist()]
    else:
        label_names = ["chirp_mass", "total_mass", "chi_eff"]

    print("train size:", len(train_idx))
    print("val size:", len(val_idx))
    print("label_names:", label_names)
    print("y_mean:", y_mean)
    print("y_std:", y_std)

    if np.any(y_std <= 0):
        raise ValueError(f"Invalid y_std values: {y_std}")

    overlap = np.intersect1d(train_idx, val_idx)
    if len(overlap) > 0:
        raise ValueError(f"Train/val overlap detected: {len(overlap)} samples")

    print()
    print("=" * 80)
    print("Creating datasets/loaders")
    print("=" * 80)

    train_dataset = HDF5RegressionDataset(
        h5_path=dataset_path,
        indices=train_idx,
        y_mean=y_mean,
        y_std=y_std,
    )

    val_dataset = HDF5RegressionDataset(
        h5_path=dataset_path,
        indices=val_idx,
        y_mean=y_mean,
        y_std=y_std,
    )

    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(training_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Sanity check
    X_batch, y_batch = next(iter(train_loader))

    print("X_batch:", X_batch.shape, X_batch.dtype)
    print("y_batch:", y_batch.shape, y_batch.dtype)
    print("X finite:", torch.isfinite(X_batch).all().item())
    print("y finite:", torch.isfinite(y_batch).all().item())
    print("y batch mean:", y_batch.mean(dim=0))
    print("y batch std:", y_batch.std(dim=0))

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
        "loss": training_cfg.get("loss", "MSELoss"),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "split_seed": seed,
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

    if bool(outputs_cfg.get("save_predictions", False)):
        print()
        print("=" * 80)
        print("Extracting predictions and embeddings")
        print("=" * 80)

        train_loader_eval = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        pred_train, emb_train, y_train_eval = extract_predictions_and_embeddings(
            model,
            train_loader_eval,
            device,
        )

        pred_val, emb_val, y_val_eval = extract_predictions_and_embeddings(
            model,
            val_loader,
            device,
        )

        pred_file_name = checkpoint_file_name.replace(
            "_checkpoint.pt",
            "_train_val_predictions_embeddings.npz",
        )

        pred_path = results_dir / pred_file_name

        np.savez_compressed(
            pred_path,
            pred_train=pred_train,
            emb_train=emb_train,
            y_train=y_train_eval,
            pred_val=pred_val,
            emb_val=emb_val,
            y_val=y_val_eval,
            y_mean=y_mean,
            y_std=y_std,
            train_idx=train_idx,
            val_idx=val_idx,
            label_names=np.array(label_names),
            checkpoint_file=checkpoint_file_name,
            dataset_path=str(dataset_path),
            split_path=str(split_path),
            label_stats_path=str(label_stats_path),
        )

        print("Saved predictions/embeddings:", pred_path)

    train_dataset.close()
    val_dataset.close()

    print()
    print("=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()