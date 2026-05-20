from __future__ import annotations
from pathlib import Path
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader




def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN baseline for CBC parameter regression.")

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/data/vserrano/gw/Gravitational-Waves-Lab/cbc_pe"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/vserrano/cbc_pe_data"),
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="bbh_processed_4s_seobnrv4opt_snr10-25_n15000_merged",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Optional explicit split filename. If omitted, uses default train12000_val3000_seed123 name.",
    )
    parser.add_argument(
        "--label-stats-file",
        type=str,
        default=None,
        help="Optional explicit label stats filename.",
    )

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-4)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dropout-conv", type=float, default=0.05)
    parser.add_argument("--dropout-dense", type=float, default=0.1)
    parser.add_argument("--pool-size", type=int, default=4)

    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="If set, save train/val predictions and embeddings after training.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    PROJECT_ROOT = args.project_root
    DATA_ROOT = args.data_root

    DATA_PROCESSED = DATA_ROOT / "processed"
    MODELS_DIR = DATA_ROOT / "models"
    RESULTS_DIR = DATA_ROOT / "results"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

    for path in [DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, CHECKPOINTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")

    os.chdir(PROJECT_ROOT)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


    from src.io import load_dataset_npz
    from src.models.dataset import ArrayRegressionDataset
    from src.models.network import SimpleCNN_Pool, CNN_Pool
    from src.models.train import train_model
    from src.models.evaluate import extract_predictions_and_embeddings
    from src.models.utils import set_seed

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Environment")
    print("=" * 80)
    print("hostname:", os.uname().nodename)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_ROOT:", DATA_ROOT)
    print("cwd:", Path.cwd())
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("device:", device)

    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        print("n GPUs visible:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))

    dataset_id = args.dataset_id

    dataset_path = DATA_PROCESSED / f"{dataset_id}.npz"

    split_file = args.split_file
    if split_file is None:
        split_file = f"{dataset_id}_splits_train12000_val3000_seed123.npz"

    label_stats_file = args.label_stats_file
    if label_stats_file is None:
        label_stats_file = f"{dataset_id}_label_stats_train_only_seed123.npz"

    split_path = DATA_PROCESSED / split_file
    label_stats_path = DATA_PROCESSED / label_stats_file

    print("=" * 80)
    print("Input files")
    print("=" * 80)
    print("dataset_path:", dataset_path)
    print("split_path:", split_path)
    print("label_stats_path:", label_stats_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not label_stats_path.exists():
        raise FileNotFoundError(f"Label stats file not found: {label_stats_path}")

    print("=" * 80)
    print("Loading dataset")
    print("=" * 80)

    batch = load_dataset_npz(dataset_path)

    X = batch.X
    y = batch.y

    splits = np.load(split_path)
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]

    X_train = X[train_idx].astype(np.float32)
    y_train_phys = y[train_idx]
    X_val = X[val_idx].astype(np.float32)
    y_val_phys = y[val_idx]

    stats = np.load(label_stats_path)
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]
    label_names = stats["label_names"].tolist()

    y_std_safe = y_std + 1e-8

    y_train_std = ((y_train_phys - y_mean) / y_std_safe).astype(np.float32)
    y_val_std = ((y_val_phys - y_mean) / y_std_safe).astype(np.float32)

    print("X:", X.shape)
    print("y:", y.shape)
    print("X_train:", X_train.shape)
    print("y_train_std:", y_train_std.shape)
    print("X_val:", X_val.shape)
    print("y_val_std:", y_val_std.shape)
    print("label_names:", label_names)
    print("y_mean:", y_mean)
    print("y_std:", y_std)
    print("train standardized mean:", y_train_std.mean(axis=0))
    print("train standardized std:", y_train_std.std(axis=0))
    print("val standardized mean:", y_val_std.mean(axis=0))
    print("val standardized std:", y_val_std.std(axis=0))

    train_dataset = ArrayRegressionDataset(X_train, y_train_std)
    val_dataset = ArrayRegressionDataset(X_val, y_val_std)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model_config = {
        "architecture": "SimpleCNN_Pool",
        "dataset_id": dataset_id,
        "n_detectors": X.shape[1],
        "n_outputs": y.shape[1],
        "embedding_dim": args.embedding_dim,
        "dropout_conv": args.dropout_conv,
        "dropout_dense": args.dropout_dense,
        "normalization": "GroupNorm",
        "activation_conv": "LeakyReLU(0.01)",
        "activation_dense": "LeakyReLU(0.1)",
        "loss": "MSELoss",
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "split_seed": args.seed,
        "pool_size": args.pool_size,
    }

    print("=" * 80)
    print("Model config")
    print("=" * 80)
    for key, value in model_config.items():
        print(f"{key}: {value}")

    model = CNN_Pool(
        n_detectors=model_config["n_detectors"],
        n_outputs=model_config["n_outputs"],
        embedding_dim=model_config["embedding_dim"],
        dropout_conv=model_config["dropout_conv"],
        dropout_dense=model_config["dropout_dense"],
        pool_size=model_config["pool_size"],
    ).to(device)

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
        model_config=model_config,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    print("=" * 80)
    print("Best checkpoint")
    print("=" * 80)
    print("Best epoch:", best_checkpoint["epoch"])
    print("Best val loss:", best_checkpoint["best_val_loss"])

    checkpoint_file_name = (
        f"{dataset_id}"
        f"_SimpleCNNPool"
        f"_pool{args.pool_size}"
        f"_MSELoss"
        f"_emb{args.embedding_dim}"
        f"_seed{args.seed}"
        f"_checkpoint.pt"
    )

    checkpoint_path = CHECKPOINTS_DIR / checkpoint_file_name
    torch.save(best_checkpoint, checkpoint_path)

    print("Saved checkpoint:", checkpoint_path)

    if args.save_predictions:
        print("=" * 80)
        print("Extracting predictions and embeddings")
        print("=" * 80)

        train_loader_eval = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
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

        pred_file = checkpoint_file_name.replace(
            ".pt",
            "_train_val_predictions_embeddings.npz",
        )

        pred_path = RESULTS_DIR / pred_file

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
        )

        print("Saved predictions/embeddings:", pred_path)

    print("=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()