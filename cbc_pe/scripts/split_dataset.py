from pathlib import Path
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to merged dataset .npz file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where split/stat files will be saved. Defaults to dataset parent.",
    )

    parser.add_argument(
        "--train-size",
        type=int,
        required=True,
        help="Number of training samples.",
    )

    parser.add_argument(
        "--val-size",
        type=int,
        required=True,
        help="Number of validation samples.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible split.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output files. Defaults to dataset stem.",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle indices before splitting.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix if args.prefix is not None else dataset_path.stem

    print(f"Loading dataset: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    if "X" not in data.files:
        raise KeyError("Dataset does not contain key 'X'.")

    if "y" not in data.files:
        raise KeyError("Dataset does not contain key 'y'.")

    X = data["X"]
    y = data["y"]

    n_samples = X.shape[0]

    print(f"X shape: {X.shape}, dtype={X.dtype}")
    print(f"y shape: {y.shape}, dtype={y.dtype}")
    print(f"Total samples: {n_samples}")

    requested = args.train_size + args.val_size

    if requested > n_samples:
        raise ValueError(
            f"Requested train+val = {requested}, but dataset has only {n_samples} samples."
        )

    rng = np.random.default_rng(args.seed)

    indices = np.arange(n_samples)

    if args.shuffle:
        rng.shuffle(indices)

    train_idx = indices[:args.train_size]
    val_idx = indices[args.train_size:args.train_size + args.val_size]

    print()
    print("Split sizes:")
    print(f"train: {len(train_idx)}")
    print(f"val:   {len(val_idx)}")

    # Sanity checks
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(train_idx) == args.train_size
    assert len(val_idx) == args.val_size

    # Label standardization stats from train only
    y_train = y[train_idx]

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    if np.any(y_std == 0):
        raise ValueError(f"At least one label has zero std: {y_std}")

    print()
    print("Train-only label statistics:")
    print(f"mean: {y_mean}")
    print(f"std:  {y_std}")

    splits_path = output_dir / f"{prefix}_splits_train{args.train_size}_val{args.val_size}_seed{args.seed}.npz"
    stats_path = output_dir / f"{prefix}_label_stats_train_only_seed{args.seed}.npz"

    np.savez(
        splits_path,
        train_idx=train_idx,
        val_idx=val_idx,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        dataset_path=str(dataset_path),
    )

    np.savez(
        stats_path,
        y_mean=y_mean,
        y_std=y_std,
        label_names=np.array(["chirp_mass", "total_mass", "chi_eff"]),
        train_idx=train_idx,
        seed=args.seed,
        dataset_path=str(dataset_path),
    )

    print()
    print(f"Saved splits to: {splits_path}")
    print(f"Saved label stats to: {stats_path}")

    data.close()


if __name__ == "__main__":
    main()