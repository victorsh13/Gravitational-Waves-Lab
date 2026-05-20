from pathlib import Path
import argparse
import json
import numpy as np
import h5py


LABEL_NAMES = ["chirp_mass", "total_mass", "chi_eff"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--val-size", type=int, required=True)

    parser.add_argument("--cal-size", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=0)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir) if args.output_dir else dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix if args.prefix else dataset_path.stem

    with h5py.File(dataset_path, "r") as f:
        n_samples = int(f.attrs["num_samples"])
        num_written = int(f.attrs.get("num_written", n_samples))
        status = f.attrs.get("dataset_status", "unknown")

        print(f"Dataset: {dataset_path}")
        print(f"num_samples: {n_samples}")
        print(f"num_written: {num_written}")
        print(f"status: {status}")

        if num_written != n_samples:
            raise ValueError(
                f"Dataset is incomplete: num_written={num_written}, num_samples={n_samples}"
            )

        if "y" not in f:
            raise KeyError("Dataset does not contain key 'y'.")

        y_ds = f["y"]

        requested = args.train_size + args.val_size + args.cal_size + args.test_size

        if requested > n_samples:
            raise ValueError(
                f"Requested split size {requested} exceeds dataset size {n_samples}."
            )

        rng = np.random.default_rng(args.seed)
        indices = np.arange(n_samples)

        if args.shuffle:
            rng.shuffle(indices)

        start = 0

        train_idx = indices[start:start + args.train_size]
        start += args.train_size

        val_idx = indices[start:start + args.val_size]
        start += args.val_size

        cal_idx = indices[start:start + args.cal_size]
        start += args.cal_size

        test_idx = indices[start:start + args.test_size]
        start += args.test_size

        # For HDF5 fancy indexing, sorted indices are safer/faster when reading.
        # But for DataLoader subsets, random order is fine because __getitem__ reads scalar indices.
        y_train = y_ds[np.sort(train_idx)]

        y_mean = y_train.mean(axis=0).astype(np.float32)
        y_std = y_train.std(axis=0).astype(np.float32)

    if np.any(y_std == 0):
        raise ValueError(f"At least one label std is zero: {y_std}")

    print()
    print("Split sizes:")
    print(f"train: {len(train_idx)}")
    print(f"val:   {len(val_idx)}")
    print(f"cal:   {len(cal_idx)}")
    print(f"test:  {len(test_idx)}")

    print()
    print("Train-only label stats:")
    print("mean:", y_mean)
    print("std: ", y_std)

    split_name = f"train{args.train_size}_val{args.val_size}"
    if args.cal_size > 0:
        split_name += f"_cal{args.cal_size}"
    if args.test_size > 0:
        split_name += f"_test{args.test_size}"
    split_name += f"_seed{args.seed}"

    splits_path = output_dir / f"{prefix}_splits_{split_name}.npz"
    stats_path = output_dir / f"{prefix}_label_stats_train_only_{split_name}.npz"

    save_payload = {
        "train_idx": train_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "seed": np.array(args.seed),
        "dataset_path": np.array(str(dataset_path)),
        "label_names": np.array(LABEL_NAMES),
    }

    if args.cal_size > 0:
        save_payload["cal_idx"] = cal_idx.astype(np.int64)

    if args.test_size > 0:
        save_payload["test_idx"] = test_idx.astype(np.int64)

    np.savez(splits_path, **save_payload)

    np.savez(
        stats_path,
        y_mean=y_mean,
        y_std=y_std,
        label_names=np.array(LABEL_NAMES),
        train_idx=train_idx.astype(np.int64),
        seed=np.array(args.seed),
        dataset_path=np.array(str(dataset_path)),
    )

    sidecar_path = output_dir / f"{prefix}_splits_{split_name}.metadata.json"

    with sidecar_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "dataset_path": str(dataset_path),
                "seed": int(args.seed),
                "shuffle": bool(args.shuffle),
                "train_size": int(args.train_size),
                "val_size": int(args.val_size),
                "cal_size": int(args.cal_size),
                "test_size": int(args.test_size),
                "splits_file": str(splits_path),
                "label_stats_file": str(stats_path),
                "label_names": LABEL_NAMES,
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
            },
            fp,
            indent=2,
        )

    print()
    print(f"Saved splits to: {splits_path}")
    print(f"Saved label stats to: {stats_path}")
    print(f"Saved split metadata to: {sidecar_path}")


if __name__ == "__main__":
    main()