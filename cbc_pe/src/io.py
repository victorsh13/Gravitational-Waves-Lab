from pathlib import Path
import numpy as np
from dataclasses import dataclass

from .dataset import DatasetBatch






@dataclass(frozen=True)
class LoadedDataset:
    X: np.ndarray
    y: np.ndarray
    injection_times: list[float | None]
    network_snrs: list[float | None]
    metadata: dict


def save_dataset_npz(
    batch: DatasetBatch,
    output_dir: Path,
    file_name: str,
    detector_names: list[str] | None = None,
    overwrite: bool = False,
    ask_before_overwrite: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / file_name

    if dataset_path.exists() and not overwrite:
        if ask_before_overwrite:
            answer = input(
                f"Dataset already exists at '{dataset_path}'. Overwrite? [y/n]: "
            ).strip().lower()

            if answer != "y":
                print("Aborting. Dataset was not overwritten.")
                return dataset_path
        else:
            raise FileExistsError(
                f"Dataset already exists at '{dataset_path}'. "
                "Use overwrite=True to replace it."
            )

    arrays = {
        "X": batch.X.astype(np.float32),
        "y": batch.y.astype(np.float32),
        "injection_times": np.array(batch.injection_times),
        "network_snrs": np.array(batch.network_snrs),
    }

    if detector_names is not None:
        arrays["detector_names"] = np.array(detector_names)

    np.savez_compressed(dataset_path, **arrays)

    print(f"Saved dataset to '{dataset_path}'.")
    return dataset_path




def load_dataset_npz(path: str | Path) -> LoadedDataset:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path)

    required_keys = ["X", "y", "injection_times", "network_snrs"]

    for key in required_keys:
        if key not in data.files:
            raise KeyError(
                f"Missing key '{key}' in dataset file. "
                f"Available keys: {data.files}"
            )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    injection_times = data["injection_times"].tolist()
    network_snrs = data["network_snrs"].tolist()

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {X.shape}")

    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (N, D), got {y.shape}")

    if len(X) != len(y):
        raise ValueError(
            f"X and y have different number of samples: {len(X)} vs {len(y)}"
        )

    if len(injection_times) != len(X):
        raise ValueError(
            f"injection_times length does not match X: "
            f"{len(injection_times)} vs {len(X)}"
        )

    if len(network_snrs) != len(X):
        raise ValueError(
            f"network_snrs length does not match X: "
            f"{len(network_snrs)} vs {len(X)}"
        )

    metadata = {}

    if "detector_names" in data.files:
        metadata["detector_names"] = data["detector_names"].astype(str).tolist()

    return LoadedDataset(
        X=X,
        y=y,
        injection_times=injection_times,
        network_snrs=network_snrs,
        metadata=metadata,
    )