from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class LoadedDataset:
    X: np.ndarray
    y: np.ndarray
    parameters: list[dict]
    metadata: list[dict]
    dataset_metadata: dict


def save_dataset_npz(
    batch,
    output_dir: Path | str,
    file_name: str,
    detector_names: list[str] | None = None,
    overwrite: bool = False,
    ask_before_overwrite: bool = True,
) -> Path:
    """
    Save a generated dataset batch.

    Arrays are saved in a compressed NPZ file:
        - X
        - y

    Per-sample parameters and metadata are saved in a sidecar JSON file:
        - <file_name>.metadata.json

    This avoids storing large or nested Python objects inside the NPZ file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not file_name.endswith(".npz"):
        file_name = f"{file_name}.npz"

    dataset_path = output_dir / file_name
    metadata_path = dataset_path.with_suffix(".metadata.json")

    _handle_existing_file(
        dataset_path,
        overwrite=overwrite,
        ask_before_overwrite=ask_before_overwrite,
    )
    _handle_existing_file(
        metadata_path,
        overwrite=overwrite,
        ask_before_overwrite=ask_before_overwrite,
    )

    _validate_batch(batch)

    arrays = {
        "X": batch.X.astype(np.float32),
        "y": batch.y.astype(np.float32),
    }

    if detector_names is not None:
        arrays["detector_names"] = np.asarray(detector_names)

    np.savez_compressed(dataset_path, **arrays)

    metadata_payload = {
        "dataset_file": dataset_path.name,
        "num_samples": int(batch.X.shape[0]),
        "X_shape": list(batch.X.shape),
        "y_shape": list(batch.y.shape),
        "parameters": [_parameters_to_dict(p) for p in batch.parameters],
        "metadata": _to_jsonable(batch.metadata),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset arrays to '{dataset_path}'.")
    print(f"Saved dataset metadata to '{metadata_path}'.")

    return dataset_path


def load_dataset_npz(path: str | Path) -> LoadedDataset:
    """
    Load a dataset saved with save_dataset_npz.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    metadata_path = path.with_suffix(".metadata.json")

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata sidecar file not found: {metadata_path}"
        )

    data = np.load(path, allow_pickle=False)

    required_keys = ["X", "y"]

    for key in required_keys:
        if key not in data.files:
            raise KeyError(
                f"Missing key '{key}' in dataset file. "
                f"Available keys: {data.files}"
            )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, C, L), got {X.shape}.")

    if y.ndim != 2:
        raise ValueError(f"Expected y with shape (N, D), got {y.shape}.")

    if len(X) != len(y):
        raise ValueError(
            f"X and y have different number of samples: {len(X)} vs {len(y)}."
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        dataset_metadata = json.load(f)

    parameters = dataset_metadata.get("parameters", [])
    metadata = dataset_metadata.get("metadata", [])

    if len(parameters) != len(X):
        raise ValueError(
            f"parameters length does not match X: {len(parameters)} vs {len(X)}."
        )

    if len(metadata) != len(X):
        raise ValueError(
            f"metadata length does not match X: {len(metadata)} vs {len(X)}."
        )

    return LoadedDataset(
        X=X,
        y=y,
        parameters=parameters,
        metadata=metadata,
        dataset_metadata=dataset_metadata,
    )


def _handle_existing_file(
    path: Path,
    overwrite: bool,
    ask_before_overwrite: bool,
) -> None:
    if not path.exists():
        return

    if overwrite:
        return

    if ask_before_overwrite:
        answer = input(
            f"File already exists at '{path}'. Overwrite? [y/n]: "
        ).strip().lower()

        if answer != "y":
            raise FileExistsError(f"Aborted. File was not overwritten: {path}")

        return

    raise FileExistsError(
        f"File already exists at '{path}'. "
        "Use overwrite=True to replace it."
    )


def _validate_batch(batch) -> None:
    if not hasattr(batch, "X"):
        raise TypeError("batch must have attribute X.")

    if not hasattr(batch, "y"):
        raise TypeError("batch must have attribute y.")

    if not hasattr(batch, "parameters"):
        raise TypeError("batch must have attribute parameters.")

    if not hasattr(batch, "metadata"):
        raise TypeError("batch must have attribute metadata.")

    if batch.X.ndim != 3:
        raise ValueError(f"Expected batch.X with shape (N, C, L), got {batch.X.shape}.")

    if batch.y.ndim != 2:
        raise ValueError(f"Expected batch.y with shape (N, D), got {batch.y.shape}.")

    n_samples = batch.X.shape[0]

    if batch.y.shape[0] != n_samples:
        raise ValueError(
            f"batch.X and batch.y have different sample counts: "
            f"{n_samples} vs {batch.y.shape[0]}."
        )

    if len(batch.parameters) != n_samples:
        raise ValueError(
            f"parameters length does not match X: "
            f"{len(batch.parameters)} vs {n_samples}."
        )

    if len(batch.metadata) != n_samples:
        raise ValueError(
            f"metadata length does not match X: "
            f"{len(batch.metadata)} vs {n_samples}."
        )

    if not np.all(np.isfinite(batch.X)):
        raise ValueError("batch.X contains NaN or Inf.")

    if not np.all(np.isfinite(batch.y)):
        raise ValueError("batch.y contains NaN or Inf.")


def _parameters_to_dict(parameters) -> dict:
    """
    Convert CBCParameters to a JSON-serializable dict.
    """
    return {
        "mass_1": float(parameters.mass_1),
        "mass_2": float(parameters.mass_2),
        "distance": float(parameters.distance),
        "inclination": float(parameters.inclination),
        "ra": float(parameters.ra),
        "dec": float(parameters.dec),
        "spin_1z": float(parameters.spin_1z),
        "spin_2z": float(parameters.spin_2z),
        "polarization_angle": float(parameters.polarization_angle),
        "total_mass": float(parameters.total_mass),
        "chirp_mass": float(parameters.chirp_mass),
        "chi_eff": float(parameters.chi_eff),
    }


def _to_jsonable(obj):
    """
    Recursively convert common NumPy/Python objects into JSON-serializable types.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Last-resort fallback. Avoids JSON crashing on harmless objects,
    # but if this appears often, metadata should be cleaned upstream.
    return str(obj)