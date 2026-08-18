from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV_VAR = "CBC_PE_DATA_ROOT"


def resolve_project_root(
    *,
    cli_project_root: str | Path | None = None,
    config_project_root: str | Path | None = None,
) -> Path:
    """
    Resolve the CBC-PE repository root.

    Precedence
    ----------
    1. Explicit CLI override.
    2. Config value, if it exists on this machine.
    3. Repository root inferred from this module location.
    """
    if cli_project_root is not None:
        path = Path(cli_project_root).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"CLI project root does not exist: {path}"
            )

        return path

    if config_project_root is not None:
        path = Path(
            config_project_root
        ).expanduser()

        if path.exists():
            return path.resolve()

    return Path(__file__).resolve().parents[1]


def resolve_data_root(
    *,
    cli_data_root: str | Path | None = None,
    config_data_root: str | Path | None = None,
) -> Path:
    """
    Resolve the external CBC-PE data root.

    Precedence
    ----------
    1. Explicit CLI override.
    2. CBC_PE_DATA_ROOT environment variable.
    3. Config value.
    """
    if cli_data_root is not None:
        return (
            Path(cli_data_root)
            .expanduser()
            .resolve()
        )

    env_value = os.environ.get(
        DATA_ROOT_ENV_VAR
    )

    if env_value:
        return (
            Path(env_value)
            .expanduser()
            .resolve()
        )

    if config_data_root is not None:
        return (
            Path(config_data_root)
            .expanduser()
            .resolve()
        )

    raise ValueError(
        "No data root configured. "
        "Use --data-root, set "
        f"{DATA_ROOT_ENV_VAR}, or provide "
        "'data_root' in the JSON config."
    )


def dataset_processed_dir(
    data_root: str | Path,
    dataset_id: str,
) -> Path:
    """
    Canonical directory for dataset and split/stat artifacts.
    """
    return (
        Path(data_root)
        / "processed"
        / dataset_id
    )


def resolve_processed_artifact(
    *,
    data_root: str | Path,
    dataset_id: str,
    file_name: str | Path,
    allow_legacy_flat: bool = True,
) -> Path:
    """
    Resolve a processed-data artifact.

    Preferred layout:
        processed/<dataset_id>/<file>

    Legacy fallback:
        processed/<file>
    """
    file_path = Path(file_name)

    if file_path.is_absolute():
        return file_path

    canonical = (
        dataset_processed_dir(
            data_root,
            dataset_id,
        )
        / file_path
    )

    if canonical.exists():
        return canonical

    legacy = (
        Path(data_root)
        / "processed"
        / file_path
    )

    if (
        allow_legacy_flat
        and legacy.exists()
    ):
        return legacy

    # Return canonical path so any eventual error
    # reports the intended modern location.
    return canonical