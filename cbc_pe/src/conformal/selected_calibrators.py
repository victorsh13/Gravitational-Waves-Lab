from __future__ import annotations

"""
Utilities for fitting and applying the final selected Mondrian calibrators.

This module bridges two stages of the conformal workflow:

    selected_configurations.csv
        -> fit calibration-derived Mondrian systems
        -> apply those systems to arbitrary target predictions

The fitted calibrators depend only on calibration data. Target ground-truth
labels are never required during application, which makes this interface
appropriate for real gravitational-wave events.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pipeline import (
    FittedMondrianRegressor,
    apply_mondrian,
    fit_mondrian,
)


_REQUIRED_SELECTION_COLUMNS = {
    "final_policy",
    "label",
    "label_index",
    "selection_policy",
    "taxonomy_mode",
    "interval_mode",
    "n_bins",
}


@dataclass(frozen=True)
class SelectedMondrianCalibrator:
    """
    One selected and fitted Mondrian calibrator.

    Parameters
    ----------
    final_policy : str
        Final selection policy, e.g. ``"conservative"`` or ``"efficient"``.

    label : str
        Regression target for which this configuration was selected.

    label_index : int
        Column index of ``label`` in the model output.

    selection_policy : str
        Detailed rule that produced the selected configuration.

    taxonomy_mode : str
        Mondrian taxonomy used by the fitted calibrator.

    interval_mode : str
        Conformal interval mode.

    n_bins : int
        Number of Mondrian bins.

    fitted : FittedMondrianRegressor
        Calibration-derived fitted Mondrian system.
    """

    final_policy: str
    label: str
    label_index: int
    selection_policy: str
    taxonomy_mode: str
    interval_mode: str
    n_bins: int
    fitted: FittedMondrianRegressor


def fit_selected_calibrators(
    selection_df: pd.DataFrame,
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    label_names: list[str] | tuple[str, ...],
    *,
    emb_cal: np.ndarray | None = None,
    confidence_level: float = 0.90,
    n_neighbors: int = 5,
    min_samples_per_bin: int = 10,
    apply_jitter: bool = True,
    jitter_variation: float = 1e-10,
) -> dict[tuple[str, str], SelectedMondrianCalibrator]:
    """
    Fit all selected Mondrian configurations using calibration data.

    Each row of ``selection_df`` defines one selected configuration for one
    target and one final policy.

    The returned dictionary is indexed by:

        (final_policy, label)

    Notes
    -----
    ``fit_mondrian`` currently calibrates all model outputs simultaneously.
    The selected target is identified through ``label_index`` when the fitted
    system is later applied.

    Parameters
    ----------
    selection_df : pd.DataFrame
        Table containing the selected Mondrian configurations.

    pred_cal : np.ndarray, shape (n_calibration_samples, n_labels)
        Calibration predictions in standardized label space.

    y_cal : np.ndarray, shape (n_calibration_samples, n_labels)
        Calibration ground-truth labels in standardized label space.

    label_names : sequence of str
        Ordered model output names.

    emb_cal : np.ndarray | None
        Calibration embeddings. Required if any selected configuration uses
        taxonomy_mode="difficulty".

    confidence_level : float
        Target conformal coverage.

    n_neighbors : int
        Number of nearest neighbours for difficulty-based taxonomy.

    min_samples_per_bin : int
        Minimum number of calibration residuals per Mondrian bin.

    apply_jitter : bool
        Whether to apply tiny jitter while fitting quantile bin edges.

    jitter_variation : float
        Uniform jitter half-width.

    Returns
    -------
    dict
        Mapping ``(final_policy, label)`` to
        ``SelectedMondrianCalibrator``.
    """
    selection_df = _validate_selection_table(
        selection_df=selection_df,
        label_names=label_names,
    )

    pred_cal = np.asarray(pred_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)

    _validate_calibration_arrays(
        pred_cal=pred_cal,
        y_cal=y_cal,
        label_names=label_names,
    )

    if emb_cal is not None:
        emb_cal = np.asarray(emb_cal, dtype=float)

        if emb_cal.ndim != 2:
            raise ValueError("emb_cal must be a 2D array.")

        if emb_cal.shape[0] != pred_cal.shape[0]:
            raise ValueError(
                "emb_cal and pred_cal must have the same number of samples."
            )

        if not np.all(np.isfinite(emb_cal)):
            raise ValueError(
                "emb_cal must contain only finite values."
            )

    uses_difficulty = bool(
        (selection_df["taxonomy_mode"] == "difficulty").any()
    )

    if uses_difficulty and emb_cal is None:
        raise ValueError(
            "emb_cal is required because at least one selected "
            "configuration uses difficulty taxonomy."
        )

    fitted_calibrators = {}

    for _, row in selection_df.iterrows():
        final_policy = str(row["final_policy"])
        label = str(row["label"])
        label_index = int(row["label_index"])
        selection_policy = str(row["selection_policy"])
        taxonomy_mode = str(row["taxonomy_mode"])
        interval_mode = str(row["interval_mode"])
        n_bins = int(row["n_bins"])

        fitted = fit_mondrian(
            pred_cal=pred_cal,
            y_cal=y_cal,
            n_bins=n_bins,
            cal_embedding=(
                emb_cal
                if taxonomy_mode == "difficulty"
                else None
            ),
            n_neighbors=(
                n_neighbors
                if taxonomy_mode == "difficulty"
                else None
            ),
            confidence_level=confidence_level,
            apply_jitter=apply_jitter,
            jitter_variation=jitter_variation,
            interval_mode=interval_mode,
            taxonomy_mode=taxonomy_mode,
            min_samples_per_bin=min_samples_per_bin,
        )

        key = (final_policy, label)

        fitted_calibrators[key] = SelectedMondrianCalibrator(
            final_policy=final_policy,
            label=label,
            label_index=label_index,
            selection_policy=selection_policy,
            taxonomy_mode=taxonomy_mode,
            interval_mode=interval_mode,
            n_bins=n_bins,
            fitted=fitted,
        )

    return fitted_calibrators


def apply_selected_calibrators(
    calibrators: dict[
        tuple[str, str],
        SelectedMondrianCalibrator,
    ],
    pred_target: np.ndarray,
    label_names: list[str] | tuple[str, ...],
    *,
    target_embedding: np.ndarray | None = None,
    event_name: str | None = None,
) -> pd.DataFrame:
    """
    Apply selected fitted calibrators to arbitrary target predictions.

    Target ground-truth labels are not required.

    Parameters
    ----------
    calibrators : dict
        Output from ``fit_selected_calibrators``.

    pred_target : np.ndarray, shape (n_target_samples, n_labels)
        Target model predictions in standardized label space.

    label_names : sequence of str
        Ordered model output names.

    target_embedding : np.ndarray | None
        Target embeddings. Required if any applied calibrator uses
        difficulty taxonomy.

    event_name : str | None
        Optional event identifier. Intended primarily for single-event real
        inference. If multiple target samples are supplied, the same name is
        attached to all rows.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per
        ``target sample x selected calibrator``.
    """
    if not calibrators:
        raise ValueError("calibrators must not be empty.")

    label_names = list(label_names)

    pred_target = np.asarray(
        pred_target,
        dtype=float,
    )

    if pred_target.ndim != 2:
        raise ValueError(
            "pred_target must be a 2D array."
        )

    if pred_target.shape[1] != len(label_names):
        raise ValueError(
            "pred_target column count must match label_names."
        )

    if not np.all(np.isfinite(pred_target)):
        raise ValueError(
            "pred_target must contain only finite values."
        )

    if target_embedding is not None:
        target_embedding = np.asarray(
            target_embedding,
            dtype=float,
        )

        if target_embedding.ndim != 2:
            raise ValueError(
                "target_embedding must be a 2D array."
            )

        if target_embedding.shape[0] != pred_target.shape[0]:
            raise ValueError(
                "target_embedding and pred_target must have "
                "the same number of samples."
            )

        if not np.all(np.isfinite(target_embedding)):
            raise ValueError(
                "target_embedding must contain only finite values."
            )

    uses_difficulty = any(
        selected.taxonomy_mode == "difficulty"
        for selected in calibrators.values()
    )

    if uses_difficulty and target_embedding is None:
        raise ValueError(
            "target_embedding is required because at least one "
            "selected calibrator uses difficulty taxonomy."
        )

    rows = []

    # Sort for deterministic output ordering.
    ordered_calibrators = sorted(
        calibrators.values(),
        key=lambda x: (
            x.final_policy,
            x.label_index,
            x.label,
        ),
    )

    for selected in ordered_calibrators:
        _validate_selected_calibrator_against_labels(
            selected=selected,
            label_names=label_names,
        )

        prediction = apply_mondrian(
            fitted=selected.fitted,
            pred_target=pred_target,
            target_embedding=(
                target_embedding
                if selected.taxonomy_mode == "difficulty"
                else None
            ),
        )

        j = selected.label_index

        for sample_index in range(pred_target.shape[0]):
            lower_std = float(
                prediction.lower[sample_index, j]
            )
            upper_std = float(
                prediction.upper[sample_index, j]
            )

            rows.append(
                {
                    "event": event_name,
                    "sample_index": sample_index,
                    "final_policy": selected.final_policy,
                    "label": selected.label,
                    "label_index": selected.label_index,
                    "selection_policy": selected.selection_policy,
                    "taxonomy_mode": selected.taxonomy_mode,
                    "interval_mode": selected.interval_mode,
                    "n_bins": selected.n_bins,
                    "pred_std": float(
                        pred_target[sample_index, j]
                    ),
                    "lower_std": lower_std,
                    "upper_std": upper_std,
                    "width_std": upper_std - lower_std,
                    "mondrian_bin": int(
                        prediction.bin_indices[sample_index, j]
                    ),
                }
            )

    return pd.DataFrame(rows)


def _validate_selection_table(
    selection_df: pd.DataFrame,
    label_names: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(selection_df, pd.DataFrame):
        raise TypeError(
            "selection_df must be a pandas DataFrame."
        )

    missing = (
        _REQUIRED_SELECTION_COLUMNS
        - set(selection_df.columns)
    )

    if missing:
        raise ValueError(
            "selection_df is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if selection_df.empty:
        raise ValueError(
            "selection_df must contain at least one selected configuration."
        )

    selection_df = selection_df.copy()
    label_names = list(label_names)

    duplicate_mask = selection_df.duplicated(
        subset=["final_policy", "label"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            selection_df.loc[
                duplicate_mask,
                ["final_policy", "label"],
            ]
            .drop_duplicates()
            .to_dict("records")
        )

        raise ValueError(
            "selection_df contains duplicate "
            "(final_policy, label) entries: "
            f"{duplicates}"
        )

    for _, row in selection_df.iterrows():
        label = str(row["label"])

        try:
            label_index = int(row["label_index"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid label_index for label={label!r}: "
                f"{row['label_index']!r}"
            )

        if label not in label_names:
            raise ValueError(
                f"Unknown label {label!r}. "
                f"Expected one of {label_names}."
            )

        if label_index < 0 or label_index >= len(label_names):
            raise ValueError(
                f"label_index={label_index} is outside the valid "
                f"range for label_names."
            )

        if label_names[label_index] != label:
            raise ValueError(
                "Inconsistent label mapping: "
                f"label={label!r}, "
                f"label_index={label_index}, "
                f"label_names[{label_index}]="
                f"{label_names[label_index]!r}."
            )

        taxonomy_mode = str(row["taxonomy_mode"])
        interval_mode = str(row["interval_mode"])

        if taxonomy_mode not in {
            "prediction",
            "difficulty",
        }:
            raise ValueError(
                f"Unsupported taxonomy_mode={taxonomy_mode!r}."
            )

        if interval_mode not in {
            "symmetric",
            "asymmetric",
        }:
            raise ValueError(
                f"Unsupported interval_mode={interval_mode!r}."
            )

        try:
            n_bins = int(row["n_bins"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid n_bins for label={label!r}: "
                f"{row['n_bins']!r}"
            )

        if n_bins < 2:
            raise ValueError(
                f"n_bins must be greater than 1 for label={label!r}."
            )

    return selection_df


def _validate_calibration_arrays(
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    label_names: list[str] | tuple[str, ...],
) -> None:
    if pred_cal.ndim != 2:
        raise ValueError(
            "pred_cal must be a 2D array."
        )

    if y_cal.ndim != 2:
        raise ValueError(
            "y_cal must be a 2D array."
        )

    if pred_cal.shape != y_cal.shape:
        raise ValueError(
            "pred_cal and y_cal must have the same shape."
        )

    if pred_cal.shape[1] != len(label_names):
        raise ValueError(
            "Calibration array column count must match label_names."
        )

    if not np.all(np.isfinite(pred_cal)):
        raise ValueError(
            "pred_cal must contain only finite values."
        )

    if not np.all(np.isfinite(y_cal)):
        raise ValueError(
            "y_cal must contain only finite values."
        )


def _validate_selected_calibrator_against_labels(
    selected: SelectedMondrianCalibrator,
    label_names: list[str],
) -> None:
    if (
        selected.label_index < 0
        or selected.label_index >= len(label_names)
    ):
        raise ValueError(
            f"Invalid label_index={selected.label_index} "
            f"for selected label={selected.label!r}."
        )

    expected_label = label_names[
        selected.label_index
    ]

    if expected_label != selected.label:
        raise ValueError(
            "Selected calibrator label mapping is inconsistent: "
            f"label={selected.label!r}, "
            f"label_index={selected.label_index}, "
            f"expected={expected_label!r}."
        )
