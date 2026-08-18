from __future__ import annotations

"""
Evaluation utilities for comparing model predictions with LVK references.

This module contains numerical and tabular comparison logic only.
Plotting and report-specific presentation remain outside the reusable
evaluation layer.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_LABELS = (
    "chirp_mass",
    "total_mass",
    "chi_eff",
)


def add_physical_clipped_intervals(
    interval_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add interval columns intended only for physical visualization.

    Original conformal intervals are preserved unchanged.

    For mass targets, negative lower bounds are clipped at zero in the
    ``*_clipped`` columns only. ``chi_eff`` intervals are not clipped.

    Notes
    -----
    These clipped intervals must not silently replace the original
    conformal intervals in scientific coverage or overlap calculations.
    """
    df = interval_df.copy()

    required_columns = {
        "label",
        "lower_phys",
        "upper_phys",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            "Missing required interval columns: "
            f"{sorted(missing)}"
        )

    df["lower_phys_clipped"] = (
        df["lower_phys"]
    )

    df["upper_phys_clipped"] = (
        df["upper_phys"]
    )

    mass_mask = df["label"].isin(
        [
            "chirp_mass",
            "total_mass",
        ]
    )

    df.loc[
        mass_mask,
        "lower_phys_clipped",
    ] = (
        df.loc[
            mass_mask,
            "lower_phys_clipped",
        ]
        .clip(lower=0.0)
    )

    return df


def cnn_results_to_wide(
    point_df: pd.DataFrame,
    interval_df: pd.DataFrame,
    *,
    labels: Sequence[str] = DEFAULT_LABELS,
    use_clipped: bool = False,
) -> pd.DataFrame:
    """
    Convert long-format CNN point predictions and intervals to one row
    per event.

    Parameters
    ----------
    point_df
        Long-format point predictions containing ``event``, ``label`` and
        ``pred_phys``.

    interval_df
        Long-format interval predictions.

    labels
        Target labels and output ordering.

    use_clipped
        If True, use ``lower_phys_clipped`` and ``upper_phys_clipped``.
        Otherwise use the original conformal interval bounds.
    """
    labels = tuple(labels)

    point_required = {
        "event",
        "label",
        "pred_phys",
    }

    missing_point = (
        point_required
        - set(point_df.columns)
    )

    if missing_point:
        raise KeyError(
            "Missing point-prediction columns: "
            f"{sorted(missing_point)}"
        )

    lower_col = (
        "lower_phys_clipped"
        if use_clipped
        else "lower_phys"
    )

    upper_col = (
        "upper_phys_clipped"
        if use_clipped
        else "upper_phys"
    )

    interval_required = {
        "event",
        "label",
        lower_col,
        upper_col,
    }

    missing_interval = (
        interval_required
        - set(interval_df.columns)
    )

    if missing_interval:
        raise KeyError(
            "Missing interval columns: "
            f"{sorted(missing_interval)}"
        )

    point_wide = (
        point_df.pivot_table(
            index="event",
            columns="label",
            values="pred_phys",
            aggfunc="first",
        )
    )

    lower_wide = (
        interval_df.pivot_table(
            index="event",
            columns="label",
            values=lower_col,
            aggfunc="first",
        )
    )

    upper_wide = (
        interval_df.pivot_table(
            index="event",
            columns="label",
            values=upper_col,
            aggfunc="first",
        )
    )

    missing_labels = [
        label
        for label in labels
        if (
            label not in point_wide.columns
            or label not in lower_wide.columns
            or label not in upper_wide.columns
        )
    ]

    if missing_labels:
        raise KeyError(
            "Missing requested labels in point/interval tables: "
            f"{missing_labels}"
        )

    out = pd.DataFrame(
        index=point_wide.index
    )

    for label in labels:
        out[
            f"{label}_cnn"
        ] = point_wide[label]

        out[
            f"{label}_cnn_lower"
        ] = lower_wide[label]

        out[
            f"{label}_cnn_upper"
        ] = upper_wide[label]

    return out.reset_index()


def add_lvk_comparison_metrics(
    comparison_df: pd.DataFrame,
    *,
    labels: Sequence[str] = DEFAULT_LABELS,
) -> pd.DataFrame:
    """
    Add point-error and interval-overlap diagnostics relative to LVK.

    The implementation preserves the closed M10 definitions:

    - delta = CNN point estimate - LVK central value
    - normalized delta = delta / LVK half-width
    - CNN point inside LVK interval
    - LVK central value inside CNN interval
    - absolute interval overlap
    - overlap fraction normalized by LVK interval width
    """
    df = comparison_df.copy()

    for label in labels:
        required_columns = [
            f"{label}_cnn",
            f"{label}_cnn_lower",
            f"{label}_cnn_upper",
            f"{label}_lvk",
            f"{label}_lvk_lower",
            f"{label}_lvk_upper",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise KeyError(
                f"Missing columns for label {label}: "
                f"{missing}"
            )

        cnn = df[
            f"{label}_cnn"
        ]

        cnn_lower = df[
            f"{label}_cnn_lower"
        ]

        cnn_upper = df[
            f"{label}_cnn_upper"
        ]

        lvk = df[
            f"{label}_lvk"
        ]

        lvk_lower = df[
            f"{label}_lvk_lower"
        ]

        lvk_upper = df[
            f"{label}_lvk_upper"
        ]

        delta = (
            cnn - lvk
        )

        df[
            f"{label}_delta_cnn_minus_lvk"
        ] = delta

        lvk_half_width = (
            0.5
            * (
                lvk_upper
                - lvk_lower
            )
        )

        df[
            f"{label}_normalized_delta_lvk"
        ] = (
            delta
            / lvk_half_width
        )

        df[
            f"{label}_cnn_point_inside_lvk"
        ] = (
            (cnn >= lvk_lower)
            & (cnn <= lvk_upper)
        )

        df[
            f"{label}_lvk_median_inside_cnn"
        ] = (
            (lvk >= cnn_lower)
            & (lvk <= cnn_upper)
        )

        overlap = np.maximum(
            0.0,
            np.minimum(
                cnn_upper,
                lvk_upper,
            )
            - np.maximum(
                cnn_lower,
                lvk_lower,
            ),
        )

        df[
            f"{label}_interval_overlap"
        ] = overlap

        df[
            f"{label}_interval_overlap_fraction_lvk"
        ] = (
            overlap
            / (
                lvk_upper
                - lvk_lower
            )
        )

    return df
