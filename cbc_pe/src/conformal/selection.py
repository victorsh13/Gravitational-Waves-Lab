from __future__ import annotations

"""
Selection policies for final Mondrian conformal configurations.

These functions reproduce the selection rules used for the closed M10
baseline in notebook:

    11_mondrian_m10_inputzscore_500k_final.ipynb

The selection is performed independently for each regression target.

Two final policies are supported:

- conservative:
    Prioritize local validity. Prefer configurations with no bins below
    the nominal 2-sigma tolerance band. Among configurations with nearly
    equivalent interval width, prefer greater Mondrian resolution.

- efficient:
    Prioritize narrow intervals subject to explicit local-validity
    constraints. If no configuration satisfies those constraints,
    fall back to the conservative policy.

The defaults below are part of the M10 conformal-selection definition.
"""

import pandas as pd


DEFAULT_MIN_COUNT_PER_BIN = 200
DEFAULT_WIDTH_TIE_FRACTION = 0.02
DEFAULT_MAX_UNDER_BIN_FRACTION_EFFICIENT = 0.10
DEFAULT_MAX_UNDERCOVERAGE_GAP_EFFICIENT = 0.05


def get_base_candidates(
    model_summary: pd.DataFrame,
    label: str,
    min_count_per_bin: int = DEFAULT_MIN_COUNT_PER_BIN,
) -> pd.DataFrame:
    """
    Return candidates satisfying the common M10 admissibility criteria.

    A candidate must:

    - correspond to the requested label;
    - have global coverage within the nominal 2-sigma tolerance band;
    - have at least ``min_count_per_bin`` test samples in every bin.
    """
    return model_summary[
        (model_summary["label"] == label)
        & model_summary["global_within_2sigma"]
        & (model_summary["min_count_per_bin"] >= min_count_per_bin)
    ].copy()


def select_conservative(
    model_summary: pd.DataFrame,
    label: str,
    min_count_per_bin: int = DEFAULT_MIN_COUNT_PER_BIN,
    width_tie_fraction: float = DEFAULT_WIDTH_TIE_FRACTION,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Select the M10 conservative Mondrian configuration for one label.

    Selection logic
    ---------------
    1. Apply the common admissibility criteria.
    2. Prefer configurations with zero bins below the 2-sigma
       lower tolerance bound.
    3. If none exist, retain configurations with the minimum number
       of such undercovered bins.
    4. Find the minimum global median physical interval width.
    5. Treat configurations within ``width_tie_fraction`` of that
       minimum as practically tied.
    6. Within that pool prefer, in order:

       - more Mondrian bins;
       - smaller maximum undercoverage gap;
       - smaller global tail-miss imbalance;
       - smaller median physical interval width.
    """
    candidates = get_base_candidates(
        model_summary,
        label,
        min_count_per_bin=min_count_per_bin,
    )

    if candidates.empty:
        raise ValueError(
            f"No globally valid candidates for label={label}"
        )

    zero_under = candidates[
        candidates["n_bins_under_2sigma"] == 0
    ].copy()

    if not zero_under.empty:
        pool = zero_under
        policy = "conservative_zero_under_bins_2sigma"
    else:
        min_under = candidates["n_bins_under_2sigma"].min()

        pool = candidates[
            candidates["n_bins_under_2sigma"] == min_under
        ].copy()

        policy = "conservative_best_available"

    min_width = pool["global_median_width_phys"].min()
    width_limit = min_width * (1.0 + width_tie_fraction)

    tied = pool[
        pool["global_median_width_phys"] <= width_limit
    ].copy()

    tied["min_width_for_label"] = min_width
    tied["width_limit"] = width_limit
    tied["relative_width_excess"] = (
        tied["global_median_width_phys"] / min_width - 1.0
    )

    ranked = tied.sort_values(
        [
            "n_bins",
            "max_undercoverage_gap",
            "global_tail_miss_imbalance",
            "global_median_width_phys",
        ],
        ascending=[False, True, True, True],
    )

    selected = ranked.iloc[0].copy()
    selected["selection_policy"] = policy

    return selected, ranked


def select_efficient(
    model_summary: pd.DataFrame,
    label: str,
    min_count_per_bin: int = DEFAULT_MIN_COUNT_PER_BIN,
    width_tie_fraction: float = DEFAULT_WIDTH_TIE_FRACTION,
    max_under_bin_fraction: float = (
        DEFAULT_MAX_UNDER_BIN_FRACTION_EFFICIENT
    ),
    max_undercoverage_gap: float = (
        DEFAULT_MAX_UNDERCOVERAGE_GAP_EFFICIENT
    ),
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Select the M10 efficient Mondrian configuration for one label.

    Efficient candidates must satisfy the common admissibility criteria
    and additionally:

    - fraction of bins below the 2-sigma tolerance band <= 0.10;
    - maximum undercoverage gap <= 0.05.

    Width is then prioritized, followed by local-validity diagnostics.

    If no candidate satisfies the efficient constraints, the function
    falls back to ``select_conservative``.
    """
    candidates = get_base_candidates(
        model_summary,
        label,
        min_count_per_bin=min_count_per_bin,
    )

    candidates = candidates[
        (
            candidates["under_bin_fraction_2sigma"]
            <= max_under_bin_fraction
        )
        & (
            candidates["max_undercoverage_gap"]
            <= max_undercoverage_gap
        )
    ].copy()

    if candidates.empty:
        return select_conservative(
            model_summary,
            label,
            min_count_per_bin=min_count_per_bin,
            width_tie_fraction=width_tie_fraction,
        )

    min_width = candidates["global_median_width_phys"].min()
    width_limit = min_width * (1.0 + width_tie_fraction)

    tied = candidates[
        candidates["global_median_width_phys"] <= width_limit
    ].copy()

    tied["min_width_for_label"] = min_width
    tied["width_limit"] = width_limit
    tied["relative_width_excess"] = (
        tied["global_median_width_phys"] / min_width - 1.0
    )

    ranked = tied.sort_values(
        [
            "global_median_width_phys",
            "n_bins_under_2sigma",
            "max_undercoverage_gap",
            "global_tail_miss_imbalance",
            "n_bins",
        ],
        ascending=[True, True, True, True, False],
    )

    selected = ranked.iloc[0].copy()
    selected["selection_policy"] = (
        "efficient_local_validity_tolerant"
    )

    return selected, ranked
