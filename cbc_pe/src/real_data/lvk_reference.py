from __future__ import annotations

"""
Construction of LVK reference quantities for real-event evaluation.

This module transforms published GWOSC/LVK parameter summaries into the
physical quantities used as reference values for comparison with the model.

Published source-frame mass summaries are converted to detector frame using

    M_det = (1 + z) M_source

with asymmetric first-order propagation of source-mass and redshift
uncertainties.

Dimensionless quantities such as chi_eff are copied without redshift
conversion.
"""

import numpy as np
import pandas as pd


def propagate_detector_frame_mass_interval(
    mass_source_best,
    mass_source_lower,
    mass_source_upper,
    redshift_best,
    redshift_lower,
    redshift_upper,
):
    """
    Convert source-frame mass summaries to detector frame.

    Source-mass and redshift uncertainties are propagated separately for
    the lower and upper interval sides using first-order error propagation.

    Returns
    -------
    mass_det_best
        Central detector-frame mass.

    mass_det_lower
        Lower detector-frame interval bound.

    mass_det_upper
        Upper detector-frame interval bound.
    """
    mass = np.asarray(
        mass_source_best,
        dtype=float,
    )

    mass_lower = np.asarray(
        mass_source_lower,
        dtype=float,
    )

    mass_upper = np.asarray(
        mass_source_upper,
        dtype=float,
    )

    redshift = np.asarray(
        redshift_best,
        dtype=float,
    )

    redshift_lower = np.asarray(
        redshift_lower,
        dtype=float,
    )

    redshift_upper = np.asarray(
        redshift_upper,
        dtype=float,
    )

    mass_det = (
        1.0 + redshift
    ) * mass

    delta_mass_minus = (
        mass - mass_lower
    )

    delta_mass_plus = (
        mass_upper - mass
    )

    delta_z_minus = (
        redshift - redshift_lower
    )

    delta_z_plus = (
        redshift_upper - redshift
    )

    delta_det_minus = np.sqrt(
        (
            (1.0 + redshift)
            * delta_mass_minus
        ) ** 2
        +
        (
            mass
            * delta_z_minus
        ) ** 2
    )

    delta_det_plus = np.sqrt(
        (
            (1.0 + redshift)
            * delta_mass_plus
        ) ** 2
        +
        (
            mass
            * delta_z_plus
        ) ** 2
    )

    mass_det_lower = (
        mass_det - delta_det_minus
    )

    mass_det_upper = (
        mass_det + delta_det_plus
    )

    return (
        mass_det,
        mass_det_lower,
        mass_det_upper,
    )


def build_lvk_reference_detector_frame(
    gwosc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the LVK real-event reference table in detector frame.

    Chirp mass, total mass and component masses are converted from source
    frame to detector frame. Their intervals include first-order redshift
    uncertainty propagation.

    ``chi_eff`` is dimensionless and is copied without transformation.
    """
    df = gwosc_df.copy()

    required_base_columns = [
        "event",
        "gps_time",
        "catalog",
        "detectors",
        "redshift_best",
        "redshift_lower",
        "redshift_upper",
        "chirp_mass_source_best",
        "chirp_mass_source_lower",
        "chirp_mass_source_upper",
        "chi_eff_best",
        "chi_eff_lower",
        "chi_eff_upper",
    ]

    missing = [
        column
        for column in required_base_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "Missing required LVK reference columns: "
            f"{missing}"
        )

    out = pd.DataFrame(
        index=df.index,
    )

    out["event"] = df["event"]
    out["gps_time"] = df["gps_time"]
    out["catalog"] = df["catalog"]
    out["detectors"] = df["detectors"]

    out["redshift"] = (
        df["redshift_best"]
        .astype(float)
    )

    out["redshift_lower"] = (
        df["redshift_lower"]
        .astype(float)
    )

    out["redshift_upper"] = (
        df["redshift_upper"]
        .astype(float)
    )

    (
        out["chirp_mass_lvk"],
        out["chirp_mass_lvk_lower"],
        out["chirp_mass_lvk_upper"],
    ) = propagate_detector_frame_mass_interval(
        mass_source_best=(
            df["chirp_mass_source_best"]
        ),
        mass_source_lower=(
            df["chirp_mass_source_lower"]
        ),
        mass_source_upper=(
            df["chirp_mass_source_upper"]
        ),
        redshift_best=(
            df["redshift_best"]
        ),
        redshift_lower=(
            df["redshift_lower"]
        ),
        redshift_upper=(
            df["redshift_upper"]
        ),
    )

    total_mass_columns = [
        "total_mass_source_best",
        "total_mass_source_lower",
        "total_mass_source_upper",
    ]

    if all(
        column in df.columns
        for column in total_mass_columns
    ):
        total_source_best = (
            df["total_mass_source_best"]
        )

        total_source_lower = (
            df["total_mass_source_lower"]
        )

        total_source_upper = (
            df["total_mass_source_upper"]
        )

    else:
        required_component_columns = [
            "mass_1_source_best",
            "mass_1_source_lower",
            "mass_1_source_upper",
            "mass_2_source_best",
            "mass_2_source_lower",
            "mass_2_source_upper",
        ]

        missing_components = [
            column
            for column
            in required_component_columns
            if column not in df.columns
        ]

        if missing_components:
            raise KeyError(
                "Cannot construct total mass. "
                "Missing columns: "
                f"{missing_components}"
            )

        total_source_best = (
            df["mass_1_source_best"]
            .astype(float)
            +
            df["mass_2_source_best"]
            .astype(float)
        )

        total_source_lower = (
            df["mass_1_source_lower"]
            .astype(float)
            +
            df["mass_2_source_lower"]
            .astype(float)
        )

        total_source_upper = (
            df["mass_1_source_upper"]
            .astype(float)
            +
            df["mass_2_source_upper"]
            .astype(float)
        )

    (
        out["total_mass_lvk"],
        out["total_mass_lvk_lower"],
        out["total_mass_lvk_upper"],
    ) = propagate_detector_frame_mass_interval(
        mass_source_best=(
            total_source_best
        ),
        mass_source_lower=(
            total_source_lower
        ),
        mass_source_upper=(
            total_source_upper
        ),
        redshift_best=(
            df["redshift_best"]
        ),
        redshift_lower=(
            df["redshift_lower"]
        ),
        redshift_upper=(
            df["redshift_upper"]
        ),
    )

    component_columns = [
        "mass_1_source_best",
        "mass_1_source_lower",
        "mass_1_source_upper",
        "mass_2_source_best",
        "mass_2_source_lower",
        "mass_2_source_upper",
    ]

    if all(
        column in df.columns
        for column in component_columns
    ):
        (
            out["mass_1_lvk"],
            out["mass_1_lvk_lower"],
            out["mass_1_lvk_upper"],
        ) = propagate_detector_frame_mass_interval(
            mass_source_best=(
                df["mass_1_source_best"]
            ),
            mass_source_lower=(
                df["mass_1_source_lower"]
            ),
            mass_source_upper=(
                df["mass_1_source_upper"]
            ),
            redshift_best=(
                df["redshift_best"]
            ),
            redshift_lower=(
                df["redshift_lower"]
            ),
            redshift_upper=(
                df["redshift_upper"]
            ),
        )

        (
            out["mass_2_lvk"],
            out["mass_2_lvk_lower"],
            out["mass_2_lvk_upper"],
        ) = propagate_detector_frame_mass_interval(
            mass_source_best=(
                df["mass_2_source_best"]
            ),
            mass_source_lower=(
                df["mass_2_source_lower"]
            ),
            mass_source_upper=(
                df["mass_2_source_upper"]
            ),
            redshift_best=(
                df["redshift_best"]
            ),
            redshift_lower=(
                df["redshift_lower"]
            ),
            redshift_upper=(
                df["redshift_upper"]
            ),
        )

    out["chi_eff_lvk"] = (
        df["chi_eff_best"]
        .astype(float)
    )

    out["chi_eff_lvk_lower"] = (
        df["chi_eff_lower"]
        .astype(float)
    )

    out["chi_eff_lvk_upper"] = (
        df["chi_eff_upper"]
        .astype(float)
    )

    return out
