import unittest

import numpy as np
import pandas as pd

from src.evaluation.lvk import (
    add_lvk_comparison_metrics,
    add_physical_clipped_intervals,
    cnn_results_to_wide,
)


LABELS = (
    "chirp_mass",
    "total_mass",
    "chi_eff",
)


class TestPhysicalClipping(unittest.TestCase):
    def test_mass_lower_bounds_are_clipped_only_in_new_columns(self):
        df = pd.DataFrame(
            {
                "event": [
                    "GW1",
                    "GW1",
                    "GW1",
                ],
                "label": LABELS,
                "lower_phys": [
                    -5.0,
                    -10.0,
                    -0.7,
                ],
                "upper_phys": [
                    30.0,
                    70.0,
                    0.2,
                ],
            }
        )

        out = (
            add_physical_clipped_intervals(
                df
            )
        )

        self.assertEqual(
            out.loc[
                0,
                "lower_phys",
            ],
            -5.0,
        )

        self.assertEqual(
            out.loc[
                0,
                "lower_phys_clipped",
            ],
            0.0,
        )

        self.assertEqual(
            out.loc[
                1,
                "lower_phys_clipped",
            ],
            0.0,
        )

        self.assertEqual(
            out.loc[
                2,
                "lower_phys_clipped",
            ],
            -0.7,
        )

        self.assertEqual(
            out.loc[
                2,
                "upper_phys_clipped",
            ],
            0.2,
        )


class TestCNNResultsWide(unittest.TestCase):
    def setUp(self):
        self.point_df = pd.DataFrame(
            {
                "event": [
                    "GW1",
                    "GW1",
                    "GW1",
                ],
                "label": LABELS,
                "pred_phys": [
                    25.0,
                    60.0,
                    0.1,
                ],
            }
        )

        self.interval_df = pd.DataFrame(
            {
                "event": [
                    "GW1",
                    "GW1",
                    "GW1",
                ],
                "label": LABELS,
                "lower_phys": [
                    20.0,
                    50.0,
                    -0.2,
                ],
                "upper_phys": [
                    30.0,
                    70.0,
                    0.4,
                ],
                "lower_phys_clipped": [
                    21.0,
                    51.0,
                    -0.2,
                ],
                "upper_phys_clipped": [
                    29.0,
                    69.0,
                    0.4,
                ],
            }
        )

    def test_wide_conversion_matches_closed_layout(self):
        out = cnn_results_to_wide(
            self.point_df,
            self.interval_df,
            labels=LABELS,
            use_clipped=False,
        )

        self.assertEqual(
            list(out["event"]),
            ["GW1"],
        )

        self.assertEqual(
            out.loc[
                0,
                "chirp_mass_cnn",
            ],
            25.0,
        )

        self.assertEqual(
            out.loc[
                0,
                "total_mass_cnn_lower",
            ],
            50.0,
        )

        self.assertEqual(
            out.loc[
                0,
                "chi_eff_cnn_upper",
            ],
            0.4,
        )

    def test_clipped_columns_can_be_selected_explicitly(self):
        out = cnn_results_to_wide(
            self.point_df,
            self.interval_df,
            labels=LABELS,
            use_clipped=True,
        )

        self.assertEqual(
            out.loc[
                0,
                "chirp_mass_cnn_lower",
            ],
            21.0,
        )

        self.assertEqual(
            out.loc[
                0,
                "total_mass_cnn_upper",
            ],
            69.0,
        )


class TestLVKComparisonMetrics(unittest.TestCase):
    def make_comparison_df(self):
        return pd.DataFrame(
            {
                "chirp_mass_cnn": [26.0],
                "chirp_mass_cnn_lower": [20.0],
                "chirp_mass_cnn_upper": [32.0],
                "chirp_mass_lvk": [25.0],
                "chirp_mass_lvk_lower": [22.0],
                "chirp_mass_lvk_upper": [28.0],

                "total_mass_cnn": [62.0],
                "total_mass_cnn_lower": [50.0],
                "total_mass_cnn_upper": [70.0],
                "total_mass_lvk": [60.0],
                "total_mass_lvk_lower": [55.0],
                "total_mass_lvk_upper": [65.0],

                "chi_eff_cnn": [0.0],
                "chi_eff_cnn_lower": [-0.3],
                "chi_eff_cnn_upper": [0.2],
                "chi_eff_lvk": [0.1],
                "chi_eff_lvk_lower": [-0.1],
                "chi_eff_lvk_upper": [0.3],
            }
        )

    def test_point_delta_and_normalized_delta(self):
        out = add_lvk_comparison_metrics(
            self.make_comparison_df(),
            labels=LABELS,
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "chirp_mass_delta_cnn_minus_lvk",
            ],
            1.0,
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "chirp_mass_normalized_delta_lvk",
            ],
            1.0 / 3.0,
        )

    def test_interval_membership_flags(self):
        out = add_lvk_comparison_metrics(
            self.make_comparison_df(),
            labels=LABELS,
        )

        self.assertTrue(
            bool(
                out.loc[
                    0,
                    "chirp_mass_cnn_point_inside_lvk",
                ]
            )
        )

        self.assertTrue(
            bool(
                out.loc[
                    0,
                    "chirp_mass_lvk_median_inside_cnn",
                ]
            )
        )

    def test_interval_overlap_matches_closed_definition(self):
        out = add_lvk_comparison_metrics(
            self.make_comparison_df(),
            labels=LABELS,
        )

        # CNN [20, 32], LVK [22, 28]
        # overlap = 6, LVK width = 6.
        self.assertAlmostEqual(
            out.loc[
                0,
                "chirp_mass_interval_overlap",
            ],
            6.0,
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "chirp_mass_interval_overlap_fraction_lvk",
            ],
            1.0,
        )

    def test_non_overlapping_intervals_have_zero_overlap(self):
        df = self.make_comparison_df()

        df[
            "chirp_mass_cnn_lower"
        ] = 40.0

        df[
            "chirp_mass_cnn_upper"
        ] = 50.0

        out = add_lvk_comparison_metrics(
            df,
            labels=LABELS,
        )

        self.assertEqual(
            out.loc[
                0,
                "chirp_mass_interval_overlap",
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
