import unittest

import numpy as np
import pandas as pd

from src.real_data.lvk_reference import (
    build_lvk_reference_detector_frame,
    propagate_detector_frame_mass_interval,
)


class TestDetectorFramePropagation(unittest.TestCase):
    def test_central_mass_conversion(self):
        best, lower, upper = (
            propagate_detector_frame_mass_interval(
                mass_source_best=30.0,
                mass_source_lower=28.0,
                mass_source_upper=33.0,
                redshift_best=0.2,
                redshift_lower=0.1,
                redshift_upper=0.3,
            )
        )

        self.assertAlmostEqual(
            float(best),
            36.0,
        )

        self.assertLess(
            float(lower),
            float(best),
        )

        self.assertGreater(
            float(upper),
            float(best),
        )

    def test_exact_asymmetric_error_propagation(self):
        M = 30.0
        M_low = 28.0
        M_high = 34.0

        z = 0.2
        z_low = 0.15
        z_high = 0.30

        best, lower, upper = (
            propagate_detector_frame_mass_interval(
                M,
                M_low,
                M_high,
                z,
                z_low,
                z_high,
            )
        )

        expected_best = (
            (1.0 + z) * M
        )

        expected_minus = np.sqrt(
            (
                (1.0 + z)
                * (M - M_low)
            ) ** 2
            +
            (
                M
                * (z - z_low)
            ) ** 2
        )

        expected_plus = np.sqrt(
            (
                (1.0 + z)
                * (M_high - M)
            ) ** 2
            +
            (
                M
                * (z_high - z)
            ) ** 2
        )

        self.assertAlmostEqual(
            float(best),
            expected_best,
        )

        self.assertAlmostEqual(
            float(lower),
            expected_best
            - expected_minus,
        )

        self.assertAlmostEqual(
            float(upper),
            expected_best
            + expected_plus,
        )

    def test_vectorized_inputs(self):
        best, lower, upper = (
            propagate_detector_frame_mass_interval(
                [20.0, 30.0],
                [18.0, 28.0],
                [22.0, 34.0],
                [0.1, 0.2],
                [0.05, 0.15],
                [0.15, 0.3],
            )
        )

        self.assertEqual(
            best.shape,
            (2,),
        )

        self.assertEqual(
            lower.shape,
            (2,),
        )

        self.assertEqual(
            upper.shape,
            (2,),
        )


class TestLVKReferenceTable(unittest.TestCase):
    def make_dataframe(
        self,
        include_total_mass=True,
    ):
        data = {
            "event": ["GWTEST"],
            "gps_time": [1234.0],
            "catalog": ["TEST"],
            "detectors": ["H1,L1,V1"],

            "redshift_best": [0.2],
            "redshift_lower": [0.1],
            "redshift_upper": [0.3],

            "chirp_mass_source_best": [25.0],
            "chirp_mass_source_lower": [23.0],
            "chirp_mass_source_upper": [28.0],

            "mass_1_source_best": [35.0],
            "mass_1_source_lower": [32.0],
            "mass_1_source_upper": [39.0],

            "mass_2_source_best": [25.0],
            "mass_2_source_lower": [22.0],
            "mass_2_source_upper": [28.0],

            "chi_eff_best": [0.1],
            "chi_eff_lower": [-0.1],
            "chi_eff_upper": [0.3],
        }

        if include_total_mass:
            data.update(
                {
                    "total_mass_source_best": [
                        60.0
                    ],
                    "total_mass_source_lower": [
                        54.0
                    ],
                    "total_mass_source_upper": [
                        67.0
                    ],
                }
            )

        return pd.DataFrame(
            data
        )

    def test_builds_detector_frame_reference(self):
        df = self.make_dataframe()

        out = (
            build_lvk_reference_detector_frame(
                df
            )
        )

        self.assertEqual(
            out.loc[
                0,
                "event",
            ],
            "GWTEST",
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "chirp_mass_lvk",
            ],
            30.0,
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "total_mass_lvk",
            ],
            72.0,
        )

    def test_chi_eff_is_not_redshifted(self):
        df = self.make_dataframe()

        out = (
            build_lvk_reference_detector_frame(
                df
            )
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "chi_eff_lvk",
            ],
            0.1,
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "chi_eff_lvk_lower",
            ],
            -0.1,
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "chi_eff_lvk_upper",
            ],
            0.3,
        )

    def test_total_mass_falls_back_to_component_masses(self):
        df = self.make_dataframe(
            include_total_mass=False
        )

        out = (
            build_lvk_reference_detector_frame(
                df
            )
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "total_mass_lvk",
            ],
            72.0,
        )

    def test_component_masses_are_converted(self):
        df = self.make_dataframe()

        out = (
            build_lvk_reference_detector_frame(
                df
            )
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "mass_1_lvk",
            ],
            42.0,
        )

        self.assertAlmostEqual(
            out.loc[
                0,
                "mass_2_lvk",
            ],
            30.0,
        )

    def test_missing_required_columns_raise(self):
        df = self.make_dataframe()

        df = df.drop(
            columns=[
                "redshift_best"
            ]
        )

        with self.assertRaisesRegex(
            KeyError,
            "Missing required",
        ):
            build_lvk_reference_detector_frame(
                df
            )


if __name__ == "__main__":
    unittest.main()
