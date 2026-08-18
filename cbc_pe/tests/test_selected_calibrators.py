import unittest

import numpy as np
import pandas as pd

from src.conformal.selected_calibrators import (
    apply_selected_calibrators,
    fit_selected_calibrators,
)


class TestSelectedMondrianCalibrators(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(321)

        self.label_names = [
            "chirp_mass",
            "total_mass",
            "chi_eff",
        ]

        self.n_cal = 1200
        self.n_target = 5
        self.n_labels = len(self.label_names)

        self.pred_cal = rng.normal(
            size=(self.n_cal, self.n_labels)
        )

        scale_cal = (
            0.15
            + 0.10 * np.abs(self.pred_cal)
        )

        self.y_cal = (
            self.pred_cal
            + rng.normal(
                size=self.pred_cal.shape
            ) * scale_cal
        )

        self.pred_target = rng.normal(
            size=(self.n_target, self.n_labels)
        )

        embedding_dim = 8

        projection = rng.normal(
            size=(self.n_labels, embedding_dim)
        )

        self.emb_cal = (
            self.pred_cal @ projection
            + 0.2 * rng.normal(
                size=(self.n_cal, embedding_dim)
            )
        )

        self.emb_target = (
            self.pred_target @ projection
            + 0.2 * rng.normal(
                size=(self.n_target, embedding_dim)
            )
        )

        # Small synthetic analogue of the closed M10 selection table.
        self.selection_df = pd.DataFrame(
            [
                {
                    "final_policy": "conservative",
                    "label": "chirp_mass",
                    "label_index": 0,
                    "selection_policy": (
                        "conservative_zero_under_bins_2sigma"
                    ),
                    "taxonomy_mode": "difficulty",
                    "interval_mode": "asymmetric",
                    "n_bins": 4,
                },
                {
                    "final_policy": "conservative",
                    "label": "total_mass",
                    "label_index": 1,
                    "selection_policy": (
                        "conservative_zero_under_bins_2sigma"
                    ),
                    "taxonomy_mode": "prediction",
                    "interval_mode": "asymmetric",
                    "n_bins": 6,
                },
                {
                    "final_policy": "conservative",
                    "label": "chi_eff",
                    "label_index": 2,
                    "selection_policy": (
                        "conservative_zero_under_bins_2sigma"
                    ),
                    "taxonomy_mode": "prediction",
                    "interval_mode": "symmetric",
                    "n_bins": 8,
                },
                {
                    "final_policy": "efficient",
                    "label": "chirp_mass",
                    "label_index": 0,
                    "selection_policy": (
                        "efficient_local_validity_tolerant"
                    ),
                    "taxonomy_mode": "difficulty",
                    "interval_mode": "symmetric",
                    "n_bins": 4,
                },
                {
                    "final_policy": "efficient",
                    "label": "total_mass",
                    "label_index": 1,
                    "selection_policy": (
                        "efficient_local_validity_tolerant"
                    ),
                    "taxonomy_mode": "difficulty",
                    "interval_mode": "symmetric",
                    "n_bins": 4,
                },
                {
                    "final_policy": "efficient",
                    "label": "chi_eff",
                    "label_index": 2,
                    "selection_policy": (
                        "efficient_local_validity_tolerant"
                    ),
                    "taxonomy_mode": "difficulty",
                    "interval_mode": "asymmetric",
                    "n_bins": 12,
                },
            ]
        )

    def fit_calibrators(self):
        return fit_selected_calibrators(
            selection_df=self.selection_df,
            pred_cal=self.pred_cal,
            y_cal=self.y_cal,
            label_names=self.label_names,
            emb_cal=self.emb_cal,
            confidence_level=0.90,
            n_neighbors=5,
            min_samples_per_bin=10,
            apply_jitter=False,
        )

    def test_fits_one_calibrator_per_selected_row(self):
        calibrators = self.fit_calibrators()

        self.assertEqual(
            len(calibrators),
            6,
        )

        expected_keys = {
            ("conservative", "chirp_mass"),
            ("conservative", "total_mass"),
            ("conservative", "chi_eff"),
            ("efficient", "chirp_mass"),
            ("efficient", "total_mass"),
            ("efficient", "chi_eff"),
        }

        self.assertEqual(
            set(calibrators.keys()),
            expected_keys,
        )

        chirp_conservative = calibrators[
            ("conservative", "chirp_mass")
        ]

        self.assertEqual(
            chirp_conservative.taxonomy_mode,
            "difficulty",
        )

        self.assertEqual(
            chirp_conservative.interval_mode,
            "asymmetric",
        )

        self.assertEqual(
            chirp_conservative.n_bins,
            4,
        )

        self.assertEqual(
            chirp_conservative.label_index,
            0,
        )

    def test_applies_all_selected_calibrators_without_truth(self):
        calibrators = self.fit_calibrators()

        result = apply_selected_calibrators(
            calibrators=calibrators,
            pred_target=self.pred_target,
            label_names=self.label_names,
            target_embedding=self.emb_target,
            event_name="GW_TEST",
        )

        # 5 target samples x 6 selected calibrators.
        self.assertEqual(
            len(result),
            self.n_target * 6,
        )

        self.assertEqual(
            set(result["event"].unique()),
            {"GW_TEST"},
        )

        self.assertEqual(
            set(result["final_policy"].unique()),
            {"conservative", "efficient"},
        )

        self.assertEqual(
            set(result["label"].unique()),
            set(self.label_names),
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    result["lower_std"].to_numpy()
                )
            )
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    result["upper_std"].to_numpy()
                )
            )
        )

        self.assertTrue(
            np.all(
                result["lower_std"].to_numpy()
                <= result["upper_std"].to_numpy()
            )
        )

        np.testing.assert_allclose(
            result["width_std"].to_numpy(),
            (
                result["upper_std"]
                - result["lower_std"]
            ).to_numpy(),
            rtol=0.0,
            atol=0.0,
        )

    def test_selected_prediction_matches_underlying_fitted_calibrator(self):
        calibrators = self.fit_calibrators()

        result = apply_selected_calibrators(
            calibrators=calibrators,
            pred_target=self.pred_target,
            label_names=self.label_names,
            target_embedding=self.emb_target,
        )

        selected = calibrators[
            ("conservative", "total_mass")
        ]

        from src.conformal.pipeline import apply_mondrian

        direct = apply_mondrian(
            fitted=selected.fitted,
            pred_target=self.pred_target,
        )

        table = result[
            (result["final_policy"] == "conservative")
            & (result["label"] == "total_mass")
        ].sort_values("sample_index")

        j = 1

        np.testing.assert_allclose(
            table["pred_std"].to_numpy(),
            self.pred_target[:, j],
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_allclose(
            table["lower_std"].to_numpy(),
            direct.lower[:, j],
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_allclose(
            table["upper_std"].to_numpy(),
            direct.upper[:, j],
            rtol=0.0,
            atol=0.0,
        )

    def test_rejects_inconsistent_label_index_mapping(self):
        bad = self.selection_df.copy()

        bad.loc[
            bad["label"] == "chirp_mass",
            "label_index",
        ] = 1

        with self.assertRaisesRegex(
            ValueError,
            "Inconsistent label mapping",
        ):
            fit_selected_calibrators(
                selection_df=bad,
                pred_cal=self.pred_cal,
                y_cal=self.y_cal,
                label_names=self.label_names,
                emb_cal=self.emb_cal,
                apply_jitter=False,
            )

    def test_rejects_duplicate_policy_label_entries(self):
        duplicate = pd.concat(
            [
                self.selection_df,
                self.selection_df.iloc[[0]],
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate",
        ):
            fit_selected_calibrators(
                selection_df=duplicate,
                pred_cal=self.pred_cal,
                y_cal=self.y_cal,
                label_names=self.label_names,
                emb_cal=self.emb_cal,
                apply_jitter=False,
            )

    def test_requires_calibration_embeddings_for_difficulty(self):
        with self.assertRaisesRegex(
            ValueError,
            "emb_cal is required",
        ):
            fit_selected_calibrators(
                selection_df=self.selection_df,
                pred_cal=self.pred_cal,
                y_cal=self.y_cal,
                label_names=self.label_names,
                emb_cal=None,
                apply_jitter=False,
            )

    def test_requires_target_embeddings_for_difficulty(self):
        calibrators = self.fit_calibrators()

        with self.assertRaisesRegex(
            ValueError,
            "target_embedding is required",
        ):
            apply_selected_calibrators(
                calibrators=calibrators,
                pred_target=self.pred_target,
                label_names=self.label_names,
                target_embedding=None,
            )


if __name__ == "__main__":
    unittest.main()
