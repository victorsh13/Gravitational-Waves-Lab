import unittest

import numpy as np

from src.conformal.apply import apply_indices
from src.conformal.binning import BinGrouper, QuantileBinner
from src.conformal.calibration import ConformalIntervalCalibrator
from src.conformal.metrics import CoverageEvaluator
from src.conformal.pipeline import (
    apply_mondrian,
    evaluate_mondrian,
    fit_mondrian,
    run_mondrian_regression,
)
from src.conformal.taxonomy import compute_binning_scores


class TestMondrianPipelineEquivalence(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)

        self.n_cal = 1200
        self.n_test = 400
        self.n_labels = 3
        self.n_bins = 4

        # Synthetic standardized-space predictions.
        self.pred_cal = rng.normal(
            0.0,
            1.0,
            size=(self.n_cal, self.n_labels),
        )
        self.pred_test = rng.normal(
            0.0,
            1.0,
            size=(self.n_test, self.n_labels),
        )

        # Heteroscedastic residual structure so that the Mondrian
        # calibration is non-trivial.
        scale_cal = 0.15 + 0.10 * np.abs(self.pred_cal)
        scale_test = 0.15 + 0.10 * np.abs(self.pred_test)

        self.y_cal = (
            self.pred_cal
            + rng.normal(size=self.pred_cal.shape) * scale_cal
        )
        self.y_test = (
            self.pred_test
            + rng.normal(size=self.pred_test.shape) * scale_test
        )

        # Synthetic embeddings for difficulty-based Mondrian taxonomy.
        # They are deliberately correlated with the predictions so that
        # nearest-neighbour structure is non-trivial.
        embedding_dim = 8

        projection = rng.normal(
            size=(self.n_labels, embedding_dim),
        )

        self.emb_cal = (
            self.pred_cal @ projection
            + 0.2 * rng.normal(
                size=(self.n_cal, embedding_dim)
            )
        )

        self.emb_test = (
            self.pred_test @ projection
            + 0.2 * rng.normal(
                size=(self.n_test, embedding_dim)
            )
        )

        self.n_neighbors = 5

        self.confidence_level = 0.90
        self.min_samples_per_bin = 10

    def test_prediction_pipeline_matches_manual_components(self):
        pipeline_result = run_mondrian_regression(
            pred_cal=self.pred_cal,
            pred_test=self.pred_test,
            y_cal=self.y_cal,
            y_test=self.y_test,
            n_bins=self.n_bins,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="symmetric",
            taxonomy_mode="prediction",
            min_samples_per_bin=self.min_samples_per_bin,
            tolerance_sigmas=(1, 2, 3),
        )

        residuals_cal = self.y_cal - self.pred_cal

        (
            scores_cal,
            scores_test,
            difficulty_model,
        ) = compute_binning_scores(
            taxonomy_mode="prediction",
            pred_cal=self.pred_cal,
            pred_test=self.pred_test,
            cal_residuals=residuals_cal,
        )

        self.assertIsNone(difficulty_model)

        binner = QuantileBinner(
            n_bins=self.n_bins,
            apply_jitter=False,
        )

        bin_indices_cal = binner.bin_edges_and_indices(
            scores_cal
        )
        bin_indices_test = binner.get_bin_indices(
            scores_test
        )

        grouper = BinGrouper()

        grouped_residuals = grouper.group_by_bin(
            residuals=residuals_cal,
            bin_indices=bin_indices_cal,
            n_bins=self.n_bins,
        )

        calibrator = ConformalIntervalCalibrator(
            confidence_level=self.confidence_level,
            interval_mode="symmetric",
            min_samples_per_bin=self.min_samples_per_bin,
        )

        calibrator.fit(grouped_residuals)

        lower, upper = apply_indices(
            values=self.pred_test,
            bin_indices=bin_indices_test,
            intervals=calibrator.intervals_,
        )

        evaluator = CoverageEvaluator(
            confidence_level=self.confidence_level,
            tolerance_sigmas=(1, 2, 3),
        )

        manual_metrics = evaluator.evaluate_intervals(
            y=self.y_test,
            lower_bound=lower,
            upper_bound=upper,
            bin_indices=bin_indices_test,
            n_bins=self.n_bins,
        )

        np.testing.assert_array_equal(
            pipeline_result.bin_indices_cal,
            bin_indices_cal,
        )

        np.testing.assert_array_equal(
            pipeline_result.bin_indices_test,
            bin_indices_test,
        )

        np.testing.assert_allclose(
            pipeline_result.intervals,
            calibrator.intervals_,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_allclose(
            pipeline_result.lower,
            lower,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_allclose(
            pipeline_result.upper,
            upper,
            rtol=0.0,
            atol=0.0,
        )

        for key in [
            "global_coverage",
            "global_median_width",
            "global_mean_width",
            "global_tail_miss_imbalance",
            "coverage_per_bin",
            "median_width_per_bin",
            "counts_per_bin",
        ]:
            np.testing.assert_allclose(
                pipeline_result.metrics[key],
                manual_metrics[key],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )

    def test_asymmetric_intervals_are_ordered(self):
        result = run_mondrian_regression(
            pred_cal=self.pred_cal,
            pred_test=self.pred_test,
            y_cal=self.y_cal,
            y_test=self.y_test,
            n_bins=self.n_bins,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="asymmetric",
            taxonomy_mode="prediction",
            min_samples_per_bin=self.min_samples_per_bin,
        )

        self.assertTrue(
            np.all(result.lower <= result.upper)
        )

        self.assertTrue(
            np.all(np.isfinite(result.lower))
        )

        self.assertTrue(
            np.all(np.isfinite(result.upper))
        )

    def test_pipeline_does_not_modify_input_arrays(self):
        pred_cal_before = self.pred_cal.copy()
        pred_test_before = self.pred_test.copy()
        y_cal_before = self.y_cal.copy()
        y_test_before = self.y_test.copy()

        run_mondrian_regression(
            pred_cal=self.pred_cal,
            pred_test=self.pred_test,
            y_cal=self.y_cal,
            y_test=self.y_test,
            n_bins=self.n_bins,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="symmetric",
            taxonomy_mode="prediction",
            min_samples_per_bin=self.min_samples_per_bin,
        )

        np.testing.assert_array_equal(
            self.pred_cal,
            pred_cal_before,
        )
        np.testing.assert_array_equal(
            self.pred_test,
            pred_test_before,
        )
        np.testing.assert_array_equal(
            self.y_cal,
            y_cal_before,
        )
        np.testing.assert_array_equal(
            self.y_test,
            y_test_before,
        )


    def test_fit_apply_evaluate_matches_legacy_prediction_pipeline(self):
        legacy = run_mondrian_regression(
            pred_cal=self.pred_cal,
            pred_test=self.pred_test,
            y_cal=self.y_cal,
            y_test=self.y_test,
            n_bins=self.n_bins,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="symmetric",
            taxonomy_mode="prediction",
            min_samples_per_bin=self.min_samples_per_bin,
            tolerance_sigmas=(1, 2, 3),
        )

        fitted = fit_mondrian(
            pred_cal=self.pred_cal,
            y_cal=self.y_cal,
            n_bins=self.n_bins,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="symmetric",
            taxonomy_mode="prediction",
            min_samples_per_bin=self.min_samples_per_bin,
        )

        prediction = apply_mondrian(
            fitted=fitted,
            pred_target=self.pred_test,
        )

        metrics = evaluate_mondrian(
            fitted=fitted,
            prediction=prediction,
            y_true=self.y_test,
            tolerance_sigmas=(1, 2, 3),
        )

        np.testing.assert_array_equal(
            fitted.bin_indices_cal,
            legacy.bin_indices_cal,
        )

        np.testing.assert_allclose(
            fitted.calibrator.intervals_,
            legacy.intervals,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_array_equal(
            prediction.bin_indices,
            legacy.bin_indices_test,
        )

        np.testing.assert_allclose(
            prediction.lower,
            legacy.lower,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_allclose(
            prediction.upper,
            legacy.upper,
            rtol=0.0,
            atol=0.0,
        )

        for key in [
            "global_coverage",
            "global_median_width",
            "global_mean_width",
            "global_tail_miss_imbalance",
            "coverage_per_bin",
            "median_width_per_bin",
            "counts_per_bin",
        ]:
            np.testing.assert_allclose(
                metrics[key],
                legacy.metrics[key],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )


    def test_fit_apply_evaluate_matches_legacy_difficulty_pipeline(self):
        legacy = run_mondrian_regression(
            pred_cal=self.pred_cal,
            pred_test=self.pred_test,
            y_cal=self.y_cal,
            y_test=self.y_test,
            n_bins=self.n_bins,
            cal_embedding=self.emb_cal,
            target_embedding=self.emb_test,
            n_neighbors=self.n_neighbors,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="asymmetric",
            taxonomy_mode="difficulty",
            min_samples_per_bin=self.min_samples_per_bin,
            tolerance_sigmas=(1, 2, 3),
        )

        fitted = fit_mondrian(
            pred_cal=self.pred_cal,
            y_cal=self.y_cal,
            n_bins=self.n_bins,
            cal_embedding=self.emb_cal,
            n_neighbors=self.n_neighbors,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="asymmetric",
            taxonomy_mode="difficulty",
            min_samples_per_bin=self.min_samples_per_bin,
        )

        prediction = apply_mondrian(
            fitted=fitted,
            pred_target=self.pred_test,
            target_embedding=self.emb_test,
        )

        metrics = evaluate_mondrian(
            fitted=fitted,
            prediction=prediction,
            y_true=self.y_test,
            tolerance_sigmas=(1, 2, 3),
        )

        self.assertIsNotNone(
            fitted.difficulty_model
        )

        np.testing.assert_allclose(
            fitted.binning_scores_cal,
            legacy.binning_scores_cal,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_array_equal(
            fitted.bin_indices_cal,
            legacy.bin_indices_cal,
        )

        np.testing.assert_allclose(
            fitted.calibrator.intervals_,
            legacy.intervals,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_allclose(
            prediction.binning_scores,
            legacy.binning_scores_test,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_array_equal(
            prediction.bin_indices,
            legacy.bin_indices_test,
        )

        np.testing.assert_allclose(
            prediction.lower,
            legacy.lower,
            rtol=0.0,
            atol=0.0,
        )

        np.testing.assert_allclose(
            prediction.upper,
            legacy.upper,
            rtol=0.0,
            atol=0.0,
        )

        for key in [
            "global_coverage",
            "global_median_width",
            "global_mean_width",
            "global_tail_miss_imbalance",
            "coverage_per_bin",
            "median_width_per_bin",
            "counts_per_bin",
        ]:
            np.testing.assert_allclose(
                metrics[key],
                legacy.metrics[key],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )


    def test_apply_mondrian_does_not_require_target_truth(self):
        fitted = fit_mondrian(
            pred_cal=self.pred_cal,
            y_cal=self.y_cal,
            n_bins=self.n_bins,
            cal_embedding=self.emb_cal,
            n_neighbors=self.n_neighbors,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="symmetric",
            taxonomy_mode="difficulty",
            min_samples_per_bin=self.min_samples_per_bin,
        )

        # Pretend these are predictions/embeddings from real GW events.
        pred_real = self.pred_test[:3]
        emb_real = self.emb_test[:3]

        prediction = apply_mondrian(
            fitted=fitted,
            pred_target=pred_real,
            target_embedding=emb_real,
        )

        self.assertEqual(
            prediction.lower.shape,
            (3, self.n_labels),
        )

        self.assertEqual(
            prediction.upper.shape,
            (3, self.n_labels),
        )

        self.assertEqual(
            prediction.bin_indices.shape,
            (3, self.n_labels),
        )

        self.assertTrue(
            np.all(np.isfinite(prediction.lower))
        )

        self.assertTrue(
            np.all(np.isfinite(prediction.upper))
        )

        self.assertTrue(
            np.all(prediction.lower <= prediction.upper)
        )


    def test_difficulty_apply_requires_target_embedding(self):
        fitted = fit_mondrian(
            pred_cal=self.pred_cal,
            y_cal=self.y_cal,
            n_bins=self.n_bins,
            cal_embedding=self.emb_cal,
            n_neighbors=self.n_neighbors,
            confidence_level=self.confidence_level,
            apply_jitter=False,
            interval_mode="symmetric",
            taxonomy_mode="difficulty",
            min_samples_per_bin=self.min_samples_per_bin,
        )

        with self.assertRaisesRegex(
            ValueError,
            "target_embedding is required",
        ):
            apply_mondrian(
                fitted=fitted,
                pred_target=self.pred_test,
            )


if __name__ == "__main__":
    unittest.main()
