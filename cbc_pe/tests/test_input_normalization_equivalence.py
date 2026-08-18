import unittest

import numpy as np

from src.models.dataset import (
    normalize_input_per_sample_per_detector_zscore,
)
from src.models.hdf5_batch_dataset import (
    normalize_batch_per_sample_per_detector_zscore,
)


class TestInputNormalizationEquivalence(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)

        self.X = rng.normal(
            size=(8, 3, 1024),
        ).astype(np.float32)

        self.eps = 1e-6

    def test_batch_matches_sample_by_sample_normalization(self):
        batch_result = (
            normalize_batch_per_sample_per_detector_zscore(
                self.X.copy(),
                eps=self.eps,
            )
        )

        sample_result = np.stack(
            [
                normalize_input_per_sample_per_detector_zscore(
                    x.copy(),
                    eps=self.eps,
                )
                for x in self.X
            ],
            axis=0,
        )

        np.testing.assert_array_equal(
            batch_result,
            sample_result,
        )

    def test_matches_closed_m10_notebook_formula(self):
        notebook_result = self.X.astype(
            np.float32
        )

        mean = notebook_result.mean(
            axis=2,
            keepdims=True,
        )

        std = notebook_result.std(
            axis=2,
            keepdims=True,
        )

        notebook_result = (
            (
                notebook_result - mean
            )
            / (
                std + self.eps
            )
        ).astype(np.float32)

        production_result = (
            normalize_batch_per_sample_per_detector_zscore(
                self.X.copy(),
                eps=self.eps,
            )
        )

        np.testing.assert_array_equal(
            production_result,
            notebook_result,
        )

    def test_output_is_float32(self):
        result = (
            normalize_batch_per_sample_per_detector_zscore(
                self.X,
                eps=self.eps,
            )
        )

        self.assertEqual(
            result.dtype,
            np.float32,
        )

    def test_each_detector_is_approximately_standardized(self):
        result = (
            normalize_batch_per_sample_per_detector_zscore(
                self.X,
                eps=self.eps,
            )
        )

        means = result.mean(
            axis=2,
        )

        stds = result.std(
            axis=2,
        )

        np.testing.assert_allclose(
            means,
            0.0,
            atol=1e-6,
        )

        np.testing.assert_allclose(
            stds,
            1.0,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()