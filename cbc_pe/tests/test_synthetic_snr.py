import unittest

import numpy as np

from src.snr import (
    compute_network_snr,
    decide_distance_rescaling,
    rescale_distance_for_target_network_snr,
    validate_snr_rescaling,
)


class TestSyntheticSNRHelpers(unittest.TestCase):
    def test_network_snr_is_euclidean_norm(self):
        self.assertAlmostEqual(
            compute_network_snr({"H1": 3.0, "L1": 4.0, "V1": 12.0}),
            13.0,
        )

    def test_network_snr_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            compute_network_snr(np.array([3.0, -1.0, 4.0]))

    def test_distance_rescaling_formula(self):
        new_distance = rescale_distance_for_target_network_snr(
            current_distance=1000.0,
            current_network_snr=20.0,
            target_network_snr=10.0,
        )
        self.assertAlmostEqual(new_distance, 2000.0)

    def test_no_rescale_when_snr_already_in_target_range(self):
        decision = decide_distance_rescaling(
            current_distance=1000.0,
            current_network_snr=15.0,
            target_network_snr_range=(10.0, 25.0),
            rng=np.random.default_rng(123),
        )
        self.assertFalse(decision.should_rescale)
        self.assertEqual(decision.reason, "already_within_target_range")
        self.assertEqual(decision.target_network_snr, 15.0)
        self.assertEqual(decision.new_distance, 1000.0)

    def test_out_of_range_snr_produces_target_inside_requested_range(self):
        decision = decide_distance_rescaling(
            current_distance=1000.0,
            current_network_snr=5.0,
            target_network_snr_range=(10.0, 25.0),
            rng=np.random.default_rng(123),
        )
        self.assertTrue(decision.should_rescale)
        self.assertGreaterEqual(decision.target_network_snr, 10.0)
        self.assertLessEqual(decision.target_network_snr, 25.0)
        self.assertAlmostEqual(
            decision.new_distance,
            1000.0 * 5.0 / decision.target_network_snr,
        )

    def test_none_target_range_disables_rescaling(self):
        decision = decide_distance_rescaling(
            current_distance=1000.0,
            current_network_snr=7.0,
            target_network_snr_range=None,
            rng=np.random.default_rng(123),
        )
        self.assertFalse(decision.should_rescale)
        self.assertEqual(decision.reason, "target_network_snr_range_is_none")
        self.assertEqual(decision.new_distance, 1000.0)

    def test_validate_snr_rescaling_accepts_within_tolerance(self):
        validate_snr_rescaling(
            final_network_snr=20.5,
            target_network_snr=20.0,
            relative_tolerance=0.05,
        )

    def test_validate_snr_rescaling_rejects_outside_tolerance(self):
        with self.assertRaises(ValueError):
            validate_snr_rescaling(
                final_network_snr=22.0,
                target_network_snr=20.0,
                relative_tolerance=0.05,
            )


if __name__ == "__main__":
    unittest.main()
