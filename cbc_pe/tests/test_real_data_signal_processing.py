import unittest
from unittest.mock import patch

import numpy as np
from pycbc.types import TimeSeries

from src.real_data.signal_processing import (
    build_real_input_like_training,
    event_window_is_available_and_finite,
)


class IdentityNetworkProcessor:
    def process_network(
        self,
        strains,
        psds=None,
    ):
        return {
            detector: strain.copy()
            for detector, strain
            in strains.items()
        }


class TestRealSignalProcessing(unittest.TestCase):
    def setUp(self):
        self.fs = 64.0
        self.epoch = 1000.0
        self.duration = 20.0
        self.center_time = 1010.0

        rng = np.random.default_rng(123)

        self.raw_strains = {}

        for detector in [
            "H1",
            "L1",
            "V1",
        ]:
            data = rng.normal(
                size=int(
                    self.fs
                    * self.duration
                )
            )

            self.raw_strains[detector] = (
                TimeSeries(
                    data,
                    delta_t=1.0 / self.fs,
                    epoch=self.epoch,
                )
            )

        self.detectors = [
            "H1",
            "L1",
            "V1",
        ]

    def test_event_window_is_available(self):
        ok, reason = (
            event_window_is_available_and_finite(
                raw_strains=self.raw_strains,
                center_time=self.center_time,
                final_duration=4.0,
                context_start_samples=64,
                context_end_samples=64,
                sampling_frequency=self.fs,
            )
        )

        self.assertTrue(ok)
        self.assertEqual(
            reason,
            "ok",
        )

    def test_event_window_outside_data_is_rejected(self):
        ok, reason = (
            event_window_is_available_and_finite(
                raw_strains=self.raw_strains,
                center_time=1001.0,
                final_duration=4.0,
                context_start_samples=64,
                context_end_samples=64,
                sampling_frequency=self.fs,
            )
        )

        self.assertFalse(ok)

        self.assertIn(
            "outside available data",
            reason,
        )

    @patch(
        "src.real_data.signal_processing."
        "estimate_offsource_psd"
    )
    def test_builds_expected_network_input(
        self,
        mock_estimate_psd,
    ):
        mock_estimate_psd.return_value = (
            object()
        )

        final_duration = 4.0
        final_length = int(
            final_duration
            * self.fs
        )

        # Identity processor means the final length equals
        # the extracted processing length in this synthetic
        # orchestration test, so use zero context here.
        processing_length = final_length

        X, processed, psds, segments, meta = (
            build_real_input_like_training(
                raw_strains=self.raw_strains,
                detectors=self.detectors,
                center_time=self.center_time,
                processor=IdentityNetworkProcessor(),
                expected_detector_order=(
                    self.detectors
                ),
                final_duration=(
                    final_duration
                ),
                final_length=(
                    final_length
                ),
                processing_length=(
                    processing_length
                ),
                context_start_samples=0,
                context_end_samples=0,
                sampling_frequency=self.fs,
                psd_delta_f=0.25,
                psd_target_flength=129,
                psd_start_offset=-8.0,
                psd_end_offset=-4.0,
                psd_segment_duration=2.0,
            )
        )

        self.assertEqual(
            X.shape,
            (1, 3, final_length),
        )

        self.assertEqual(
            set(processed),
            set(self.detectors),
        )

        self.assertEqual(
            set(psds),
            set(self.detectors),
        )

        self.assertEqual(
            set(segments),
            set(self.detectors),
        )

        self.assertEqual(
            mock_estimate_psd.call_count,
            3,
        )

        self.assertAlmostEqual(
            meta["output_start"],
            self.center_time - 2.0,
        )

        self.assertAlmostEqual(
            meta["output_end"],
            self.center_time + 2.0,
        )

    @patch(
        "src.real_data.signal_processing."
        "estimate_offsource_psd"
    )
    def test_rejects_wrong_detector_order(
        self,
        mock_estimate_psd,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "Detector order mismatch",
        ):
            build_real_input_like_training(
                raw_strains=self.raw_strains,
                detectors=[
                    "L1",
                    "H1",
                    "V1",
                ],
                center_time=self.center_time,
                processor=IdentityNetworkProcessor(),
                expected_detector_order=(
                    self.detectors
                ),
                final_duration=4.0,
                final_length=256,
                processing_length=256,
                context_start_samples=0,
                context_end_samples=0,
                sampling_frequency=self.fs,
                psd_delta_f=0.25,
                psd_target_flength=129,
            )

    @patch(
        "src.real_data.signal_processing."
        "estimate_offsource_psd"
    )
    def test_rejects_missing_detector(
        self,
        mock_estimate_psd,
    ):
        incomplete = dict(
            self.raw_strains
        )

        del incomplete["V1"]

        with self.assertRaisesRegex(
            KeyError,
            "Missing detector strains",
        ):
            build_real_input_like_training(
                raw_strains=incomplete,
                detectors=self.detectors,
                center_time=self.center_time,
                processor=IdentityNetworkProcessor(),
                expected_detector_order=(
                    self.detectors
                ),
                final_duration=4.0,
                final_length=256,
                processing_length=256,
                context_start_samples=0,
                context_end_samples=0,
                sampling_frequency=self.fs,
                psd_delta_f=0.25,
                psd_target_flength=129,
            )


if __name__ == "__main__":
    unittest.main()