import unittest

import pandas as pd
import numpy as np

from src.real_data.event_runner import (
    RealEventRunContext,
    run_single_event,
    classify_failure_reason,
    events_df_to_event_configs,
)


class TestEventConfigBuilder(unittest.TestCase):
    def setUp(self):
        self.events_df = pd.DataFrame(
            {
                "event": [
                    "GWGOOD",
                    "GWMISSING",
                    "GWPARTIAL",
                ],
                "gps_time": [
                    1000.0,
                    2000.0,
                    3000.0,
                ],
            }
        )

        self.urls = {
            "GWGOOD": {
                "H1": "h1",
                "L1": "l1",
                "V1": "v1",
            },
            "GWPARTIAL": {
                "H1": "h1",
                "L1": "l1",
            },
        }

    def test_builds_only_complete_event_configs(self):
        configs, skipped = (
            events_df_to_event_configs(
                self.events_df,
                self.urls,
                detector_order=[
                    "H1",
                    "L1",
                    "V1",
                ],
            )
        )

        self.assertEqual(
            len(configs),
            1,
        )

        self.assertEqual(
            configs[0]["event"],
            "GWGOOD",
        )

        self.assertEqual(
            configs[0]["detectors"],
            ["H1", "L1", "V1"],
        )

        self.assertEqual(
            len(skipped),
            2,
        )

    def test_default_execution_policy_matches_closed_m10(self):
        configs, _ = (
            events_df_to_event_configs(
                self.events_df.iloc[[0]],
                self.urls,
                detector_order=[
                    "H1",
                    "L1",
                    "V1",
                ],
            )
        )

        cfg = configs[0]

        self.assertEqual(
            cfg["psd_window"],
            (-1024.0, -640.0),
        )

        self.assertEqual(
            cfg["center_offset"],
            0.0,
        )

        self.assertEqual(
            cfg["psd_segment_duration"],
            8.0,
        )

    def test_dead_fields_are_not_emitted(self):
        configs, _ = (
            events_df_to_event_configs(
                self.events_df.iloc[[0]],
                self.urls,
                detector_order=[
                    "H1",
                    "L1",
                    "V1",
                ],
            )
        )

        cfg = configs[0]

        for field in [
            "sample_rate",
            "duration",
            "adaptive_psd_window",
            "notes",
        ]:
            self.assertNotIn(
                field,
                cfg,
            )

    def test_missing_detector_is_reported(self):
        _, skipped = (
            events_df_to_event_configs(
                self.events_df.iloc[[2]],
                self.urls,
                detector_order=[
                    "H1",
                    "L1",
                    "V1",
                ],
            )
        )

        self.assertEqual(
            skipped.loc[
                0,
                "reason",
            ],
            "missing_required_detector_urls",
        )

        self.assertEqual(
            skipped.loc[
                0,
                "missing_detectors",
            ],
            "V1",
        )


class TestFailureClassification(unittest.TestCase):
    def test_known_failure_classes(self):
        cases = {
            (
                "event processing window "
                "contains NaN/Inf"
            ): "nonfinite_event_window",

            (
                "No valid finite PSD window "
                "found"
            ): "no_valid_psd_window",

            (
                "PSD contains non-finite values"
            ): "nonfinite_psd",

            (
                "window outside available data"
            ): "window_outside_available_data",

            (
                "truncated file"
            ): "corrupted_or_truncated_hdf5",

            (
                "GWTEST not found in "
                "GWOSC_URLS_ACTIVE"
            ): "missing_url",

            (
                "Downloaded file is not a "
                "valid GWOSC HDF5"
            ): "invalid_hdf5_download",
        }

        for message, expected in cases.items():
            with self.subTest(
                message=message
            ):
                self.assertEqual(
                    classify_failure_reason(
                        message
                    ),
                    expected,
                )

    def test_unknown_failure_is_other(self):
        self.assertEqual(
            classify_failure_reason(
                "something unexpected"
            ),
            "other",
        )


class TestRealEventRunContext(
    unittest.TestCase
):
    def test_context_can_be_constructed(self):
        context = RealEventRunContext(
            detector_order=[
                "H1",
                "L1",
                "V1",
            ],
            gwosc_urls={},
            gwosc_cache_dir="/tmp",
            final_duration=4.0,
            final_length=16384,
            processing_length=19712,
            context_start_samples=1664,
            context_end_samples=1664,
            sampling_frequency=4096.0,
            processing_delta_f=0.25,
            processing_flength=8193,
            psd_candidate_windows=(
                (-1024.0, -640.0),
                (-768.0, -384.0),
            ),
            processor=object(),
            input_normalization_eps=1e-6,
            model=object(),
            device="cpu",
            y_mean=np.zeros(
                3,
                dtype=np.float32,
            ),
            y_std=np.ones(
                3,
                dtype=np.float32,
            ),
            label_names=[
                "chirp_mass",
                "total_mass",
                "chi_eff",
            ],
            apply_intervals=lambda **kwargs: None,
        )

        self.assertEqual(
            list(
                context.detector_order
            ),
            ["H1", "L1", "V1"],
        )


    def test_run_rejects_wrong_detector_order_before_io(self):
        context = RealEventRunContext(
            detector_order=[
                "H1",
                "L1",
                "V1",
            ],
            gwosc_urls={},
            gwosc_cache_dir="/tmp",
            final_duration=4.0,
            final_length=16384,
            processing_length=19712,
            context_start_samples=1664,
            context_end_samples=1664,
            sampling_frequency=4096.0,
            processing_delta_f=0.25,
            processing_flength=8193,
            psd_candidate_windows=(),
            processor=object(),
            input_normalization_eps=1e-6,
            model=object(),
            device="cpu",
            y_mean=np.zeros(3),
            y_std=np.ones(3),
            label_names=[
                "chirp_mass",
                "total_mass",
                "chi_eff",
            ],
            apply_intervals=lambda **kwargs: None,
        )

        event_cfg = {
            "event": "GWTEST",
            "gps_time": 1000.0,
            "detectors": [
                "L1",
                "H1",
                "V1",
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "Detector order",
        ):
            run_single_event(
                event_cfg,
                context,
            )


if __name__ == "__main__":
    unittest.main()