import unittest

import numpy as np

from pycbc.psd import (
    interpolate,
    inverse_spectrum_truncation,
)
from pycbc.types import TimeSeries

from src.real_data.psd import (
    estimate_offsource_psd,
    psd_window_is_available_and_finite,
    select_valid_psd_window,
)


class TestRealDataPSD(unittest.TestCase):
    def setUp(self):
        self.fs = 256.0
        self.duration = 64.0

        rng = np.random.default_rng(123)

        data = rng.normal(
            size=int(
                self.fs * self.duration
            )
        )

        self.strain = TimeSeries(
            data,
            delta_t=1.0 / self.fs,
            epoch=1000.0,
        )

        self.event_time = 1050.0

        self.delta_f = 0.25

        # At fs=256 and delta_f=0.25:
        # Nyquist/delta_f + 1
        self.target_flength = int(
            self.fs / 2
            / self.delta_f
        ) + 1

    def legacy_reference(
        self,
        *,
        psd_start_offset,
        psd_end_offset,
        psd_segment_duration,
        low_frequency_cutoff,
        max_filter_duration,
    ):
        """
        Exact closed-notebook PSD implementation used as
        regression reference.
        """
        psd_start = (
            self.event_time
            + psd_start_offset
        )

        psd_end = (
            self.event_time
            + psd_end_offset
        )

        psd_data = (
            self.strain.time_slice(
                psd_start,
                psd_end,
            )
        )

        psd = psd_data.psd(
            psd_segment_duration
        )

        psd = interpolate(
            psd,
            self.delta_f,
        )

        max_filter_len = int(
            round(
                max_filter_duration
                * self.fs
            )
        )

        psd = (
            inverse_spectrum_truncation(
                psd,
                max_filter_len=max_filter_len,
                low_frequency_cutoff=(
                    low_frequency_cutoff
                ),
                trunc_method="hann",
            )
        )

        if len(psd) > self.target_flength:
            psd = psd[
                :self.target_flength
            ]

        elif len(psd) < self.target_flength:
            raise ValueError

        return psd

    def test_matches_closed_notebook_implementation(self):
        kwargs = dict(
            psd_start_offset=-40.0,
            psd_end_offset=-8.0,
            psd_segment_duration=4.0,
            low_frequency_cutoff=20.0,
            max_filter_duration=0.5,
        )

        reference = self.legacy_reference(
            **kwargs
        )

        result = estimate_offsource_psd(
            strain=self.strain,
            event_time=self.event_time,
            delta_f=self.delta_f,
            sampling_frequency=self.fs,
            target_flength=(
                self.target_flength
            ),
            **kwargs,
        )

        self.assertEqual(
            len(result),
            len(reference),
        )

        self.assertEqual(
            float(result.delta_f),
            float(reference.delta_f),
        )

        np.testing.assert_array_equal(
            result.numpy(),
            reference.numpy(),
        )

    def test_output_has_requested_length(self):
        result = estimate_offsource_psd(
            strain=self.strain,
            event_time=self.event_time,
            delta_f=self.delta_f,
            sampling_frequency=self.fs,
            target_flength=(
                self.target_flength
            ),
            psd_start_offset=-40.0,
            psd_end_offset=-8.0,
            psd_segment_duration=4.0,
            low_frequency_cutoff=20.0,
            max_filter_duration=0.5,
        )

        self.assertEqual(
            len(result),
            self.target_flength,
        )

    def test_output_is_finite(self):
        result = estimate_offsource_psd(
            strain=self.strain,
            event_time=self.event_time,
            delta_f=self.delta_f,
            sampling_frequency=self.fs,
            target_flength=(
                self.target_flength
            ),
            psd_start_offset=-40.0,
            psd_end_offset=-8.0,
            psd_segment_duration=4.0,
            low_frequency_cutoff=20.0,
            max_filter_duration=0.5,
        )

        self.assertTrue(
            np.all(
                np.isfinite(
                    result.numpy()
                )
            )
        )

    def test_rejects_window_outside_available_data(self):
        with self.assertRaisesRegex(
            ValueError,
            "outside available",
        ):
            estimate_offsource_psd(
                strain=self.strain,
                event_time=self.event_time,
                delta_f=self.delta_f,
                sampling_frequency=self.fs,
                target_flength=(
                    self.target_flength
                ),
                psd_start_offset=-100.0,
                psd_end_offset=-80.0,
                psd_segment_duration=4.0,
                low_frequency_cutoff=20.0,
                max_filter_duration=0.5,
            )

    def test_rejects_invalid_window_order(self):
        with self.assertRaisesRegex(
            ValueError,
            "greater",
        ):
            estimate_offsource_psd(
                strain=self.strain,
                event_time=self.event_time,
                delta_f=self.delta_f,
                sampling_frequency=self.fs,
                target_flength=(
                    self.target_flength
                ),
                psd_start_offset=-8.0,
                psd_end_offset=-40.0,
            )

class TestPSDWindowSelection(unittest.TestCase):
    def setUp(self):
        self.fs = 64.0
        self.duration = 100.0

        rng = np.random.default_rng(456)

        self.event_time = 1050.0

        self.raw_strains = {}

        for ifo in [
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

            self.raw_strains[ifo] = (
                TimeSeries(
                    data,
                    delta_t=1.0 / self.fs,
                    epoch=1000.0,
                )
            )

    def test_psd_window_available_for_all_detectors(self):
        valid = (
            psd_window_is_available_and_finite(
                raw_strains=self.raw_strains,
                event_time=self.event_time,
                psd_window=(-40.0, -20.0),
            )
        )

        self.assertTrue(valid)

    def test_psd_window_rejected_if_outside_data(self):
        valid = (
            psd_window_is_available_and_finite(
                raw_strains=self.raw_strains,
                event_time=self.event_time,
                psd_window=(-80.0, -60.0),
            )
        )

        self.assertFalse(valid)

    def test_selector_prefers_preferred_window(self):
        result = select_valid_psd_window(
            raw_strains=self.raw_strains,
            event_time=self.event_time,
            preferred_window=(-40.0, -20.0),
            candidate_windows=(
                (-30.0, -10.0),
            ),
        )

        self.assertEqual(
            result,
            (-40.0, -20.0),
        )

    def test_selector_falls_back_to_candidate(self):
        result = select_valid_psd_window(
            raw_strains=self.raw_strains,
            event_time=self.event_time,
            preferred_window=(-80.0, -60.0),
            candidate_windows=(
                (-40.0, -20.0),
                (-30.0, -10.0),
            ),
        )

        self.assertEqual(
            result,
            (-40.0, -20.0),
        )

    def test_selector_raises_if_no_window_is_valid(self):
        with self.assertRaisesRegex(
            ValueError,
            "No valid finite PSD window",
        ):
            select_valid_psd_window(
                raw_strains=self.raw_strains,
                event_time=self.event_time,
                preferred_window=(-90.0, -80.0),
                candidate_windows=(
                    (-80.0, -70.0),
                    (-70.0, -60.0),
                ),
            )

if __name__ == "__main__":
    unittest.main()