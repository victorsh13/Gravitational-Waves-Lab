import unittest

import numpy as np
from pycbc.types import TimeSeries

from src.config import SimulationConfig
from src.injection import SignalInjector
from src.windowing import ProjectedNetworkWindowSelector


def make_ts(values, *, delta_t, start_time):
    return TimeSeries(
        np.asarray(values, dtype=float),
        delta_t=delta_t,
        epoch=start_time,
    )


class TestProjectedNetworkWindowSelector(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(
            sampling_frequency=16.0,
            duration=4.0,
            required_final_duration=1.0,
            truncation_policy="keep_last_segment",
            target_network_snr_range=None,
        )
        self.selector = ProjectedNetworkWindowSelector(self.config)

    def test_short_network_is_not_truncated(self):
        n = int(3.0 * self.config.sampling_frequency)
        projected = {
            "H1": make_ts(np.ones(n), delta_t=self.config.delta_t, start_time=100.0),
            "L1": make_ts(2*np.ones(n), delta_t=self.config.delta_t, start_time=100.0),
            "V1": make_ts(3*np.ones(n), delta_t=self.config.delta_t, start_time=100.0),
        }
        result = self.selector.select(projected)
        self.assertFalse(result.metadata.is_truncated)
        self.assertEqual(result.metadata.detector_names, ["H1", "L1", "V1"])

    def test_long_projected_network_keeps_last_analysis_window(self):
        n = int(6.0 * self.config.sampling_frequency)
        values = np.arange(n, dtype=float)
        projected = {
            detector: make_ts(values, delta_t=self.config.delta_t, start_time=100.0)
            for detector in ("H1", "L1", "V1")
        }
        result = self.selector.select(projected)
        self.assertTrue(result.metadata.is_truncated)
        self.assertAlmostEqual(result.metadata.full_network_end_time, 106.0)
        self.assertAlmostEqual(result.metadata.used_window_start_time, 102.0)
        self.assertAlmostEqual(result.metadata.used_window_end_time, 106.0)
        for strain in result.strains.values():
            self.assertGreaterEqual(float(strain.start_time), 102.0 - self.config.delta_t)
            self.assertLessEqual(len(strain), self.config.length)
            self.assertGreater(len(strain), 0)

    def test_truncation_policy_none_rejects_long_network(self):
        config = SimulationConfig(
            sampling_frequency=16.0,
            duration=4.0,
            required_final_duration=1.0,
            truncation_policy="none",
            target_network_snr_range=None,
        )
        selector = ProjectedNetworkWindowSelector(config)
        n = int(5.0 * config.sampling_frequency)
        projected = {
            detector: make_ts(np.ones(n), delta_t=config.delta_t, start_time=10.0)
            for detector in ("H1", "L1", "V1")
        }
        with self.assertRaises(ValueError):
            selector.select(projected)


class TestSignalInjector(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(
            sampling_frequency=16.0,
            duration=4.0,
            required_final_duration=1.0,
            target_network_snr_range=None,
        )
        self.injector = SignalInjector(
            self.config,
            rng=np.random.default_rng(123),
        )

    def test_inject_uses_absolute_start_time(self):
        strain = self.injector.build_zero_strain(start_time=100.0)
        signal = make_ts(
            [1.0, 2.0, 3.0, 4.0],
            delta_t=self.config.delta_t,
            start_time=101.0,
        )
        result = self.injector.inject(strain, signal)
        expected = int(round((101.0 - 100.0) / self.config.delta_t))
        self.assertEqual(result.signal_start_index, expected)
        self.assertEqual(result.n_injected_samples, len(signal))
        self.assertFalse(result.is_partially_clipped)
        np.testing.assert_array_equal(
            result.strain[expected:expected + len(signal)],
            signal,
        )

    def test_partial_clipping_before_segment_is_recorded(self):
        strain = self.injector.build_zero_strain(start_time=100.0)
        signal = make_ts(
            np.ones(8),
            delta_t=self.config.delta_t,
            start_time=100.0 - 4 * self.config.delta_t,
        )
        result = self.injector.inject(strain, signal)
        self.assertTrue(result.is_partially_clipped)
        self.assertEqual(result.n_clipped_before, 4)
        self.assertEqual(result.n_clipped_after, 0)
        self.assertEqual(result.n_injected_samples, 4)

    def test_nonoverlapping_signal_is_rejected(self):
        strain = self.injector.build_zero_strain(start_time=100.0)
        signal = make_ts(
            np.ones(8),
            delta_t=self.config.delta_t,
            start_time=200.0,
        )
        with self.assertRaises(ValueError):
            self.injector.inject(strain, signal)

    def test_random_contained_placement_contains_network(self):
        n = int(2.0 * self.config.sampling_frequency)
        signals = {
            "H1": make_ts(np.ones(n), delta_t=self.config.delta_t, start_time=100.10),
            "L1": make_ts(np.ones(n), delta_t=self.config.delta_t, start_time=100.12),
            "V1": make_ts(np.ones(n), delta_t=self.config.delta_t, start_time=100.08),
        }
        placement = self.injector.choose_segment_placement_containing_network(
            signals,
            placement_policy="random_contained",
        )
        self.assertLessEqual(
            placement.segment_start_time,
            placement.earliest_signal_start_time,
        )
        self.assertGreaterEqual(
            placement.segment_end_time,
            placement.latest_signal_end_time,
        )
        self.assertTrue(placement.margins_respected)


if __name__ == "__main__":
    unittest.main()
