import unittest

import numpy as np
from pycbc.types import TimeSeries

from src.config import SimulationConfig
from src.processing import SignalProcessor


class TestSyntheticProcessingContract(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(
            sampling_frequency=2048.0,
            duration=4.0,
            required_final_duration=1.0,
            processing_context_start_samples=8,
            processing_context_end_samples=8,
            target_network_snr_range=None,
        )

    def make_processing_series(self, start_time=100.0):
        return TimeSeries(
            np.arange(self.config.processing_length, dtype=float),
            delta_t=self.config.delta_t,
            epoch=start_time,
        )

    def test_identity_processing_crops_exact_final_window(self):
        processor = SignalProcessor(
            config=self.config,
            whitening_method="none",
            apply_highpass=False,
            apply_lowpass=False,
            apply_standardization=False,
            output_mode="crop_to_config",
            remove_corrupted=False,
        )
        strain = self.make_processing_series(start_time=100.0)
        result = processor.process(strain)

        self.assertEqual(len(result), self.config.length)
        self.assertAlmostEqual(
            float(result.start_time),
            100.0 + self.config.processing_context_start_seconds,
        )

        expected = strain[
            self.config.processing_context_start_samples:
            self.config.processing_context_start_samples + self.config.length
        ]
        np.testing.assert_array_equal(result.numpy(), expected.numpy())

    def test_crop_to_config_requires_processing_length_input(self):
        processor = SignalProcessor(
            config=self.config,
            whitening_method="none",
            apply_highpass=False,
            apply_lowpass=False,
            output_mode="crop_to_config",
            remove_corrupted=False,
        )
        too_short = TimeSeries(
            np.zeros(self.config.length),
            delta_t=self.config.delta_t,
            epoch=100.0,
        )
        with self.assertRaises(ValueError):
            processor.process(too_short)

    def test_standardization_produces_zero_mean_unit_std(self):
        config = SimulationConfig(
            sampling_frequency=2048.0,
            duration=4.0,
            required_final_duration=1.0,
            target_network_snr_range=None,
        )
        processor = SignalProcessor(
            config=config,
            whitening_method="none",
            apply_highpass=False,
            apply_lowpass=False,
            apply_standardization=True,
            output_mode="restore_length",
            remove_corrupted=False,
        )
        strain = TimeSeries(
            np.linspace(-3.0, 7.0, config.length),
            delta_t=config.delta_t,
            epoch=50.0,
        )
        result = processor.process(strain)
        self.assertAlmostEqual(float(np.mean(result.numpy())), 0.0, places=12)
        self.assertAlmostEqual(float(np.std(result.numpy())), 1.0, places=12)

    def test_standardization_of_constant_series_returns_zeros(self):
        config = SimulationConfig(
            sampling_frequency=2048.0,
            duration=4.0,
            required_final_duration=1.0,
            target_network_snr_range=None,
        )
        processor = SignalProcessor(
            config=config,
            whitening_method="none",
            apply_highpass=False,
            apply_lowpass=False,
            apply_standardization=True,
            output_mode="restore_length",
            remove_corrupted=False,
        )
        strain = TimeSeries(
            np.ones(config.length),
            delta_t=config.delta_t,
            epoch=10.0,
        )
        result = processor.process(strain)
        np.testing.assert_array_equal(result.numpy(), np.zeros(config.length))

    def test_metadata_records_processing_context_contract(self):
        processor = SignalProcessor(
            config=self.config,
            whitening_method="none",
            apply_highpass=False,
            apply_lowpass=False,
            output_mode="crop_to_config",
            remove_corrupted=False,
        )
        metadata = processor.metadata()
        self.assertTrue(metadata["uses_processing_context"])
        self.assertEqual(metadata["output_length"], self.config.length)
        self.assertEqual(
            metadata["processing_input_length"],
            self.config.processing_length,
        )
        self.assertEqual(metadata["processing_context_start_samples"], 8)
        self.assertEqual(metadata["processing_context_end_samples"], 8)


if __name__ == "__main__":
    unittest.main()
