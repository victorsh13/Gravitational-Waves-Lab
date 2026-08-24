import unittest
import numpy as np

from src.config import SimulationConfig
from src.dataset import DatasetBuilder
from src.parameters import CBCParameters


class TestDatasetBuilderIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(
            sampling_frequency=4096.0,
            duration=4.0,
            low_frequency_cutoff=30.0,
            waveform_approximant="SEOBNRv4_opt",
            target_network_snr_range=None,
            required_final_duration=1.0,
            processing_context_start_samples=1664,
            processing_context_end_samples=1664,
        )
        cls.detectors = ["H1", "L1", "V1"]
        cls.builder = DatasetBuilder.from_config(
            config=cls.config,
            detector_names=cls.detectors,
            signal_processor_kwargs={
                "whitening_method": "none",
                "apply_highpass": False,
                "apply_lowpass": False,
                "apply_standardization": False,
                "output_mode": "crop_to_config",
                "remove_corrupted": False,
            },
            label_transformer_kwargs={},
            parameter_sampler_kwargs={
                "regime": "BBH",
                "fixed": {
                    "mass_1": 60.0,
                    "mass_2": 40.0,
                    "distance": 1000.0,
                    "inclination": 0.8,
                    "ra": 1.2,
                    "dec": -0.4,
                    "spin_1z": 0.15,
                    "spin_2z": -0.10,
                    "polarization_angle": 0.7,
                },
            },
            rng=np.random.default_rng(123),
        )
        cls.params = CBCParameters(
            mass_1=60.0,
            mass_2=40.0,
            distance=1000.0,
            inclination=0.8,
            ra=1.2,
            dec=-0.4,
            spin_1z=0.15,
            spin_2z=-0.10,
            polarization_angle=0.7,
        )
        cls.sample_gw_only = cls.builder.build_sample(
            params=cls.params,
            standardize_labels=False,
            geocentric_coalescence_time=1126259462.0,
            placement_policy="centered",
            strain_mode="gw_only",
        )

    def test_gw_only_sample_has_closed_m10_shape(self):
        s = self.sample_gw_only
        self.assertEqual(s.X.shape, (3, 16384))
        self.assertEqual(s.y.shape, (3,))
        self.assertTrue(np.all(np.isfinite(s.X)))
        self.assertTrue(np.all(np.isfinite(s.y)))

    def test_label_order_is_chirp_total_chi_eff(self):
        s = self.sample_gw_only
        expected = np.asarray(
            [
                s.parameters.chirp_mass,
                s.parameters.total_mass,
                s.parameters.chi_eff,
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            s.y,
            expected,
            rtol=0.0,
            atol=1e-12,
        )

    def test_detector_order_and_mode_are_recorded(self):
        m = self.sample_gw_only.metadata
        self.assertEqual(m["detectors"], ["H1", "L1", "V1"])
        self.assertEqual(m["strain_mode"], "gw_only")
        self.assertEqual(m["placement_policy"], "centered")

    def test_metadata_contains_closed_pipeline_sections(self):
        m = self.sample_gw_only.metadata
        required = {
            "simulation",
            "initial_parameters",
            "final_parameters",
            "geocentric_coalescence_time",
            "detectors",
            "strain_mode",
            "placement_policy",
            "waveform",
            "windowing",
            "projection",
            "placement",
            "snr",
            "injection",
            "noise",
            "processing",
            "processing_context",
            "labels",
        }
        self.assertTrue(
            required.issubset(m.keys()),
            msg=f"Missing metadata sections: {sorted(required - set(m.keys()))}",
        )

    def test_no_snr_rescaling_when_target_range_is_none(self):
        snr = self.sample_gw_only.metadata["snr"]
        self.assertFalse(snr["snr_rescaled"])
        self.assertEqual(
            snr["snr_rescaling_reason"],
            "target_network_snr_range_is_none",
        )
        self.assertAlmostEqual(snr["distance_before_rescale"], 1000.0)
        self.assertAlmostEqual(snr["distance_after_rescale"], 1000.0)

    def test_gw_only_output_contains_nonzero_signal(self):
        maxima = np.max(
            np.abs(self.sample_gw_only.X),
            axis=1,
        )
        self.assertTrue(
            np.all(maxima > 0.0),
            msg=f"Per-detector maxima: {maxima}",
        )

    def test_processing_context_metadata_matches_m10_context(self):
        meta = self.sample_gw_only.metadata["processing_context"]

        self.assertEqual(
            meta["output_length"],
            16384,
        )
        self.assertEqual(
            meta["processing_input_length"],
            19712,
        )
        self.assertEqual(
            meta["context_start_samples"],
            1664,
        )
        self.assertEqual(
            meta["context_end_samples"],
            1664,
        )

        self.assertAlmostEqual(
            meta["context_start_seconds"],
            1664 / 4096,
        )
        self.assertAlmostEqual(
            meta["context_end_seconds"],
            1664 / 4096,
        )

    def test_in_noise_mode_uses_same_output_contract(self):
        sample = self.builder.build_sample(
            params=self.params,
            standardize_labels=False,
            geocentric_coalescence_time=1126259462.0,
            placement_policy="centered",
            strain_mode="in_noise",
        )
        self.assertEqual(sample.X.shape, (3, 16384))
        self.assertEqual(sample.y.shape, (3,))
        self.assertTrue(np.all(np.isfinite(sample.X)))
        self.assertEqual(sample.metadata["strain_mode"], "in_noise")


if __name__ == "__main__":
    unittest.main()
