import unittest
import numpy as np

from src.config import SimulationConfig
from src.detectors import DetectorProjector
from src.parameters import CBCParameters
from src.waveform import WaveformGenerator


class TestWaveformAndProjectionSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SimulationConfig(
            sampling_frequency=4096.0,
            duration=4.0,
            low_frequency_cutoff=30.0,
            waveform_approximant="SEOBNRv4_opt",
            target_network_snr_range=None,
            required_final_duration=1.0,
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
        cls.waveform = WaveformGenerator(cls.config).generate(cls.params)
        cls.geocentric_time = 1126259462.0
        cls.projection = DetectorProjector(
            ["H1", "L1", "V1"]
        ).project(
            cls.waveform.h_plus,
            cls.waveform.h_cross,
            cls.params,
            cls.geocentric_time,
        )

    def test_generated_waveform_pair_contract(self):
        hp = self.waveform.h_plus
        hc = self.waveform.h_cross
        self.assertGreater(len(hp), 0)
        self.assertEqual(len(hp), len(hc))
        self.assertEqual(float(hp.delta_t), self.config.delta_t)
        self.assertEqual(float(hc.delta_t), self.config.delta_t)
        self.assertEqual(float(hp.start_time), float(hc.start_time))
        self.assertTrue(np.all(np.isfinite(hp.numpy())))
        self.assertTrue(np.all(np.isfinite(hc.numpy())))
        self.assertGreater(float(np.max(np.abs(hp.numpy()))), 0.0)

    def test_waveform_metadata_matches_timeseries(self):
        m = self.waveform.metadata
        hp = self.waveform.h_plus
        self.assertEqual(m.approximant, "SEOBNRv4_opt")
        self.assertEqual(m.low_frequency_cutoff, 30.0)
        self.assertEqual(m.n_samples, len(hp))
        self.assertAlmostEqual(m.duration, len(hp) * self.config.delta_t)
        self.assertAlmostEqual(m.start_time, float(hp.start_time))
        self.assertAlmostEqual(
            m.end_time,
            float(hp.start_time) + len(hp) * self.config.delta_t,
        )

    def test_projection_returns_closed_detector_set(self):
        self.assertEqual(
            list(self.projection.strains.keys()),
            ["H1", "L1", "V1"],
        )
        self.assertEqual(
            self.projection.metadata.detector_names,
            ["H1", "L1", "V1"],
        )

    def test_projected_strains_are_finite_and_sampled_correctly(self):
        for detector, strain in self.projection.strains.items():
            self.assertGreater(len(strain), 0, msg=detector)
            self.assertEqual(
                float(strain.delta_t),
                self.config.delta_t,
                msg=detector,
            )
            self.assertTrue(
                np.all(np.isfinite(strain.numpy())),
                msg=detector,
            )

    def test_projection_timing_metadata_is_self_consistent(self):
        meta = self.projection.metadata
        for detector, strain in self.projection.strains.items():
            delay = meta.expected_detector_time_delays[detector]
            self.assertAlmostEqual(
                meta.detector_arrival_times[detector],
                self.geocentric_time + delay,
                places=9,
            )
            self.assertAlmostEqual(
                meta.projected_start_times[detector],
                float(strain.start_time),
                places=9,
            )
            self.assertAlmostEqual(
                meta.projected_end_times[detector],
                float(strain.start_time)
                + len(strain) * float(strain.delta_t),
                places=9,
            )

    def test_projection_contains_detector_dependent_timing(self):
        delays = np.asarray(
            list(
                self.projection.metadata
                .expected_detector_time_delays.values()
            ),
            dtype=float,
        )
        self.assertTrue(np.all(np.isfinite(delays)))
        self.assertGreater(float(np.ptp(delays)), 0.0)


if __name__ == "__main__":
    unittest.main()
