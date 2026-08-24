import unittest

import numpy as np

from src.parameters import CBCParameters
from src.sampling import ParameterSampler, PriorConfig


class SequenceRNG:
    def __init__(self, values):
        self.values = iter(values)

    def uniform(self, low, high):
        value = float(next(self.values))
        if not (low <= value <= high):
            raise AssertionError(
                f"Stub value {value} outside requested interval [{low}, {high}]."
            )
        return value


class TestCBCParameters(unittest.TestCase):
    def test_reorders_masses_and_corresponding_spins(self):
        params = CBCParameters(
            mass_1=20.0, mass_2=40.0, distance=1000.0,
            inclination=1.0, ra=1.0, dec=0.2,
            spin_1z=0.1, spin_2z=-0.4,
            polarization_angle=0.5,
        )
        self.assertEqual(params.mass_1, 40.0)
        self.assertEqual(params.mass_2, 20.0)
        self.assertEqual(params.spin_1z, -0.4)
        self.assertEqual(params.spin_2z, 0.1)

    def test_derived_labels_match_closed_formulas(self):
        params = CBCParameters(
            mass_1=40.0, mass_2=20.0, distance=1000.0,
            inclination=1.0, ra=1.0, dec=0.2,
            spin_1z=0.3, spin_2z=-0.2,
            polarization_angle=0.5,
        )
        total = 60.0
        chirp = (40.0 * 20.0) ** (3.0 / 5.0) / total ** (1.0 / 5.0)
        chi_eff = (40.0 * 0.3 + 20.0 * (-0.2)) / total
        self.assertAlmostEqual(params.total_mass, total)
        self.assertAlmostEqual(params.chirp_mass, chirp)
        self.assertAlmostEqual(params.chi_eff, chi_eff)

    def test_with_distance_changes_only_distance(self):
        params = CBCParameters(
            mass_1=50.0, mass_2=30.0, distance=800.0,
            inclination=0.7, ra=2.0, dec=-0.3,
            spin_1z=0.25, spin_2z=-0.1,
            polarization_angle=1.2,
        )
        moved = params.with_distance(1600.0)
        self.assertEqual(moved.distance, 1600.0)
        self.assertEqual(moved.mass_1, params.mass_1)
        self.assertEqual(moved.mass_2, params.mass_2)
        self.assertEqual(moved.inclination, params.inclination)
        self.assertEqual(moved.ra, params.ra)
        self.assertEqual(moved.dec, params.dec)
        self.assertEqual(moved.spin_1z, params.spin_1z)
        self.assertEqual(moved.spin_2z, params.spin_2z)
        self.assertEqual(moved.polarization_angle, params.polarization_angle)

    def test_angles_are_wrapped_modulo_two_pi(self):
        params = CBCParameters(
            mass_1=30.0, mass_2=20.0, distance=1000.0,
            inclination=1.0, ra=2.0 * np.pi + 0.25, dec=0.0,
            spin_1z=0.0, spin_2z=0.0,
            polarization_angle=4.0 * np.pi + 0.5,
        )
        self.assertAlmostEqual(params.ra, 0.25)
        self.assertAlmostEqual(params.polarization_angle, 0.5)


class TestParameterSampler(unittest.TestCase):
    def test_bbh_prior_defaults(self):
        prior = PriorConfig.bbh()
        self.assertEqual(prior.regime, "BBH")
        self.assertEqual(prior.component_mass_range, (5.0, 90.0))
        self.assertEqual(prior.distance_range, (200.0, 5000.0))
        self.assertEqual(prior.spin_1z_range, (-1.0, 1.0))
        self.assertEqual(prior.spin_2z_range, (-1.0, 1.0))

    def test_fixed_parameters_override_sampled_values(self):
        prior = PriorConfig.bbh(
            fixed_parameters={
                "mass_1": 60.0,
                "mass_2": 40.0,
                "spin_1z": 0.0,
                "spin_2z": 0.0,
            }
        )
        sample = ParameterSampler(
            prior_config=prior,
            rng=np.random.default_rng(123),
        ).sample_one()
        self.assertEqual(sample.mass_1, 60.0)
        self.assertEqual(sample.mass_2, 40.0)
        self.assertEqual(sample.spin_1z, 0.0)
        self.assertEqual(sample.spin_2z, 0.0)

    def test_sampling_uses_isotropic_coordinate_transforms(self):
        rng = SequenceRNG([
            60.0, 40.0, 1000.0, 0.5, 1.3,
            -0.25, 0.2, -0.1, 2.1,
        ])
        sample = ParameterSampler(
            prior_config=PriorConfig.bbh(),
            rng=rng,
        ).sample_one()
        self.assertAlmostEqual(sample.inclination, np.arccos(0.5))
        self.assertAlmostEqual(sample.dec, np.arcsin(-0.25))
        self.assertAlmostEqual(sample.ra, 1.3)
        self.assertAlmostEqual(sample.polarization_angle, 2.1)

    def test_sample_many_rejects_negative_count(self):
        sampler = ParameterSampler(
            prior_config=PriorConfig.bbh(),
            rng=np.random.default_rng(123),
        )
        with self.assertRaises(ValueError):
            sampler.sample_many(-1)


if __name__ == "__main__":
    unittest.main()
