import numpy as np 
from .parameters import CBCParameters

class ParameterSampler:
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_one(self) -> CBCParameters:
        """
        This implementatios uses isotropic priors for inclinacion and dec and uniform priors for the rest. This is
        a difference to the legacy implementation which used non isotopric priors for these parameters.

        Returns
        -------
        CBCParameters
            A random sample from the priors.
        """
        mass_1 = self.rng.uniform(5.0, 90.0) # In Msun
        mass_2 = self.rng.uniform(5.0, 90.0) # in Msun
        distance = self.rng.uniform(200.0, 5000.0) # in Mpc
        inclination = np.arccos(self.rng.uniform(-1.0, 1.0))  # in [0, pi], angle between line of sight and the orbital plane of the binary system
        ra = self.rng.uniform(0.0, 2.0 * np.pi) # in [0, 2*pi], like longitude
        dec = np.arcsin(self.rng.uniform(-1.0, 1.0)) # in [-pi/2, pi/2], similar to latitude
        spin_1z = self.rng.uniform(-1.0, 1.0)
        spin_2z = self.rng.uniform(-1.0, 1.0)
        polarization_angle = self.rng.uniform(0.0, 2.0*np.pi)

        return CBCParameters(
            mass_1=mass_1,
            mass_2=mass_2,
            distance=distance,
            inclination=inclination,
            ra=ra,
            dec=dec,
            spin_1z=spin_1z,
            spin_2z=spin_2z,
            polarization_angle=polarization_angle,
        )

    def sample_many(self, n_samples: int) -> list[CBCParameters]:
        return [self.sample_one() for _ in range(n_samples)]