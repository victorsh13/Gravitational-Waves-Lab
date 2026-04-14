from dataclasses import dataclass
import numpy as np

@dataclass
class CBCParameters:
    mass_1: float # in Msun
    mass_2: float # in Msun
    distance: float # in Mpc
    inclination: float # in [0, pi], angle between line of sight and the orbital plane of the binary system
    ra: float # in [0, 2*pi], like longitude
    dec: float # in [-pi/2, pi/2], similar to latitude
    spin_1z: float # in [-1, 1]
    spin_2z: float # in [-1, 1]

    def __post_init__(self): # Is used after initialization, useful for validation, reordering, normalization, consistency...
        if self.mass_1 < self.mass_2: # Sort masses
            self.mass_1, self.mass_2 = self.mass_2, self.mass_1
            self.spin_1z, self.spin_2z = self.spin_2z, self.spin_1z

        ## Validation ##
        if self.mass_1 <= 0 or self.mass_2 <= 0:
            raise ValueError("Masses must be positive.")
        if self.distance <= 0:
            raise ValueError("Distance must be positive.")
        if not (0.0 <= self.inclination <= np.pi):
            raise ValueError("Inclination must be in [0, pi].")
        if not (0.0 <= self.ra <= 2 * np.pi):
            raise ValueError("Right ascension must be in [0, 2*pi].")
        if not (-np.pi / 2 <= self.dec <= np.pi / 2):
            raise ValueError("Declination must be in [-pi/2, pi/2].")
        if not (-1.0 <= self.spin_1z <= 1.0):
            raise ValueError("spin_1z must be in [-1, 1].")
        if not (-1.0 <= self.spin_2z <= 1.0):
            raise ValueError("spin_2z must be in [-1, 1].")

    @property
    def total_mass(self) -> float:
        return self.mass_1 + self.mass_2

    @property
    def chirp_mass(self) -> float:
        return (self.mass_1 * self.mass_2) ** (3.0 / 5.0) / (self.total_mass ** (1.0 / 5.0))

    @property
    def chi_eff(self) -> float:
        return (self.mass_1 * self.spin_1z + self.mass_2 * self.spin_2z) / self.total_mass
