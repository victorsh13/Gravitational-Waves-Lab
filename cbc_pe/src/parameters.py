from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CBCParameters:
    """
    Parameters of a binary compact object.

    Attributes
    ----------
    mass_1 : float
        Mass of the first component in solar masses.
    mass_2 : float
        Mass of the second component in solar masses.
    distance : float
        Distance to the source in megaparsecs.
    inclination : float
        Inclination of the line of sight in radians.
    ra : float
        Right ascension of the source in radians.
    dec : float
        Declination of the source in radians.
    spin_1z : float
        Spin of the first component along the z-axis in units of the total mass.
    spin_2z : float
        Spin of the second component along the z-axis in units of the total mass.
    polarization_angle : float
        Polarization angle of the source in radians.
    """
    mass_1: float # in Msun
    mass_2: float # in Msun
    distance: float # in Mpc
    inclination: float # in [0, pi], angle between line of sight and the orbital plane of the binary system
    ra: float # in [0, 2*pi], like longitude
    dec: float # in [-pi/2, pi/2], similar to latitude
    spin_1z: float # in [-1, 1]
    spin_2z: float # in [-1, 1]
    polarization_angle: float = 0.0 # in [0, 2*pi], the polarization angle of the source.

    def __post_init__(self):
        """
        Post initialization method. Used for validation, reordering, normalization, consistency...

        Reorders masses and spins if necessary.
        Validates all parameters.
        """
        if self.mass_1 < self.mass_2:  # Sort masses
            # We need to use the __setattr__ method to change the value when frozen=True.
            m1_old = self.mass_1
            m2_old = self.mass_2
            s1_old = self.spin_1z
            s2_old = self.spin_2z

            object.__setattr__(self, "mass_1", m2_old)
            object.__setattr__(self, "mass_2", m1_old)
            object.__setattr__(self, "spin_1z", s2_old)
            object.__setattr__(self, "spin_2z", s1_old)

        ## Validation##
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
        if not (0.0 <= self.polarization_angle <= 2 * np.pi):
            raise ValueError("Polarization angle must be in [0, 2*pi].")

    @property
    def total_mass(self) -> float:
        return self.mass_1 + self.mass_2

    @property
    def chirp_mass(self) -> float:
        return (self.mass_1 * self.mass_2) ** (3.0 / 5.0) / (self.total_mass ** (1.0 / 5.0))

    @property
    def chi_eff(self) -> float:
        return (self.mass_1 * self.spin_1z + self.mass_2 * self.spin_2z) / self.total_mass
