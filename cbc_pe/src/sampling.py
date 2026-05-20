from __future__ import annotations
from typing import Literal

import numpy as np
from dataclasses import dataclass

from .parameters import CBCParameters



@dataclass(frozen=True)
class PriorConfig:
    regime: Literal["BBH", "BNS"]
    component_mass_range: tuple[float, float] = (1.0, 90.0)
    distance_range: tuple[float, float] = (200.0, 5000.0)
    spin_1z_range: tuple[float, float] = (-1.0, 1.0)
    spin_2z_range: tuple[float, float] = (-1.0, 1.0)

    @classmethod
    def bbh(cls) -> "PriorConfig":
        return cls(
            regime="BBH",
            component_mass_range=(5.0, 90.0),
            distance_range=(200.0, 5000.0),
            spin_1z_range=(-1.0, 1.0),
            spin_2z_range=(-1.0, 1.0),
        )

    @classmethod
    def bns(cls) -> "PriorConfig":
        return cls(
            regime="BNS",
            component_mass_range=(1.0, 3.0),
            distance_range=(20.0, 500.0),  # revisable
            spin_1z_range=(-0.05, 0.05),   # revisable
            spin_2z_range=(-0.05, 0.05),
        )


    def __post_init__(self) -> None:
        self._validate_range(self.component_mass_range, "component_mass_range", positive=True)
        self._validate_range(self.distance_range, "distance_range", positive=True)
        self._validate_range(self.spin_1z_range, "spin_1z_range", lower_bound=-1.0, upper_bound=1.0)
        self._validate_range(self.spin_2z_range, "spin_2z_range", lower_bound=-1.0, upper_bound=1.0)

    @staticmethod
    def _validate_range(
        value: tuple[float, float],
        name: str,
        positive: bool = False,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> None:
        low, high = value

        if low > high:
            raise ValueError(f"{name} must be ordered as (min, max).")

        if positive and low <= 0:
            raise ValueError(f"{name} values must be positive.")

        if lower_bound is not None and low < lower_bound:
            raise ValueError(f"{name} lower value must be >= {lower_bound}.")

        if upper_bound is not None and high > upper_bound:
            raise ValueError(f"{name} upper value must be <= {upper_bound}.")


class ParameterSampler:
    def __init__(
        self,
        prior_config: PriorConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.prior_config = prior_config if prior_config is not None else PriorConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_one(self) -> CBCParameters:
        """
        Sample CBC parameters from the configured priors.

        Inclination and declination are sampled isotropically:
        cos(inclination) ~ Uniform(-1, 1)
        sin(declination) ~ Uniform(-1, 1)

        Other parameters are sampled uniformly within their configured ranges.
        """
        m_min, m_max = self.prior_config.component_mass_range
        d_min, d_max = self.prior_config.distance_range
        s1_min, s1_max = self.prior_config.spin_1z_range
        s2_min, s2_max = self.prior_config.spin_2z_range

        mass_1 = self.rng.uniform(m_min, m_max)
        mass_2 = self.rng.uniform(m_min, m_max)
        distance = self.rng.uniform(d_min, d_max)

        inclination = np.arccos(self.rng.uniform(-1.0, 1.0))
        ra = self.rng.uniform(0.0, 2.0 * np.pi)
        dec = np.arcsin(self.rng.uniform(-1.0, 1.0))

        spin_1z = self.rng.uniform(s1_min, s1_max)
        spin_2z = self.rng.uniform(s2_min, s2_max)
        polarization_angle = self.rng.uniform(0.0, 2.0 * np.pi)

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
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative.")

        return [self.sample_one() for _ in range(n_samples)]
    

    @classmethod
    def from_regime(cls, regime: Literal["BBH", "BNS"]) -> "PriorConfig":
        if regime == "BBH":
            return cls.bbh()
        if regime == "BNS":
            return cls.bns()
        raise ValueError(f"Unknown regime: {regime}")