import numpy as np

from pycbc.noise import gaussian
from pycbc.psd import analytical
from pycbc.types import FrequencySeries, TimeSeries

from .config import SimulationConfig


class NoiseModel:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

        self.psd_models = {
            "H1": "aLIGOZeroDetHighPower",
            "L1": "aLIGOZeroDetHighPower",
            "V1": "AdvVirgo",
        }

        self.psds: dict[str, FrequencySeries] = {
            "H1": analytical.aLIGOZeroDetHighPower(
                config.flength,
                config.delta_f,
                config.low_frequency_cutoff,
            ),
            "L1": analytical.aLIGOZeroDetHighPower(
                config.flength,
                config.delta_f,
                config.low_frequency_cutoff,
            ),
            "V1": analytical.AdvVirgo(
                config.flength,
                config.delta_f,
                config.low_frequency_cutoff,
            ),
        }

        self._validate_psds()

    def get_psd(self, detector: str) -> FrequencySeries:
        if detector not in self.psds:
            raise ValueError(f"Unsupported detector: {detector}")

        return self.psds[detector]

    def sample(self, detector_name: str, seed: int | None = None) -> TimeSeries:
        return gaussian.noise_from_psd(
            psd=self.get_psd(detector_name),
            length=self.config.length,
            delta_t=self.config.delta_t,
            seed=seed,
        )

    def sample_network(
        self,
        detector_names: list[str],
        seed: int | None = None,
    ) -> dict[str, TimeSeries]:
        rng = np.random.default_rng(seed)

        noises: dict[str, TimeSeries] = {}

        for name in detector_names:
            detector_seed = int(rng.integers(0, 2**32 - 1))
            noises[name] = self.sample(name, seed=detector_seed)

        return noises

    def metadata(self) -> dict:
        return {
            "psd_models": dict(self.psd_models),
            "psd_low_frequency_cutoff": self.config.low_frequency_cutoff,
            "delta_f": self.config.delta_f,
            "flength": self.config.flength,
            "noise_length": self.config.length,
            "delta_t": self.config.delta_t,
            "duration": self.config.duration,
        }

    def _validate_psds(self) -> None:
        for detector, psd in self.psds.items():
            if len(psd) != self.config.flength:
                raise ValueError(
                    f"PSD length mismatch for {detector}: "
                    f"got {len(psd)}, expected {self.config.flength}."
                )

            if psd.delta_f != self.config.delta_f:
                raise ValueError(
                    f"PSD delta_f mismatch for {detector}: "
                    f"got {psd.delta_f}, expected {self.config.delta_f}."
                )