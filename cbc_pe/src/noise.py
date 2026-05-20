from __future__ import annotations
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

        self._psd_cache: dict[tuple[str, int, float], FrequencySeries] = {}

        # Precompute standard output-length PSDs for compatibility and early validation.
        for detector in self.psd_models:
            self.get_psd(detector)

    def _build_psd(
        self,
        detector: str,
        flength: int,
        delta_f: float,
    ) -> FrequencySeries:
        if detector in {"H1", "L1"}:
            return analytical.aLIGOZeroDetHighPower(
                flength,
                delta_f,
                self.config.low_frequency_cutoff,
            )

        if detector == "V1":
            return analytical.AdvVirgo(
                flength,
                delta_f,
                self.config.low_frequency_cutoff,
            )

        raise ValueError(f"Unsupported detector: {detector}")

    def get_psd(
        self,
        detector: str,
        length: int | None = None,
    ) -> FrequencySeries:
        if detector not in self.psd_models:
            raise ValueError(f"Unsupported detector: {detector}")

        if length is None:
            length = self.config.length

        if length <= 0:
            raise ValueError("length must be positive.")

        flength = length // 2 + 1
        duration = length * self.config.delta_t
        delta_f = 1.0 / duration

        key = (detector, length, delta_f)

        if key not in self._psd_cache:
            psd = self._build_psd(
                detector=detector,
                flength=flength,
                delta_f=delta_f,
            )
            self._validate_psd(detector=detector, psd=psd, length=length)
            self._psd_cache[key] = psd

        return self._psd_cache[key]

    def sample(
        self,
        detector_name: str,
        seed: int | None = None,
        length: int | None = None,
    ) -> TimeSeries:
        if length is None:
            length = self.config.length

        return gaussian.noise_from_psd(
            psd=self.get_psd(detector_name, length=length),
            length=length,
            delta_t=self.config.delta_t,
            seed=seed,
        )

    def sample_network(
        self,
        detector_names: list[str],
        seed: int | None = None,
        length: int | None = None,
    ) -> dict[str, TimeSeries]:
        rng = np.random.default_rng(seed)

        noises: dict[str, TimeSeries] = {}

        for name in detector_names:
            detector_seed = int(rng.integers(0, 2**32 - 1))
            noises[name] = self.sample(
                name,
                seed=detector_seed,
                length=length,
            )

        return noises

    def metadata(self) -> dict:
        return {
            "psd_models": dict(self.psd_models),
            "psd_low_frequency_cutoff": self.config.low_frequency_cutoff,

            "output_delta_f": self.config.delta_f,
            "output_flength": self.config.flength,
            "output_noise_length": self.config.length,
            "output_duration": self.config.duration,

            "processing_delta_f": self.config.processing_delta_f,
            "processing_flength": self.config.processing_flength,
            "processing_noise_length": self.config.processing_length,
            "processing_duration": self.config.processing_duration,

            "delta_t": self.config.delta_t,
        }

    def _validate_psd(
        self,
        detector: str,
        psd: FrequencySeries,
        length: int,
    ) -> None:
        expected_flength = length // 2 + 1
        expected_delta_f = 1.0 / (length * self.config.delta_t)

        if len(psd) != expected_flength:
            raise ValueError(
                f"PSD length mismatch for {detector}: "
                f"got {len(psd)}, expected {expected_flength}."
            )

        if psd.delta_f != expected_delta_f:
            raise ValueError(
                f"PSD delta_f mismatch for {detector}: "
                f"got {psd.delta_f}, expected {expected_delta_f}."
            )