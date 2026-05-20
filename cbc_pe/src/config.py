from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuration object for CBC signal simulation.

    This object defines the numerical and physical assumptions used across
    waveform generation, detector projection, noise generation, injection,
    SNR computation, and dataset construction.

    Attributes
    ----------
    sampling_frequency : float
        Sampling frequency in Hz.
    duration : float
        Fixed strain segment duration in seconds.
    low_frequency_cutoff : float
        Low-frequency cutoff used for waveform generation and, unless explicitly
        overridden elsewhere, for SNR/PSD-related computations.
    waveform_approximant : str
        PyCBC/LAL waveform approximant.
    target_network_snr_range : tuple[float, float] | None
        Target network SNR range. If None, distance/SNR rescaling is disabled.
    snr_relative_tolerance : float
        Relative tolerance for accepting the rescaled network SNR.
    truncation_policy : str
        Policy for waveforms longer than the fixed segment duration.
    required_final_duration : float
        Minimum amount of final pre-merger/coalescence signal that must be
        present inside the selected segment.
    snr_on_truncated_signal : bool
        If True, SNR and distance rescaling are computed using the waveform
        segment actually injected into the strain.
    safe_margin_start : float
        Minimum margin in seconds before the injected signal region, useful
        for later whitening/filtering.
    safe_margin_end : float
        Minimum margin in seconds after the injected signal region.
    """

    # Regime to build the dataset BBH or BNS
    simulation_regime: Literal["BBH", "BNS"] = "BBH"
    waveform_family: Literal["IMR", "inspiral_tidal", "inspiral_only"] = "IMR"

    # Numerical setup
    sampling_frequency: float = 4096.0  # Hz
    duration: float = 4.0  # s

    # Waveform setup
    low_frequency_cutoff: float = 30.0  # Hz
    waveform_approximant: str = "SEOBNRv4_opt"

    # SNR targeting
    target_network_snr_range: tuple[float, float] | None = (10.0, 25.0)
    snr_relative_tolerance: float = 0.05
    snr_on_truncated_signal: bool = True

    # Long-waveform handling
    truncation_policy: Literal["none", "keep_full_if_possible", "keep_last_segment"] = (
        "keep_last_segment"
    )
    required_final_duration: float = 1.0  # s

    # Network timing
    event_time_reference: Literal["geocentric"] = "geocentric"

    # Injection margins inside the fixed strain segment
    safe_margin_start: float = 0.0  # s
    safe_margin_end: float = 0.0  # s

    # Processing context around the final CNN segment.
    # These are extra samples generated before/after the final output segment
    # so whitening/FIR edge corruption happens outside the returned 4 s window.
    processing_context_start_samples: int = 0
    processing_context_end_samples: int = 0

    @property
    def delta_t(self) -> float:
        return 1.0 / self.sampling_frequency

    @property
    def length(self) -> int:
        return int(round(self.duration * self.sampling_frequency))

    @property
    def delta_f(self) -> float:
        return 1.0 / self.duration

    @property
    def flength(self) -> int:
        return self.length // 2 + 1

    @property
    def required_final_samples(self) -> int:
        return int(round(self.required_final_duration * self.sampling_frequency))

    @property
    def safe_margin_start_samples(self) -> int:
        return int(round(self.safe_margin_start * self.sampling_frequency))

    @property
    def safe_margin_end_samples(self) -> int:
        return int(round(self.safe_margin_end * self.sampling_frequency))


    #Processing context
    @property
    def processing_context_start_seconds(self) -> float:
        return self.processing_context_start_samples * self.delta_t

    @property
    def processing_context_end_seconds(self) -> float:
        return self.processing_context_end_samples * self.delta_t

    @property
    def processing_length(self) -> int:
        return (
            self.length
            + self.processing_context_start_samples
            + self.processing_context_end_samples
        )

    @property
    def processing_duration(self) -> float:
        return self.processing_length * self.delta_t

    @property
    def processing_delta_f(self) -> float:
        return 1.0 / self.processing_duration

    @property
    def processing_flength(self) -> int:
        return self.processing_length // 2 + 1

    @property
    def has_processing_context(self) -> bool:
        return (
            self.processing_context_start_samples > 0
            or self.processing_context_end_samples > 0
        )

    def __post_init__(self) -> None:
        if self.sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive.")

        if self.duration <= 0:
            raise ValueError("duration must be positive.")

        if self.low_frequency_cutoff <= 0:
            raise ValueError("low_frequency_cutoff must be positive.")

        if self.snr_relative_tolerance < 0:
            raise ValueError("snr_relative_tolerance must be non-negative.")

        if self.target_network_snr_range is not None:
            low, high = self.target_network_snr_range

            if low <= 0 or high <= 0:
                raise ValueError("target_network_snr_range values must be positive.")

            if low > high:
                raise ValueError(
                    "target_network_snr_range must be ordered as (min_snr, max_snr)."
                )

        if self.required_final_duration <= 0:
            raise ValueError("required_final_duration must be positive.")

        if self.required_final_duration > self.duration:
            raise ValueError(
                "required_final_duration cannot be larger than the segment duration."
            )

        if self.safe_margin_start < 0 or self.safe_margin_end < 0:
            raise ValueError("safe margins must be non-negative.")

        if self.safe_margin_start + self.safe_margin_end >= self.duration:
            raise ValueError(
                "safe_margin_start + safe_margin_end must be smaller than duration."
            )

        expected_length = self.duration * self.sampling_frequency
        if abs(expected_length - round(expected_length)) > 1e-9:
            raise ValueError(
                "duration * sampling_frequency must be an integer number of samples."
            )

        if self.simulation_regime == "BBH":
            if self.waveform_family != "IMR":
                raise ValueError("BBH regime expects waveform_family='IMR'.")

        if self.waveform_approximant == "SEOBNRv4_opt" and self.simulation_regime != "BBH":
            raise ValueError("SEOBNRv4_opt is intended here for BBH simulations.")
        
        if not self.snr_on_truncated_signal:
            raise NotImplementedError(
                "snr_on_truncated_signal=False is not implemented in DatasetBuilder."
            )
        
        # Processing validations
        if self.processing_context_start_samples < 0:
            raise ValueError("processing_context_start_samples must be non-negative.")

        if self.processing_context_end_samples < 0:
            raise ValueError("processing_context_end_samples must be non-negative.")

        if not isinstance(self.processing_context_start_samples, int):
            raise TypeError("processing_context_start_samples must be an integer.")

        if not isinstance(self.processing_context_end_samples, int):
            raise TypeError("processing_context_end_samples must be an integer.")