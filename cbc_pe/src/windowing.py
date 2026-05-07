from dataclasses import dataclass
from typing import Literal

from pycbc.types import TimeSeries

from .config import SimulationConfig


@dataclass(frozen=True)
class WindowMetadata:
    """
    Metadata describing how a waveform was selected for a fixed-duration segment.

    Times are expressed in seconds relative to the waveform time coordinates.
    """

    is_truncated: bool
    truncation_policy: str

    full_duration: float
    used_duration: float

    full_n_samples: int
    used_n_samples: int

    full_start_time: float
    full_end_time: float

    used_start_time: float
    used_end_time: float

    segment_duration: float
    required_final_duration: float
    required_available_final_duration: float

    seconds_before_waveform_end_in_window: float
    fraction_duration_used: float


@dataclass(frozen=True)
class WindowedWaveform:
    h_plus: TimeSeries
    h_cross: TimeSeries
    metadata: WindowMetadata


class WaveformWindowSelector:
    """
    Selects a fixed-duration window from a generated waveform.

    The selector does not modify amplitudes, does not inject into noise, and does
    not compute SNR. It only chooses which part of the waveform is retained.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def select(self, h_plus: TimeSeries, h_cross: TimeSeries) -> WindowedWaveform:
        self._validate_input_pair(h_plus, h_cross)

        if self.config.truncation_policy == "none":
            return self._select_none(h_plus, h_cross)

        if self.config.truncation_policy in {
            "keep_full_if_possible",
            "keep_last_segment",
        }:
            return self._select_keep_last_segment(h_plus, h_cross)

        raise ValueError(
            f"Unknown truncation_policy: {self.config.truncation_policy}"
        )

    def _select_none(
        self,
        h_plus: TimeSeries,
        h_cross: TimeSeries,
    ) -> WindowedWaveform:
        full_duration = self._duration(h_plus)

        if full_duration > self.config.duration:
            raise ValueError(
                "Waveform is longer than the configured segment duration, "
                "but truncation_policy='none'. "
                f"waveform_duration={full_duration}, "
                f"segment_duration={self.config.duration}"
            )

        return self._build_result(
            h_plus=h_plus,
            h_cross=h_cross,
            full_h_plus=h_plus,
            is_truncated=False,
        )

    def _select_keep_last_segment(
        self,
        h_plus: TimeSeries,
        h_cross: TimeSeries,
    ) -> WindowedWaveform:
        full_duration = self._duration(h_plus)

        if full_duration <= self.config.duration:
            return self._build_result(
                h_plus=h_plus,
                h_cross=h_cross,
                full_h_plus=h_plus,
                is_truncated=False,
            )

        n_keep = self.config.length

        if n_keep <= 0:
            raise ValueError("config.length must be positive.")

        if n_keep > len(h_plus):
            raise ValueError(
                "Internal error: requested more samples than waveform length."
            )

        start_index = len(h_plus) - n_keep
        end_index = len(h_plus)

        h_plus_window = h_plus[start_index:end_index]
        h_cross_window = h_cross[start_index:end_index]

        if len(h_plus_window) != self.config.length:
            raise ValueError(
                f"Window length mismatch: got {len(h_plus_window)}, "
                f"expected {self.config.length}."
    )

        # PyCBC slicing should already update start_time correctly,
        # but we enforce it explicitly for clarity and robustness.
        new_start = h_plus.start_time + start_index * h_plus.delta_t
        h_plus_window.start_time = new_start
        h_cross_window.start_time = new_start

        return self._build_result(
            h_plus=h_plus_window,
            h_cross=h_cross_window,
            full_h_plus=h_plus,
            is_truncated=True,
        )

    def _build_result(
        self,
        h_plus: TimeSeries,
        h_cross: TimeSeries,
        full_h_plus: TimeSeries,
        is_truncated: bool,
    ) -> WindowedWaveform:
        self._validate_input_pair(h_plus, h_cross)

        full_duration = self._duration(full_h_plus)
        used_duration = self._duration(h_plus)

        full_start_time = float(full_h_plus.start_time)
        full_end_time = self._end_time(full_h_plus)

        used_start_time = float(h_plus.start_time)
        used_end_time = self._end_time(h_plus)

        seconds_before_waveform_end_in_window = full_end_time - used_start_time

        required_available_final_duration = min(
            self.config.required_final_duration,
            full_duration,
        )

        tolerance = 1e-9 # avoid rejection because of floating point

        if seconds_before_waveform_end_in_window + tolerance < required_available_final_duration:
            raise ValueError(
                "Selected window does not contain the required final duration. "
                f"seconds_before_waveform_end_in_window="
                f"{seconds_before_waveform_end_in_window}, "
                f"required_available_final_duration={required_available_final_duration}, "
                f"configured_required_final_duration="
                f"{self.config.required_final_duration}"
            )

        metadata = WindowMetadata(
            is_truncated=is_truncated,
            truncation_policy=self.config.truncation_policy,
            full_duration=full_duration,
            used_duration=used_duration,
            full_n_samples=len(full_h_plus),
            used_n_samples=len(h_plus),
            full_start_time=full_start_time,
            full_end_time=full_end_time,
            used_start_time=used_start_time,
            used_end_time=used_end_time,
            segment_duration=self.config.duration,
            required_final_duration=self.config.required_final_duration,
            required_available_final_duration=required_available_final_duration,
            seconds_before_waveform_end_in_window=seconds_before_waveform_end_in_window,
            fraction_duration_used=used_duration / full_duration,
        )

        return WindowedWaveform(
            h_plus=h_plus,
            h_cross=h_cross,
            metadata=metadata,
        )

    def _validate_input_pair(self, h_plus: TimeSeries, h_cross: TimeSeries) -> None:
        if len(h_plus) != len(h_cross):
            raise ValueError("h_plus and h_cross must have the same length.")

        if h_plus.delta_t != h_cross.delta_t:
            raise ValueError("h_plus and h_cross must have the same delta_t.")

        if h_plus.start_time != h_cross.start_time:
            raise ValueError("h_plus and h_cross must have the same start_time.")

        if h_plus.delta_t != self.config.delta_t:
            raise ValueError(
                "Waveform delta_t does not match config.delta_t. "
                f"waveform_delta_t={h_plus.delta_t}, "
                f"config_delta_t={self.config.delta_t}"
            )

        if len(h_plus) == 0:
            raise ValueError("Waveform cannot be empty.")

    @staticmethod
    def _duration(series: TimeSeries) -> float:
        return len(series) * float(series.delta_t)

    @staticmethod
    def _end_time(series: TimeSeries) -> float:
        return float(series.start_time) + len(series) * float(series.delta_t)