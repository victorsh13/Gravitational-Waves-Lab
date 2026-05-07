from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pycbc.types import TimeSeries

from .config import SimulationConfig


@dataclass(frozen=True)
class NetworkWindowMetadata:
    """
    Metadata describing how a projected detector network was selected for a
    fixed-duration training/inference segment.

    All times are expressed in the absolute time coordinates of the projected
    detector TimeSeries objects.
    """

    is_truncated: bool
    truncation_policy: str

    detector_names: list[str]

    full_network_start_time: float
    full_network_end_time: float
    full_network_duration: float

    used_window_start_time: float
    used_window_end_time: float
    used_window_duration: float

    segment_duration: float
    required_final_duration: float
    required_available_final_duration: float

    seconds_before_network_end_in_window: float
    fraction_network_duration_used: float

    full_start_times: dict[str, float]
    full_end_times: dict[str, float]
    full_n_samples: dict[str, int]

    used_start_times: dict[str, float]
    used_end_times: dict[str, float]
    used_n_samples: dict[str, int]


@dataclass(frozen=True)
class WindowedProjectedNetwork:
    """
    Windowed projected detector strains.

    The strains remain shorter than or equal to config.duration. They are not
    padded to config.length here. Padding/placement into a fixed 4 s segment is
    handled later by injection.py.
    """

    strains: dict[str, TimeSeries]
    metadata: NetworkWindowMetadata


class ProjectedNetworkWindowSelector:
    """
    Select a fixed-duration time window from a projected detector network.

    This class operates after detector projection. That is deliberate:
    detector projection can introduce physical time delays and small differences
    in start time / length between detectors. Therefore the robust place to
    decide the common training window is after projection, not before.

    It does not compute SNR, does not inject noise, and does not process the
    strain. It only chooses which part of the projected signal network is kept.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def select(
        self,
        projected_strains: dict[str, TimeSeries],
    ) -> WindowedProjectedNetwork:
        self._validate_projected_strains(projected_strains)

        if self.config.truncation_policy == "none":
            return self._select_none(projected_strains)

        if self.config.truncation_policy in {
            "keep_full_if_possible",
            "keep_last_segment",
        }:
            return self._select_keep_last_network_window(projected_strains)

        raise ValueError(
            f"Unknown truncation_policy: {self.config.truncation_policy}"
        )

    def _select_none(
        self,
        projected_strains: dict[str, TimeSeries],
    ) -> WindowedProjectedNetwork:
        network_start, network_end = self._network_time_bounds(projected_strains)
        network_duration = network_end - network_start

        if network_duration > self.config.duration:
            raise ValueError(
                "Projected network signal is longer than the configured "
                "segment duration, but truncation_policy='none'. "
                f"network_duration={network_duration}, "
                f"segment_duration={self.config.duration}."
            )

        return self._build_result(
            full_projected_strains=projected_strains,
            used_projected_strains=projected_strains,
            is_truncated=False,
            used_window_start_time=network_start,
            used_window_end_time=network_end,
        )

    def _select_keep_last_network_window(
        self,
        projected_strains: dict[str, TimeSeries],
    ) -> WindowedProjectedNetwork:
        network_start, network_end = self._network_time_bounds(projected_strains)
        network_duration = network_end - network_start

        if network_duration <= self.config.duration:
            return self._build_result(
                full_projected_strains=projected_strains,
                used_projected_strains=projected_strains,
                is_truncated=False,
                used_window_start_time=network_start,
                used_window_end_time=network_end,
            )

        used_window_end_time = network_end
        used_window_start_time = network_end - self.config.duration

        used_projected_strains = {
            detector_name: self._slice_timeseries_by_time(
                series=strain,
                window_start_time=used_window_start_time,
                window_end_time=used_window_end_time,
            )
            for detector_name, strain in projected_strains.items()
        }

        return self._build_result(
            full_projected_strains=projected_strains,
            used_projected_strains=used_projected_strains,
            is_truncated=True,
            used_window_start_time=used_window_start_time,
            used_window_end_time=used_window_end_time,
        )

    def _build_result(
        self,
        full_projected_strains: dict[str, TimeSeries],
        used_projected_strains: dict[str, TimeSeries],
        is_truncated: bool,
        used_window_start_time: float,
        used_window_end_time: float,
    ) -> WindowedProjectedNetwork:
        self._validate_projected_strains(full_projected_strains)
        self._validate_projected_strains(used_projected_strains)

        if set(full_projected_strains.keys()) != set(used_projected_strains.keys()):
            raise ValueError(
                "Full and used projected strain detector sets must match."
            )

        full_network_start_time, full_network_end_time = self._network_time_bounds(
            full_projected_strains
        )
        full_network_duration = full_network_end_time - full_network_start_time

        used_network_start_time, used_network_end_time = self._network_time_bounds(
            used_projected_strains
        )
        used_network_duration = used_window_end_time - used_window_start_time

        required_available_final_duration = min(
            self.config.required_final_duration,
            full_network_duration,
        )

        seconds_before_network_end_in_window = (
            full_network_end_time - used_window_start_time
        )

        tolerance = self.config.delta_t

        if (
            seconds_before_network_end_in_window + tolerance
            < required_available_final_duration
        ):
            raise ValueError(
                "Selected network window does not contain the required final "
                "duration. "
                f"seconds_before_network_end_in_window="
                f"{seconds_before_network_end_in_window}, "
                f"required_available_final_duration="
                f"{required_available_final_duration}, "
                f"configured_required_final_duration="
                f"{self.config.required_final_duration}."
            )

        if used_network_duration > self.config.duration + tolerance:
            raise ValueError(
                "Used network window is longer than config.duration. "
                f"used_network_duration={used_network_duration}, "
                f"segment_duration={self.config.duration}."
            )

        full_start_times = {
            detector_name: float(strain.start_time)
            for detector_name, strain in full_projected_strains.items()
        }

        full_end_times = {
            detector_name: self._end_time(strain)
            for detector_name, strain in full_projected_strains.items()
        }

        full_n_samples = {
            detector_name: len(strain)
            for detector_name, strain in full_projected_strains.items()
        }

        used_start_times = {
            detector_name: float(strain.start_time)
            for detector_name, strain in used_projected_strains.items()
        }

        used_end_times = {
            detector_name: self._end_time(strain)
            for detector_name, strain in used_projected_strains.items()
        }

        used_n_samples = {
            detector_name: len(strain)
            for detector_name, strain in used_projected_strains.items()
        }

        metadata = NetworkWindowMetadata(
            is_truncated=is_truncated,
            truncation_policy=self.config.truncation_policy,
            detector_names=list(used_projected_strains.keys()),
            full_network_start_time=full_network_start_time,
            full_network_end_time=full_network_end_time,
            full_network_duration=full_network_duration,
            used_window_start_time=used_window_start_time,
            used_window_end_time=used_window_end_time,
            used_window_duration=used_network_duration,
            segment_duration=self.config.duration,
            required_final_duration=self.config.required_final_duration,
            required_available_final_duration=required_available_final_duration,
            seconds_before_network_end_in_window=seconds_before_network_end_in_window,
            fraction_network_duration_used=used_network_duration / full_network_duration,
            full_start_times=full_start_times,
            full_end_times=full_end_times,
            full_n_samples=full_n_samples,
            used_start_times=used_start_times,
            used_end_times=used_end_times,
            used_n_samples=used_n_samples,
        )

        return WindowedProjectedNetwork(
            strains=used_projected_strains,
            metadata=metadata,
        )

    def _slice_timeseries_by_time(
        self,
        series: TimeSeries,
        window_start_time: float,
        window_end_time: float,
    ) -> TimeSeries:
        """
        Return the part of `series` inside [window_start_time, window_end_time].

        If the series only partially overlaps the window, only the overlapping
        samples are returned. If there is no overlap, this raises an error.
        """

        series_start_time = float(series.start_time)
        series_end_time = self._end_time(series)
        delta_t = float(series.delta_t)

        overlap_start_time = max(series_start_time, window_start_time)
        overlap_end_time = min(series_end_time, window_end_time)

        tolerance = 0.5 * delta_t

        if overlap_end_time <= overlap_start_time + tolerance:
            raise ValueError(
                "Detector strain has no overlap with selected network window. "
                f"series_start_time={series_start_time}, "
                f"series_end_time={series_end_time}, "
                f"window_start_time={window_start_time}, "
                f"window_end_time={window_end_time}."
            )

        start_index = int(np.ceil(
            (overlap_start_time - series_start_time) / delta_t
        ))
        end_index = int(np.floor(
            (overlap_end_time - series_start_time) / delta_t
        ))

        start_index = max(start_index, 0)
        end_index = min(end_index, len(series))

        if end_index <= start_index:
            raise ValueError(
                "Empty slice after index conversion. "
                f"start_index={start_index}, end_index={end_index}, "
                f"len(series)={len(series)}."
            )

        sliced = series[start_index:end_index]

        new_start_time = series_start_time + start_index * delta_t
        sliced.start_time = new_start_time

        return sliced

    def _validate_projected_strains(
        self,
        projected_strains: dict[str, TimeSeries],
    ) -> None:
        if len(projected_strains) == 0:
            raise ValueError("projected_strains cannot be empty.")

        for detector_name, strain in projected_strains.items():
            if not isinstance(strain, TimeSeries):
                raise TypeError(
                    f"Projected strain for {detector_name} must be a TimeSeries."
                )

            if len(strain) == 0:
                raise ValueError(
                    f"Projected strain for {detector_name} cannot be empty."
                )

            if strain.delta_t != self.config.delta_t:
                raise ValueError(
                    f"Projected strain delta_t mismatch for {detector_name}: "
                    f"got {strain.delta_t}, expected {self.config.delta_t}."
                )

            if not np.all(np.isfinite(strain.numpy())):
                raise ValueError(
                    f"Projected strain for {detector_name} contains NaN or Inf."
                )

    def _network_time_bounds(
        self,
        projected_strains: dict[str, TimeSeries],
    ) -> tuple[float, float]:
        start_times = [
            float(strain.start_time)
            for strain in projected_strains.values()
        ]

        end_times = [
            self._end_time(strain)
            for strain in projected_strains.values()
        ]

        return min(start_times), max(end_times)

    @staticmethod
    def _end_time(series: TimeSeries) -> float:
        return float(series.start_time) + len(series) * float(series.delta_t)