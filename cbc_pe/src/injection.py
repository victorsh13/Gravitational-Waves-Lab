from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pycbc.types.timeseries import TimeSeries

from .config import SimulationConfig


@dataclass(frozen=True)
class InjectionResult:
    strain: TimeSeries

    signal_start_time: float
    signal_end_time: float

    segment_start_time: float
    segment_end_time: float

    signal_start_index: int
    signal_end_index: int

    overlap_start_index_strain: int
    overlap_end_index_strain: int

    overlap_start_index_signal: int
    overlap_end_index_signal: int

    n_signal_samples: int
    n_injected_samples: int

    n_clipped_before: int
    n_clipped_after: int
    is_partially_clipped: bool



@dataclass(frozen=True)
class SegmentPlacement:
    segment_start_time: float
    segment_end_time: float

    earliest_signal_start_time: float
    latest_signal_end_time: float

    valid_start_min: float
    valid_start_max: float

    placement_policy: str
    signal_network_duration: float

    safe_margin_start: float
    safe_margin_end: float
    margin_before_signal: float
    margin_after_signal: float
    margins_respected: bool
    enforce_safe_margins: bool


class SignalInjector:
    def __init__(
        self,
        config: SimulationConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

    def inject(
        self,
        strain: TimeSeries,
        signal: TimeSeries,
        ) -> InjectionResult:
        """
        Inject the part of a projected detector signal that overlaps a fixed
        strain segment.

        The signal and the strain must already live on the same absolute time axis.
        No physical detector delay is applied here.
        """

        self._validate_inputs(strain, signal)

        signal_start_time = float(signal.start_time)
        signal_end_time = signal_start_time + len(signal) * float(signal.delta_t)

        segment_start_time = float(strain.start_time)
        segment_end_time = segment_start_time + len(strain) * float(strain.delta_t)

        signal_start_index = int(
            round((signal_start_time - segment_start_time) / float(strain.delta_t))
        )
        signal_end_index = signal_start_index + len(signal)

        overlap_start_index_strain = max(signal_start_index, 0)
        overlap_end_index_strain = min(signal_end_index, len(strain))

        if overlap_start_index_strain >= overlap_end_index_strain:
            raise ValueError(
                "Signal does not overlap the strain segment. "
                f"signal_start_time={signal_start_time}, "
                f"signal_end_time={signal_end_time}, "
                f"segment_start_time={segment_start_time}, "
                f"segment_end_time={segment_end_time}."
            )

        overlap_start_index_signal = overlap_start_index_strain - signal_start_index
        overlap_end_index_signal = overlap_start_index_signal + (
            overlap_end_index_strain - overlap_start_index_strain
        )

        n_injected_samples = overlap_end_index_strain - overlap_start_index_strain
        n_clipped_before = overlap_start_index_signal
        n_clipped_after = len(signal) - overlap_end_index_signal

        injected_strain = strain.copy()

        injected_strain[
            overlap_start_index_strain:overlap_end_index_strain
        ] += np.asarray(signal)[
            overlap_start_index_signal:overlap_end_index_signal
        ]

        return InjectionResult(
            strain=injected_strain,
            signal_start_time=signal_start_time,
            signal_end_time=signal_end_time,
            segment_start_time=segment_start_time,
            segment_end_time=segment_end_time,
            signal_start_index=signal_start_index,
            signal_end_index=signal_end_index,
            overlap_start_index_strain=overlap_start_index_strain,
            overlap_end_index_strain=overlap_end_index_strain,
            overlap_start_index_signal=overlap_start_index_signal,
            overlap_end_index_signal=overlap_end_index_signal,
            n_signal_samples=len(signal),
            n_injected_samples=n_injected_samples,
            n_clipped_before=n_clipped_before,
            n_clipped_after=n_clipped_after,
            is_partially_clipped=(n_clipped_before > 0 or n_clipped_after > 0),
        )

    def inject_network(
        self,
        noises: dict[str, TimeSeries],
        signals: dict[str, TimeSeries],
    ) -> dict[str, InjectionResult]:
        """
        Inject a projected signal into each detector noise segment.

        This method assumes all noise segments share the same absolute time
        window and all signals have already been projected with physical
        detector delays.
        """
        if set(noises.keys()) != set(signals.keys()):
            raise ValueError(
                "Noise and signal detector sets must match. "
                f"noise_detectors={set(noises.keys())}, "
                f"signal_detectors={set(signals.keys())}"
            )

        return {
            detector: self.inject(noises[detector], signals[detector])
            for detector in signals
        }

    def build_zero_strain(
        self,
        start_time: float,
        length: int | None = None,
    ) -> TimeSeries:
        """
        Build an empty zero-valued strain segment.

        By default this builds the final model-length segment. Passing length allows
        building a longer processing-context segment.
        """
        if length is None:
            length = self.config.length

        if length <= 0:
            raise ValueError("length must be positive.")

        return TimeSeries(
            initial_array=np.zeros(length),
            delta_t=self.config.delta_t,
            epoch=start_time,
        )

    def set_strain_start_time(
        self,
        strain: TimeSeries,
        start_time: float,
        expected_length: int | None = None,
    ) -> TimeSeries:
        """
        Return a copy of a strain segment with a new absolute start time.

        If expected_length is provided, validate against it. Otherwise accept the
        current strain length.
        """
        if expected_length is not None and len(strain) != expected_length:
            raise ValueError(
                f"Strain length mismatch: got {len(strain)}, "
                f"expected {expected_length}."
            )

        if strain.delta_t != self.config.delta_t:
            raise ValueError(
                f"Strain delta_t mismatch: got {strain.delta_t}, "
                f"expected {self.config.delta_t}."
            )

        out = strain.copy()
        out.start_time = start_time
        return out

    def _validate_inputs(
        self,
        strain: TimeSeries,
        signal: TimeSeries,
    ) -> None:
        if strain.delta_t != signal.delta_t:
            raise ValueError(
                "Strain and signal must have the same delta_t. "
                f"strain.delta_t={strain.delta_t}, signal.delta_t={signal.delta_t}"
            )

        if strain.delta_t != self.config.delta_t:
            raise ValueError(
                "Strain delta_t does not match config.delta_t. "
                f"strain.delta_t={strain.delta_t}, config.delta_t={self.config.delta_t}"
            )

        allowed_lengths = {
            self.config.length,
            self.config.processing_length,
        }

        if len(strain) not in allowed_lengths:
            raise ValueError(
                "Strain length mismatch: got "
                f"{len(strain)}, expected one of {sorted(allowed_lengths)}."
            )

        if len(signal) == 0:
            raise ValueError("Signal cannot be empty.")

        if signal.delta_t != self.config.delta_t:
            raise ValueError(
                "Signal delta_t does not match config.delta_t. "
                f"signal.delta_t={signal.delta_t}, config.delta_t={self.config.delta_t}"
            )
        

    def choose_segment_placement_containing_network(
        self,
        signals: dict[str, TimeSeries],
        placement_policy: str = "random_contained",
        safe_margin_start: float | None = None,
        safe_margin_end: float | None = None,
        enforce_safe_margins: bool = True,
    ) -> SegmentPlacement:

        if len(signals) == 0:
            raise ValueError("signals cannot be empty.")

        delta_ts = {
            ifo: float(signal.delta_t)
            for ifo, signal in signals.items()
        }

        if any(abs(dt - self.config.delta_t) > 0.0 for dt in delta_ts.values()):
            raise ValueError(
                "All signals must have delta_t matching config.delta_t. "
                f"delta_ts={delta_ts}, config_delta_t={self.config.delta_t}"
            )

        empty = [
            ifo for ifo, signal in signals.items()
            if len(signal) == 0
        ]

        if empty:
            raise ValueError(f"Empty projected signals found for detectors: {empty}")

        if safe_margin_start is None:
            safe_margin_start = self.config.safe_margin_start

        if safe_margin_end is None:
            safe_margin_end = self.config.safe_margin_end

        safe_margin_start = float(safe_margin_start)
        safe_margin_end = float(safe_margin_end)

        if safe_margin_start < 0.0 or safe_margin_end < 0.0:
            raise ValueError(
                "safe_margin_start and safe_margin_end must be non-negative. "
                f"got {safe_margin_start}, {safe_margin_end}"
            )

        signal_start_times = {
            ifo: float(signal.start_time)
            for ifo, signal in signals.items()
        }

        signal_end_times = {
            ifo: float(signal.start_time) + len(signal) * float(signal.delta_t)
            for ifo, signal in signals.items()
        }

        earliest_signal_start_time = min(signal_start_times.values())
        latest_signal_end_time = max(signal_end_times.values())

        signal_network_duration = (
            latest_signal_end_time - earliest_signal_start_time
        )

        if enforce_safe_margins:
            valid_start_min = (
                latest_signal_end_time
                + safe_margin_end
                - self.config.duration
            )
            valid_start_max = (
                earliest_signal_start_time
                - safe_margin_start
            )
        else:
            valid_start_min = latest_signal_end_time - self.config.duration
            valid_start_max = earliest_signal_start_time

        tolerance = self.config.delta_t

        if valid_start_min > valid_start_max + tolerance:
            raise ValueError(
                "Projected network signal does not fit inside the configured segment "
                "duration with the requested safe margins. "
                f"network_signal_duration={signal_network_duration}, "
                f"segment_duration={self.config.duration}, "
                f"safe_margin_start={safe_margin_start}, "
                f"safe_margin_end={safe_margin_end}, "
                f"enforce_safe_margins={enforce_safe_margins}, "
                f"valid_start_min={valid_start_min}, "
                f"valid_start_max={valid_start_max}."
            )

        # If the interval is only numerically negative within tolerance,
        # collapse it to a single valid point.
        if valid_start_max < valid_start_min:
            valid_start_max = valid_start_min

        if placement_policy == "end_aligned":
            segment_start_time = valid_start_min

        elif placement_policy == "start_aligned":
            segment_start_time = valid_start_max

        elif placement_policy == "centered":
            segment_start_time = (
                0.5 * (earliest_signal_start_time + latest_signal_end_time)
                - 0.5 * self.config.duration
            )

            segment_start_time = min(
                max(segment_start_time, valid_start_min),
                valid_start_max,
            )

        elif placement_policy == "random_contained":
            if abs(valid_start_max - valid_start_min) <= tolerance:
                segment_start_time = valid_start_min
            else:
                segment_start_time = self.rng.uniform(
                    valid_start_min,
                    valid_start_max,
                )

        else:
            raise ValueError(f"Unknown placement_policy: {placement_policy}")

        segment_end_time = segment_start_time + self.config.duration

        margin_before_signal = earliest_signal_start_time - segment_start_time
        margin_after_signal = segment_end_time - latest_signal_end_time

        margins_respected = (
            margin_before_signal + tolerance >= safe_margin_start
            and margin_after_signal + tolerance >= safe_margin_end
        )

        if enforce_safe_margins and not margins_respected:
            raise RuntimeError(
                "Internal placement error: safe margins were enforced but not respected. "
                f"margin_before_signal={margin_before_signal}, "
                f"margin_after_signal={margin_after_signal}, "
                f"safe_margin_start={safe_margin_start}, "
                f"safe_margin_end={safe_margin_end}, "
                f"segment_start_time={segment_start_time}, "
                f"segment_end_time={segment_end_time}."
            )

        return SegmentPlacement(
            segment_start_time=segment_start_time,
            segment_end_time=segment_end_time,
            earliest_signal_start_time=earliest_signal_start_time,
            latest_signal_end_time=latest_signal_end_time,
            valid_start_min=valid_start_min,
            valid_start_max=valid_start_max,
            placement_policy=placement_policy,
            signal_network_duration=signal_network_duration,
            safe_margin_start=safe_margin_start,
            safe_margin_end=safe_margin_end,
            margin_before_signal=margin_before_signal,
            margin_after_signal=margin_after_signal,
            margins_respected=margins_respected,
            enforce_safe_margins=enforce_safe_margins,
        )
        