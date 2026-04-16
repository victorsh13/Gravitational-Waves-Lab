from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig
from pycbc.types.timeseries import TimeSeries


@dataclass(frozen=True)
class InjectionResult:
    strain: object
    injection_time: float
    injection_index: int


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
        injection_time: float | None = None,
    ) -> InjectionResult:
        """
        Inject a signal into a strain.

        Parameters
        ----------
        strain : TimeSeries
            The strain to inject the signal into.
        signal : TimeSeries
            The signal to inject.
        injection_time : float | None = None
            The time at which to inject the signal into the strain. If None,
            a random time between 0.25 and 0.75 of the maximum possible injection
            time is chosen.

        Returns
        -------
        InjectionResult
            An object containing the injected strain, the injection time, and the
            injection index.
        """

        if len(strain) < len(signal):
            raise ValueError("The strain must be longer than or equal to the signal.")

        if strain.delta_t != signal.delta_t:
            raise ValueError("Strain and signal must have the same delta_t.")

        max_start_index = len(strain) - len(signal)
        max_injection_time = max_start_index * signal.delta_t

        if injection_time is None:
            # Choose a random injection time between 25% and 75% of the maximum possible injection time
            min_index = int(0.25 * max_start_index)
            max_index = int(0.75 * max_start_index)

            if max_index < min_index:
                raise ValueError("Not enough room in the strain to choose a safe injection window.")

            injection_index = self.rng.integers(min_index, max_index, endpoint=True)
            injection_time = injection_index * signal.delta_t
        else:
            # Check if the given injection time is within the allowed range
            if not (0.0 <= injection_time <= max_injection_time):
                raise ValueError(
                    f"injection_time must be between 0 and {max_injection_time:.6f} s, "
                    f"got {injection_time:.6f} s."
                )

            injection_index = round(injection_time / signal.delta_t)


        if injection_index < 0 or injection_index + len(signal) > len(strain):
            raise ValueError(
                f"The injected signal does not fit within the strain. Injection index: {injection_index}, signal length: {len(signal)}, strain length: {len(strain)}."
            )

        signal_array = np.array(signal)

        injected_strain = strain.copy()
        injected_strain[injection_index:injection_index + len(signal)] += signal_array

        ts_strain = TimeSeries(
            initial_array=injected_strain,
            delta_t=strain.delta_t,
            epoch=strain.start_time,
        )

        return InjectionResult(
            strain=ts_strain,
            injection_time=injection_time,
            injection_index=injection_index,
        )

