from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig
from pycbc.types.timeseries import TimeSeries


@dataclass(frozen=True)
class InjectionResult:
    """
    A class to store the result of the injection process.

    Attributes
    ----------
    strain : TimeSeries
        The injected strain.
    injection_time : float
        The injection time in the event reference frame. (i.e. considering the strain start time as 0).
    injection_index : int
        The index of the injection time within the strain in the event reference frame.
    detector_start_time : float
        The detector start time in the event reference frame (i.e. the geocentric time of the injection).
    detector_start_index : int
        The detector start index in the event reference frame (i.e. the geocentric index of the injection).
    """
    strain: TimeSeries
    injection_time: float
    injection_index: int
    detector_start_time: float
    detector_start_index: int


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
        injection_time: float,
    ) -> InjectionResult:
        """
        Inject a signal into a strain.

        Parameters
        ----------
        strain : TimeSeries
            The strain to inject the signal into (i.e. the noise signal).
        signal : TimeSeries
            The signal to inject (i.e. the gw signal).
        injection_time : float | None = None
            The time at which to inject the signal into the strain. 

        Returns
        -------
        InjectionResult
            An object containing the injected strain, the injection time, and the
            injection index.

        """

        if strain.delta_t != signal.delta_t:
            raise ValueError("Strain and signal must have the same delta_t.")

        t_start_in_strain = float(signal.start_time + injection_time)

        # Compute the start and end index of the signal within the strain applying the time delays
        idx_start_signal = round((t_start_in_strain - float(strain.start_time)) / signal.delta_t)
        idx_end_signal = idx_start_signal + len(signal)

        if idx_start_signal < 0 or idx_end_signal > len(strain):
            raise ValueError("The signal is not within the strain, the injection time for this sample is too early or too late.")

        signal_array = np.array(signal)

        injected_strain = strain.copy()
        injected_strain[idx_start_signal:idx_end_signal] += signal_array

        ts_strain = TimeSeries(
            initial_array=injected_strain,
            delta_t=strain.delta_t,
            epoch=strain.start_time,
        )

        return InjectionResult(
            strain=ts_strain,
            injection_time=injection_time ,
            injection_index=round((injection_time - float(strain.start_time)) / signal.delta_t),
            detector_start_time=t_start_in_strain,
            detector_start_index=idx_start_signal,
        )

