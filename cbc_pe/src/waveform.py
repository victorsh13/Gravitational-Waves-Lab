from dataclasses import dataclass

from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform

from .config import SimulationConfig
from .parameters import CBCParameters


@dataclass(frozen=True)
class WaveformMetadata:
    approximant: str
    low_frequency_cutoff: float
    delta_t: float
    n_samples: int
    duration: float
    start_time: float
    end_time: float


@dataclass(frozen=True)
class GeneratedWaveform:
    h_plus: TimeSeries
    h_cross: TimeSeries
    metadata: WaveformMetadata


class WaveformGenerator:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate(self, parameters: CBCParameters) -> GeneratedWaveform:
        h_plus, h_cross = get_td_waveform(
            mass1=parameters.mass_1,
            mass2=parameters.mass_2,
            distance=parameters.distance,
            inclination=parameters.inclination,
            spin1z=parameters.spin_1z,
            spin2z=parameters.spin_2z,
            f_lower=self.config.low_frequency_cutoff,
            delta_t=self.config.delta_t,
            approximant=self.config.waveform_approximant,
        )

        self._validate_waveform_pair(h_plus, h_cross)

        metadata = WaveformMetadata(
            approximant=self.config.waveform_approximant,
            low_frequency_cutoff=self.config.low_frequency_cutoff,
            delta_t=float(h_plus.delta_t),
            n_samples=len(h_plus),
            duration=len(h_plus) * float(h_plus.delta_t),
            start_time=float(h_plus.start_time),
            end_time=float(h_plus.start_time) + len(h_plus) * float(h_plus.delta_t),
        )

        return GeneratedWaveform(
            h_plus=h_plus,
            h_cross=h_cross,
            metadata=metadata,
        )

    def _validate_waveform_pair(self, h_plus: TimeSeries, h_cross: TimeSeries) -> None:
        if len(h_plus) != len(h_cross):
            raise ValueError("h_plus and h_cross must have the same length.")

        if h_plus.delta_t != h_cross.delta_t:
            raise ValueError("h_plus and h_cross must have the same delta_t.")

        if h_plus.start_time != h_cross.start_time:
            raise ValueError("h_plus and h_cross must have the same start_time.")

        if h_plus.delta_t != self.config.delta_t:
            raise ValueError("Generated waveform delta_t does not match config.delta_t.")