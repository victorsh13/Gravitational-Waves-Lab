from .config import SimulationConfig
from .parameters import CBCParameters
from pycbc.waveform import get_td_waveform


class WaveformGenerator:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate(self, parameters: CBCParameters):
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
        return h_plus, h_cross