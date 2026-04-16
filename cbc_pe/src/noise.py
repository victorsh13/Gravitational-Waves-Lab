from .config import SimulationConfig

from pycbc.psd import analytical
from pycbc.noise import gaussian

class NoiseModel:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

        # Initialize detectors PSDs
        # WARNING: the legacy code uses low_frequency_cutoff=15, but here we use the config low_frequency_cutoff=30
        self.psds ={
            "H1": analytical.aLIGOZeroDetHighPower(config.flength, config.delta_f, config.low_frequency_cutoff),
            "L1": analytical.aLIGOZeroDetHighPower(config.flength, config.delta_f, config.low_frequency_cutoff),
            "V1": analytical.AdvVirgo(config.flength, config.delta_f, config.low_frequency_cutoff),
        }

        
    def get_psd(self, detector: str):
        if detector not in self.psds:
            raise ValueError(f"Unsupported detector: {detector}")
        return self.psds[detector]
        
    def sample(self, detector_name: str, seed: int | None = None):
        return gaussian.noise_from_psd(
                psd=self.get_psd(detector_name),
                length=self.config.length,
                delta_t=self.config.delta_t,
                seed=seed)
    
    # def sample_network(self, detector_names: list[str], seed: int | None = None):
    #     return {name: self.sample(name, seed) for name in detector_names}