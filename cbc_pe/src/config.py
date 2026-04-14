from dataclasses import dataclass

@dataclass(frozen=True) # With this the config is immutable, cannot be changed during runtime
class SimulationConfig:
    sampling_frequency: float = 4096.0  # Hz
    duration: float = 4.0  # s
    low_frequency_cutoff: float = 30.0  # Hz
    waveform_approximant: str = "SEOBNRv4_opt"

    @property
    def delta_t(self) -> float:
        return 1.0 / self.sampling_frequency

    @property
    def length(self) -> int:
        return int(self.duration * self.sampling_frequency)

    @property
    def delta_f(self) -> float:
        return 1.0 / self.duration

    @property
    def flength(self) -> int:
        return self.length // 2 + 1
