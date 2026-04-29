from dataclasses import dataclass

@dataclass(frozen=True) # With this the config is immutable, cannot be changed during runtime
class SimulationConfig:
    """
    Configuration object for the simulation.

    Attributes
    ----------
    sampling_frequency : float
        The sampling frequency of the simulation.
    duration : float
        The duration of the simulation in seconds.
    low_frequency_cutoff : float
        The low frequency cutoff of the simulation.
    waveform_approximant : str
        The waveform approximation method to use.
    target_network_snr_range : tuple[float, float] | None
        The target network SNR range for the simulation. If None, the simulation will not be rescaled.
    snr_relative_tolerance : float
        The relative tolerance for the SNR. If the network SNR is within this tolerance, the simulation will not be rescaled.
    """

    sampling_frequency: float = 4096.0  # Hz
    duration: float = 4.0  # s
    low_frequency_cutoff: float = 30.0  # Hz
    waveform_approximant: str = "SEOBNRv4_opt"
    target_network_snr_range: tuple[float, float] | None = (10.0, 25.0)
    snr_relative_tolerance: float = 0.05

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

    if target_network_snr_range is not None:
        if not (target_network_snr_range[0] <= target_network_snr_range[1]):
            raise ValueError("target_network_snr_range must be a tuple with the first element less than or equal to the second element.")   
        if target_network_snr_range[0] < 0:
            raise ValueError("The first element of target_network_snr_range must be positive.")
        if target_network_snr_range[1] < 0:
            raise ValueError("The second element of target_network_snr_range must be positive.")