import numpy as np
from pycbc.filter.matchedfilter import sigma
from pycbc.types.timeseries import TimeSeries, FrequencySeries
from .config import SimulationConfig

"""
Matched filtering used to compute the SNR of the detector. This is optimal correlation between data and a sample, ponderated by the PSD.
"""

def compute_detector_optimal_snr(
        waveform: TimeSeries, 
        psd: FrequencySeries,
        config: SimulationConfig,
        ) -> float:
    """
    Compute the expected optimal SNR. This is the PSD-weighted norm of the detector waveform,
    i.e., the matched-filter norm / expected optimal SNR of the waveform respect to the PSD (the optimal expected magnitude of the signal in that PSD). 
    

    Parameters
    ----------
    waveform : TimeSeries
        The waveform array containing the signal for the detector.
    psd : FrequencySeries
        The power spectral density array containing the noise for the detector. This is the PSD calculated in NoiseModel.
    config : SimulationConfig
        The simulation configuration object. With the delta_f and low_frequency_cutoff attributes used to compute the SNR.

    Returns
    -------
    float
        The "SNR".
    """

    if not isinstance(waveform, TimeSeries):
        raise TypeError("waveform must be a TimeSeries object.")
    if not isinstance(psd, FrequencySeries):
        raise TypeError("psd must be a FrequencySeries object.")
    
    if not np.all(np.isfinite(waveform.numpy())):
        raise ValueError("waveform must not contain NaN or Inf values.")
    if not np.all(np.isfinite(psd.numpy())):
        raise ValueError("psd must not contain NaN or Inf values.")

    waveform_fs = waveform.to_frequencyseries(config.delta_f) # Convert to frequency series

    if len(waveform_fs) != len(psd) or not(np.isclose(waveform_fs.delta_f, psd.delta_f)):
        raise ValueError("waveform and psd must have the same length and delta_f.")
    

    snr = sigma(
        htilde=waveform_fs,
        psd=psd,
        low_frequency_cutoff=config.low_frequency_cutoff,
        high_frequency_cutoff=None)

    return snr


def compute_network_snr(
        detectors_snrs: np.ndarray,
        ) -> float:
    """
    Compute the network SNR as the cuadratic norm of the detectors SNRs.
    
    Parameters
    ----------
    detectors_snrs : np.ndarray
        The array containing the SNRs of the detectors.

    Returns
    -------
    float
        The network SNR.
    """
    if not np.all(np.isfinite(detectors_snrs)):
        raise ValueError("detectors_snrs must not contain NaN or Inf values.")
    
    return float(np.linalg.norm(detectors_snrs))


def rescale_distance_for_target_network_snr(
        current_distance: float,
        current_network_snr: float,
        target_network_snr: float,
        ) -> float:
    """
    Rescale the distance of the binary compact object to achieve a target network SNR.

    Parameters
    ----------
    current_distance : float
        The current distance of the binary compact object.
    current_network_snr : float
        The current network SNR of the binary compact object.
    target_network_snr : float
        The target network SNR of the binary compact object.

    Returns
    -------
    new_distance : float
        The new rescaled distance of the binary compact object.
    """
    if current_network_snr <= 0:
        raise ValueError("current_network_snr must be positive.")
    if target_network_snr <= 0:
        raise ValueError("target_network_snr must be positive.")    
    if current_distance <= 0:
        raise ValueError("current_distance must be positive.")
    if not np.isfinite(current_network_snr):
        raise ValueError("current_network_snr must be finite.")
    if not np.isfinite(target_network_snr):
        raise ValueError("target_network_snr must be finite.")


    new_distance = current_distance * (current_network_snr / target_network_snr) 

    return new_distance