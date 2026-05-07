import numpy as np

from pycbc.filter.matchedfilter import sigma
from pycbc.types.timeseries import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from dataclasses import dataclass

from .config import SimulationConfig



@dataclass(frozen=True)
class SNRRescalingDecision:
    should_rescale: bool
    current_network_snr: float
    target_network_snr: float
    old_distance: float
    new_distance: float
    reason: str


def compute_detector_optimal_snr(
    signal_segment: TimeSeries,
    psd: FrequencySeries,
    config: SimulationConfig,
) -> float:
    """
    Compute the optimal matched-filter SNR of a detector signal segment.

    signal_segment must have the same duration and delta_t as the configured
    analysis segment. It may contain zeros outside the actual injected signal.

    Parameters
    ----------
    signal_segment: TimeSeries



    """

    if not isinstance(signal_segment, TimeSeries):
        raise TypeError("signal_segment must be a TimeSeries object.")

    if not isinstance(psd, FrequencySeries):
        raise TypeError("psd must be a FrequencySeries object.")

    if len(signal_segment) != config.length:
        raise ValueError(
            f"signal_segment length mismatch: got {len(signal_segment)}, "
            f"expected {config.length}."
        )

    if signal_segment.delta_t != config.delta_t:
        raise ValueError(
            f"signal_segment delta_t mismatch: got {signal_segment.delta_t}, "
            f"expected {config.delta_t}."
        )

    if len(psd) != config.flength:
        raise ValueError(
            f"PSD length mismatch: got {len(psd)}, expected {config.flength}."
        )

    if psd.delta_f != config.delta_f:
        raise ValueError(
            f"PSD delta_f mismatch: got {psd.delta_f}, expected {config.delta_f}."
        )

    if not np.all(np.isfinite(signal_segment.numpy())):
        raise ValueError("signal_segment must not contain NaN or Inf values.")

    if not np.all(np.isfinite(psd.numpy())):
        raise ValueError("psd must not contain NaN or Inf values.")

    signal_fs = signal_segment.to_frequencyseries()

    if len(signal_fs) != len(psd) or not np.isclose(signal_fs.delta_f, psd.delta_f):
        raise ValueError(
            "signal_segment and psd are incompatible after FFT. "
            f"signal_fs_len={len(signal_fs)}, psd_len={len(psd)}, "
            f"signal_delta_f={signal_fs.delta_f}, psd_delta_f={psd.delta_f}."
        )

    snr = sigma(
        htilde=signal_fs,
        psd=psd,
        low_frequency_cutoff=config.low_frequency_cutoff,
        high_frequency_cutoff=None,
    )

    return float(snr)


def compute_network_snr(detector_snrs: dict[str, float] | np.ndarray) -> float:
    if isinstance(detector_snrs, dict):
        values = np.asarray(list(detector_snrs.values()), dtype=float)
    else:
        values = np.asarray(detector_snrs, dtype=float)

    if values.ndim != 1:
        raise ValueError("detector_snrs must be one-dimensional.")

    if len(values) == 0:
        raise ValueError("detector_snrs cannot be empty.")

    if not np.all(np.isfinite(values)):
        raise ValueError("detector_snrs must not contain NaN or Inf values.")

    if np.any(values < 0):
        raise ValueError("detector_snrs must be non-negative.")

    return float(np.linalg.norm(values))


def compute_network_optimal_snr(
    signal_segments: dict[str, TimeSeries],
    psds: dict[str, FrequencySeries],
    config: SimulationConfig,
) -> tuple[dict[str, float], float]:
    """
    Compute detector and network optimal SNRs for a projected signal network.

    Each signal segment must be a fixed-duration zero-padded segment with
    length config.length.
    """

    if set(signal_segments.keys()) != set(psds.keys()):
        raise ValueError(
            "signal_segments and psds must contain the same detectors. "
            f"signal_detectors={set(signal_segments.keys())}, "
            f"psd_detectors={set(psds.keys())}"
        )

    detector_snrs = {
        detector: compute_detector_optimal_snr(
            signal_segment=signal_segments[detector],
            psd=psds[detector],
            config=config,
        )
        for detector in signal_segments
    }

    network_snr = compute_network_snr(detector_snrs)

    return detector_snrs, network_snr



def rescale_distance_for_target_network_snr(
    current_distance: float,
    current_network_snr: float,
    target_network_snr: float,
) -> float:
    if not np.isfinite(current_distance):
        raise ValueError("current_distance must be finite.")

    if not np.isfinite(current_network_snr):
        raise ValueError("current_network_snr must be finite.")

    if not np.isfinite(target_network_snr):
        raise ValueError("target_network_snr must be finite.")

    if current_distance <= 0:
        raise ValueError("current_distance must be positive.")

    if current_network_snr <= 0:
        raise ValueError("current_network_snr must be positive.")

    if target_network_snr <= 0:
        raise ValueError("target_network_snr must be positive.")

    return float(current_distance * current_network_snr / target_network_snr)


def decide_distance_rescaling(
    current_distance: float,
    current_network_snr: float,
    target_network_snr_range: tuple[float, float] | None,
    rng: np.random.Generator,
) -> SNRRescalingDecision:
    if not np.isfinite(current_distance):
        raise ValueError("current_distance must be finite.")

    if not np.isfinite(current_network_snr):
        raise ValueError("current_network_snr must be finite.")

    if current_distance <= 0:
        raise ValueError("current_distance must be positive.")

    if current_network_snr <= 0:
        raise ValueError("current_network_snr must be positive.")

    if target_network_snr_range is None:
        return SNRRescalingDecision(
            should_rescale=False,
            current_network_snr=current_network_snr,
            target_network_snr=current_network_snr,
            old_distance=current_distance,
            new_distance=current_distance,
            reason="target_network_snr_range_is_none",
        )

    low, high = target_network_snr_range

    if low <= 0 or high <= 0:
        raise ValueError("target_network_snr_range values must be positive.")

    if low > high:
        raise ValueError("target_network_snr_range must be ordered as (low, high).")

    if low <= current_network_snr <= high:
        return SNRRescalingDecision(
            should_rescale=False,
            current_network_snr=current_network_snr,
            target_network_snr=current_network_snr,
            old_distance=current_distance,
            new_distance=current_distance,
            reason="already_within_target_range",
        )

    target_network_snr = float(rng.uniform(low, high))

    new_distance = rescale_distance_for_target_network_snr(
        current_distance=current_distance,
        current_network_snr=current_network_snr,
        target_network_snr=target_network_snr,
    )

    return SNRRescalingDecision(
        should_rescale=True,
        current_network_snr=current_network_snr,
        target_network_snr=target_network_snr,
        old_distance=current_distance,
        new_distance=new_distance,
        reason="outside_target_range",
    )

def validate_snr_rescaling(
    final_network_snr: float,
    target_network_snr: float,
    relative_tolerance: float,
) -> None:
    if not np.isfinite(final_network_snr):
        raise ValueError("final_network_snr must be finite.")

    if not np.isfinite(target_network_snr):
        raise ValueError("target_network_snr must be finite.")

    if final_network_snr <= 0:
        raise ValueError("final_network_snr must be positive.")

    if target_network_snr <= 0:
        raise ValueError("target_network_snr must be positive.")

    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative.")

    relative_error = abs(final_network_snr - target_network_snr) / target_network_snr

    if relative_error > relative_tolerance:
        raise ValueError(
            "SNR rescaling validation failed. "
            f"final_network_snr={final_network_snr}, "
            f"target_network_snr={target_network_snr}, "
            f"relative_error={relative_error}, "
            f"relative_tolerance={relative_tolerance}."
        )