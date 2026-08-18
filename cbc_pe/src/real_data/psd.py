from __future__ import annotations

"""
PSD estimation utilities for real gravitational-wave strain data.
"""

import numpy as np

from dataclasses import dataclass
from pycbc.psd import (
    interpolate,
    inverse_spectrum_truncation,
)
from collections.abc import Mapping, Sequence
from pycbc.types import FrequencySeries, TimeSeries
from src.real_data.gwosc_utils import (
    time_slice_is_finite,
)


@dataclass(frozen=True)
class PSDWindowPolicy:
    """
    Policy defining the off-source windows considered for real-data
    PSD estimation.

    All offsets are expressed in seconds relative to the event GPS time.
    """

    preferred_window: tuple[float, float]

    candidate_windows: tuple[
        tuple[float, float],
        ...
    ]


def estimate_offsource_psd(
    strain: TimeSeries,
    event_time: float,
    *,
    delta_f: float,
    sampling_frequency: float,
    target_flength: int,
    psd_start_offset: float = -512.0,
    psd_end_offset: float = -128.0,
    psd_segment_duration: float = 8.0,
    low_frequency_cutoff: float = 30.0,
    max_filter_duration: float = 0.5,
    trunc_method: str = "hann",
) -> FrequencySeries:
    """
    Estimate an off-source PSD from a long real strain segment.

    The PSD is estimated from the requested off-source time window,
    interpolated to the target frequency spacing, truncated in the
    inverse-spectrum domain, and adjusted to the target frequency length.

    Parameters
    ----------
    strain
        Long PyCBC TimeSeries containing the requested PSD window.

    event_time
        GPS reference time of the event.

    delta_f
        Target PSD frequency spacing.

    sampling_frequency
        Sampling frequency of the strain in Hz.

    target_flength
        Required final PSD length.

    psd_start_offset, psd_end_offset
        PSD-window bounds relative to event_time, in seconds.

    psd_segment_duration
        Segment duration used by PyCBC for PSD estimation.

    low_frequency_cutoff
        Low-frequency cutoff passed to inverse-spectrum truncation.

    max_filter_duration
        Maximum whitening-filter duration in seconds.

    trunc_method
        Truncation window passed to inverse_spectrum_truncation.

    Returns
    -------
    FrequencySeries
        PSD with the requested delta_f and target_flength.
    """
    event_time = float(event_time)
    delta_f = float(delta_f)
    sampling_frequency = float(sampling_frequency)
    target_flength = int(target_flength)

    psd_start_offset = float(psd_start_offset)
    psd_end_offset = float(psd_end_offset)
    psd_segment_duration = float(psd_segment_duration)
    low_frequency_cutoff = float(low_frequency_cutoff)
    max_filter_duration = float(max_filter_duration)

    if delta_f <= 0:
        raise ValueError("delta_f must be positive.")

    if sampling_frequency <= 0:
        raise ValueError(
            "sampling_frequency must be positive."
        )

    if target_flength <= 0:
        raise ValueError(
            "target_flength must be positive."
        )

    if psd_segment_duration <= 0:
        raise ValueError(
            "psd_segment_duration must be positive."
        )

    if max_filter_duration <= 0:
        raise ValueError(
            "max_filter_duration must be positive."
        )

    if psd_end_offset <= psd_start_offset:
        raise ValueError(
            "psd_end_offset must be greater than "
            "psd_start_offset."
        )

    psd_start = (
        event_time + psd_start_offset
    )

    psd_end = (
        event_time + psd_end_offset
    )

    available_start = float(
        strain.start_time
    )

    available_end = float(
        strain.end_time
    )

    if (
        psd_start < available_start
        or psd_end > available_end
    ):
        raise ValueError(
            f"PSD window [{psd_start}, {psd_end}] "
            f"outside available "
            f"[{available_start}, {available_end}]"
        )

    psd_data = strain.time_slice(
        psd_start,
        psd_end,
    )

    if len(psd_data) == 0:
        raise ValueError(
            "PSD window produced an empty TimeSeries."
        )

    if not np.all(
        np.isfinite(
            psd_data.numpy()
        )
    ):
        raise ValueError(
            "PSD input window contains non-finite values."
        )

    psd = psd_data.psd(
        psd_segment_duration
    )

    psd = interpolate(
        psd,
        delta_f,
    )

    max_filter_len = int(
        round(
            max_filter_duration
            * sampling_frequency
        )
    )

    psd = inverse_spectrum_truncation(
        psd,
        max_filter_len=max_filter_len,
        low_frequency_cutoff=low_frequency_cutoff,
        trunc_method=trunc_method,
    )

    if len(psd) > target_flength:
        psd = psd[:target_flength]

    elif len(psd) < target_flength:
        raise ValueError(
            f"PSD too short: "
            f"{len(psd)} < {target_flength}"
        )

    if not np.all(
        np.isfinite(
            psd.numpy()
        )
    ):
        raise ValueError(
            "PSD contains non-finite values."
        )

    return psd

def psd_window_is_available_and_finite(
    raw_strains: Mapping,
    event_time: float,
    psd_window: tuple[float, float],
) -> bool:
    """
    Check whether an off-source PSD window is available and finite
    for every detector.

    Parameters
    ----------
    raw_strains
        Mapping from detector name to PyCBC TimeSeries.

    event_time
        GPS reference time of the event.

    psd_window
        Pair ``(start_offset, end_offset)`` in seconds relative
        to ``event_time``.

    Returns
    -------
    bool
        True only if the complete window exists and contains finite
        data for every detector.
    """
    if len(psd_window) != 2:
        raise ValueError(
            "psd_window must contain exactly "
            "(start_offset, end_offset)."
        )

    start_offset = float(
        psd_window[0]
    )

    end_offset = float(
        psd_window[1]
    )

    if end_offset <= start_offset:
        return False

    start = (
        float(event_time)
        + start_offset
    )

    end = (
        float(event_time)
        + end_offset
    )

    if len(raw_strains) == 0:
        return False

    return all(
        time_slice_is_finite(
            ts,
            start,
            end,
        )
        for ts in raw_strains.values()
    )


def select_valid_psd_window(
    raw_strains: Mapping,
    event_time: float,
    *,
    preferred_window: tuple[float, float],
    candidate_windows: Sequence[
        tuple[float, float]
    ],
) -> tuple[float, float]:
    """
    Select the first valid finite off-source PSD window.

    Selection priority is explicit:

    1. ``preferred_window``.
    2. Remaining ``candidate_windows`` in the supplied order.

    The function contains no hard-coded M10 window policy. The caller
    supplies that scientific policy explicitly.
    """
    preferred_window = tuple(
        float(x)
        for x in preferred_window
    )

    windows_to_try = [
        preferred_window
    ]

    for window in candidate_windows:
        window = tuple(
            float(x)
            for x in window
        )

        if window not in windows_to_try:
            windows_to_try.append(
                window
            )

    for window in windows_to_try:
        if psd_window_is_available_and_finite(
            raw_strains=raw_strains,
            event_time=event_time,
            psd_window=window,
        ):
            return window

    raise ValueError(
        "No valid finite PSD window found "
        "for all detectors."
    )