from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.real_data.gwosc_utils import time_slice_is_finite
from src.real_data.psd import estimate_offsource_psd


def event_window_is_available_and_finite(
    raw_strains: Mapping,
    center_time: float,
    *,
    final_duration: float,
    context_start_samples: int,
    context_end_samples: int,
    sampling_frequency: float,
) -> tuple[bool, str]:
    """
    Check whether the complete processing window around an event is
    available and finite for every detector.

    The processing window consists of the final output window plus the
    left/right processing context required by the signal-processing chain.
    """
    center_time = float(center_time)
    final_duration = float(final_duration)
    sampling_frequency = float(sampling_frequency)

    if final_duration <= 0:
        raise ValueError(
            "final_duration must be positive."
        )

    if sampling_frequency <= 0:
        raise ValueError(
            "sampling_frequency must be positive."
        )

    if context_start_samples < 0:
        raise ValueError(
            "context_start_samples must be non-negative."
        )

    if context_end_samples < 0:
        raise ValueError(
            "context_end_samples must be non-negative."
        )

    if len(raw_strains) == 0:
        return False, "no detector strains provided"

    output_start = (
        center_time
        - final_duration / 2.0
    )

    output_end = (
        center_time
        + final_duration / 2.0
    )

    processing_start = (
        output_start
        - context_start_samples
        / sampling_frequency
    )

    processing_end = (
        output_end
        + context_end_samples
        / sampling_frequency
    )

    for detector, ts in raw_strains.items():
        ts_start = float(ts.start_time)
        ts_end = float(ts.end_time)

        if (
            processing_start < ts_start
            or processing_end > ts_end
        ):
            return (
                False,
                f"{detector}: event processing window "
                "outside available data",
            )

        if not time_slice_is_finite(
            ts,
            processing_start,
            processing_end,
        ):
            return (
                False,
                f"{detector}: event processing window "
                "contains NaN/Inf",
            )

    return True, "ok"


def build_real_input_like_training(
    *,
    raw_strains: Mapping,
    detectors: Sequence[str],
    center_time: float,
    processor,
    expected_detector_order: Sequence[str],
    final_duration: float,
    final_length: int,
    processing_length: int,
    context_start_samples: int,
    context_end_samples: int,
    sampling_frequency: float,
    psd_delta_f: float,
    psd_target_flength: int,
    psd_start_offset: float = -512.0,
    psd_end_offset: float = -128.0,
    psd_segment_duration: float = 8.0,
    psd_low_frequency_cutoff: float = 30.0,
    psd_max_filter_duration: float = 0.5,
):
    """
    Build one real detector-network input using the same signal-processing
    contract used for model training.

    The function:
      1. extracts the event processing window with context;
      2. estimates one off-source PSD per detector;
      3. applies the supplied SignalProcessor to the detector network;
      4. stacks processed detector channels into shape (1, C, T).

    No model-specific input normalization is applied here.
    """
    detectors = list(detectors)
    expected_detector_order = list(
        expected_detector_order
    )

    if detectors != expected_detector_order:
        raise ValueError(
            "Detector order mismatch: "
            f"expected {expected_detector_order}, "
            f"got {detectors}."
        )

    if len(detectors) == 0:
        raise ValueError(
            "detectors must contain at least one detector."
        )

    missing = [
        detector
        for detector in detectors
        if detector not in raw_strains
    ]

    if missing:
        raise KeyError(
            f"Missing detector strains: {missing}"
        )

    center_time = float(center_time)
    final_duration = float(final_duration)
    sampling_frequency = float(
        sampling_frequency
    )

    output_start = (
        center_time
        - final_duration / 2.0
    )

    output_end = (
        center_time
        + final_duration / 2.0
    )

    processing_start = (
        output_start
        - context_start_samples
        / sampling_frequency
    )

    processing_end = (
        output_end
        + context_end_samples
        / sampling_frequency
    )

    real_segments = {}

    for detector in detectors:
        segment = raw_strains[
            detector
        ].time_slice(
            processing_start,
            processing_end,
        )

        if len(segment) != processing_length:
            raise ValueError(
                f"{detector}: expected "
                f"{processing_length} samples, "
                f"got {len(segment)}"
            )

        if not np.all(
            np.isfinite(
                segment.numpy()
            )
        ):
            raise ValueError(
                f"{detector}: processing segment "
                "contains non-finite values."
            )

        real_segments[detector] = segment

    psds = {}

    for detector in detectors:
        psds[detector] = (
            estimate_offsource_psd(
                strain=raw_strains[detector],
                event_time=center_time,
                delta_f=psd_delta_f,
                sampling_frequency=(
                    sampling_frequency
                ),
                target_flength=(
                    psd_target_flength
                ),
                psd_start_offset=(
                    psd_start_offset
                ),
                psd_end_offset=(
                    psd_end_offset
                ),
                psd_segment_duration=(
                    psd_segment_duration
                ),
                low_frequency_cutoff=(
                    psd_low_frequency_cutoff
                ),
                max_filter_duration=(
                    psd_max_filter_duration
                ),
            )
        )

    processed = processor.process_network(
        strains=real_segments,
        psds=psds,
    )

    X = np.stack(
        [
            processed[detector].numpy()
            for detector in detectors
        ],
        axis=0,
    ).astype(np.float32)

    X = X[None, :, :]

    expected_shape = (
        1,
        len(detectors),
        int(final_length),
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"Bad X shape: {X.shape}; "
            f"expected {expected_shape}"
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains non-finite values."
        )

    metadata = {
        "center_time": center_time,
        "output_start": float(
            output_start
        ),
        "output_end": float(
            output_end
        ),
        "processing_start": float(
            processing_start
        ),
        "processing_end": float(
            processing_end
        ),
        "psd_start_offset": float(
            psd_start_offset
        ),
        "psd_end_offset": float(
            psd_end_offset
        ),
        "psd_segment_duration": float(
            psd_segment_duration
        ),
    }

    return (
        X,
        processed,
        psds,
        real_segments,
        metadata,
    )
