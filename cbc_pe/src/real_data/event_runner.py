"""
Real-event execution configuration and run-status utilities.

This module contains reusable orchestration helpers for running the
real-data inference pipeline.

It does not implement strain processing, PSD estimation, model inference
or LVK reference construction; those responsibilities live in the
corresponding real_data modules.

event_cfg
    = which event I want to process

RealEventRunContext
    = with which pipeline/model I want to process it
"""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.hdf5_batch_dataset import (
    normalize_batch_per_sample_per_detector_zscore,
)

from src.real_data.gwosc_utils import (
    download_if_needed,
    read_gwosc_hdf5_as_pycbc_timeseries,
)

from src.real_data.inference import (
    predict_real_with_embeddings,
)

from src.real_data.psd import (
    select_valid_psd_window,
)

from src.real_data.signal_processing import (
    build_real_input_like_training,
    event_window_is_available_and_finite,
)


@dataclass(frozen=True)
class RealEventRunContext:
    """
    Dependencies and processing parameters required to run real-event
    inference.

    This separates stable event configuration from experiment/runtime
    dependencies such as the trained model, processing chain and
    calibration objects.
    """

    detector_order: Sequence[str]

    gwosc_urls: Mapping
    gwosc_cache_dir: str

    final_duration: float
    final_length: int
    processing_length: int

    context_start_samples: int
    context_end_samples: int

    sampling_frequency: float
    processing_delta_f: float
    processing_flength: int

    psd_candidate_windows: Sequence[
        tuple[float, float]
    ]

    processor: object

    input_normalization_eps: float

    model: object
    device: object

    y_mean: np.ndarray
    y_std: np.ndarray
    label_names: Sequence[str]

    apply_intervals: Callable


def run_single_event(
    event_cfg: dict,
    context: RealEventRunContext,
    *,
    final_policy: str = "conservative",
) -> dict:
    """
    Run real-data model inference and calibrated intervals for one event.
    """
    event_name = event_cfg["event"]
    gps_time = float(
        event_cfg["gps_time"]
    )

    detectors = list(
        event_cfg["detectors"]
    )

    center_offset = float(
        event_cfg.get(
            "center_offset",
            0.0,
        )
    )

    psd_window = tuple(
        float(x)
        for x in event_cfg.get(
            "psd_window",
            (-1024.0, -640.0),
        )
    )

    psd_segment_duration = float(
        event_cfg.get(
            "psd_segment_duration",
            8.0,
        )
    )

    detector_order = list(
        context.detector_order
    )

    if detectors != detector_order:
        raise ValueError(
            "Detector order must match training order "
            f"{detector_order}; got {detectors}."
        )

    if event_name not in context.gwosc_urls:
        raise KeyError(
            f"{event_name} not found in GWOSC URLs."
        )

    raw_strains_long = {}

    for detector in detectors:
        event_urls = context.gwosc_urls[
            event_name
        ]

        if detector not in event_urls:
            raise KeyError(
                f"{event_name}/{detector} not found "
                "in GWOSC URLs."
            )

        local_path = download_if_needed(
            event_urls[detector],
            cache_dir=(
                context.gwosc_cache_dir
            ),
        )

        raw_strains_long[
            detector
        ] = (
            read_gwosc_hdf5_as_pycbc_timeseries(
                local_path
            )
        )

    center_time = (
        gps_time + center_offset
    )

    event_ok, event_message = (
        event_window_is_available_and_finite(
            raw_strains=(
                raw_strains_long
            ),
            center_time=center_time,
            final_duration=(
                context.final_duration
            ),
            context_start_samples=(
                context.context_start_samples
            ),
            context_end_samples=(
                context.context_end_samples
            ),
            sampling_frequency=(
                context.sampling_frequency
            ),
        )
    )

    if not event_ok:
        raise ValueError(
            f"Invalid event window: "
            f"{event_message}"
        )

    selected_psd_window = (
        select_valid_psd_window(
            raw_strains=(
                raw_strains_long
            ),
            event_time=center_time,
            preferred_window=(
                psd_window
            ),
            candidate_windows=(
                context.psd_candidate_windows
            ),
        )
    )

    (
        psd_start_offset,
        psd_end_offset,
    ) = selected_psd_window

    (
        X_real_raw,
        processed,
        psds,
        real_segments,
        metadata,
    ) = build_real_input_like_training(
        raw_strains=(
            raw_strains_long
        ),
        detectors=detector_order,
        center_time=center_time,
        processor=context.processor,
        expected_detector_order=(
            detector_order
        ),
        final_duration=(
            context.final_duration
        ),
        final_length=(
            context.final_length
        ),
        processing_length=(
            context.processing_length
        ),
        context_start_samples=(
            context.context_start_samples
        ),
        context_end_samples=(
            context.context_end_samples
        ),
        sampling_frequency=(
            context.sampling_frequency
        ),
        psd_delta_f=(
            context.processing_delta_f
        ),
        psd_target_flength=(
            context.processing_flength
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
        psd_low_frequency_cutoff=30.0,
        psd_max_filter_duration=0.5,
    )

    X_real_z = (
        normalize_batch_per_sample_per_detector_zscore(
            X_real_raw,
            eps=(
                context.input_normalization_eps
            ),
        )
    )

    (
        pred_real_std,
        pred_real_phys,
        emb_real,
    ) = predict_real_with_embeddings(
        model=context.model,
        X_real=X_real_z,
        device=context.device,
        y_mean=context.y_mean,
        y_std=context.y_std,
        batch_size=1,
    )

    point_df = pd.DataFrame(
        {
            "event": (
                [event_name]
                * len(
                    context.label_names
                )
            ),
            "label": list(
                context.label_names
            ),
            "pred_std": (
                pred_real_std[0]
            ),
            "pred_phys": (
                pred_real_phys[0]
            ),
        }
    )

    interval_df = (
        context.apply_intervals(
            event_name=event_name,
            final_policy=final_policy,
            pred_real_std=(
                pred_real_std
            ),
            emb_real=emb_real,
            pred_real_phys=(
                pred_real_phys
            ),
        )
    )

    return {
        "event": event_name,
        "config": event_cfg,
        "selected_psd_window": (
            selected_psd_window
        ),
        "meta": metadata,
        "raw_strains_long": (
            raw_strains_long
        ),
        "processed": processed,
        "psds": psds,
        "real_segments": (
            real_segments
        ),
        "X_real_raw": X_real_raw,
        "X_real_z": X_real_z,
        "pred_real_std": (
            pred_real_std
        ),
        "pred_real_phys": (
            pred_real_phys
        ),
        "emb_real": emb_real,
        "point_df": point_df,
        "interval_df": interval_df,
    }


def events_df_to_event_configs(
    events_df: pd.DataFrame,
    gwosc_urls_dict: dict,
    *,
    detector_order: Sequence[str],
    psd_window: tuple[float, float] = (-1024.0, -640.0),
    center_offset: float = 0.0,
    psd_segment_duration: float = 8.0,
) -> tuple[list[dict], pd.DataFrame]:
    """
    Convert a candidate-event table into real-event run configurations.

    Only events with URLs for every detector in ``detector_order`` are
    retained.

    The returned configuration contains only parameters actually consumed
    by the current single-event runner.
    """
    detector_order = list(detector_order)

    if not detector_order:
        raise ValueError(
            "detector_order must contain at least one detector."
        )

    if len(psd_window) != 2:
        raise ValueError(
            "psd_window must contain exactly two offsets."
        )

    configs = []
    skipped = []

    for _, row in events_df.iterrows():
        event_name = row["event"]

        if event_name not in gwosc_urls_dict:
            skipped.append(
                {
                    "event": event_name,
                    "reason": (
                        "missing_event_in_gwosc_urls_dict"
                    ),
                }
            )
            continue

        available_detectors = set(
            gwosc_urls_dict[
                event_name
            ].keys()
        )

        missing_detectors = [
            detector
            for detector in detector_order
            if detector not in available_detectors
        ]

        if missing_detectors:
            skipped.append(
                {
                    "event": event_name,
                    "reason": (
                        "missing_required_detector_urls"
                    ),
                    "missing_detectors": ",".join(
                        missing_detectors
                    ),
                }
            )
            continue

        configs.append(
            {
                "event": event_name,
                "gps_time": float(
                    row["gps_time"]
                ),
                "detectors": (
                    detector_order.copy()
                ),
                "psd_window": (
                    float(psd_window[0]),
                    float(psd_window[1]),
                ),
                "center_offset": float(
                    center_offset
                ),
                "psd_segment_duration": float(
                    psd_segment_duration
                ),
            }
        )

    return (
        configs,
        pd.DataFrame(skipped),
    )


def classify_failure_reason(
    error_text,
) -> str:
    """
    Map known real-event pipeline failures to stable reason labels.
    """
    error_text = str(error_text)

    if (
        "event processing window contains NaN/Inf"
        in error_text
    ):
        return "nonfinite_event_window"

    if (
        "Invalid event window"
        in error_text
        and "NaN/Inf" in error_text
    ):
        return "nonfinite_event_window"

    if (
        "No valid finite PSD window"
        in error_text
    ):
        return "no_valid_psd_window"

    if (
        "PSD contains non-finite values"
        in error_text
    ):
        return "nonfinite_psd"

    if (
        "outside available"
        in error_text
    ):
        return "window_outside_available_data"

    if (
        "truncated file"
        in error_text
    ):
        return "corrupted_or_truncated_hdf5"

    if (
        "not found in GWOSC_URLS_ACTIVE"
        in error_text
    ):
        return "missing_url"

    if (
        "Downloaded file is not a valid GWOSC HDF5"
        in error_text
    ):
        return "invalid_hdf5_download"

    return "other"
