from __future__ import annotations

"""
GWOSC catalog access and metadata utilities.

Responsibilities
----------------
- Query GWOSC catalog API endpoints.
- Handle paginated catalog responses.
- Flatten GWOSC event metadata and published parameters.
- Resolve event/detector strain-file metadata and URLs.

This module does not:
- download or read strain HDF5 files;
- process detector strain;
- construct detector-frame LVK reference quantities.
"""

import numpy as np
import pandas as pd
import requests

from collections.abc import Sequence

from gwosc.locate import get_event_urls


GWOSC_API_BASE = "https://gwosc.org/api/v2"


def fetch_paginated_json(
    url: str,
    params: dict | None = None,
    timeout: float = 60,
) -> list[dict]:
    """
    Fetch all pages from a paginated GWOSC API endpoint.

    Returns
    -------
    list[dict]
        Concatenated dictionaries from all ``results`` fields.
    """
    results = []

    next_url = url

    next_params = (
        params.copy()
        if params is not None
        else None
    )

    while next_url is not None:
        response = requests.get(
            next_url,
            params=next_params,
            timeout=timeout,
        )

        response.raise_for_status()

        payload = response.json()

        if "results" not in payload:
            raise KeyError(
                f"No 'results' field in response "
                f"from {next_url}"
            )

        results.extend(
            payload["results"]
        )

        next_url = payload.get(
            "next",
            None,
        )

        # Pagination URLs returned by GWOSC already contain the
        # required query state. Do not reuse original parameters.
        next_params = None

    return results


def fetch_gwosc_catalog_events(
    catalog: str = "GWTC-3-confident",
    include_default_parameters: bool = True,
) -> list[dict]:
    """
    Fetch events from a GWOSC catalog.
    """
    url = (
        f"{GWOSC_API_BASE}"
        f"/catalogs/{catalog}/events"
    )

    params = {
        "format": "json",
    }

    if include_default_parameters:
        params[
            "include-default-parameters"
        ] = "true"

    return fetch_paginated_json(
        url,
        params=params,
    )


def parameters_list_to_dict(
    parameters,
) -> dict:
    """
    Flatten a GWOSC ``default_parameters`` list.

    For each parameter ``name`` the output contains:

        name_best
        name_lower_error
        name_upper_error
        name_lower
        name_upper
        name_unit
        name_is_lower_limit
        name_is_upper_limit

    Absolute bounds are reconstructed as:

        lower = best + lower_error
        upper = best + upper_error
    """
    out = {}

    if parameters is None:
        return out

    for parameter in parameters:
        name = parameter.get(
            "name"
        )

        if name is None:
            continue

        best = parameter.get(
            "best",
            np.nan,
        )

        lower_error = parameter.get(
            "lower_error",
            np.nan,
        )

        upper_error = parameter.get(
            "upper_error",
            np.nan,
        )

        out[
            f"{name}_best"
        ] = best

        out[
            f"{name}_lower_error"
        ] = lower_error

        out[
            f"{name}_upper_error"
        ] = upper_error

        try:
            out[
                f"{name}_lower"
            ] = best + lower_error
        except Exception:
            out[
                f"{name}_lower"
            ] = np.nan

        try:
            out[
                f"{name}_upper"
            ] = best + upper_error
        except Exception:
            out[
                f"{name}_upper"
            ] = np.nan

        out[
            f"{name}_unit"
        ] = parameter.get(
            "unit",
            "",
        )

        out[
            f"{name}_is_lower_limit"
        ] = parameter.get(
            "is_lower_limit",
            False,
        )

        out[
            f"{name}_is_upper_limit"
        ] = parameter.get(
            "is_upper_limit",
            False,
        )

    return out


def gwosc_events_to_parameter_df(
    events,
    catalog_name: str,
) -> pd.DataFrame:
    """
    Flatten GWOSC catalog events into one row per event.
    """
    rows = []

    for event in events:
        row = {
            "event": event.get(
                "name"
            ),
            "shortName": event.get(
                "shortName"
            ),
            "catalog": event.get(
                "catalog",
                catalog_name,
            ),
            "gps_time": event.get(
                "gps"
            ),
            "version": event.get(
                "version"
            ),
            "detectors": ",".join(
                event.get(
                    "detectors",
                    [],
                )
            ),
            "detail_url": event.get(
                "detail_url"
            ),
            "parameters_url": event.get(
                "parameters_url"
            ),
            "timelines_url": event.get(
                "timelines_url"
            ),
        }

        row.update(
            parameters_list_to_dict(
                event.get(
                    "default_parameters",
                    [],
                )
            )
        )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def has_required_detectors(
    detector_string,
    required: Sequence[str] = ("H1", "L1", "V1"),
) -> bool:
    """
    Check whether an event includes all required detectors.

    ``detector_string`` follows the flattened catalog representation,
    e.g. ``"H1,L1,V1"``.
    """
    detectors = {
        detector.strip()
        for detector
        in str(detector_string).split(",")
        if detector.strip()
    }

    return all(
        detector in detectors
        for detector in required
    )


def choose_best_4096s_hdf5_url(
    urls,
):
    """
    Select the preferred 4096-second HDF5 strain URL.

    Preference
    ----------
    1. HDF5 files only.
    2. 4096-second files only.
    3. Prefer R1 release files when several are available.
    4. Resolve remaining ambiguity deterministically by sorting.
    """
    if urls is None or len(urls) == 0:
        return None

    hdf5_urls = [
        url
        for url in urls
        if str(url).endswith(".hdf5")
    ]

    if not hdf5_urls:
        return None

    long_urls = [
        url
        for url in hdf5_urls
        if "-4096.hdf5" in str(url)
    ]

    if not long_urls:
        return None

    r1_urls = [
        url
        for url in long_urls
        if (
            "_R1-" in str(url)
            or "R1-" in str(url)
        )
    ]

    if r1_urls:
        return sorted(r1_urls)[0]

    return sorted(long_urls)[0]


def get_event_detector_url(
    event_name: str,
    detector: str,
    catalog: str = "GWTC-3-confident",
    sample_rate: int = 4096,
):
    """
    Resolve the preferred 4096-second HDF5 strain URL for one
    event/detector pair.
    """
    urls = get_event_urls(
        event=event_name,
        catalog=catalog,
        detector=detector,
        format="hdf5",
        sample_rate=sample_rate,
    )

    return choose_best_4096s_hdf5_url(
        urls
    )


def build_gwosc_urls_for_events(
    events_df: pd.DataFrame,
    catalog: str = "GWTC-3-confident",
    sample_rate: int = 4096,
):
    """
    Resolve strain URLs for all detectors of each catalog event.

    Returns
    -------
    urls_by_event
        Nested mapping:

            urls_by_event[event][detector] = url

        Only events for which every listed detector has a valid URL
        are retained.

    failed_df
        One row for each unresolved event/detector pair.
    """
    urls_by_event = {}
    failed_rows = []

    for _, row in events_df.iterrows():
        event_name = row["event"]

        detectors = [
            detector.strip()
            for detector
            in str(
                row["detectors"]
            ).split(",")
            if detector.strip()
        ]

        event_urls = {}
        complete = True

        for detector in detectors:
            url = get_event_detector_url(
                event_name=event_name,
                detector=detector,
                catalog=catalog,
                sample_rate=sample_rate,
            )

            if url is None:
                complete = False

                failed_rows.append(
                    {
                        "event": event_name,
                        "detector": detector,
                        "reason": (
                            "no_4096s_hdf5_url"
                        ),
                    }
                )
            else:
                event_urls[
                    detector
                ] = url

        if complete:
            urls_by_event[
                event_name
            ] = event_urls

    failed_df = pd.DataFrame(
        failed_rows
    )

    return (
        urls_by_event,
        failed_df,
    )