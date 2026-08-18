from __future__ import annotations

"""
Low-level GWOSC strain-file utilities.

Responsibilities
----------------
- Validate cached HDF5 files.
- Download GWOSC strain files.
- Read GWOSC strain into PyCBC TimeSeries objects.
- Inspect TimeSeries temporal bounds and finite-data availability.

Catalog metadata and LVK parameter transformations live in separate modules.
"""

from pathlib import Path
import urllib.request

import h5py
import numpy as np
from pycbc.types import TimeSeries as PyCBCTimeSeries


def is_valid_hdf5(path: str | Path) -> bool:
    """
    Return True if ``path`` exists and can be opened as an HDF5 file.
    """
    path = Path(path)

    if not path.is_file():
        return False

    try:
        with h5py.File(path, "r") as f:
            # Touch the root keys so obviously malformed/truncated files fail
            # during validation rather than later in the analysis.
            list(f.keys())
    except (OSError, ValueError):
        return False

    return True


def download_if_needed(
    url: str,
    cache_dir: str | Path,
    force: bool = False,
) -> Path:
    """
    Download a GWOSC HDF5 file if it is missing or invalid.

    Existing valid files are reused. Invalid or corrupted cached files are
    removed and downloaded again.

    Parameters
    ----------
    url : str
        Remote GWOSC HDF5 URL.

    cache_dir : str or Path
        Local cache directory.

    force : bool
        If True, remove an existing cached file and download it again.

    Returns
    -------
    Path
        Local path to the validated HDF5 file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]

    if not filename:
        raise ValueError(
            f"Could not determine filename from URL: {url!r}"
        )

    local_path = cache_dir / filename

    if force and local_path.exists():
        print(
            f"Force removing cached file: {filename}"
        )
        local_path.unlink()

    if local_path.exists():
        if is_valid_hdf5(local_path):
            print(
                f"Using cached file: {filename}"
            )
            return local_path

        print(
            f"Removing corrupted cached file: {filename}"
        )
        local_path.unlink()

    print(
        f"Downloading {filename}..."
    )

    urllib.request.urlretrieve(
        url,
        local_path,
    )

    if not is_valid_hdf5(local_path):
        raise OSError(
            "Downloaded file is not a valid GWOSC HDF5 file: "
            f"{local_path}"
        )

    return local_path


def read_gwosc_hdf5_as_pycbc_timeseries(
    path: str | Path,
) -> PyCBCTimeSeries:
    """
    Read a GWOSC HDF5 strain file as a PyCBC TimeSeries.
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        strain = (
            f["strain"]["Strain"][:]
            .astype(np.float64)
        )

        gps_start = float(
            f["meta"]["GPSstart"][()]
        )

        duration_file = float(
            f["meta"]["Duration"][()]
        )

        delta_t = (
            duration_file / len(strain)
        )

    return PyCBCTimeSeries(
        strain,
        delta_t=delta_t,
        epoch=gps_start,
    )


def get_timeseries_bounds(
    ts: PyCBCTimeSeries,
) -> tuple[float, float]:
    """
    Return the available [start, end] GPS interval of a TimeSeries.
    """
    return (
        float(ts.start_time),
        float(ts.end_time),
    )


def time_slice_is_finite(
    ts: PyCBCTimeSeries,
    start: float,
    end: float,
) -> bool:
    """
    Check that a requested TimeSeries interval is available and finite.
    """
    available_start, available_end = (
        get_timeseries_bounds(ts)
    )

    if start < available_start:
        return False

    if end > available_end:
        return False

    if end <= start:
        return False

    try:
        segment = ts.time_slice(
            start,
            end,
        )
    except (ValueError, IndexError):
        return False

    if len(segment) == 0:
        return False

    return bool(
        np.all(
            np.isfinite(
                segment.numpy()
            )
        )
    )
