import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.real_data.gwosc_utils import (
    get_timeseries_bounds,
    is_valid_hdf5,
    read_gwosc_hdf5_as_pycbc_timeseries,
    time_slice_is_finite,
)


class TestGWOSCUtilities(unittest.TestCase):
    def _create_test_hdf5(
        self,
        path,
        *,
        gps_start=1000.0,
        duration=4.0,
        n_samples=16,
    ):
        strain = np.arange(
            n_samples,
            dtype=np.float64,
        )

        with h5py.File(path, "w") as f:
            strain_group = f.create_group(
                "strain"
            )

            strain_group.create_dataset(
                "Strain",
                data=strain,
            )

            meta_group = f.create_group(
                "meta"
            )

            meta_group.create_dataset(
                "GPSstart",
                data=gps_start,
            )

            meta_group.create_dataset(
                "Duration",
                data=duration,
            )

        return strain


    def test_valid_hdf5_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.hdf5"

            self._create_test_hdf5(path)

            self.assertTrue(
                is_valid_hdf5(path)
            )


    def test_invalid_hdf5_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.hdf5"

            path.write_bytes(
                b"not an hdf5 file"
            )

            self.assertFalse(
                is_valid_hdf5(path)
            )


    def test_missing_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.hdf5"

            self.assertFalse(
                is_valid_hdf5(path)
            )


    def test_reads_gwosc_hdf5_as_timeseries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.hdf5"

            strain = self._create_test_hdf5(
                path,
                gps_start=1000.0,
                duration=4.0,
                n_samples=16,
            )

            ts = (
                read_gwosc_hdf5_as_pycbc_timeseries(
                    path
                )
            )

            self.assertEqual(
                len(ts),
                16,
            )

            self.assertAlmostEqual(
                float(ts.start_time),
                1000.0,
            )

            self.assertAlmostEqual(
                float(ts.delta_t),
                0.25,
            )

            np.testing.assert_array_equal(
                ts.numpy(),
                strain,
            )


    def test_timeseries_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.hdf5"

            self._create_test_hdf5(
                path,
                gps_start=1000.0,
                duration=4.0,
                n_samples=16,
            )

            ts = (
                read_gwosc_hdf5_as_pycbc_timeseries(
                    path
                )
            )

            start, end = get_timeseries_bounds(
                ts
            )

            self.assertAlmostEqual(
                start,
                1000.0,
            )

            self.assertAlmostEqual(
                end,
                float(ts.end_time),
            )


    def test_finite_time_slice_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.hdf5"

            self._create_test_hdf5(
                path,
                gps_start=1000.0,
                duration=4.0,
                n_samples=16,
            )

            ts = (
                read_gwosc_hdf5_as_pycbc_timeseries(
                    path
                )
            )

            self.assertTrue(
                time_slice_is_finite(
                    ts,
                    1001.0,
                    1002.0,
                )
            )


    def test_out_of_bounds_slice_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.hdf5"

            self._create_test_hdf5(
                path,
                gps_start=1000.0,
                duration=4.0,
                n_samples=16,
            )

            ts = (
                read_gwosc_hdf5_as_pycbc_timeseries(
                    path
                )
            )

            self.assertFalse(
                time_slice_is_finite(
                    ts,
                    999.0,
                    1002.0,
                )
            )


if __name__ == "__main__":
    unittest.main()
