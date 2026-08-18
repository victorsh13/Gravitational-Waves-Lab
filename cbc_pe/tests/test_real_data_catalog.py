import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.real_data.catalog import (
    GWOSC_API_BASE,
    build_gwosc_urls_for_events,
    choose_best_4096s_hdf5_url,
    fetch_gwosc_catalog_events,
    fetch_paginated_json,
    get_event_detector_url,
    gwosc_events_to_parameter_df,
    has_required_detectors,
    parameters_list_to_dict,
)


class TestCatalogParameterParsing(unittest.TestCase):
    def test_parameters_list_to_dict(self):
        parameters = [
            {
                "name": "chirp_mass_source",
                "best": 25.0,
                "lower_error": -2.0,
                "upper_error": 3.0,
                "unit": "M_sun",
                "is_lower_limit": False,
                "is_upper_limit": False,
            }
        ]

        result = (
            parameters_list_to_dict(
                parameters
            )
        )

        self.assertEqual(
            result[
                "chirp_mass_source_best"
            ],
            25.0,
        )

        self.assertEqual(
            result[
                "chirp_mass_source_lower"
            ],
            23.0,
        )

        self.assertEqual(
            result[
                "chirp_mass_source_upper"
            ],
            28.0,
        )

        self.assertEqual(
            result[
                "chirp_mass_source_unit"
            ],
            "M_sun",
        )

    def test_none_parameters_returns_empty_dict(self):
        self.assertEqual(
            parameters_list_to_dict(
                None
            ),
            {},
        )

    def test_parameter_without_name_is_skipped(self):
        result = (
            parameters_list_to_dict(
                [
                    {
                        "best": 10.0,
                    }
                ]
            )
        )

        self.assertEqual(
            result,
            {},
        )

    def test_invalid_arithmetic_produces_nan_bound(self):
        parameters = [
            {
                "name": "x",
                "best": None,
                "lower_error": -1.0,
                "upper_error": 1.0,
            }
        ]

        result = (
            parameters_list_to_dict(
                parameters
            )
        )

        self.assertTrue(
            np.isnan(
                result["x_lower"]
            )
        )

        self.assertTrue(
            np.isnan(
                result["x_upper"]
            )
        )

    def test_events_are_flattened_to_dataframe(self):
        events = [
            {
                "name": "GWTEST",
                "shortName": "GWTEST",
                "gps": 1234567890.0,
                "version": 1,
                "detectors": [
                    "H1",
                    "L1",
                ],
                "default_parameters": [
                    {
                        "name": "chi_eff",
                        "best": 0.1,
                        "lower_error": -0.2,
                        "upper_error": 0.3,
                    }
                ],
            }
        ]

        df = (
            gwosc_events_to_parameter_df(
                events,
                catalog_name="TEST",
            )
        )

        self.assertIsInstance(
            df,
            pd.DataFrame,
        )

        self.assertEqual(
            len(df),
            1,
        )

        self.assertEqual(
            df.loc[
                0,
                "event",
            ],
            "GWTEST",
        )

        self.assertEqual(
            df.loc[
                0,
                "catalog",
            ],
            "TEST",
        )

        self.assertEqual(
            df.loc[
                0,
                "detectors",
            ],
            "H1,L1",
        )

        self.assertEqual(
            df.loc[
                0,
                "chi_eff_lower",
            ],
            -0.1,
        )

        self.assertEqual(
            df.loc[
                0,
                "chi_eff_upper",
            ],
            0.4,
        )


class TestCatalogAPI(unittest.TestCase):
    @patch(
        "src.real_data.catalog.requests.get"
    )
    def test_fetch_paginated_json_combines_pages(
        self,
        mock_get,
    ):
        first_response = Mock()

        first_response.json.return_value = {
            "results": [
                {"id": 1},
                {"id": 2},
            ],
            "next": (
                "https://gwosc.org/"
                "api/v2/example?page=2"
            ),
        }

        second_response = Mock()

        second_response.json.return_value = {
            "results": [
                {"id": 3},
            ],
            "next": None,
        }

        mock_get.side_effect = [
            first_response,
            second_response,
        ]

        result = fetch_paginated_json(
            "https://example.test/api",
            params={
                "format": "json",
            },
            timeout=10,
        )

        self.assertEqual(
            result,
            [
                {"id": 1},
                {"id": 2},
                {"id": 3},
            ],
        )

        self.assertEqual(
            mock_get.call_count,
            2,
        )

        first_response.raise_for_status\
            .assert_called_once()

        second_response.raise_for_status\
            .assert_called_once()

        first_call = (
            mock_get.call_args_list[0]
        )

        second_call = (
            mock_get.call_args_list[1]
        )

        self.assertEqual(
            first_call.kwargs[
                "params"
            ],
            {
                "format": "json",
            },
        )

        self.assertIsNone(
            second_call.kwargs[
                "params"
            ]
        )

    @patch(
        "src.real_data.catalog.requests.get"
    )
    def test_missing_results_field_raises(
        self,
        mock_get,
    ):
        response = Mock()

        response.json.return_value = {
            "next": None,
        }

        mock_get.return_value = (
            response
        )

        with self.assertRaisesRegex(
            KeyError,
            "No 'results' field",
        ):
            fetch_paginated_json(
                "https://example.test/api"
            )

    @patch(
        "src.real_data.catalog."
        "fetch_paginated_json"
    )
    def test_fetch_catalog_events_builds_expected_request(
        self,
        mock_fetch,
    ):
        mock_fetch.return_value = [
            {"name": "GWTEST"}
        ]

        result = (
            fetch_gwosc_catalog_events(
                catalog="TEST-CATALOG",
                include_default_parameters=True,
            )
        )

        self.assertEqual(
            result,
            [
                {"name": "GWTEST"}
            ],
        )

        expected_url = (
            f"{GWOSC_API_BASE}"
            "/catalogs/"
            "TEST-CATALOG/events"
        )

        mock_fetch.assert_called_once_with(
            expected_url,
            params={
                "format": "json",
                "include-default-parameters": (
                    "true"
                ),
            },
        )

    @patch(
        "src.real_data.catalog."
        "fetch_paginated_json"
    )
    def test_default_parameters_can_be_disabled(
        self,
        mock_fetch,
    ):
        mock_fetch.return_value = []

        fetch_gwosc_catalog_events(
            catalog="TEST",
            include_default_parameters=False,
        )

        _, kwargs = (
            mock_fetch.call_args
        )

        self.assertEqual(
            kwargs["params"],
            {
                "format": "json",
            },
        )


class TestDetectorAndURLResolution(
    unittest.TestCase
):
    def test_has_required_detectors(self):
        self.assertTrue(
            has_required_detectors(
                "H1,L1,V1"
            )
        )

        self.assertTrue(
            has_required_detectors(
                "V1,H1,L1"
            )
        )

        self.assertFalse(
            has_required_detectors(
                "H1,L1"
            )
        )

    def test_detector_check_accepts_subset_requirement(self):
        self.assertTrue(
            has_required_detectors(
                "H1,L1,V1",
                required=("H1", "L1"),
            )
        )

    def test_best_url_prefers_r1_4096_hdf5(self):
        urls = [
            (
                "https://example.test/"
                "H-H1_TEST-123-4096.hdf5"
            ),
            (
                "https://example.test/"
                "H-H1_O2_4KHZ_R1-123-4096.hdf5"
            ),
            (
                "https://example.test/"
                "H-H1_OTHER-123-32.hdf5"
            ),
        ]

        result = (
            choose_best_4096s_hdf5_url(
                urls
            )
        )

        self.assertEqual(
            result,
            urls[1],
        )

    def test_best_url_requires_4096_hdf5(self):
        self.assertIsNone(
            choose_best_4096s_hdf5_url(
                [
                    "x-32.hdf5",
                    "x-4096.txt",
                ]
            )
        )

    def test_best_url_handles_empty_input(self):
        self.assertIsNone(
            choose_best_4096s_hdf5_url(
                None
            )
        )

        self.assertIsNone(
            choose_best_4096s_hdf5_url(
                []
            )
        )

    @patch(
        "src.real_data.catalog."
        "get_event_urls"
    )
    def test_get_event_detector_url(
        self,
        mock_get_event_urls,
    ):
        expected = (
            "https://example.test/"
            "H-H1_O2_4KHZ_R1-1-4096.hdf5"
        )

        mock_get_event_urls.return_value = [
            expected
        ]

        result = get_event_detector_url(
            event_name="GWTEST",
            detector="H1",
            catalog="TEST",
            sample_rate=4096,
        )

        self.assertEqual(
            result,
            expected,
        )

        mock_get_event_urls\
            .assert_called_once_with(
                event="GWTEST",
                catalog="TEST",
                detector="H1",
                format="hdf5",
                sample_rate=4096,
            )

    @patch(
        "src.real_data.catalog."
        "get_event_detector_url"
    )
    def test_build_urls_keeps_only_complete_events(
        self,
        mock_get_url,
    ):
        events_df = pd.DataFrame(
            {
                "event": [
                    "GWGOOD",
                    "GWBAD",
                ],
                "detectors": [
                    "H1,L1,V1",
                    "H1,L1,V1",
                ],
            }
        )

        def fake_get_url(
            event_name,
            detector,
            catalog,
            sample_rate,
        ):
            if (
                event_name == "GWBAD"
                and detector == "V1"
            ):
                return None

            return (
                f"https://example.test/"
                f"{event_name}-{detector}"
                "-4096.hdf5"
            )

        mock_get_url.side_effect = (
            fake_get_url
        )

        urls, failed = (
            build_gwosc_urls_for_events(
                events_df,
                catalog="TEST",
                sample_rate=4096,
            )
        )

        self.assertIn(
            "GWGOOD",
            urls,
        )

        self.assertNotIn(
            "GWBAD",
            urls,
        )

        self.assertEqual(
            len(urls["GWGOOD"]),
            3,
        )

        self.assertEqual(
            len(failed),
            1,
        )

        self.assertEqual(
            failed.loc[
                0,
                "event",
            ],
            "GWBAD",
        )

        self.assertEqual(
            failed.loc[
                0,
                "detector",
            ],
            "V1",
        )

        self.assertEqual(
            failed.loc[
                0,
                "reason",
            ],
            "no_4096s_hdf5_url",
        )

if __name__ == "__main__":
    unittest.main()