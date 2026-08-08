import unittest

import pandas as pd

from src.conformal.selection import (
    get_base_candidates,
    select_conservative,
    select_efficient,
)


def make_candidate(
    *,
    candidate_id,
    label="chirp_mass",
    global_within_2sigma=True,
    min_count_per_bin=500,
    n_bins_under_2sigma=0,
    under_bin_fraction_2sigma=0.0,
    max_undercoverage_gap=0.0,
    global_median_width_phys=10.0,
    global_tail_miss_imbalance=0.01,
    n_bins=8,
):
    return {
        "candidate_id": candidate_id,
        "label": label,
        "global_within_2sigma": global_within_2sigma,
        "min_count_per_bin": min_count_per_bin,
        "n_bins_under_2sigma": n_bins_under_2sigma,
        "under_bin_fraction_2sigma": under_bin_fraction_2sigma,
        "max_undercoverage_gap": max_undercoverage_gap,
        "global_median_width_phys": global_median_width_phys,
        "global_tail_miss_imbalance": global_tail_miss_imbalance,
        "n_bins": n_bins,
    }


class TestBaseCandidates(unittest.TestCase):
    def test_filters_global_validity_and_minimum_bin_count(self):
        df = pd.DataFrame(
            [
                make_candidate(
                    candidate_id="valid",
                    global_within_2sigma=True,
                    min_count_per_bin=200,
                ),
                make_candidate(
                    candidate_id="global_invalid",
                    global_within_2sigma=False,
                    min_count_per_bin=500,
                ),
                make_candidate(
                    candidate_id="too_few_samples",
                    global_within_2sigma=True,
                    min_count_per_bin=199,
                ),
                make_candidate(
                    candidate_id="other_label",
                    label="total_mass",
                ),
            ]
        )

        result = get_base_candidates(df, "chirp_mass")

        self.assertEqual(
            result["candidate_id"].tolist(),
            ["valid"],
        )


class TestConservativeSelection(unittest.TestCase):
    def test_prefers_more_bins_inside_width_tie(self):
        df = pd.DataFrame(
            [
                make_candidate(
                    candidate_id="minimum_width",
                    global_median_width_phys=10.0,
                    n_bins=4,
                ),
                make_candidate(
                    candidate_id="more_bins_within_2pct",
                    global_median_width_phys=10.1,
                    n_bins=12,
                ),
                make_candidate(
                    candidate_id="outside_width_tie",
                    global_median_width_phys=10.3,
                    n_bins=20,
                ),
            ]
        )

        selected, ranked = select_conservative(
            df,
            "chirp_mass",
        )

        self.assertEqual(
            selected["candidate_id"],
            "more_bins_within_2pct",
        )
        self.assertEqual(
            selected["selection_policy"],
            "conservative_zero_under_bins_2sigma",
        )

        self.assertNotIn(
            "outside_width_tie",
            ranked["candidate_id"].tolist(),
        )

    def test_uses_best_available_if_no_zero_undercoverage_candidate(self):
        df = pd.DataFrame(
            [
                make_candidate(
                    candidate_id="two_under",
                    n_bins_under_2sigma=2,
                    under_bin_fraction_2sigma=0.2,
                ),
                make_candidate(
                    candidate_id="one_under",
                    n_bins_under_2sigma=1,
                    under_bin_fraction_2sigma=0.1,
                ),
            ]
        )

        selected, _ = select_conservative(
            df,
            "chirp_mass",
        )

        self.assertEqual(
            selected["candidate_id"],
            "one_under",
        )
        self.assertEqual(
            selected["selection_policy"],
            "conservative_best_available",
        )


class TestEfficientSelection(unittest.TestCase):
    def test_prioritizes_minimum_width_among_valid_candidates(self):
        df = pd.DataFrame(
            [
                make_candidate(
                    candidate_id="narrow",
                    global_median_width_phys=9.0,
                    n_bins_under_2sigma=1,
                    under_bin_fraction_2sigma=0.08,
                    max_undercoverage_gap=0.03,
                    n_bins=6,
                ),
                make_candidate(
                    candidate_id="slightly_wider",
                    global_median_width_phys=9.1,
                    n_bins_under_2sigma=0,
                    under_bin_fraction_2sigma=0.0,
                    max_undercoverage_gap=0.0,
                    n_bins=12,
                ),
            ]
        )

        selected, _ = select_efficient(
            df,
            "chirp_mass",
        )

        self.assertEqual(
            selected["candidate_id"],
            "narrow",
        )
        self.assertEqual(
            selected["selection_policy"],
            "efficient_local_validity_tolerant",
        )

    def test_rejects_candidates_outside_efficient_limits(self):
        df = pd.DataFrame(
            [
                make_candidate(
                    candidate_id="too_many_under_bins",
                    global_median_width_phys=8.0,
                    n_bins_under_2sigma=2,
                    under_bin_fraction_2sigma=0.20,
                    max_undercoverage_gap=0.03,
                ),
                make_candidate(
                    candidate_id="gap_too_large",
                    global_median_width_phys=8.5,
                    n_bins_under_2sigma=1,
                    under_bin_fraction_2sigma=0.05,
                    max_undercoverage_gap=0.06,
                ),
                make_candidate(
                    candidate_id="valid",
                    global_median_width_phys=9.0,
                    n_bins_under_2sigma=1,
                    under_bin_fraction_2sigma=0.05,
                    max_undercoverage_gap=0.04,
                ),
            ]
        )

        selected, _ = select_efficient(
            df,
            "chirp_mass",
        )

        self.assertEqual(
            selected["candidate_id"],
            "valid",
        )

    def test_falls_back_to_conservative_when_no_efficient_candidate(self):
        df = pd.DataFrame(
            [
                make_candidate(
                    candidate_id="fallback_candidate",
                    n_bins_under_2sigma=1,
                    under_bin_fraction_2sigma=0.20,
                    max_undercoverage_gap=0.06,
                    global_median_width_phys=10.0,
                ),
                make_candidate(
                    candidate_id="worse_fallback_candidate",
                    n_bins_under_2sigma=2,
                    under_bin_fraction_2sigma=0.30,
                    max_undercoverage_gap=0.08,
                    global_median_width_phys=9.0,
                ),
            ]
        )

        selected, _ = select_efficient(
            df,
            "chirp_mass",
        )

        self.assertEqual(
            selected["candidate_id"],
            "fallback_candidate",
        )
        self.assertEqual(
            selected["selection_policy"],
            "conservative_best_available",
        )


if __name__ == "__main__":
    unittest.main()
