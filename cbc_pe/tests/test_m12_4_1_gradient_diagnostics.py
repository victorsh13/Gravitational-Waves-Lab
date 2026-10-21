"""Small deterministic CPU checks; no checkpoints, strain, inference or training."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import io
from contextlib import redirect_stderr

import h5py
import numpy as np
import pandas as pd
import torch

from scripts.m12.run_m12_4_1_gradient_diagnostics import (
    LABELS, SimpleCNN_ResidualDilated, make_regions, make_schedule,
    grouped_read, gradient_statistics, shared_groups, state_unchanged, summarize,
    large_schedule, cumulative_counts, geometry_from_gram, StreamingGradients,
    jackknife_interval, batch_bootstrap, large_inference, parse_args, LARGE_METRICS,
)


class GradientDiagnosticChecks(unittest.TestCase):
    def test_region_boundary_and_discordance(self):
        marginal = np.array([[79., 80., 80.], [80., 79., 79.], [50., 50., 50.]])
        regions = make_regions(np.array([79., 80., 50.]), marginal)
        np.testing.assert_array_equal(regions["global_peripheral"], [False, True, False])
        np.testing.assert_array_equal(regions["chirp_mass_central_spin_peripheral"], [True, False, False])
        np.testing.assert_array_equal(regions["chirp_mass_peripheral_spin_central"], [False, True, False])
        self.assertFalse(regions["total_mass_central_spin_peripheral"].any())
        for label in LABELS:
            self.assertTrue(np.all(regions[label+"_central"] ^ regions[label+"_peripheral"]))

    def test_schedule_repeatability_budget_and_membership(self):
        rng = np.random.default_rng(9)
        ids = np.arange(1000, 5000, dtype=np.int64)
        regions = make_regions(rng.uniform(0, 100, len(ids)), rng.uniform(0, 100, (len(ids), 3)))
        for pilot, count in [(True, 4), (False, 28)]:
            a = make_schedule(ids, regions, 123, pilot, 28, 8)
            b = make_schedule(ids, regions, 123, pilot, 28, 8)
            self.assertEqual(len(a), count)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x["ids"], y["ids"])
                self.assertEqual(x["ids_sha256"], y["ids_sha256"])
                self.assertTrue(regions[x["region"]][x["positions"]].all())
                np.testing.assert_array_equal(x["ids"], ids[x["positions"]])
            for region in regions:
                drawn = np.concatenate([x["ids"] for x in a if x["region"] == region] or [np.array([])])
                self.assertEqual(len(drawn), len(np.unique(drawn)))
        bad = dict(regions)
        bad["general"] = np.zeros(len(ids), dtype=bool)
        with self.assertRaisesRegex(ValueError, "Insufficient train population"):
            make_schedule(ids, bad, 123, True, 4, 8)

    def test_grouped_reads_exact_order_duplicates_and_bound(self):
        values = np.arange(160*2*5, dtype=np.float32).reshape(160, 2, 5)
        rows = np.array([159, 3, 2, 65, 63, 64, 3, 0, 100])
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp)/"small.h5", "w") as h:
                for name, chunks in [("chunked", (16, 2, 5)), ("contiguous", None)]:
                    ds = h.create_dataset(name, data=values, chunks=chunks)
                    got, fetched = grouped_read(ds, rows)
                    np.testing.assert_array_equal(got, values[rows])
                    self.assertLessEqual(fetched, 4*len(np.unique(rows)))

    def test_cosine_none_zero_and_norm_ratios(self):
        a = [torch.tensor([3., 4.], dtype=torch.float64), None]
        b = [torch.tensor([-6., -8.], dtype=torch.float64), torch.tensor([2.], dtype=torch.float64)]
        got = gradient_statistics(a, b, [0, 1])
        self.assertAlmostEqual(got["dot"], -50.)
        self.assertAlmostEqual(got["cosine"], -50/(5*np.sqrt(104)))
        self.assertAlmostEqual(got["norm_ratio_a_over_b"], 5/np.sqrt(104))
        self.assertEqual(got["negative_cosine"], 1.)
        zero = gradient_statistics([None], [torch.ones(1)], [0])
        self.assertTrue(np.isnan(zero["cosine"]))
        self.assertTrue(np.isnan(zero["negative_cosine"]))
        orthogonal = gradient_statistics([torch.tensor([1., 0.])], [torch.tensor([0., 1.])], [0])
        self.assertEqual(orthogonal["cosine"], 0.)

    def test_autograd_preserves_parameters_and_buffers(self):
        # Analytic scalar expression, no model forward/inference. Exercises state audit.
        model = torch.nn.Module()
        model.register_parameter("w", torch.nn.Parameter(torch.tensor([1., 2., 3.])))
        model.register_buffer("sentinel", torch.tensor([7.]))
        model.eval()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        losses = model.w.square()
        for k in range(3):
            g, = torch.autograd.grad(losses[k], [model.w], retain_graph=k < 2)
            expected = torch.zeros(3)
            expected[k] = 2*model.w.detach()[k]
            np.testing.assert_array_equal(g.numpy(), expected.numpy())
        self.assertTrue(state_unchanged(model, before))
        self.assertIsNone(model.w.grad)
        model.sentinel.add_(1)
        self.assertFalse(state_unchanged(model, before))

    def test_actual_architecture_group_coverage_without_forward(self):
        model = SimpleCNN_ResidualDilated()
        named, groups = shared_groups(model)
        selected = {id(named[i][1]) for i in groups["all_shared"]}
        head = {id(p) for p in model.head.parameters()}
        self.assertFalse(selected & head)
        self.assertEqual(selected | head, {id(p) for p in model.parameters()})
        self.assertIn(id(model.block1.conv.bias), selected)

    def test_summary_denominator_and_sparse_ci(self):
        rows = []
        for bid, cosine in enumerate([-1., np.nan, 1.]):
            rows.append(dict(checkpoint="M11", pair="a / b", group="all_shared", region="general",
                             batch_id=bid, cosine=cosine, dot=cosine, norm_a=1., norm_b=1.,
                             norm_ratio_a_over_b=1., norm_ratio_b_over_a=1.))
        row = summarize(pd.DataFrame(rows), 123).iloc[0]
        self.assertEqual(row.negative_fraction, .5)
        self.assertEqual(row.undefined_cosines, 1)
        self.assertTrue(np.isnan(row.cosine_median_ci_low))

    def test_notebook_is_unexecuted_result_only_and_syntax_valid(self):
        root = Path(__file__).resolve().parents[1]
        nb = json.loads((root/"notebooks/m12/M12_4_1_multitask_gradient_diagnostics.ipynb").read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            # User-executed initial-study outputs are historical and preserved.
            if cell.get("id", "").startswith("m1241-large-"):
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.split(".")[0] in {"torch", "h5py", "src", "scripts"} for n in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], {"torch", "h5py", "src", "scripts"})

    def test_large_unique_nested_paired_budget_and_overlap(self):
        ids = np.arange(10000, 14000, dtype=np.int64)
        rng = np.random.default_rng(9)
        regions = make_regions(rng.uniform(0, 100, len(ids)), rng.uniform(0, 100, (len(ids), 3)))
        a, d = large_schedule(ids, regions, 123, 300, 16, ["global_peripheral"], 80)
        b, e = large_schedule(ids, regions, 123, 300, 16, ["global_peripheral"], 80)
        self.assertEqual(d, e)
        self.assertEqual(d["general_examples"], 220)
        self.assertEqual(d["total_cohort_exposures"], 300)
        self.assertLessEqual(d["distinct_union"], 300)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x["ids"], y["ids"])
            self.assertEqual(x["ids_sha256"], y["ids_sha256"])
            self.assertTrue(regions[x["region"]][x["positions"]].all())
        for region in d["cohorts"]:
            drawn = np.concatenate([x["ids"] for x in a if x["region"] == region])
            self.assertEqual(len(drawn), len(np.unique(drawn)))
        c, _ = large_schedule(ids, regions, 123, 500, 16, ["global_peripheral"], 80)
        general = lambda s: np.concatenate([x["ids"] for x in s if x["region"] == "general"])
        np.testing.assert_array_equal(general(a), general(c)[:220])
        with self.assertRaisesRegex(ValueError, "exhaust"):
            large_schedule(ids, regions, 123, 80, 16, ["global_peripheral"], 80)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            large_schedule(ids, regions, 123, 2000, 16, ["global_peripheral"], 1500)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            large_schedule(ids, regions, 123, 4001, 16)
        # Exact overlap audit; repeated cohorts do not silently enlarge the union budget.
        pools = [set(np.concatenate([x["ids"] for x in a if x["region"] == r])) for r in d["overlap_order"]]
        self.assertEqual(d["overlap"], [[len(x & y) for y in pools] for x in pools])

    def test_cli_budget_defaults_and_cumulative_boundaries(self):
        self.assertEqual(parse_args([]).batch_size, 32)
        self.assertEqual(parse_args([]).max_batches, 28)
        self.assertEqual(parse_args(["--large-sample"]).batch_size, 64)
        self.assertEqual(parse_args(["--large-sample"]).examples, 40000)
        self.assertEqual(parse_args(["--large-sample", "--examples", "400000"]).examples, 400000)
        for argv in [["--large-sample", "--examples", "400001"], ["--large-sample", "--max-batches", "4"],
                     ["--large-sample", "--pilot"], ["--examples", "1000"]]:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)
        self.assertEqual(cumulative_counts(40000, 64), [2560, 5056, 10048, 20032, 40000])
        self.assertEqual(cumulative_counts(5000, 64), [2560, 5000])

    def test_streaming_mean_equals_direct_with_partial_batch(self):
        rng = np.random.default_rng(5)
        examples = torch.tensor(rng.normal(size=(11, 3, 7)), dtype=torch.float64)
        acc = StreamingGradients(7, {"all_shared": slice(0, 7)}, blocks=10)
        for lo, hi in [(0, 4), (4, 8), (8, 11)]:
            acc.add(examples[lo:hi].mean(0), hi-lo)
        np.testing.assert_allclose(acc.mean().numpy(), examples.mean(0).numpy(), atol=1e-14)
        rows, theta, deleted = acc.snapshot()
        expected = geometry_from_gram((examples.mean(0) @ examples.mean(0).T).numpy())
        np.testing.assert_allclose(theta, [[r[m] for m in LARGE_METRICS] for r in expected], atol=1e-14)
        self.assertEqual(acc.counts.sum(), 11)
        with self.assertRaisesRegex(ValueError, "workspace"):
            StreamingGradients(100000000, {"all_shared": slice(0, 1)})

    def test_mean_cosine_is_not_cosine_of_mean(self):
        g1 = torch.tensor([[1., 0.], [-1., 0.], [1., 1.]], dtype=torch.float64)
        g2 = torch.tensor([[0., 2.], [0., 2.], [1., 1.]], dtype=torch.float64)
        acc = StreamingGradients(2, {"all_shared": slice(0, 2)}, blocks=10)
        acc.add(g1, 4); acc.add(g2, 4)
        avg = np.mean([geometry_from_gram((g @ g.T).numpy())[0]["cosine"] for g in [g1, g2]])
        rows, _, _ = acc.snapshot()
        self.assertAlmostEqual(avg, 0.)
        self.assertAlmostEqual(rows[0]["cosine"], .6)

    def test_combined_alignment_and_tiny_norms(self):
        g = np.array([[1., 0.], [-2., 1.], [0., 2.]])
        rows = geometry_from_gram(g @ g.T)
        total = g.mean(0)
        for k in range(3):
            row = rows[3+k]
            self.assertAlmostEqual(row["dot"], np.dot(g[k], total))
            self.assertAlmostEqual(row["cosine"], np.dot(g[k], total)/(np.linalg.norm(g[k])*np.linalg.norm(total)))
        tiny = geometry_from_gram(np.eye(3)*1e-20, reference=np.eye(3))
        self.assertTrue(all(np.isnan(row["cosine"]) for row in tiny))
        zero = geometry_from_gram(np.zeros((3, 3)))
        self.assertTrue(all(np.isnan(row["negative_cosine"]) for row in zero))

    def test_jackknife_equal_groups_and_census(self):
        theta = np.array([[2.]])
        deleted = np.array([[[1.8]], [[2.1]], [[2.2]], [[1.9]]])
        counts = np.array([10, 10, 10, 10])
        lo, hi, se = jackknife_interval(theta, deleted, counts, 1000, min_blocks=4)
        expected = np.sqrt(3/4*np.sum((deleted-deleted.mean(0))**2, axis=0))*np.sqrt(960/999)
        np.testing.assert_allclose(se, expected)
        lo, hi, se = jackknife_interval(theta, deleted, counts, 40, min_blocks=4)
        np.testing.assert_array_equal(lo, theta)
        np.testing.assert_array_equal(hi, theta)
        bad = deleted.copy(); bad[0] = np.nan
        self.assertTrue(np.isnan(jackknife_interval(theta, bad, counts, 1000, min_blocks=4)[0]).all())

    def test_paired_bootstrap_preserves_constant_difference(self):
        base = np.arange(40, dtype=float).reshape(40, 1)
        point, boot = batch_bootstrap(np.stack([base, base+7]), 123, replicates=20)
        np.testing.assert_allclose(point[1]-point[0], 7)
        np.testing.assert_allclose(boot[:, 1]-boot[:, 0], 7)

    def test_scalar_inference_on_synthetic_gradients(self):
        ids = np.arange(1000, dtype=np.int64)
        regions = {"general": np.ones(1000, dtype=bool)}
        schedule, design = large_schedule(ids, regions, 123, 80, 2)
        groups = {g: slice(0, 2) for g in ["shallow", "middle", "deep", "all_shared"]}
        values = np.empty((2, 40, 24, 7))
        snapshots = {}
        for model, name in enumerate(["M11", "M12.1"]):
            acc = StreamingGradients(2, groups, blocks=10)
            for batch in schedule:
                # Scaled gradients have exactly paired cosines and norm ratios.
                g = torch.tensor([[1., .2], [.4, 1.], [.7, .8]], dtype=torch.float64)*(1+batch["batch_id"]*.01)*(model+1)
                acc.add(g, 2)
                rows = geometry_from_gram((g @ g.T).numpy())*4
                values[model, batch["batch_id"]] = [[r[m] for m in LARGE_METRICS+["negative_cosine"]] for r in rows]
            _, theta, deleted = acc.snapshot()
            snapshots[(name, "general", 80)] = dict(theta=theta, deleted=deleted, counts=acc.counts.copy(), batches=40)
        conv, paired = large_inference(values, schedule, snapshots, ["M11", "M12.1"], design, 123, 2)
        delta = paired[paired.metric.eq("cosine")]
        np.testing.assert_allclose(delta.estimate, 0, atol=1e-14)
        self.assertTrue(np.isfinite(delta.ci_low).all())
        self.assertFalse(conv.stable_geometry.any())  # one cumulative point cannot establish convergence


if __name__ == "__main__":
    unittest.main()
