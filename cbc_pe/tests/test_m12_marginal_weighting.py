"""Small CPU-only checks; no scientific artifacts, training or inference."""
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from src.models.sample_weighting import (
    marginal_label_weights, mahalanobis_sample_weights,
    load_train_sample_weights, sample_weighted_mse, array_hash,
)
from src.models.hdf5_batch_dataset import HDF5BatchIterableDataset

ROOT = Path(__file__).resolve().parents[1]


class MarginalWeightingChecks(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[1., 2., 0.], [2., 4., 1.], [4., 3., -1.],
                           [5., 8., .5], [3., 6., -.5], [1., 2., 0.]])

    def test_formula_and_metadata(self):
        weights, q, meta = marginal_label_weights(self.y)
        # The exact independently per-column formula from M12.2 diagnostics.
        distances = abs((self.y-self.y.mean(axis=0))/self.y.std(axis=0))
        expected_q = np.column_stack([
            np.searchsorted(np.sort(distances[:, j]), distances[:, j], side="right") / len(self.y)
            for j in range(3)])
        raw = 1 + .5*expected_q
        expected = (raw/raw.mean(axis=0)).astype(np.float32)
        np.testing.assert_array_equal(q, expected_q)
        np.testing.assert_array_equal(weights, expected)
        np.testing.assert_allclose(weights.mean(axis=0, dtype=float), 1., atol=1e-7)
        self.assertTrue(np.isfinite(weights).all() and (weights > 0).all())
        np.testing.assert_array_equal(q[0], q[-1])  # tied rows
        for j, label in enumerate(meta["label_names"]):
            self.assertEqual(meta["reference_distance_sha256"][label], array_hash(np.sort(distances[:, j])))
            self.assertEqual(meta["normalized_weight_sha256_per_label"][label], array_hash(weights[:, j]))

    def test_loss_and_gradients(self):
        pred = torch.tensor(self.y, dtype=torch.float64, requires_grad=True)
        target = torch.zeros_like(pred)
        base = torch.nn.functional.mse_loss(pred, target)
        zero, _, _ = marginal_label_weights(self.y, alpha=0.)
        global_zero, _, _ = mahalanobis_sample_weights(self.y, alpha=0.)
        for weights in (torch.from_numpy(zero), torch.from_numpy(global_zero)):
            got = sample_weighted_mse(pred, target, weights)
            self.assertTrue(torch.allclose(got, base, rtol=1e-12, atol=1e-12))
            self.assertTrue(torch.allclose(torch.autograd.grad(got, pred, retain_graph=True)[0],
                                          torch.autograd.grad(base, pred, retain_graph=True)[0],
                                          rtol=1e-12, atol=1e-12))
        gw, _, _ = mahalanobis_sample_weights(self.y)
        weights = torch.from_numpy(gw)
        historical = torch.mean(weights * torch.mean((pred-target)**2, dim=1))
        got = sample_weighted_mse(pred, target, weights)
        self.assertTrue(torch.equal(got, historical))
        self.assertTrue(torch.equal(torch.autograd.grad(got, pred, retain_graph=True)[0],
                                    torch.autograd.grad(historical, pred, retain_graph=True)[0]))
        mw, _, _ = marginal_label_weights(self.y)
        got = sample_weighted_mse(pred, target, torch.from_numpy(mw))
        expected = torch.mean(torch.from_numpy(mw)*(pred-target)**2)
        self.assertTrue(torch.equal(got, expected))
        self.assertTrue(torch.equal(torch.autograd.grad(got, pred, retain_graph=True)[0],
                                    torch.autograd.grad(expected, pred)[0]))
        for bad in (torch.ones(6, 1), torch.full((6, 3), float("nan"))):
            with self.assertRaises(ValueError):
                sample_weighted_mse(pred, target, bad)
        for p, t in ((pred.detach()*float("nan"), target), (pred, target+float("inf"))):
            with self.assertRaises(ValueError):
                sample_weighted_mse(p, t, torch.ones(6))

    def test_row_lookup_and_loader(self):
        ids = np.array([1, 3, 5, 7, 9, 11])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"tiny.h5"
            y = np.full((12, 3), np.nan)
            y[ids] = self.y  # NaN nontrain labels must never enter reference fitting.
            splits = np.full(12, b"val", dtype="S5"); splits[ids] = b"train"
            with h5py.File(path, "w") as h:
                h.attrs["labels"] = "chirp_mass,total_mass,chi_eff"
                h["split"] = splits; h["y"] = y
                h["X"] = np.arange(12, dtype=np.float32).reshape(12, 1, 1)
            for method in ("train_label_mahalanobis", "train_label_marginal"):
                lookup, _, _ = load_train_sample_weights(path, ids[::-1], method=method)
                self.assertEqual(lookup.shape, (12, 3) if method.endswith("marginal") else (12,))
                self.assertTrue(np.isnan(lookup[np.setdiff1d(np.arange(12), ids)]).all())
                expected, _, _ = (marginal_label_weights(self.y) if method.endswith("marginal")
                                  else mahalanobis_sample_weights(self.y))
                np.testing.assert_array_equal(lookup[ids], expected)
                for overread in (1., 4.):  # fancy and contiguous-slice branches
                    ds = HDF5BatchIterableDataset(path, ids[::-1], batch_size=2, drop_last=False,
                        shuffle_batches=True, shuffle_within_batch=True, max_slice_overread=overread,
                        sample_weights_by_row=lookup)
                    ds.set_epoch(2)
                    seen = []
                    for worker in (0, 1):
                        with patch("src.models.hdf5_batch_dataset.get_worker_info",
                                   return_value=SimpleNamespace(id=worker, num_workers=2)):
                            for xb, yb, wb in ds:
                                rows = xb[:, 0, 0].numpy().astype(int)
                                seen.extend(rows.tolist())
                                np.testing.assert_array_equal(wb.numpy(), lookup[rows])
                                np.testing.assert_array_equal(yb.numpy(), y[rows].astype(np.float32))
                    self.assertEqual(sorted(seen), ids.tolist())
            plain = HDF5BatchIterableDataset(path, ids, batch_size=2)
            self.assertEqual(len(next(iter(plain))), 2)
            with self.assertRaises(ValueError):
                load_train_sample_weights(path, ids[:-1], method="train_label_marginal")

    def test_config_and_unweighted_selection_contract(self):
        old = json.loads((ROOT/"configs/experiments/train_m12_1a_weighted_mse.json").read_text())
        new = json.loads((ROOT/"configs/experiments/train_m12_2_marginal_alpha0p5.json").read_text())
        old["training"]["sample_weighting"]["method"] = "train_label_marginal"
        old["outputs"]["checkpoint_tag"] = "M12_2_G1_marginal_alpha0p5_meanone_seed123"
        self.assertEqual(old, new)
        # Static guard: no training or validation forward pass needed.
        source = (ROOT/"src/models/train.py").read_text()
        tree = ast.parse(source)
        validate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_one_epoch")
        self.assertNotIn("sample_weighted_mse", ast.get_source_segment(source, validate))
        self.assertIn("loss_fn = nn.MSELoss()", source)
        self.assertIn("improved = val_loss < best_val_loss", source)


if __name__ == "__main__":
    unittest.main()
