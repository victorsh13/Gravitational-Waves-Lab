"""Small CPU-only checks; no scientific dataset, model training or inference.

Run from the repository root:
python -m unittest discover -s tests -p test_m12_balanced_sampling.py -v
"""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.balanced_sampling import MarginalMixtureSampler, load_train_marginal_sampler
from src.models.hdf5_batch_dataset import (
    HDF5BatchIterableDataset, apply_input_normalization_to_batch,
)


def diagnostic_schedule(ids, q, epoch, count, seed=123):
    """Independent transcription of the diagnostic's S1 draw order."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 35, epoch]))
    label = rng.integers(0, 3, size=count)
    side = rng.random(count) < 0.35
    positions = np.empty(count, dtype=np.int64)
    for j in range(3):
        for peripheral in (False, True):
            pool = np.flatnonzero((q[:, j] >= 0.8) == peripheral)
            selected = np.flatnonzero((label == j) & (side == peripheral))
            positions[selected] = pool[rng.integers(0, len(pool), size=len(selected))]
    return ids[positions]


class BalancedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "tiny.h5"
        self.ids = np.arange(1, 28, 2, dtype=np.int64)
        t = np.arange(len(self.ids), dtype=np.float64)
        self.y = np.column_stack((t ** 2 + 1, 3 * t + 7, np.sin(t)))
        # Non-train labels are invalid deliberately: they cannot fit references.
        X = np.arange(30, dtype=np.float32)[:, None, None] + np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 10
        y = np.full((30, 3), np.nan, dtype=np.float32)
        y[self.ids] = self.y
        self.y = y[self.ids].astype(np.float64)
        split = np.full(30, b"val", dtype="S5")
        split[self.ids] = b"train"
        with h5py.File(self.path, "w") as h:
            h.create_dataset("X", data=X)
            h.create_dataset("y", data=y)
            h.create_dataset("split", data=split)
            h.attrs["labels"] = "chirp_mass,total_mass,chi_eff"
        distance = np.abs(self.y - self.y.mean(axis=0)) / self.y.std(axis=0, ddof=0)
        self.q = np.column_stack([
            np.searchsorted(np.sort(distance[:, j]), distance[:, j], side="right") / len(self.ids)
            for j in range(3)])
        self.sampler = load_train_marginal_sampler(self.path, self.ids[::-1])

    def dataset(self, sampler=True, **kwargs):
        options = dict(batch_size=4, drop_last=True, shuffle_batches=True,
                       shuffle_within_batch=True)
        options.update(kwargs)
        return HDF5BatchIterableDataset(
            self.path, self.ids, balanced_sampling=self.sampler if sampler else None, **options)

    def test_probabilities_frequencies_and_reference(self):
        peripheral = self.q >= 0.8
        expected = np.zeros(len(self.ids))
        for j in range(3):
            expected += np.where(peripheral[:, j],
                                 0.35 / peripheral[:, j].sum(),
                                 0.65 / (~peripheral[:, j]).sum()) / 3
        np.testing.assert_allclose(self.sampler.probabilities, expected, rtol=0, atol=1e-16)
        self.assertAlmostEqual(expected.sum(), 1)
        self.assertTrue(np.isfinite(expected).all() and (expected > 0).all())
        np.testing.assert_allclose(
            self.sampler.metadata["induced_marginal_peripheral_frequencies"],
            expected @ peripheral, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(self.sampler.train_ids, self.ids)
        with self.assertRaises(ValueError):
            load_train_marginal_sampler(self.path, self.ids[:-1])
        with self.assertRaises(ValueError):
            load_train_marginal_sampler(self.path, np.append(self.ids, 0))

    def test_schedule_matches_notebook_and_replacement(self):
        schedule = self.sampler.schedule(2, 101)
        np.testing.assert_array_equal(schedule, diagnostic_schedule(self.ids, self.q, 2, 101))
        np.testing.assert_array_equal(schedule, self.sampler.schedule(2, 101))
        self.assertFalse(np.array_equal(schedule, self.sampler.schedule(3, 101)))
        self.assertTrue(np.isin(schedule, self.ids).all())
        self.assertLess(len(np.unique(schedule)), len(schedule))

    def test_unique_reads_restore_duplicates_order_and_scaling(self):
        rows = np.array([27, 1, 27, 3], dtype=np.int64)
        for overread in (1.0, 100.0):
            ds = self.dataset(max_slice_overread=overread,
                              y_mean=np.array([1, 2, 3]), y_std=np.array([2, 3, 4]))
            # Explicit schedule checks reconstruction at different overread limits.
            ds._sampling_schedule.numpy()[:4] = rows
            with h5py.File(self.path, "r") as h:
                X, y = ds._read_batch(h, 0)
                expected_X = np.stack([h["X"][int(i)] for i in rows])
                expected_y = np.stack([h["y"][int(i)] for i in rows])
            np.testing.assert_array_equal(X.numpy(), expected_X)
            np.testing.assert_array_equal(y.numpy(),
                                          (expected_y - ds.y_mean) / (ds.y_std + 1e-8))
            self.assertEqual(X.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.float32)

    def test_workers_epochs_and_drop_last(self):
        for workers, persistent in ((0, False), (4, False), (4, True)):
            for drop in (False, True):
                ds = self.dataset(drop_last=drop)
                loader = DataLoader(ds, batch_size=None, num_workers=workers,
                                    persistent_workers=persistent)
                try:
                    for epoch in (0, 1):
                        ds.set_epoch(epoch)
                        batches = list(loader)
                        self.assertEqual(len(batches), len(ds))
                        self.assertTrue(all(len(batch) == 2 for batch in batches))
                        observed = np.concatenate([X[:, 0, 0].numpy() for X, y in batches])
                        consumed = (len(self.ids) // 4 * 4) if drop else len(self.ids)
                        np.testing.assert_array_equal(observed, self.sampler.schedule(epoch)[:consumed])
                        self.assertEqual(ds.sampling_epoch_provenance["consumed_draws"], consumed)
                finally:
                    # DataLoader destruction shuts down persistent workers as well.
                    del loader

    def test_frozen_epoch_budget_without_signal_iteration(self):
        # HDF5 shape metadata only: no signal payload is allocated or read.
        nominal = 400000
        path = Path(self.temp.name) / "shape_only.h5"
        with h5py.File(path, "w") as h:
            h.create_dataset("X", shape=(nominal, 1, 1), dtype="f4")
            h.create_dataset("y", shape=(nominal, 3), dtype="f4")
        ids = np.arange(nominal, dtype=np.int64)
        q = np.resize(self.q, (nominal, 3))
        sampler = MarginalMixtureSampler(ids, q)
        ds = HDF5BatchIterableDataset(path, ids, batch_size=256, drop_last=True,
                                      balanced_sampling=sampler)
        self.assertEqual(len(ds), 1562)
        self.assertEqual(len(ds) * ds.batch_size, 399872)
        self.assertEqual(nominal % ds.batch_size, 128)
        self.assertEqual(ds.sampling_epoch_provenance["consumed_draws"], 399872)
        self.assertEqual(ds.sampling_epoch_provenance["nominal_draws"], 400000)

    def test_legacy_contract_loss_and_rejection(self):
        legacy = HDF5BatchIterableDataset(self.path, self.ids, batch_size=4,
                                           shuffle_batches=False, shuffle_within_batch=False)
        disabled = self.dataset(sampler=False, shuffle_batches=False, shuffle_within_batch=False)
        for left, right in zip(legacy, disabled):
            self.assertEqual(len(left), 2)
            for a, b in zip(left, right):
                self.assertTrue(torch.equal(a, b))
            target = left[1]
            pred = torch.zeros_like(target)
            self.assertTrue(torch.equal(torch.nn.MSELoss()(pred, target),
                                        torch.nn.MSELoss()(pred, right[1])))
        with self.assertRaises(ValueError):
            self.dataset(sample_weights_by_row=np.ones(30, dtype=np.float32))
        with self.assertRaises(ValueError):
            HDF5BatchIterableDataset(self.path, self.ids[:-1], balanced_sampling=self.sampler)

    def test_config_only_two_changes(self):
        root = Path(__file__).resolve().parents[1]
        base = json.loads((root / "configs/experiments/train_m11_final_G1_500k.json").read_text())
        candidate = json.loads((root / "configs/experiments/train_m12_3_S1_balanced_sampling.json").read_text())
        policy = candidate["training"].pop("balanced_sampling")
        self.assertEqual(policy, dict(enabled=True, mode="marginal_mixture", threshold=0.8,
                                      p_peripheral=0.35, seed=123))
        self.assertTrue(candidate["outputs"]["checkpoint_tag"].startswith("M12_3_S1_"))
        candidate["outputs"]["checkpoint_tag"] = base["outputs"]["checkpoint_tag"]
        self.assertEqual(candidate, base)

    def test_grouped_reads_bitwise_equal_and_bounded(self):
        path = Path(self.temp.name) / "chunked.h5"
        rng = np.random.default_rng(123)
        data = rng.standard_normal((192, 3, 8)).astype(np.float32)
        data[0, 0, 0] = -0.0
        targets = rng.standard_normal((192, 3)).astype(np.float32)
        with h5py.File(path, "w") as h:
            h.create_dataset("X", data=data, chunks=(64, 3, 8))
            h.create_dataset("y", data=targets)
        ids = np.arange(192, dtype=np.int64)
        sampler = MarginalMixtureSampler(ids, np.resize(self.q, (192, 3)))
        cases = {
            "random_unordered": rng.integers(0, 192, size=16),
            "duplicates": np.array([127, 0, 127, 64] * 4),
            "contiguous": np.arange(20, 36)[::-1],
            "isolated": np.arange(0, 192, 12)[::-1],
            "within_chunk": np.array([7, 0, 7, 8, 20, 21, 28, 29] * 2),
            "boundaries": np.array([63, 64, 65, 62, 127, 128, 129, 191] * 2),
        }

        class RecordingDataset:
            def __init__(self, source):
                self.source = source
                self.shape, self.dtype, self.chunks = source.shape, source.dtype, source.chunks
                self.reads = []

            def __getitem__(self, key):
                self.reads.append(key)
                return self.source[key]

        for enabled in (False, True):
            normalization = dict(enabled=enabled, mode="per_sample_per_detector_zscore", eps=1e-6)
            ds = HDF5BatchIterableDataset(
                path, ids, batch_size=16, balanced_sampling=sampler,
                input_normalization=normalization,
                y_mean=np.array([1, 2, 3]), y_std=np.array([2, 3, 4]))
            for name, rows in cases.items():
                with self.subTest(case=name, normalization=enabled):
                    ds._sampling_schedule.numpy()[:16] = rows
                    unique, inverse = np.unique(rows, return_inverse=True)
                    with h5py.File(path, "r") as h:
                        source = RecordingDataset(h["X"])
                        actual = ds._read_batch({"X": source, "y": h["y"]}, 0)
                        expected_X = apply_input_normalization_to_batch(
                            np.asarray(h["X"][unique][inverse], dtype=np.float32), normalization)
                        expected_y = (h["y"][unique][inverse] - ds.y_mean) / (ds.y_std + 1e-8)
                    self.assertEqual(len(actual), 2)
                    self.assertEqual(tuple(actual[0].shape), (16, 3, 8))
                    self.assertEqual(tuple(actual[1].shape), (16, 3))
                    for tensor, expected in zip(actual, (expected_X, expected_y)):
                        self.assertEqual(tensor.dtype, torch.float32)
                        np.testing.assert_array_equal(tensor.numpy().view(np.uint8), expected.view(np.uint8))
                    for selection in source.reads:
                        self.assertIsInstance(selection, slice)
                        self.assertEqual(selection.start // 64, (selection.stop - 1) // 64)
                        count = np.count_nonzero((unique >= selection.start) & (unique < selection.stop))
                        self.assertLessEqual(selection.stop - selection.start, 4 * count)
                        self.assertLessEqual(selection.stop - selection.start, 64)

        # Real chunked HDF5 reads through both single- and multiworker iteration.
        for workers in (0, 4):
            ds = HDF5BatchIterableDataset(path, ids, batch_size=16, balanced_sampling=sampler)
            ds.set_epoch(1)
            loader = DataLoader(ds, batch_size=None, num_workers=workers)
            observed_X, observed_y = [], []
            for X, y in loader:
                observed_X.append(X.numpy())
                observed_y.append(y.numpy())
            rows = sampler.schedule(1)
            np.testing.assert_array_equal(np.concatenate(observed_X).view(np.uint8), data[rows].view(np.uint8))
            np.testing.assert_array_equal(np.concatenate(observed_y).view(np.uint8), targets[rows].view(np.uint8))

    def test_legacy_global_and_marginal_paths_bypass_sampled_reads(self):
        for shape in (None, (30,), (30, 3)):
            weights = None if shape is None else np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
            ds = self.dataset(sampler=False, shuffle_batches=False, shuffle_within_batch=False,
                              sample_weights_by_row=weights)
            with patch.object(ds, "_read_sampled_batch", side_effect=AssertionError("Legacy path changed")):
                with h5py.File(self.path, "r") as h:
                    actual = ds._read_batch(h, 0)
                    rows = ds.sorted_indices[:4]
                    np.testing.assert_array_equal(actual[0].numpy(), h["X"][rows])
                    np.testing.assert_array_equal(actual[1].numpy(), h["y"][rows])
            self.assertEqual(len(actual), 2 if weights is None else 3)
            if weights is not None:
                np.testing.assert_array_equal(actual[2].numpy(), weights[rows])


if __name__ == "__main__":
    unittest.main()
