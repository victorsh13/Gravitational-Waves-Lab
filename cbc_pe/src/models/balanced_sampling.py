"""Train-only M12.3-S1 replacement sampling; no loss weights are applied.

Ordinary batch MSE estimates risk under this mixture, not uniform train risk.
"""
from __future__ import annotations

import numpy as np

from src.models.sample_weighting import array_hash, load_train_sample_weights


class MarginalMixtureSampler:
    """Pool positions and probabilities are aligned to sorted original row IDs."""

    def __init__(self, train_ids, marginal_q, seed=123, threshold=0.8,
                 p_peripheral=0.35):
        ids = np.asarray(train_ids)
        q = np.asarray(marginal_q, dtype=np.float64)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 2
                or np.any(ids < 0) or np.any(np.diff(ids.astype(np.int64)) <= 0)):
            raise ValueError("Expected sorted, unique, nonnegative train row IDs")
        if (q.shape != (len(ids), 3) or not np.isfinite(q).all()
                or np.any((q < 0) | (q > 1))):
            raise ValueError("Expected finite train marginal percentiles with shape (N_train, 3)")
        if not 0 < threshold < 1 or not 0 < p_peripheral < 1:
            raise ValueError("Threshold and peripheral probability must lie in (0, 1)")
        if int(seed) != seed or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.train_ids = ids.astype(np.int64, copy=True)
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.p_peripheral = float(p_peripheral)
        peripheral = q >= threshold
        self.pools = {}
        self.probabilities = np.zeros(len(ids), dtype=np.float64)
        for j in range(3):
            for side in (False, True):
                pool = np.flatnonzero(peripheral[:, j] == side)
                if len(pool) == 0:
                    raise ValueError(f"Empty pool: label={j}, peripheral={side}")
                self.pools[j, side] = pool
                selection = p_peripheral if side else 1 - p_peripheral
                self.probabilities[pool] += selection / (3 * len(pool))
        if (not np.isfinite(self.probabilities).all()
                or np.any(self.probabilities <= 0)
                or not np.isclose(self.probabilities.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("Invalid mixture probabilities")
        self.metadata = {
            "policy": "M12.3-S1", "mode": "marginal_mixture",
            "label_names": ["chirp_mass", "total_mass", "chi_eff"],
            "threshold": self.threshold, "p_peripheral": self.p_peripheral,
            "seed": self.seed, "replacement": True,
            "seed_scheme": "SeedSequence([seed, 35, zero_based_epoch]); diagnostic S1 stream",
            "n_train": len(ids), "train_indices_sha256": array_hash(self.train_ids),
            "probabilities_sha256": array_hash(self.probabilities),
            "probability_summary": {
                "sum": float(self.probabilities.sum()),
                "min": float(self.probabilities.min()),
                "max": float(self.probabilities.max()),
                "weight_ess": float(1 / np.square(self.probabilities).sum()),
            },
            "pool_sizes": {f"{j}_{'peripheral' if side else 'central'}": len(pool)
                           for (j, side), pool in self.pools.items()},
            "induced_marginal_peripheral_frequencies":
                (self.probabilities @ peripheral).tolist(),
            "training_objective": "ordinary standardized MSE under the replacement mixture",
            "validation_objective": "unchanged unweighted standardized MSELoss",
        }

    def schedule(self, epoch, draw_count=None):
        """Match the diagnostic's label/side/pool RNG call order exactly."""
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        count = len(self.train_ids) if draw_count is None else int(draw_count)
        if count < 1 or (draw_count is not None and count != draw_count):
            raise ValueError("draw_count must be a positive integer")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 35, int(epoch)]))
        label = rng.integers(0, 3, size=count)
        side = rng.random(count) < self.p_peripheral
        positions = np.empty(count, dtype=np.int64)
        for j in range(3):
            for is_peripheral in (False, True):
                selected = np.flatnonzero((label == j) & (side == is_peripheral))
                pool = self.pools[j, is_peripheral]
                positions[selected] = pool[rng.integers(0, len(pool), size=len(selected))]
        return self.train_ids[positions]


def load_train_marginal_sampler(h5_path, train_idx, seed=123,
                                threshold=0.8, p_peripheral=0.35):
    """Reuse M12.2's membership-checked, chunked physical TRAIN label reader.

    alpha=0 obtains exactly the established train-only percentile references.
    Its temporary unit-weight lookup is discarded, never attached to a loader.
    No strain arrays or non-train targets are read here.
    """
    unused_weights, q, reference = load_train_sample_weights(
        h5_path, train_idx, alpha=0.0, method="train_label_marginal")
    del unused_weights
    sampler = MarginalMixtureSampler(np.sort(np.asarray(train_idx, dtype=np.int64)), q,
                                     seed, threshold, p_peripheral)
    sampler.metadata["train_reference"] = {
        key: reference[key] for key in (
            "mean", "std_ddof0", "percentile_mapping", "reference_distance_sha256")
    }
    return sampler
