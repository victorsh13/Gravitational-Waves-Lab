#!/usr/bin/env python3
"""Bounded M12.4.1 train-only gradients at frozen checkpoints; never trains.

Moved from the original unexecuted diagnostic notebook. --max-batches is a
TOTAL per-checkpoint budget across regions, not a per-region budget. Default
28 batches of 32; --pilot forces four. Outputs are exclusively small scalar
statistics and provenance. Existing output directories are never overwritten.
"""
from pathlib import Path
import os, sys, json, hashlib, inspect, random, time, platform
import numpy as np
import pandas as pd
import h5py
import torch

import argparse
import csv
import warnings
from itertools import combinations
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.paths import resolve_data_root, resolve_processed_artifact
from src.models.network import SimpleCNN_ResidualDilated, ConvBlock, ResidualDilatedBlock
from src.models.hdf5_batch_dataset import apply_input_normalization_to_batch
from src.models.sample_weighting import array_hash

def require(ok, message):
    if not ok:
        raise ValueError(message)

LABELS = ["chirp_mass", "total_mass", "chi_eff"]
TASK_PAIRS = [(0, 1), (0, 2), (1, 2)]
CONFIG_NAMES = {
    "M11": "train_m11_final_G1_500k.json",
    "M12.1": "train_m12_1a_weighted_mse.json",
    "M12.2": "train_m12_2_marginal_alpha0p5.json",
    "M12.3-S1": "train_m12_3_S1_balanced_sampling.json",
}
CHECKPOINT_TAGS = {
    "M11": "M11_final_G1_inputzscore_resdilated_emb64_d124_train400k",
    "M12.1": "M12_1a_G1_mahalanobis_alpha0p5_meanone_seed123",
    "M12.2": "M12_2_G1_marginal_alpha0p5_meanone_seed123",
    "M12.3-S1": "M12_3_S1_G1_balanced_sampling_p035_seed123",
}

def file_record(path, digest=False):
    require(path.is_file(), f"Missing required frozen input: {path}; no generation fallback")
    st = path.stat()
    out = dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    if digest:
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(block)
        out["sha256"] = sha.hexdigest()
    return out


REGION_DEFINITIONS = {
    "version": "M12.4.1-train-regions-v1",
    "reference": "frozen G1 train only; M12.0 geometry",
    "percentile": "100 * searchsorted(sorted_train_distance, distance, side=right) / n_train",
    "marginal_distance": "abs((physical_label - train_mean) / train_std_ddof0)",
    "global_distance": "sqrt(z.T @ frozen_precision @ z)",
    "central": "percentile < 80", "peripheral": "percentile >= 80",
    "core_annotation": "marginal percentile <= 50",
    "discordant": "each mass central AND spin peripheral; each mass peripheral AND spin central",
}
LAYER_GROUPS = {"shallow": ["block1", "block2"],
                "middle": ["block3", "residual_blocks"],
                "deep": ["projection", "embedding_layer"]}
OBJECTIVE = "unweighted mean((pred_std[:,k]-target_std[:,k])**2) per task"


def make_regions(global_pct, marginal_pct):
    require(marginal_pct.shape == (len(global_pct), 3)
            and np.isfinite(global_pct).all() and np.isfinite(marginal_pct).all(),
            "Invalid region percentile arrays")
    regions = {"general": np.ones(len(global_pct), dtype=bool),
               "global_central": global_pct < 80, "global_peripheral": global_pct >= 80}
    for j, label in enumerate(LABELS):
        regions[label + "_central"] = marginal_pct[:, j] < 80
        regions[label + "_peripheral"] = marginal_pct[:, j] >= 80
    for j, label in enumerate(LABELS[:2]):
        regions[label + "_central_spin_peripheral"] = (marginal_pct[:, j] < 80) & (marginal_pct[:, 2] >= 80)
        regions[label + "_peripheral_spin_central"] = (marginal_pct[:, j] >= 80) & (marginal_pct[:, 2] < 80)
    return regions


def make_schedule(train_ids, regions, seed, pilot, max_batches, batch_size):
    """Global per-checkpoint budget, independent of selected models and device."""
    require(1 <= batch_size <= 64 and 4 <= max_batches <= 32, "Batch budget outside bounds")
    pilot_regions = ["general", "global_peripheral", "chirp_mass_central_spin_peripheral",
                     "total_mass_peripheral_spin_central"]
    order = list(regions)
    selected = pilot_regions if pilot else [order[i % len(order)] for i in range(max_batches)]
    rng = np.random.default_rng(seed)
    schedule = []
    for region in order:
        batch_numbers = [i for i, r in enumerate(selected) if r == region]
        pool = np.flatnonzero(regions[region])
        require(len(pool) >= len(batch_numbers)*batch_size, f"Insufficient train population: {region}")
        picks = rng.choice(pool, size=len(batch_numbers)*batch_size, replace=False)
        for offset, batch_id in enumerate(batch_numbers):
            positions = picks[offset*batch_size:(offset+1)*batch_size]
            ids = train_ids[positions].copy()
            schedule.append(dict(batch_id=batch_id, region=region, positions=positions,
                                 ids=ids, ids_sha256=array_hash(ids)))
    return sorted(schedule, key=lambda b: b["batch_id"])


def grouped_read(ds, rows):
    """Existing sampled-loader strategy: bounded spans, exact order/multiplicity."""
    unique, inverse = np.unique(rows, return_inverse=True)
    chunk_rows = min(64, ds.chunks[0]) if ds.chunks else 64
    values = np.empty((len(unique), *ds.shape[1:]), dtype=ds.dtype)
    start, fetched = 0, 0
    while start < len(unique):
        end = start + 1
        while end < len(unique):
            if unique[end] // chunk_rows != unique[start] // chunk_rows:
                break
            if unique[end] - unique[start] + 1 > 4 * (end - start + 1):
                break
            end += 1
        lo, hi = int(unique[start]), int(unique[end-1])+1
        block = ds[lo:hi]
        values[start:end] = block[unique[start:end]-lo]
        fetched += hi-lo
        del block
        start = end
    return values[inverse], fetched


def gradient_statistics(ga, gb, idx):
    # None is a documented zero contribution, not a silently dropped coordinate.
    aa = sum(float(torch.sum(ga[i] ** 2)) for i in idx if ga[i] is not None)
    bb = sum(float(torch.sum(gb[i] ** 2)) for i in idx if gb[i] is not None)
    dot = sum(float(torch.sum(ga[i]*gb[i])) for i in idx if ga[i] is not None and gb[i] is not None)
    na, nb = np.sqrt(aa), np.sqrt(bb)
    cosine = dot/(na*nb) if na > 0 and nb > 0 else np.nan
    return dict(cosine=cosine, dot=dot, norm_a=na, norm_b=nb,
                norm_ratio_a_over_b=na/nb if nb > 0 else np.nan,
                norm_ratio_b_over_a=nb/na if na > 0 else np.nan,
                negative_cosine=float(cosine < 0) if np.isfinite(cosine) else np.nan)


def state_unchanged(model, before):
    after = model.state_dict()
    return (set(before) == set(after)
            and all(torch.equal(before[k], after[k].detach().cpu()) for k in before))

def shared_groups(model):
    require(type(model) is SimpleCNN_ResidualDilated, "Unexpected architecture")
    require(set(dict(model.named_children())) == {"block1", "block2", "block3", "residual_blocks",
            "projection", "pool", "embedding_layer", "head"}, "Unexpected module hierarchy")
    require(all(isinstance(getattr(model, f"block{i}"), ConvBlock) for i in (1, 2, 3)), "Initial blocks")
    require(len(model.residual_blocks) == 3 and all(isinstance(b, ResidualDilatedBlock)
            for b in model.residual_blocks), "Residual blocks")
    require(isinstance(model.head, torch.nn.Linear) and model.head.out_features == 3
            and model.head.in_features == 64, "Output head contract")
    modules = {group: [getattr(model, name) for name in names]
               for group, names in LAYER_GROUPS.items()}
    named = list(model.named_parameters())
    groups = {g: [i for i, (_, p) in enumerate(named)
                  if id(p) in {id(q) for module in ms for q in module.parameters()}]
              for g, ms in modules.items()}
    groups["all_shared"] = sorted(sum(groups.values(), []))
    head_ids = {id(p) for p in model.head.parameters()}
    require(len(groups["all_shared"]) == len(set(groups["all_shared"])), "Overlapping layer groups")
    require({id(named[i][1]) for i in groups["all_shared"]} | head_ids == {id(p) for _, p in named},
            "Unclassified parameters")
    return named, groups


def summarize(results, seed, min_ci_batches=5, replicates=400):
    rng_boot = np.random.default_rng(seed + 1)
    def describe(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return dict(n=0, q25=np.nan, median=np.nan, q75=np.nan)
        q = np.quantile(values, [.25, .5, .75])
        return dict(n=len(values), q25=q[0], median=q[1], q75=q[2])

    def interval(values, statistic):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < min_ci_batches:
            return np.nan, np.nan
        draws = rng_boot.choice(values, size=(replicates, len(values)), replace=True)
        stats = np.median(draws, axis=1) if statistic == "median" else np.mean(draws < 0, axis=1)
        return tuple(np.quantile(stats, [.025, .975]))

    summary_rows = []
    keys = ["checkpoint", "pair", "group", "region"]
    for key, frame in results.groupby(keys, sort=False):
        finite = frame.cosine[np.isfinite(frame.cosine)]
        row = dict(zip(keys, key), batches=len(frame), undefined_cosines=len(frame)-len(finite),
                   negative_fraction=float((finite < 0).mean()) if len(finite) else np.nan,
                   uncertainty="descriptive batch bootstrap" if len(finite) >= min_ci_batches else "insufficient batches")
        for metric in ["cosine", "dot", "norm_a", "norm_b", "norm_ratio_a_over_b", "norm_ratio_b_over_a"]:
            row.update({metric+"_"+k: v for k, v in describe(frame[metric]).items()})
        row["cosine_median_ci_low"], row["cosine_median_ci_high"] = interval(finite, "median")
        row["negative_fraction_ci_low"], row["negative_fraction_ci_high"] = interval(finite, "negative")
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    return summary

# v2 is opt-in. The legacy schedule, reductions and export schema remain unchanged.
LARGE_METRICS = ["cosine", "dot", "norm_a", "norm_b", "norm_ratio_a_over_b", "norm_ratio_b_over_a"]
LARGE_PAIRS = [(LABELS[a]+" / "+LABELS[b], a, b) for a, b in TASK_PAIRS] + [
    (label+" / combined_MSE", k, 3) for k, label in enumerate(LABELS)]
STABILITY = dict(min_batches=30, min_finite_fraction=.95, cosine_max_halfwidth=.10,
                 cosine_max_change=.03, consecutive_changes=2,
                 interpretation="predefined descriptive adequacy, not a sequential test or stopping rule")


def large_schedule(train_ids, regions, seed, examples=40000, batch_size=64,
                   selected_regions=(), region_examples=5000):
    """Conservative global budget: sum of cohort sizes <= examples, union <= sum.

    Region draws are independent SRSWOR samples. General receives the remainder.
    Overlap is retained/reported, never silently refilled or pooled. Full random
    permutations make prefixes invariant to extending a cohort's requested size.
    """
    require(1 <= examples <= len(train_ids) and 1 <= batch_size <= 64, "Large budget exceeds train population")
    require(len(set(selected_regions)) == len(selected_regions), "Duplicate region request")
    require(all(r in regions and r != "general" for r in selected_regions), "Unknown/nonregional cohort")
    require(region_examples > 0, "Region target must be positive")
    n_general = examples - len(selected_regions)*region_examples
    require(n_general > 0, "Regional targets exhaust global example budget; reduce regions/targets or explicitly raise budget")
    targets = {"general": n_general, **{r: region_examples for r in selected_regions}}
    schedule, cohorts, used = [], {}, []
    for region, target in targets.items():
        pool = np.flatnonzero(regions[region])
        require(target <= len(pool), f"Insufficient eligible population for {region}: requested {target}, available {len(pool)}")
        code = list(regions).index(region)
        rng = np.random.default_rng(np.random.SeedSequence([seed, code]))
        positions = rng.permutation(pool)[:target]
        cohorts[region] = dict(population=len(pool), selected=target,
                               selection_fraction=target/len(pool), seed_code=code)
        used.append(train_ids[positions])
        for start in range(0, target, batch_size):
            pos = positions[start:start+batch_size]
            ids = train_ids[pos].copy()
            schedule.append(dict(batch_id=len(schedule), region=region, region_batch=start//batch_size,
                                 positions=pos, ids=ids, ids_sha256=array_hash(ids),
                                 cumulative_examples=min(start+batch_size, target)))
    overlap = [[int(len(np.intersect1d(a, b))) for b in used] for a in used]
    union_count = len(np.unique(np.concatenate(used)))
    require(union_count <= examples, "Internal global distinct-example budget violation")
    return schedule, dict(cohorts=cohorts, overlap=overlap, overlap_order=list(targets),
                          distinct_union=union_count, total_cohort_exposures=sum(targets.values()),
                          example_budget=examples, general_examples=n_general)


def cumulative_counts(n, batch_size):
    return sorted({min(n, ((t+batch_size-1)//batch_size)*batch_size)
                   for t in [2500, 5000, 10000, 20000, 40000] if t <= n} | {n})


def task_gradients(model, X, target, named):
    """Same eval-mode, unweighted task losses as v1; one graph, three VJPs."""
    require(not any(m.training for m in model.modules()), "Model left eval mode")
    with torch.enable_grad():
        pred = model(X)
        require(pred.shape == target.shape and torch.isfinite(pred).all().item(), "Invalid prediction")
        losses = ((pred-target)**2).mean(dim=0)
        grads = []
        head_ids = {id(p) for p in model.head.parameters()}
        for k in range(3):
            raw = torch.autograd.grad(losses[k], [p for _, p in named],
                                      retain_graph=k < 2, create_graph=False, allow_unused=True)
            detached = [None if g is None else g.detach().cpu().double() for g in raw]
            require(all(g is None or torch.isfinite(g).all().item() for g in detached), "Nonfinite gradient")
            for i, (_, p) in enumerate(named):
                if id(p) in head_ids:
                    require(detached[i] is not None, "Disconnected output head")
                    require(torch.count_nonzero(detached[i][[j for j in range(3) if j != k]]).item() == 0,
                            "Unexpected cross-output head coupling")
            grads.append(detached)
            del raw, detached
    return grads, losses.detach().cpu().tolist()


def flat_layout(named, groups):
    offsets, end = {}, 0
    for i in groups["all_shared"]:
        offsets[i] = (end, end+named[i][1].numel())
        end = offsets[i][1]
    slices = {}
    for group, indices in groups.items():
        bounds = [offsets[i] for i in indices]
        require(all(a[1] == b[0] for a, b in zip(bounds, bounds[1:])), "Noncontiguous verified layer group")
        slices[group] = slice(bounds[0][0], bounds[-1][1])
    return offsets, slices, end


def flatten_shared(grads, named, offsets, width):
    flat = torch.zeros((3, width), dtype=torch.float64)
    for i, (lo, hi) in offsets.items():
        for k in range(3):
            if grads[k][i] is not None:
                flat[k, lo:hi] = grads[k][i].reshape(-1)
    return flat


def geometry_from_gram(gram, reference=None, atol=1e-12, rtol=1e-6):
    """3-task Gram -> task pairs AND alignment to (g0+g1+g2)/3.

    Reference is the sample-weighted mean of batch Gram matrices, used only
    to flag tiny/cancelled aggregate norms relative to batch-gradient RMS.
    """
    G = np.asarray(gram, dtype=float)
    R = G if reference is None else np.asarray(reference, dtype=float)
    transform = np.vstack([np.eye(3), np.ones(3)/3])
    A, B = transform @ G @ transform.T, transform @ R @ transform.T
    norms = np.sqrt(np.maximum(np.diag(A), 0))
    floors = np.maximum(atol, rtol*np.sqrt(np.maximum(np.diag(B), 0)))
    out = []
    for pair, a, b in LARGE_PAIRS:
        dot, na, nb = float(A[a, b]), float(norms[a]), float(norms[b])
        valid = na > floors[a] and nb > floors[b]
        cosine = float(np.clip(dot/(na*nb), -1, 1)) if valid else np.nan
        out.append(dict(pair=pair, cosine=cosine, dot=dot, norm_a=na, norm_b=nb,
                        norm_ratio_a_over_b=na/nb if nb > floors[b] else np.nan,
                        norm_ratio_b_over_a=nb/na if na > floors[a] else np.nan,
                        negative_cosine=float(cosine < 0) if valid else np.nan,
                        norm_floor_a=float(floors[a]), norm_floor_b=float(floors[b]),
                        undefined_cosine=not valid))
    return out


class StreamingGradients:
    """Bounded delete-group sums. No per-example or per-batch vectors retained.

    Batch b is assigned to block b % K, identically for every model. Counts
    weight all means, including final partial batches. Only one cohort/model
    accumulator is live. Delete-group pseudovalues below yield approximate
    normal intervals; unequal blocks use weighted pseudovalues explicitly.
    """
    def __init__(self, width, slices, blocks=20, max_mib=256):
        # Include sum/mean/leave-group workspaces, not just the persistent sums.
        needed = (blocks+6)*3*width*8
        require(needed <= max_mib*2**20, f"Aggregate workspace requires {needed/2**20:.1f} MiB > {max_mib} MiB")
        self.sums = torch.zeros((blocks, 3, width), dtype=torch.float64)
        self.counts = np.zeros(blocks, dtype=np.int64)
        self.slices, self.batches = slices, 0
        self.energy = {g: np.zeros((3, 3)) for g in slices}
        self.workspace_bound_bytes = needed

    def add(self, flat, count):
        require(count > 0 and torch.isfinite(flat).all().item(), "Invalid streaming batch")
        block = self.batches % len(self.counts)
        self.sums[block].add_(flat, alpha=int(count))
        self.counts[block] += count
        self.batches += 1
        for group, sl in self.slices.items():
            part = flat[:, sl]
            self.energy[group] += count*(part @ part.T).numpy()

    def mean(self):
        return self.sums.sum(dim=0)/int(self.counts.sum())

    def snapshot(self, atol=1e-12, rtol=1e-6):
        n = int(self.counts.sum())
        summed = self.sums.sum(dim=0)
        def evaluate(values):
            rows = []
            for group, sl in self.slices.items():
                part = values[:, sl]
                for row in geometry_from_gram((part @ part.T).numpy(), self.energy[group]/n, atol, rtol):
                    rows.append(dict(group=group, **row))
            return rows
        rows = evaluate(summed/n)
        theta = np.array([[r[m] for m in LARGE_METRICS] for r in rows])
        deleted = np.full((len(self.counts), *theta.shape), np.nan)
        for block, count in enumerate(self.counts):
            if 0 < count < n:
                leave = evaluate((summed-self.sums[block])/(n-int(count)))
                deleted[block] = [[r[m] for m in LARGE_METRICS] for r in leave]
        return rows, theta, deleted


def jackknife_interval(theta, deleted, counts, population, min_blocks=10):
    """Weighted delete-group pseudo-values; asymptotic normal + finite-population correction.

    Equal blocks reduce to the standard delete-group jackknife. Unequal block
    sizes use p_i=(theta-(1-w_i)*theta_-i)/w_i, weighted centered variance.
    Coverage is approximate, especially for nonlinear near-zero norms. Batch
    groups partition the SRSWOR sample, not independently trained models.
    """
    theta = np.asarray(theta, dtype=float)
    n = int(np.sum(counts))
    w = np.asarray(counts, dtype=float)/n
    mask = w > 0
    w, deleted = w[mask], np.asarray(deleted)[mask]
    shaped = w.reshape((-1,)+(1,)*theta.ndim)
    pseudo = (theta-(1-shaped)*deleted)/shaped
    center = np.sum(shaped*pseudo, axis=0)
    se = np.sqrt(np.sum(shaped**2*(pseudo-center)**2, axis=0)/(1-np.sum(w*w))) if len(w) > 1 else np.full_like(theta, np.nan)
    # At census the aggregate is exact conditional on this finite population.
    fpc = np.sqrt(max(0., (population-n)/max(1, population-1)))
    se *= fpc
    if n == population:
        se = np.where(np.isfinite(theta), 0., np.nan)
    elif len(w) < min_blocks:
        se[:] = np.nan
    return theta-1.96*se, theta+1.96*se, se


def batch_bootstrap(values, seed, replicates=400):
    """Paired batch indices for all model/metric columns, one replicate at a time.

    Input (models, batches, columns). Primary estimands: mean, median and
    negative-cosine fraction. Caller supplies full-size batches only and NaN
    for undefined entries. No FPC: nonlinear minibatch quantities depend on
    grouping even at a census. Return paired differences using same resamples.
    """
    values = np.asarray(values, dtype=float)
    m, n, d = values.shape
    require(n > 0, "No full-size batches for uncertainty")
    def stats(v):
        # All-undefined columns stay NaN without warning floods.
        finite = np.isfinite(v)
        count = finite.sum(axis=1)
        avg = np.divide(np.where(finite, v, 0).sum(axis=1), count,
                        out=np.full((m, d), np.nan), where=count > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-undefined columns remain NaN
            med = np.nanmedian(v, axis=1)
        return np.stack([avg, med], axis=-1)
    point = stats(values)
    boot = np.empty((replicates, *point.shape))
    rng = np.random.default_rng(seed)
    for b in range(replicates):
        boot[b] = stats(values[:, rng.integers(0, n, size=n), :])
    return point, boot


def append_csv(handle, rows, writer=None):
    if rows:
        if writer is None:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
        writer.writerows(rows)
    return writer


def atomic_json(path, value):
    # Only files created inside this run directory are replaced, never old runs.
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def large_inference(values, schedule, snapshots, models, design, seed, batch_size):
    """Scalar-only inference; same bootstrap indices across models, never across regions.

    Values shape M x total_batches x 24 coordinates x 7 metrics. This bounded
    array is <= ~34 MiB at four models and 400000 observations (batch size64).
    No gradient vectors enter this function or the exports.
    """
    columns = LARGE_METRICS + ["negative_cosine"]
    coordinates = [(g, pair) for g in [*LAYER_GROUPS, "all_shared"] for pair, _, _ in LARGE_PAIRS]
    convergence, paired = [], []
    for region, cohort in design["cohorts"].items():
        batches = [b for b in schedule if b["region"] == region]
        for n in cumulative_counts(cohort["selected"], batch_size):
            chosen = [b["batch_id"] for b in batches if b["cumulative_examples"] <= n and len(b["ids"]) == batch_size]
            if chosen:
                data = values[:, chosen].reshape(len(models), len(chosen), -1)
                # Paired inference conditions on the same finite batches for every model.
                joint = np.isfinite(data).all(axis=0)
                data = np.where(joint[None], data, np.nan)
                point, boot = batch_bootstrap(data, np.random.SeedSequence([seed, list(design["cohorts"]).index(region), n]))
                fraction = np.mean(joint, axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bounds = np.nanquantile(boot, [.025, .975], axis=0)
                    quartiles = np.nanquantile(data, [.25, .75], axis=1)
                for ci, (group, pair) in enumerate(coordinates):
                    for mi, metric in enumerate(columns):
                        col = ci*len(columns)+mi
                        eligible = len(chosen) >= STABILITY["min_batches"] and fraction[col] >= STABILITY["min_finite_fraction"]
                        statistics = ["mean"] if metric == "negative_cosine" else ["mean", "median"]
                        for si, statistic in enumerate(statistics):
                            for model_index, name in enumerate(models):
                                convergence.append(dict(checkpoint=name, region=region, group=group, pair=pair,
                                    n_examples=n, n_batches=len(chosen), kind="minibatch", metric=metric, statistic=statistic,
                                    estimate=point[model_index, col, si],
                                    ci_low=bounds[0, model_index, col, si] if eligible else np.nan,
                                    ci_high=bounds[1, model_index, col, si] if eligible else np.nan,
                                    q25=quartiles[0, model_index, col], q75=quartiles[1, model_index, col],
                                    finite_fraction=fraction[col], adequate=eligible,
                                    method="paired batch percentile bootstrap; common finite full-size batches"))
                            for a, b in combinations(range(len(models)), 2):
                                delta = boot[:, b, col, si]-boot[:, a, col, si]
                                interval = np.quantile(delta, [.025, .975]) if eligible and np.isfinite(delta).all() else [np.nan, np.nan]
                                paired.append(dict(reference=models[a], checkpoint=models[b], region=region,
                                    group=group, pair=pair, n_examples=n, kind="minibatch", metric=metric, statistic=statistic,
                                    estimate=point[b, col, si]-point[a, col, si], ci_low=interval[0], ci_high=interval[1],
                                    method="paired batch bootstrap; no between-region or between-seed inference"))
            # Aggregate point and leave-group estimates computed online, never rerun.
            for model_index, name in enumerate(models):
                snap = snapshots[(name, region, n)]
                theta, deleted, counts = snap["theta"], snap["deleted"], snap["counts"]
                lo, hi, se = jackknife_interval(theta, deleted, counts, cohort["population"])
                for ci, (group, pair) in enumerate(coordinates):
                    for mi, metric in enumerate(LARGE_METRICS):
                        finite = np.isfinite([theta[ci, mi], lo[ci, mi], hi[ci, mi]]).all()
                        eligible = bool(finite and (snap["batches"] >= STABILITY["min_batches"] or n == cohort["population"]))
                        convergence.append(dict(checkpoint=name, region=region, group=group, pair=pair,
                            n_examples=n, n_batches=snap["batches"], kind="sample_aggregate", metric=metric, statistic="geometry_of_mean_gradient",
                            estimate=theta[ci, mi], ci_low=lo[ci, mi] if eligible else np.nan,
                            ci_high=hi[ci, mi] if eligible else np.nan, q25=np.nan, q75=np.nan,
                            finite_fraction=float(np.isfinite(deleted[:, ci, mi]).mean()), adequate=eligible,
                            method="weighted delete-group normal approximation with aggregate FPC"))
            for a, b in combinations(models, 2):
                sa, sb = snapshots[(a, region, n)], snapshots[(b, region, n)]
                require(np.array_equal(sa["counts"], sb["counts"]), "Unpaired aggregate delete groups")
                delta, deleted_delta = sb["theta"]-sa["theta"], sb["deleted"]-sa["deleted"]
                lo, hi, se = jackknife_interval(delta, deleted_delta, sa["counts"], cohort["population"])
                for ci, (group, pair) in enumerate(coordinates):
                    for mi, metric in enumerate(LARGE_METRICS):
                        eligible = sa["batches"] >= STABILITY["min_batches"] or n == cohort["population"]
                        paired.append(dict(reference=a, checkpoint=b, region=region, group=group, pair=pair,
                            n_examples=n, kind="sample_aggregate", metric=metric, statistic="geometry_of_mean_gradient",
                            estimate=delta[ci, mi], ci_low=lo[ci, mi] if eligible else np.nan,
                            ci_high=hi[ci, mi] if eligible else np.nan,
                            method="paired delete-group delta; aggregate FPC; no cross-region inference"))
    convergence = pd.DataFrame(convergence)
    convergence["stable_geometry"] = False
    convergence["absolute_change"] = np.nan
    keys = ["checkpoint", "region", "group", "pair", "kind", "metric", "statistic"]
    for _, frame in convergence[convergence.metric.isin(["cosine", "negative_cosine"])].groupby(keys, sort=False):
        frame = frame.sort_values("n_examples")
        changes = frame.estimate.diff().abs()
        halfwidth = (frame.ci_high-frame.ci_low)/2
        steady = (changes <= STABILITY["cosine_max_change"]).rolling(STABILITY["consecutive_changes"]).sum().eq(STABILITY["consecutive_changes"])
        convergence.loc[frame.index, "absolute_change"] = changes
        convergence.loc[frame.index, "stable_geometry"] = (steady & frame.adequate
            & (halfwidth <= STABILITY["cosine_max_halfwidth"])).to_numpy()
    return convergence, pd.DataFrame(paired, columns=["reference", "checkpoint", "region", "group", "pair",
        "n_examples", "kind", "metric", "statistic", "estimate", "ci_low", "ci_high", "method"])


def run_large(args, output_dir, provenance, train_ids, regions, global_pct, marginal_pct,
              load_frozen, read_batch, dataset_path, device):
    schedule, design = large_schedule(train_ids, regions, args.seed, args.examples, args.batch_size,
                                     args.regions, args.region_examples)
    models = args.models
    # Scalar cache is bounded separately from vector accumulators. Small batch
    # requests at full-population size must fail rather than allocate many GiB.
    cache_bytes = len(models)*len(schedule)*24*7*8
    require(cache_bytes <= 128*2**20, f"Scalar cache would require {cache_bytes/2**20:.1f} MiB; increase batch size (max64) or reduce explicit budget")
    values = np.full((len(models), len(schedule), 24, 7), np.nan)
    snapshots, invariants, inventory, coverage, timings = {}, [], [], [], []
    output_dir.mkdir(parents=True, exist_ok=False)
    progress = dict(schema_version=2, status="running", resume_supported=False,
                    restart="An interrupted run must restart in a NEW directory; partial files are not completed results",
                    completed=[], planned_batches_per_model=len(schedule), models=models)
    atomic_json(output_dir/"progress.json", progress)
    # Compact normalized manifests: no repeated row-ID lists or quadratic batch-overlap matrix.
    with (output_dir/"batch_manifest.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_id", "region", "region_batch", "sample_order", "row_id"])
        writer.writeheader()
        for batch in schedule:
            for order, row_id in enumerate(batch["ids"]):
                writer.writerow(dict(batch_id=batch["batch_id"], region=batch["region"], region_batch=batch["region_batch"],
                                     sample_order=order, row_id=int(row_id)))
    batch_index = pd.DataFrame([dict(batch_id=b["batch_id"], region=b["region"], region_batch=b["region_batch"],
        n_examples=len(b["ids"]), cumulative_examples=b["cumulative_examples"], ids_sha256=b["ids_sha256"],
        jackknife_block=b["region_batch"] % args.jackknife_blocks) for b in schedule])
    batch_index.to_csv(output_dir/"batch_index.csv", index=False)
    chosen_positions = np.unique(np.concatenate([b["positions"] for b in schedule]))
    sample_info = pd.DataFrame(dict(row_id=train_ids[chosen_positions], global_percentile=global_pct[chosen_positions]))
    for j, label in enumerate(LABELS):
        sample_info[label+"_percentile"] = marginal_pct[chosen_positions, j]
    sample_info.to_csv(output_dir/"sample_annotations.csv", index=False)
    cohort_rows = [dict(region=r, **row) for r, row in design["cohorts"].items()]
    # Report eligible sizes even for cohorts not explicitly selected.
    provenance["population"] = [dict(region=r, population=int(mask.sum()),
        selected=design["cohorts"].get(r, {}).get("selected", 0)) for r, mask in regions.items()]
    print(json.dumps(dict(design=design, batches_per_model=len(schedule),
                         total_forward_calls=len(models)*len(schedule), total_task_backward_calls=3*len(models)*len(schedule)), indent=2), flush=True)
    coordinates = [(g, pair) for g in [*LAYER_GROUPS, "all_shared"] for pair, _, _ in LARGE_PAIRS]
    scalar_fields = LARGE_METRICS + ["negative_cosine"]
    with (output_dir/"gradients_per_batch.csv").open("x", newline="") as batch_file, \
         (output_dir/"aggregate_gradients.csv").open("x", newline="") as aggregate_file, \
         (output_dir/"aggregate_delete_groups.csv").open("x", newline="") as delete_file:
        batch_writer = aggregate_writer = delete_writer = None
        for model_index, name in enumerate(models):
            model = load_frozen(name)
            named, groups = shared_groups(model)
            offsets, slices, width = flat_layout(named, groups)
            inventory.extend(dict(checkpoint=name, group=group, parameter=named[i][0],
                                  shape=list(named[i][1].shape), elements=named[i][1].numel())
                             for group in LAYER_GROUPS for i in groups[group])
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            accumulator = None
            previous_region = None
            coverage_counts = {}
            try:
                with h5py.File(dataset_path, "r") as h:
                    for batch in schedule:
                        if batch["region"] != previous_region:
                            # Release prior cohort sums before allocating the next.
                            accumulator = None
                            accumulator = StreamingGradients(width, slices, args.jackknife_blocks)
                            previous_region = batch["region"]
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        X, target, fetched = read_batch(h, batch["ids"])
                        X, target = X.to(device), target.to(device)
                        grads, losses = task_gradients(model, X, target, named)
                        del X, target
                        # Persistent coverage counters are O(parameters), not O(batches*parameters).
                        for k, label in enumerate(LABELS):
                            for i in groups["all_shared"]:
                                key = (batch["region"], label, named[i][0])
                                count = coverage_counts.setdefault(key, [0, 0, 0])
                                count[0] += 1
                                count[1] += int(grads[k][i] is None)
                                count[2] += int(grads[k][i] is not None and torch.count_nonzero(grads[k][i]).item() == 0)
                        flat = flatten_shared(grads, named, offsets, width)
                        del grads
                        accumulator.add(flat, len(batch["ids"]))
                        rows = []
                        for group, sl in slices.items():
                            part = flat[:, sl]
                            for row in geometry_from_gram((part @ part.T).numpy(), atol=args.norm_atol, rtol=args.norm_rtol):
                                rows.append(dict(checkpoint=name, batch_id=batch["batch_id"], region=batch["region"],
                                    region_batch=batch["region_batch"], ids_sha256=batch["ids_sha256"],
                                    n_examples=len(batch["ids"]), cumulative_examples=batch["cumulative_examples"],
                                    group=group, **row, loss_chirp=losses[0], loss_total_mass=losses[1], loss_spin=losses[2]))
                        del flat, part
                        values[model_index, batch["batch_id"]] = [[r[m] for m in scalar_fields] for r in rows]
                        batch_writer = append_csv(batch_file, rows, batch_writer)
                        n = batch["cumulative_examples"]
                        cohort = design["cohorts"][batch["region"]]
                        if n in cumulative_counts(cohort["selected"], args.batch_size):
                            rows, theta, deleted = accumulator.snapshot(args.norm_atol, args.norm_rtol)
                            counts = accumulator.counts.copy()
                            snapshots[(name, batch["region"], n)] = dict(theta=theta, deleted=deleted,
                                counts=counts, batches=accumulator.batches)
                            lo, hi, se = jackknife_interval(theta, deleted, counts, cohort["population"])
                            aggregate_rows, delete_rows = [], []
                            for ci, row in enumerate(rows):
                                record = dict(checkpoint=name, region=batch["region"], n_examples=n,
                                    n_batches=accumulator.batches, population=cohort["population"], selection_fraction=n/cohort["population"],
                                    exact_finite_population=(n == cohort["population"]), **row)
                                for mi, metric in enumerate(LARGE_METRICS):
                                    record[metric+"_ci_low"] = lo[ci, mi]
                                    record[metric+"_ci_high"] = hi[ci, mi]
                                    record[metric+"_se"] = se[ci, mi]
                                aggregate_rows.append(record)
                                for block, count in enumerate(counts):
                                    delete_rows.append(dict(checkpoint=name, region=batch["region"], n_examples=n,
                                        group=row["group"], pair=row["pair"], block=block, removed_examples=int(count),
                                        **{m: deleted[block, ci, j] for j, m in enumerate(LARGE_METRICS)}))
                            aggregate_writer = append_csv(aggregate_file, aggregate_rows, aggregate_writer)
                            delete_writer = append_csv(delete_file, delete_rows, delete_writer)
                            require(state_unchanged(model, before) and all(p.grad is None for p in model.parameters()),
                                    "Frozen-state audit failed at cumulative checkpoint")
                            for f in (batch_file, aggregate_file, delete_file):
                                f.flush(); os.fsync(f.fileno())
                            progress["completed"].append(dict(checkpoint=name, region=batch["region"], n_examples=n,
                                batch_id=batch["batch_id"], parameters_and_buffers_unchanged=True))
                            atomic_json(output_dir/"progress.json", progress)
                            print(f"{name} {batch['region']}: {n}/{cohort['selected']} examples", flush=True)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        timings.append(dict(checkpoint=name, batch_id=batch["batch_id"], seconds=time.perf_counter()-started,
                                            fetched_storage_rows=fetched, selected_rows=len(batch["ids"])))
            finally:
                unchanged = state_unchanged(model, before)
                unset = all(p.grad is None for p in model.parameters())
                invariants.append(dict(checkpoint=name, parameters_and_buffers_unchanged=unchanged,
                    parameter_grad_fields_unset=unset, cuda_peak_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None))
                require(unchanged and unset, f"{name}: frozen-state invariant failed")
                accumulator = None
                del before, named, model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            coverage.extend(dict(checkpoint=name, region=key[0], task=key[1], parameter=key[2],
                batches=count[0], none_batches=count[1], zero_batches=count[2]) for key, count in coverage_counts.items())
    convergence, paired = large_inference(values, schedule, snapshots, models, design, args.seed, args.batch_size)
    convergence.to_csv(output_dir/"convergence.csv", index=False)
    paired.to_csv(output_dir/"paired_comparisons.csv", index=False)
    final = convergence[convergence.apply(lambda r: r.n_examples == design["cohorts"][r.region]["selected"], axis=1)]
    final.to_csv(output_dir/"gradients_summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(output_dir/"coverage.csv", index=False)
    pd.DataFrame(timings).to_csv(output_dir/"timings.csv", index=False)
    provenance.update(schema_version=2, status="complete", dataset_id="M11_final_G1_500k", labels=LABELS,
        models=models, objective=OBJECTIVE, region_definitions=REGION_DEFINITIONS, layer_groups=LAYER_GROUPS,
        parameter_inventory=inventory, design=design, actual_batches=len(schedule), invariants=invariants,
        preprocessing=dict(enabled=False, already_normalized=True, mode="per_sample_per_detector_zscore", eps=1e-6),
        settings=dict(vars(args), data_root=str(args.data_root) if args.data_root else None, output_dir=str(output_dir)),
        aggregation="sum(batch_size * batch_mean_gradient) / cumulative_distinct_cohort_examples; float64 CPU accumulators",
        combined_gradient="(g_chirp_mass + g_total_mass + g_chi_eff)/3; local negative SGD-direction derivative is -dot",
        uncertainty=dict(batch="400 paired percentile resamples of common finite full-size batches, no FPC",
            aggregate="weighted delete-group pseudovalues, 1.96 normal approximation, FPC sqrt((N-n)/(N-1))",
            blocks=args.jackknife_blocks, assignment="region_batch modulo K; identical across models",
            unequal_blocks="w_i=n_i/n, p_i=(theta-(1-w_i)*theta_minus_i)/w_i; SE^2=sum(w_i^2*(p_i-sum(w*p))^2)/(1-sum(w_i^2))",
            dependence="No cross-region pooled or independence-based intervals; cohort overlap is reported",
            limits="Conditional fixed-checkpoint finite-sample approximations; no training-seed variability or multiplicity correction"),
        stability=STABILITY, resume_supported=False, script=file_record(Path(__file__).resolve(), True),
        memory=dict(scalar_cache_bytes=cache_bytes, aggregate_workspace_limit_MiB=256,
                    scalar_cache_limit_MiB=128, one_model_one_cohort_one_batch_graph=True),
        sampling="per-cohort train-only SRSWOR permutation SeedSequence([seed,frozen_region_index]); overlap retained; no budget refill",
        cumulative_thresholds=[2500, 5000, 10000, 20000, 40000],
        limitations=["Endpoint gradients, not training-trajectory interference or exact Adam-step effects",
                     "Mean minibatch cosine is not cosine of mean gradients",
                     "Non-census aggregate is a finite sample gradient, not the exact population gradient",
                     "Thresholds and nested convergence looks are descriptive, not a sequential stopping test",
                     "No full strain digest; hashes identify checkpoint files rather than independently authenticate them"])
    export_names = [p.name for p in output_dir.glob("*.csv")]
    provenance["exports"] = {name: file_record(output_dir/name, True) for name in export_names}
    atomic_json(output_dir/"run_metadata.json", provenance)
    progress["status"] = "complete"
    atomic_json(output_dir/"progress.json", progress)
    print(f"Completed: {output_dir}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(CONFIG_NAMES), default=list(CONFIG_NAMES))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--pilot", action="store_true", help="Four total batches per model (overrides --max-batches)")
    parser.add_argument("--max-batches", type=int, default=None, help="Compatibility mode: TOTAL across regions per model, default28, 4–32")
    parser.add_argument("--batch-size", type=int, default=None, help="1–64; scientific runs use 32–64")
    parser.add_argument("--seed", type=int, default=123, help="Diagnostic sampling seed; frozen split/training seed remains 123")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, help="New directory under resolved external data root; never overwritten")
    parser.add_argument("--large-sample", action="store_true", help="Opt-in v2: fixed distinct examples and streaming aggregates")
    parser.add_argument("--examples", type=int, default=None, help="Total cohort exposure/distinct-union ceiling; default 40000 in large mode; max 400000")
    parser.add_argument("--regions", nargs="*", default=[], help="Explicit regional cohorts, charged to --examples")
    parser.add_argument("--region-examples", type=int, default=5000)
    parser.add_argument("--jackknife-blocks", type=int, default=20, help="10–32 bounded in-memory delete groups")
    parser.add_argument("--norm-atol", type=float, default=1e-12)
    parser.add_argument("--norm-rtol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.large_sample and args.max_batches is not None:
        parser.error("--max-batches is compatibility-only; large mode is bounded by --examples")
    args.max_batches = 28 if args.max_batches is None else args.max_batches
    if args.batch_size is None:
        args.batch_size = 64 if args.large_sample else 32
    if args.large_sample and args.pilot:
        parser.error("--pilot is the compatibility mode; do not combine with --large-sample")
    if not args.large_sample and (args.examples is not None or args.regions):
        parser.error("--examples and --regions require --large-sample")
    args.examples = 40000 if args.examples is None else args.examples
    if not 1 <= args.examples <= 400000 or not 10 <= args.jackknife_blocks <= 32 or args.region_examples < 1:
        parser.error("Examples must be 1–400000, jackknife blocks 10–32, regional target positive")
    if not np.isfinite([args.norm_atol, args.norm_rtol]).all() or min(args.norm_atol, args.norm_rtol) < 0:
        parser.error("Norm tolerances must be finite and nonnegative")
    if len(args.models) != len(set(args.models)):
        parser.error("Duplicate model names")
    if not 4 <= args.max_batches <= 32 or not 1 <= args.batch_size <= 64 or args.seed < 0 or args.seed >= 2**32:
        parser.error("Require 4 <= max-batches <= 32, 1 <= batch-size <= 64, 0 <= seed < 2**32")
    return args


def main(argv=None):
    args = parse_args(argv)
    PILOT_ONLY, MAX_BATCHES, BATCH_SIZE = args.pilot, args.max_batches, args.batch_size
    DEVICE, CPU_THREADS, SEED = args.device, 2, args.seed
    require(1 <= BATCH_SIZE <= 64 and 4 <= MAX_BATCHES <= 32, "Bound exceeded: batch <=64, batches 4–32")
    require(1 <= CPU_THREADS <= 4, "CPU thread budget must be 1–4")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    device = torch.device(DEVICE)
    require(device.type in ("cpu", "cuda"), "Use CPU or CUDA")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA explicitly requested but unavailable")
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)  # supported in 1.9.1; fail on unsupported kernels
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    CONFIG_PATHS = {m: ROOT / "configs/experiments" / CONFIG_NAMES[m] for m in dict.fromkeys(["M11"] + args.models)}
    for p in CONFIG_PATHS.values():
        file_record(p)
    configs = {m: json.loads(p.read_text()) for m, p in CONFIG_PATHS.items()}
    base = configs["M11"]
    DATA_ROOT = resolve_data_root(cli_data_root=args.data_root, config_data_root=base.get("data_root"))
    configs = {m: configs[m] for m in args.models}
    output_dir = (args.output_dir or DATA_ROOT / "results/M11_final_G1_500k/M12_4_1" /
                  datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).expanduser().resolve()
    require(DATA_ROOT != ROOT and ROOT not in DATA_ROOT.parents, "Data root must be outside the repository")
    require(DATA_ROOT in output_dir.parents and ROOT not in output_dir.parents,
            "Output directory must be below the external data root and outside Git")
    require(not any((parent / ".git").exists() for parent in (output_dir, *output_dir.parents)),
            "Output directory must be outside all Git working trees")
    require(not output_dir.exists(), f"Output destination already exists: {output_dir}; choose a new tag")
    DATASET_ID = "M11_final_G1_500k"
    PATHS = {k: resolve_processed_artifact(data_root=DATA_ROOT, dataset_id=DATASET_ID,
        file_name=base["dataset"][k], allow_legacy_flat=False)
        for k in ("dataset_file", "split_file", "label_stats_file")}
    CONTRACT_PATH = ROOT / "notebooks/m12/outputs/m12_0_baseline_contract.json"
    CHECKPOINTS = {}
    for name, cfg in configs.items():
        require(cfg["outputs"]["checkpoint_tag"] == CHECKPOINT_TAGS[name],
                f"{name}: expected the exact frozen checkpoint tag {CHECKPOINT_TAGS[name]}")
        for key in ("dataset", "model", "input_normalization"):
            require(cfg[key] == base[key], f"{name}: changed {key}")
        require(cfg["training"]["seed"] == 123 and cfg["training"]["loss"] == "MSELoss", "Training contract")
        filename = (f"{DATASET_ID}_{cfg['model']['class_name']}_{cfg['outputs']['checkpoint_tag']}"
                    "_MSELoss_seed123_checkpoint.pt")
        canonical = DATA_ROOT / "models/checkpoints" / DATASET_ID / filename
        legacy = DATA_ROOT / "models/checkpoints" / filename
        CHECKPOINTS[name] = canonical if canonical.is_file() else legacy if legacy.is_file() else canonical
    provenance = dict(configs={m: file_record(p, True) for m, p in CONFIG_PATHS.items()},
        artifacts={k: file_record(p, k != "dataset_file") for k, p in PATHS.items()},
        contract=file_record(CONTRACT_PATH, True),
        checkpoints={m: file_record(p, True) for m, p in CHECKPOINTS.items()},
        sources={f: file_record(ROOT / f, True) for f in
                 ["src/models/network.py", "src/models/hdf5_batch_dataset.py", "src/models/sample_weighting.py"]},
        environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                         h5py=h5py.__version__, pandas=pd.__version__, platform=platform.platform(),
                         cpu=platform.processor(), cpu_count=os.cpu_count(), threads=torch.get_num_threads(),
                         device=str(device), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                         gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        settings=dict(seed=SEED, pilot=PILOT_ONLY, max_batches=MAX_BATCHES, batch_size=BATCH_SIZE))
    print(provenance["environment"], flush=True)
    frozen = json.loads(CONTRACT_PATH.read_text())
    require(frozen["schema_version"] == 1 and frozen["dataset_id"] == DATASET_ID
            and frozen["labels"] == LABELS, "Wrong M12.0 contract")
    geometry = frozen["geometry"]
    for k in ("split_file", "label_stats_file"):
        require(provenance["artifacts"][k]["sha256"] ==
                frozen["provenance"]["local_artifacts"][k]["sha256"], f"Frozen {k} hash mismatch")
    del frozen
    with np.load(PATHS["split_file"], allow_pickle=False) as z:
        train_ids = np.sort(z["train_idx"].copy())
        require(int(z["seed"].item()) == 123, "Split seed mismatch")
    require(train_ids.shape == (400000,) and np.issubdtype(train_ids.dtype, np.integer)
            and len(np.unique(train_ids)) == 400000
            and ((train_ids >= 0) & (train_ids < 500000)).all(), "Invalid train IDs")
    require(array_hash(train_ids) == geometry["train_indices_sha256"], "Frozen train ID mismatch")
    with np.load(PATHS["label_stats_file"], allow_pickle=False) as z:
        require(z["label_names"].tolist() == LABELS, "Scaler label order")
        require(np.array_equal(np.sort(z["train_idx"]), train_ids), "Scaler train membership")
        mean = np.asarray(z["y_mean"], dtype=np.float32)
        std = np.asarray(z["y_std"], dtype=np.float32)
    require(mean.shape == std.shape == (3,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all(), "Invalid scaler")
    normalization = base["input_normalization"]
    require(normalization == dict(enabled=False, already_normalized=True,
        mode="per_sample_per_detector_zscore", eps=1e-6), "Unexpected input preprocessing")
    with h5py.File(PATHS["dataset_file"], "r") as h:
        require({"X", "y", "split"} <= set(h.keys()), "Incompatible HDF5 schema: require X, y, split")
        require(h.attrs.get("experiment_id") == DATASET_ID
                and h.attrs.get("labels") == ",".join(LABELS), "HDF5 identity/order")
        require(h["X"].shape == (500000, 3, 16384) and h["y"].shape == (500000, 3), "HDF5 shapes")
        membership = h["split"].asstr()[:]
        require(np.array_equal(np.flatnonzero(membership == "train"), train_ids), "HDF5 train mismatch")
        del membership
        train_y = np.empty((len(train_ids), 3), dtype=np.float64)
        for start in range(0, len(train_ids), 4096):
            train_y[start:start+4096] = h["y"][train_ids[start:start+4096]]
    require(np.isfinite(train_y).all(), "Nonfinite labels")
    mu, sigma = np.asarray(geometry["mean"]), np.asarray(geometry["std_ddof0"])
    np.testing.assert_allclose(train_y.mean(0), mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(train_y.std(0), sigma, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.cov(train_y, rowvar=False), geometry["covariance_ddof1"], rtol=1e-12, atol=1e-12)
    z_train = (train_y - mu) / sigma
    squared = np.einsum("ni,ij,nj->n", z_train, np.asarray(geometry["precision"]), z_train)
    require(np.isfinite(squared).all() and (squared >= -1e-10).all(), "Invalid Mahalanobis geometry")
    distance = np.sqrt(np.maximum(squared, 0))
    reference = np.sort(distance)
    require(array_hash(reference) == geometry["reference_distance_sha256"], "Global reference hash")
    global_pct = 100 * np.searchsorted(reference, distance, side="right") / len(train_ids)
    marginal_pct = np.empty_like(train_y)
    marginal_hashes = {}
    for j, label in enumerate(LABELS):
        ref = np.sort(np.abs(z_train[:, j]))
        marginal_hashes[label] = array_hash(ref)
        marginal_pct[:, j] = 100 * np.searchsorted(ref, np.abs(z_train[:, j]), side="right") / len(train_ids)
    regions = make_regions(global_pct, marginal_pct)
    provenance.update(train_ids_sha256=array_hash(train_ids), train_ids_dtype=train_ids.dtype.str,
                      train_labels_sha256=array_hash(train_y),
                      marginal_reference_hashes=marginal_hashes)
    del z_train, squared, distance, reference, ref
    if not args.large_sample:
        schedule = make_schedule(train_ids, regions, SEED, PILOT_ONLY, MAX_BATCHES, BATCH_SIZE)
        selected_regions = [b["region"] for b in schedule]
        batch_manifest = pd.DataFrame([dict(batch_id=b["batch_id"], region=b["region"],
            row_ids=b["ids"].tolist(), ids_sha256=b["ids_sha256"]) for b in schedule])
        sample_manifest = pd.DataFrame([dict(batch_id=b["batch_id"], sample_order=i, region=b["region"],
            row_id=int(train_ids[p]), global_percentile=float(global_pct[p]),
            memberships=[r for r, mask in regions.items() if mask[p]],
            **{label+"_percentile": float(marginal_pct[p, j]) for j, label in enumerate(LABELS)},
            **{label+"_core_Ple50": bool(marginal_pct[p, j] <= 50) for j, label in enumerate(LABELS)})
            for b in schedule for i, p in enumerate(b["positions"])])
        population_table = pd.DataFrame([dict(region=r, population=int(mask.sum()),
            batches=selected_regions.count(r), sampled_rows=selected_regions.count(r)*BATCH_SIZE)
            for r, mask in regions.items()])
        overlap = pd.DataFrame([[len(np.intersect1d(a["ids"], b["ids"])) for b in schedule] for a in schedule],
                               index=[b["batch_id"] for b in schedule], columns=[b["batch_id"] for b in schedule])
        print(population_table.to_string(index=False), flush=True)
        print("Draws / unique rows per checkpoint:", len(sample_manifest), sample_manifest.row_id.nunique())
        provenance["batch_manifest"] = batch_manifest.to_dict("records")
    def read_batch(h, rows):
        require(np.isin(rows, train_ids).all(), "Non-train diagnostic row")
        X, fetched = grouped_read(h["X"], rows)
        unique, inverse = np.unique(rows, return_inverse=True)
        X = apply_input_normalization_to_batch(np.asarray(X, dtype=np.float32), normalization)
        y = np.asarray(h["y"][unique][inverse], dtype=np.float32)
        np.testing.assert_array_equal(y, train_y[np.searchsorted(train_ids, rows)].astype(np.float32))
        y = (y-mean)/(std+1e-8)
        require(np.isfinite(X).all() and np.isfinite(y).all(), "Nonfinite input/target")
        return torch.from_numpy(X), torch.from_numpy(y), fetched

    def load_frozen(name):
        # Only trusted repository research checkpoints: current torch defaults differ from 1.9.1.
        kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
        checkpoint = torch.load(CHECKPOINTS[name], **kwargs)
        require(isinstance(checkpoint, dict) and {"model_config", "model_state_dict"} <= set(checkpoint),
                f"{name}: incompatible checkpoint schema; require model_config and model_state_dict")
        mc = checkpoint["model_config"]
        cfg = configs[name]
        expected = dict(class_name="SimpleCNN_ResidualDilated", dataset_id=DATASET_ID,
            label_names=LABELS, model_kwargs=cfg["model"]["kwargs"], input_normalization=normalization,
            training_seed=123, split_seed=123, train_size=400000, val_size=40000,
            cal_size=30000, test_size=30000, n_detectors=3, signal_length=16384, n_outputs=3,
            normalization="GroupNorm", loss="MSELoss")
        for key, value in expected.items():
            require(mc.get(key) == value, f"{name}: checkpoint contract mismatch: {key}")
        for ck, pk in [("dataset_path", "dataset_file"), ("split_path", "split_file"), ("label_stats_path", "label_stats_file")]:
            require(Path(mc[ck]).name == PATHS[pk].name, f"{name}: artifact mismatch {ck}")
        if name in ("M12.1", "M12.2"):
            ref = mc["sample_weighting"]
            for k, v in cfg["training"]["sample_weighting"].items():
                require(ref[k] == v, f"{name}: weighting provenance {k}")
            require(ref["train_indices_sha256"] == geometry["train_indices_sha256"], "Weight train IDs")
            if name == "M12.1":
                require(ref["reference_distance_sha256"] == geometry["reference_distance_sha256"], "Global weight reference")
            else:
                require(ref["reference_distance_sha256"] == marginal_hashes and ref["label_names"] == LABELS,
                        "Marginal weight reference")
        elif name == "M12.3-S1":
            require(mc.get("sample_weighting") is None, "Unexpected weighted sampling model")
            ref = mc["balanced_sampling"]
            for k, v in dict(policy="M12.3-S1", mode="marginal_mixture", threshold=.8,
                             p_peripheral=.35, seed=123, replacement=True, n_train=400000).items():
                require(ref[k] == v, f"Sampling provenance {k}")
            require(ref["train_indices_sha256"] == geometry["train_indices_sha256"] and ref["label_names"] == LABELS,
                    "Sampling train IDs/labels")
            require(ref["train_reference"]["reference_distance_sha256"] == marginal_hashes, "Sampling references")
        else:
            require(mc.get("sample_weighting") is None and mc.get("balanced_sampling") is None, "M11 interventions")
        model = SimpleCNN_ResidualDilated(**mc["model_kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        provenance["checkpoints"][name].update(epoch=checkpoint.get("epoch"), model_config=mc)
        del checkpoint
        return model.to(device)
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if args.large_sample:
        run_large(args, output_dir, provenance, train_ids, regions, global_pct, marginal_pct,
                  load_frozen, read_batch, PATHS["dataset_file"], device)
        return

    records, coverage_rows, timing_rows, invariants, parameter_rows = [], [], [], [], []
    for name in configs:
        model = load_frozen(name)
        named, groups = shared_groups(model)
        print(name, model)
        for group in ("shallow", "middle", "deep"):
            parameter_rows.extend(dict(checkpoint=name, group=group, parameter=named[i][0],
                                       shape=list(named[i][1].shape), elements=named[i][1].numel()) for i in groups[group])
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            with h5py.File(PATHS["dataset_file"], "r") as h:
                for batch in schedule:
                    synchronize()
                    start = time.perf_counter()
                    X, target, fetched = read_batch(h, batch["ids"])
                    X, target = X.to(device), target.to(device)
                    grads, loss_values = task_gradients(model, X, target, named)
                    del X, target
                    for group, idx in groups.items():
                        for k, label in enumerate(LABELS):
                            missing = [named[i][0] for i in idx if grads[k][i] is None]
                            coverage_rows.append(dict(checkpoint=name, batch_id=batch["batch_id"], region=batch["region"],
                                group=group, task=label, tensors=len(idx), elements=sum(named[i][1].numel() for i in idx),
                                none_parameters=missing, none_tensors=len(missing),
                                nonzero_elements=sum(int(torch.count_nonzero(grads[k][i])) for i in idx if grads[k][i] is not None)))
                        for a, b in TASK_PAIRS:
                            metrics = gradient_statistics(grads[a], grads[b], idx)
                            records.append(dict(checkpoint=name, batch_id=batch["batch_id"], region=batch["region"],
                                ids_sha256=batch["ids_sha256"], group=group, pair=LABELS[a]+" / "+LABELS[b],
                                **metrics,
                                loss_a=loss_values[a], loss_b=loss_values[b]))
                    del grads
                    print(f"{name}: completed batch {batch['batch_id']+1}/{len(schedule)}", flush=True)
                    synchronize()
                    timing_rows.append(dict(checkpoint=name, batch_id=batch["batch_id"], seconds=time.perf_counter()-start,
                                            fetched_storage_rows=fetched, selected_rows=len(batch["ids"])))
        finally:
            after = model.state_dict()
            unchanged = state_unchanged(model, before)
            grads_unset = all(p.grad is None for p in model.parameters())
            invariants.append(dict(checkpoint=name, parameters_and_buffers_unchanged=unchanged,
                                   parameter_grad_fields_unset=grads_unset,
                                   cuda_peak_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else None))
            require(unchanged and grads_unset, f"{name}: frozen-state invariant failed")
            del before, after, named, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    results = pd.DataFrame(records)
    coverage = pd.DataFrame(coverage_rows)
    timings = pd.DataFrame(timing_rows)
    parameter_inventory = pd.DataFrame(parameter_rows)
    print(pd.DataFrame(invariants).to_string(index=False), flush=True)
    require(len(results) == len(configs)*len(schedule)*4*3, "Incomplete diagnostic")
    summary = summarize(results, SEED)
    # Only successful, fully checked runs receive a manifest/complete metadata file.
    provenance.update(schema_version=1, status="complete", dataset_id=DATASET_ID,
        labels=LABELS, models=list(configs), objective=OBJECTIVE,
        region_definitions=REGION_DEFINITIONS, layer_groups=LAYER_GROUPS,
        shared_parameter_rule="all backbone/embedding parameters including biases; exclude output-specific head rows",
        unused_gradient_rule="None recorded and treated as zero contribution; zero norm cosine undefined",
        preprocessing=normalization, label_standardization="float32 (y - saved_train_mean) / (saved_train_std + 1e-8)",
        invariants=invariants, coverage=coverage_rows, timings=timing_rows,
        parameter_inventory=parameter_rows, population=population_table.to_dict("records"),
        overlap=overlap.to_numpy().tolist(), actual_batches=len(schedule),
        summary_method=dict(seed=SEED+1, bootstrap_replicates=400, min_ci_batches=5,
                            scope="within-region diagnostic batches; not training seeds"),
        sampling=dict(method="numpy.default_rng(seed).choice without replacement within each region",
                      region_iteration_order=list(regions), batch_budget="total across regions per checkpoint",
                      identical_schedule_for_all_models=True),
        script=file_record(Path(__file__).resolve(), True),
        limitations=["Checkpoint hashes identify inputs, not independent authenticity attestations",
                     "No full strain-file digest; frozen metadata and train geometry are verified",
                     "Across-region sample overlap; sparse region batches; trained endpoints only"])
    # Each sample row preserves order, membership and percentiles for local verification.
    sample_manifest["ids_sha256"] = sample_manifest.batch_id.map(batch_manifest.set_index("batch_id").ids_sha256)
    sample_manifest["memberships"] = sample_manifest.memberships.map(json.dumps)
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, frame in [("gradients_per_batch.csv", results), ("gradients_summary.csv", summary),
                            ("batch_manifest.csv", sample_manifest)]:
        with (output_dir / filename).open("x") as f:
            frame.to_csv(f, index=False)
    provenance["exports"] = {filename: file_record(output_dir / filename, True)
                             for filename in ("gradients_per_batch.csv", "gradients_summary.csv", "batch_manifest.csv")}
    with (output_dir / "run_metadata.json").open("x") as f:
        json.dump(provenance, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
