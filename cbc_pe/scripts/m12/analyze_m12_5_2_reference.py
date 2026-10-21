"""Scalar-only paired validation analysis, also callable from the M12.5.2 notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.m12.run_m12_5_2_reference import (
    LABELS, PINNED, VERSION, require, digest, write_json)
from src.models.evaluate import regression_metrics, inverse_standardize


def paired_errors(truth, baseline, reference, std):
    """Positive delta means smaller reference squared error; no loss-objective proxy."""
    arrays = [np.asarray(v, dtype=np.float64) for v in (truth, baseline, reference)]
    require(arrays[0].ndim == 2 and arrays[0].shape[1] == 3 and
            all(a.shape == arrays[0].shape and np.isfinite(a).all() for a in arrays),
            "Invalid paired arrays")
    scale = np.asarray(std, dtype=np.float64)
    require(scale.shape == (3,) and np.isfinite(scale).all() and (scale > 0).all(),
            "Invalid scale")
    mse_m11, mse_ref = (arrays[1]-arrays[0])**2, (arrays[2]-arrays[0])**2
    return mse_m11, mse_ref, mse_m11-mse_ref, (mse_m11-mse_ref)*scale**2


def analyze(inference_dir, bootstrap=2000):
    """No HDF5, checkpoints, calibration/test or model inference is opened here.

    Intervals condition on the observed held-out environment mixture. They do
    not cover training-seed variation or generalization to new environments.
    Validation checkpoint selection also makes these exploratory intervals.
    """
    directory = Path(inference_dir)
    receipt = json.loads((directory / "complete.json").read_text())
    require(receipt["complete"] and receipt["split"] == "val" and
            receipt["identity"]["version"] == VERSION and
            receipt["identity"]["pinned"] == PINNED and
            receipt["baseline_sha256"] == PINNED["baseline"], "Invalid inference receipt")
    path = directory / "val_predictions.npz"
    require(digest(path) == receipt["predictions_sha256"], "Prediction archive changed")
    with np.load(path, allow_pickle=False) as a:
        data = {k: a[k] for k in a.files}
    ids, truth, baseline, reference = [data[k] for k in
        ("idx_val", "y_val", "m11_pred_val", "pred_val")]
    require(data["label_names"].tolist() == LABELS and ids.shape == (40000,) and
            np.array_equal(ids, np.arange(400000, 440000)) and
            truth.shape == baseline.shape == reference.shape == (40000, 3),
            "Frozen validation membership/order mismatch")
    for key in ("source_id", "environment", "snr"):
        require(data[key].shape == (40000,), f"Bad {key} alignment")
    require(len(np.unique(data["source_id"])) == 40000, "Duplicate validation sources")
    for key in ("y_mean", "y_std", "snr", "train_center", "train_precision",
                "train_distance_cuts", "train_label_cuts"):
        require(np.isfinite(data[key]).all(), f"Nonfinite {key}")
    std = data["y_std"].astype(np.float64)
    mean = data["y_mean"].astype(np.float64)
    bse, rse, delta, physical_delta = paired_errors(truth, baseline, reference, std)
    require(bootstrap >= 1000, "At least 1000 replicates for exploratory tail intervals")
    rng = np.random.default_rng(125202)
    groups = [np.flatnonzero(data["environment"] == e) for e in np.unique(data["environment"])]
    # One resample at a time: O(N), not O(N * bootstrap), working memory.
    draws = np.empty((bootstrap, 3))
    for b in range(bootstrap):
        draws[b] = sum(delta[g[rng.integers(len(g), size=len(g))]].sum(axis=0)
                       for g in groups) / len(ids)
    lo, hi = np.quantile(draws, [0.05/6, 1-0.05/6], axis=0)
    rows = []
    for j, label in enumerate(LABELS):
        for unit, factor in (("standardized", 1.), ("physical", std[j]**2)):
            margin = .01*bse[:, j].mean()*factor
            rows.append(dict(label=label, units=unit, n=len(ids),
                m11_mse=bse[:, j].mean()*factor, reference_mse=rse[:, j].mean()*factor,
                favorable_delta=delta[:, j].mean()*factor,
                ci_low=lo[j]*factor, ci_high=hi[j]*factor, practical_margin=margin,
                practical_improvement=bool(lo[j]*factor > margin),
                practical_equivalence=bool(lo[j]*factor > -margin and hi[j]*factor < margin),
                rmse_ratio=float(np.sqrt(rse[:, j].mean()/bse[:, j].mean())),
                rmse_guardrail=bool(rse[:, j].mean() <= 1.02**2*bse[:, j].mean())))
    endpoint = pd.DataFrame(rows)
    physical_truth = inverse_standardize(truth.astype(np.float64), mean, std)
    physical_predictions = {"M11": inverse_standardize(baseline.astype(np.float64), mean, std),
                            "PatchMLP": inverse_standardize(reference.astype(np.float64), mean, std)}
    metrics = []
    for model, pred in physical_predictions.items():
        frame = regression_metrics(physical_truth, pred, LABELS, split_name="val")
        frame["model"] = model
        for j in range(3):
            y, yhat = physical_truth[:, j], pred[:, j]
            slope = np.sum((y-y.mean())*(yhat-yhat.mean())) / np.sum((y-y.mean())**2)
            frame.loc[j, "OLS_slope"] = slope
            frame.loc[j, "OLS_intercept"] = yhat.mean()-slope*y.mean()
            for tail, mask in (("Q10", y <= data["train_label_cuts"][0, j]),
                               ("Q90", y >= data["train_label_cuts"][1, j])):
                frame.loc[j, tail+"_signed_bias"] = (yhat-y)[mask].mean()
        metrics.append(frame)
    centered = truth.astype(np.float64)-data["train_center"]
    distance = np.einsum("ni,ij,nj->n", centered, data["train_precision"], centered)
    strata = {"all": np.ones(len(ids), dtype=bool)}
    bin_id = np.searchsorted(data["train_distance_cuts"], distance, side="right")
    strata.update({"Mahalanobis_"+name: bin_id == i for i, name in
                   enumerate(("0_80", "80_95", "95_99", "99_100"))})
    for low, high in ((10, 15), (15, 20), (20, 25.000001)):
        strata[f"SNR_{low}_{min(high, 25):g}"] = (data["snr"] >= low) & (data["snr"] < high)
    for env in np.unique(data["environment"]):
        strata["environment_"+str(env)] = data["environment"] == env
    tails = {}
    for j, label in enumerate(LABELS):
        for tail, mask in (("Q10", physical_truth[:, j] <= data["train_label_cuts"][0, j]),
                           ("Q90", physical_truth[:, j] >= data["train_label_cuts"][1, j])):
            tails[label+"_"+tail] = mask
    strata.update(tails)
    # Predeclared mass/spin cross-tail failure diagnostics; no significance tests.
    for mass in LABELS[:2]:
        for mt in ("Q10", "Q90"):
            for st in ("Q10", "Q90"):
                strata[mass+"_"+mt+"__chi_eff_"+st] = tails[mass+"_"+mt] & tails["chi_eff_"+st]
    stratified = []
    for name, mask in strata.items():
        if not mask.any():
            continue
        for j, label in enumerate(LABELS):
            stratified.append(dict(stratum=name, label=label, n=int(mask.sum()),
                m11_mse_std=bse[mask, j].mean(), reference_mse_std=rse[mask, j].mean(),
                m11_mse_physical=bse[mask, j].mean()*std[j]**2,
                reference_mse_physical=rse[mask, j].mean()*std[j]**2,
                favorable_delta_physical=physical_delta[mask, j].mean(),
                m11_bias=((baseline-truth)[mask, j]*std[j]).mean(),
                reference_bias=((reference-truth)[mask, j]*std[j]).mean()))
    # Preserve observation-level pairing for later pre-test precision planning.
    paired = pd.DataFrame(dict(row_id=ids, source_id=data["source_id"],
                               environment=data["environment"], snr=data["snr"]))
    for j, label in enumerate(LABELS):
        paired[label+"_delta_std"] = delta[:, j]
        paired[label+"_delta_physical"] = physical_delta[:, j]
    return endpoint, pd.concat(metrics, ignore_index=True), pd.DataFrame(stratified), paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    out = a.output_dir.resolve()
    require(ROOT not in out.parents and out != ROOT, "Results must be outside Git")
    require(not out.exists(), "Output already exists")
    tables = analyze(a.inference_dir, a.bootstrap)
    out.mkdir(parents=True, exist_ok=False)
    names = ("endpoint", "metrics", "strata", "paired_errors")
    for name, table in zip(names, tables):
        table.to_csv(out / (name+".csv"), index=False)
    write_json(out / "analysis.json", dict(complete=True, bootstrap=a.bootstrap,
        seed=125202, code_sha256=digest(__file__),
        inference_receipt_sha256=digest(a.inference_dir / "complete.json"),
        interval="exploratory paired bootstrap within fixed environment counts; Bonferroni 3 targets",
        selection="validation-selected checkpoint; no confirmatory claim",
        tables={n: digest(out / (n+".csv")) for n in names}))
    print(tables[0].to_string(index=False))


if __name__ == "__main__":
    main()
