#!/usr/bin/env python3
"""Scalar-only M12.5.1 analysis, also callable by the local notebook.

Independent conditions; paired models within realization. Percentile bootstrap
CIs describe repeated-noise Monte Carlo uncertainty at fixed source/SNR, not
posterior uncertainty, training-seed uncertainty or population generalization.
"""
import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.m12 import m12_5_common as c
from src.paths import resolve_data_root


def analyze(predictions, receipt, bootstrap=2000):
    c.require(receipt.get("schema") == "M12.5.1-inference-v1"
              and receipt.get("complete") is True, "Incomplete inference")
    c.require(c.digest(predictions) == receipt["predictions"]["sha256"],
              "Predictions do not match inference receipt")
    c.require(100 <= bootstrap <= 10000, "Bootstrap budget must be 100..10000")
    df = pd.read_csv(predictions)
    keys = ["observation_id", "model", "label"]
    c.require(len(df) == 576 * 4 * 3 and not df.duplicated(keys).any(),
              "Missing/duplicate prediction rows")
    c.require(set(df.model) == set(c.MODEL_TAGS) and set(df.label) == set(c.LABELS),
              "Unexpected checkpoint/target")
    c.require(np.isfinite(df[["truth", "prediction", "target_snr"]]).all().all(),
              "Nonfinite scalar results")
    c.require(set(df.observation_id) == set(range(576))
              and (df.observation_id == df.condition_id * 16 + df.realization_id).all()
              and df.realization_id.between(0, 15).all(), "Invalid observation IDs")
    c.require(df.groupby("observation_id").tensor_sha256.nunique().eq(1).all(),
              "Models did not receive identical tensors")
    summaries, pairs = [], []
    names = list(c.MODEL_TAGS)
    for (cid, label), group in df.groupby(["condition_id", "label"], sort=True):
        j = c.LABELS.index(label)
        source = int(cid) // 4
        m1, m2 = c.MASSES[source // 3]
        spin = c.SPINS[source % 3]
        expected_truth = ((m1*m2)**0.6 / (m1+m2)**0.2, m1+m2, spin)[j]
        c.require(group.source_id.eq(f"M1251_SRC_{source:02d}").all()
                  and group.target_snr.eq(c.SNRS[int(cid) % 4]).all()
                  and group.unit.eq(c.UNITS[j]).all()
                  and np.allclose(group.truth, expected_truth, rtol=1e-12, atol=1e-12),
                  "Frozen source/SNR/unit contract violated")
        table = group.pivot(index="realization_id", columns="model", values="prediction")
        table = table.reindex(index=range(16), columns=names)
        c.require(table.shape == (16, 4) and np.isfinite(table).all().all(),
                  "Unpaired or incomplete noise realizations")
        p = table.to_numpy()
        truth = float(group.truth.iloc[0])
        rng = np.random.default_rng(np.random.SeedSequence([125002, int(cid), j]))
        draws = rng.integers(0, 16, size=(bootstrap, 16))
        samples = p[draws]  # Same resampled realization IDs for every checkpoint.
        boot_mean = samples.mean(axis=1)
        boot = dict(mean_prediction=boot_mean, bias=boot_mean-truth,
                    sample_variance=samples.var(axis=1, ddof=1),
                    conditional_mse=((samples-truth)**2).mean(axis=1))
        base = dict(condition_id=int(cid), source_id=group.source_id.iloc[0],
                    target_snr=float(group.target_snr.iloc[0]), label=label,
                    unit=c.UNITS[j], truth=truth, R=16)
        values = [c.moments(p[:, k], truth) for k in range(4)]
        for k, name in enumerate(names):
            row = dict(base, model=name, **values[k])
            for metric, distribution in boot.items():
                lo, hi = np.quantile(distribution[:, k], [.025, .975])
                row.update({metric+"_ci_low": float(lo), metric+"_ci_high": float(hi)})
            summaries.append(row)
        for a, b in combinations(range(4), 2):
            row = dict(base, model_a=names[a], model_b=names[b], sign="a_minus_b")
            for metric, distribution in boot.items():
                row["delta_"+metric] = values[a][metric] - values[b][metric]
                lo, hi = np.quantile(distribution[:, a]-distribution[:, b], [.025, .975])
                row.update({"delta_"+metric+"_ci_low": float(lo),
                            "delta_"+metric+"_ci_high": float(hi)})
            pairs.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = resolve_data_root(cli_data_root=args.data_root)
    receipt = c.read_json(args.inference / "inference.json")
    summary, paired = analyze(args.inference / "predictions.csv", receipt)
    out = c.output_directory(root, args.output_dir)
    c.write_csv(out / "conditional_metrics.csv", summary.to_dict("records"))
    c.write_csv(out / "paired_differences.csv", paired.to_dict("records"))
    c.write_json(out / "analysis.json", dict(
        complete=True, inference=c.record(args.inference / "inference.json", True),
        analysis_code_sha256=c.digest(__file__), bootstrap=2000, bootstrap_seed=125002,
        ci="pointwise 95% percentile; independent within condition; paired across models",
        estimand="fixed-condition repeated-noise bias, variance and MSE; not posterior",
        finite_R_identity="MSE = bias^2 + 15/16 * sample_variance",
        limitations="R=16 is a pilot; no cross-source pooling, multiplicity correction or training-seed inference"))


if __name__ == "__main__":
    main()
