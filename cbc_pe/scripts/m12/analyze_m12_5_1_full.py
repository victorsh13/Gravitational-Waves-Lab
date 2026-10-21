#!/usr/bin/env python3
"""CSV/JSON-only full-study analysis; no torch, HDF5 or waveform imports.

Noise bootstrap is conditional on a fixed designed source/SNR/environment.
The 16/32/64/128 prefixes are nested convergence diagnostics, not independent
experiments or evidence about the G1 population or posterior width.
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
from scripts.m12 import run_m12_5_1_full as full
from src.paths import resolve_data_root

BOOTSTRAP_SEED = 125102
PREFIXES = (16, 32, 64, 128)
METRICS = ("mean_prediction", "bias", "sample_variance", "noise_std",
           "conditional_mse", "conditional_rmse")


def extended_moments(p, truth):
    values = c.moments(p, truth)
    err = np.asarray(p)-truth
    values.update(absolute_bias=abs(values["bias"]),
                  noise_std=float(np.sqrt(values["sample_variance"])),
                  conditional_rmse=float(np.sqrt(values["conditional_mse"])),
                  squared_bias_fraction=(values["bias_squared"]/values["conditional_mse"]
                                         if values["conditional_mse"] > 0 else np.nan),
                  error_q05=float(np.quantile(err, .05)), error_q50=float(np.quantile(err, .5)),
                  error_q95=float(np.quantile(err, .95)),
                  absolute_error_q90=float(np.quantile(abs(err), .90)),
                  absolute_error_q95=float(np.quantile(abs(err), .95)),
                  max_absolute_error=float(np.max(abs(err))))
    return values


def bootstrap_metrics(p, truth, cid, label_index, n_boot):
    """Bound temporary resampling arrays to 128 draws; share IDs across models."""
    R = len(p)
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SEED, cid, label_index, R]))
    boot = {metric: np.empty((n_boot, p.shape[1])) for metric in METRICS}
    for start in range(0, n_boot, 128):
        stop = min(start+128, n_boot)
        draws = rng.integers(0, R, size=(stop-start, R))
        samples = p[draws]
        mean = samples.mean(axis=1)
        variance = samples.var(axis=1, ddof=1)
        mse = ((samples-truth)**2).mean(axis=1)
        bias = mean-truth
        c.require(np.allclose(mse, bias**2+(R-1)/R*variance, rtol=1e-10, atol=1e-12),
                  "Bootstrap finite-R identity failed")
        for metric, value in zip(METRICS, (mean, bias, variance, np.sqrt(variance), mse, np.sqrt(mse))):
            boot[metric][start:stop] = value
    return boot


def absolute_interval(low, high):
    return (0. if low <= 0 <= high else min(abs(low), abs(high)), max(abs(low), abs(high)))


def summarize_condition(p, truth, base, n_boot):
    c.require(p.shape == (base["R"], 4) and np.isfinite(p).all(), "Invalid paired prediction matrix")
    names = list(c.MODEL_TAGS)
    values = [extended_moments(p[:, k], truth) for k in range(4)]
    boot = bootstrap_metrics(p, truth, base["condition_id"], c.LABELS.index(base["label"]), n_boot)
    summaries, pairs = [], []
    for k, name in enumerate(names):
        row = dict(base, model=name, **values[k])
        for metric in METRICS:
            lo, hi = np.quantile(boot[metric][:, k], [.025, .975])
            row.update({metric+"_ci_low": float(lo), metric+"_ci_high": float(hi)})
        lo, hi = absolute_interval(row["bias_ci_low"], row["bias_ci_high"])
        row.update(absolute_bias_ci_low=lo, absolute_bias_ci_high=hi)
        summaries.append(row)
    for a, b in combinations(range(4), 2):
        row = dict(base, model_a=names[a], model_b=names[b], sign="a_minus_b")
        for metric in METRICS:
            lo, hi = np.quantile(boot[metric][:, a]-boot[metric][:, b], [.025, .975])
            row.update({"delta_"+metric: values[a][metric]-values[b][metric],
                        "delta_"+metric+"_ci_low": float(lo), "delta_"+metric+"_ci_high": float(hi)})
        # Abs(bias) is nonregular at zero. Use image of two 97.5% signed-bias
        # intervals, with a Bonferroni envelope (approximate, possibly wide).
        la, ha = absolute_interval(*np.quantile(boot["bias"][:, a], [.0125, .9875]))
        lb, hb = absolute_interval(*np.quantile(boot["bias"][:, b], [.0125, .9875]))
        row.update(delta_absolute_bias=values[a]["absolute_bias"]-values[b]["absolute_bias"],
                   delta_absolute_bias_ci_low=la-hb, delta_absolute_bias_ci_high=ha-lb)
        pairs.append(row)
    return summaries, pairs


def load_inputs(inference, study):
    """Validate scalar lineage without opening large HDF5 or checkpoint files."""
    receipt = c.read_json(inference / "inference.json")
    generation = c.read_json(study / "generation.json")
    c.require(receipt.get("schema") == full.INF_SCHEMA and receipt.get("complete") is True,
              "Incomplete full inference")
    c.require(generation.get("schema") == full.GEN_SCHEMA and generation.get("complete") is True,
              "Incomplete full generation")
    c.require(receipt["generation_sha256"] == c.digest(study / "generation.json"), "Generation receipt mismatch")
    c.require(receipt["design"] == generation["design"] == full.design(), "Design mismatch")
    c.require(receipt["code"] == generation["code"] == full.code_identity(), "Implementation changed")
    c.require(c.digest(inference / "predictions.csv") == receipt["predictions"]["sha256"], "Prediction digest mismatch")
    for name in ("conditions.csv", "observations.csv", "design.csv", "design.json"):
        c.require(c.digest(study / name) == generation["artifacts"][name]["sha256"], f"Scalar artifact mismatch: {name}")
    c.require(set(receipt["checkpoints"]) == set(c.MODEL_TAGS), "Missing checkpoints")
    for name in c.MODEL_TAGS:
        c.require(receipt["checkpoints"][name]["sha256"] == c.MODEL_HASHES[name], "Unrecognized checkpoint receipt")
    reference = generation["provenance"]["selected_rows"]["0"]
    expected = pd.DataFrame(full.conditions(reference)).set_index("condition_id")
    physical = pd.read_csv(study / "conditions.csv").set_index("condition_id")
    c.require(physical.index.is_unique and set(physical.index) == set(expected.index), "Condition IDs mismatch")
    physical = physical.loc[expected.index]
    for key in expected:
        if pd.api.types.is_numeric_dtype(expected[key]):
            c.require(np.allclose(physical[key], expected[key], rtol=0, atol=1e-12), f"Design field mismatch: {key}")
        else:
            c.require(physical[key].eq(expected[key]).all(), f"Design field mismatch: {key}")
    c.require(physical.groupby("source_id").actual_geocentric_time.nunique().eq(1).all(), "Unfixed placement")
    c.require(physical.psd_identity.nunique() == 1, "PSD changed across conditions")
    c.require(np.isfinite(physical[["snr_H1", "snr_L1", "snr_V1", "final_network_snr", "final_distance"]]).all().all()
              and (physical.final_distance > 0).all(), "Invalid physical metadata")
    c.require(np.allclose(np.sqrt((physical[["snr_H1", "snr_L1", "snr_V1"]]**2).sum(axis=1)),
                          physical.final_network_snr, rtol=1e-10), "Detector SNR norm mismatch")
    # Same historical rescaling acceptance; this analysis does not redefine it.
    c.require(np.allclose(physical.final_network_snr, physical.target_network_snr, rtol=.05, atol=0), "SNR rescaling mismatch")
    obs = pd.read_csv(study / "observations.csv")
    c.require(len(obs) == full.N_OBSERVATIONS and np.array_equal(obs.observation_id, np.arange(full.N_OBSERVATIONS)),
              "Observation order mismatch")
    seeds, _ = full.seed_schedule()
    c.require(np.array_equal(obs[[f"seed_{d}" for d in c.DETECTORS]], seeds.reshape(-1, 3))
              and generation["seed_sha256"] == c.array_digest(seeds), "Seed schedule mismatch")
    c.require((obs.condition_id == obs.observation_id//full.REPEATS).all()
              and (obs.realization_id == obs.observation_id % full.REPEATS).all(), "Observation IDs mismatch")
    c.require(obs.source_id.eq(obs.condition_id.map(physical.source_id)).all(), "Observation source mismatch")
    df = pd.read_csv(inference / "predictions.csv")
    validate_predictions(df, physical, obs)
    return df, physical, receipt, generation


def validate_predictions(df, physical, obs):
    c.require(len(df) == full.N_OBSERVATIONS*12
              and not df.duplicated(["observation_id", "model", "label"]).any(), "Missing/duplicate predictions")
    c.require(set(df.model) == set(c.MODEL_TAGS) and set(df.label) == set(c.LABELS), "Unexpected model/label")
    c.require(np.isfinite(df[["truth", "prediction", "target_snr"]]).all().all(), "Nonfinite predictions")
    c.require(set(df.observation_id) == set(range(full.N_OBSERVATIONS))
              and (df.observation_id == df.condition_id*full.REPEATS+df.realization_id).all()
              and df.realization_id.between(0, full.REPEATS-1).all(), "Prediction IDs mismatch")
    c.require(df.tensor_sha256.eq(df.observation_id.map(obs.set_index("observation_id").tensor_sha256)).all(),
              "Models did not receive identical saved tensors")
    for (cid, label), group in df.groupby(["condition_id", "label"], sort=True):
        row = physical.loc[cid]
        c.require(group.source_id.eq(row.source_id).all()
                  and group.target_snr.eq(row.target_network_snr).all()
                  and group.unit.eq(c.UNITS[c.LABELS.index(label)]).all()
                  and np.allclose(group.truth, row[label], rtol=1e-12, atol=1e-12), "Source/SNR/unit/truth mismatch")


def analyze(df, physical, bootstrap=2000):
    c.require(100 <= bootstrap <= 10000, "Bootstrap budget must be 100..10000")
    summaries, pairs = [], []
    metadata = ("mass_1", "mass_2", "spin_1z", "spin_2z", "mass_ratio", "chirp_mass", "total_mass", "chi_eff")
    for (cid, label), group in df.groupby(["condition_id", "label"], sort=True):
        row = physical.loc[cid]
        table = group.pivot(index="realization_id", columns="model", values="prediction")
        table = table.reindex(index=range(full.REPEATS), columns=list(c.MODEL_TAGS))
        c.require(np.isfinite(table).all().all(), "Unpaired/incomplete repetitions")
        for R in PREFIXES:
            base = dict(condition_id=int(cid), source_id=row.source_id,
                        **{k: float(row[k]) for k in metadata},
                        target_snr=float(row.target_network_snr), label=label,
                        unit=c.UNITS[c.LABELS.index(label)], truth=float(row[label]), R=R)
            a, b = summarize_condition(table.to_numpy()[:R], float(row[label]), base, bootstrap)
            summaries.extend(a)
            pairs.extend(b)
    return pd.DataFrame(summaries), pd.DataFrame(pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    df, physical, receipt, generation = load_inputs(args.inference, args.study)
    # Reserve destination first; failed analyses remain distinguishable from success.
    out = c.output_directory(Path(resolve_data_root(cli_data_root=args.data_root)), args.output_dir)
    result = dict(complete=False)
    try:
        summary, pairs = analyze(df, physical, args.bootstrap)
        outputs = {"conditional_metrics.csv": summary[summary.R == full.REPEATS],
                   "paired_differences.csv": pairs[pairs.R == full.REPEATS],
                   "convergence_metrics.csv": summary, "convergence_paired.csv": pairs}
        for name, frame in outputs.items():
            c.write_csv(out / name, frame.to_dict("records"))
        result.update(complete=True, schema="M12.5.1-B-analysis-v1",
            inference=c.record(args.inference / "inference.json", True),
            generation=c.record(args.study / "generation.json", True),
            analysis_code_sha256=c.digest(__file__), common_code_sha256=c.digest(c.__file__),
            bootstrap=args.bootstrap, bootstrap_seed=BOOTSTRAP_SEED, prefixes=list(PREFIXES),
            ci="pointwise approximate 95% noise-bootstrap percentile; pairing across models; no multiplicity correction",
            absolute_bias_ci="image of signed bias interval; paired difference uses Bonferroni 97.5% component interval envelope",
            tails="empirical descriptive quantiles/maxima, no CI; q95 supported by only ~6 upper-tail observations at R=128",
            convergence="nested first R realization IDs, not independent estimates; no choice of preferred prefix",
            finite_R_identity="MSE = estimated_bias^2 + (R-1)/R * unbiased_sample_variance",
            undefined_fraction="CSV nan if MSE=0; variance units are label unit squared",
            limitations="25 designed sources, one environment/extrinsic setting, one training seed per checkpoint; no posterior or population uncertainty; R=128 is not asymptotic reliability",
            artifacts={name: dict(c.record(out / name, True), rows=len(frame)) for name, frame in outputs.items()})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        c.write_json(out / "analysis.json", result)


if __name__ == "__main__":
    main()
