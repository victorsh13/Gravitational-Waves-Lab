#!/usr/bin/env python3
"""Materialize frozen M10/M11 selections using cal only; accept after test replay.

No grid or selection is run. Outputs are staged in --output-dir, never over an
existing bundle. Reports compare the reloaded object against both its pre-save
predictions (exact) and historical test CSVs (fixed numerical tolerances).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import resolve_data_root, resolve_processed_artifact, resolve_project_root
from src.conformal.selected_calibrators import fit_selected_calibrators
from src.conformal.pipeline import apply_mondrian, evaluate_mondrian

LABELS = ["chirp_mass", "total_mass", "chi_eff"]
# Fixed before evaluating test; far smaller than one test sample (1/30000).
RTOL, ATOL = 1e-9, 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def metrics_for_label(fitted, result, y, scale, j):
    metrics = evaluate_mondrian(fitted, result, y)
    widths = result.upper[:, j] - result.lower[:, j]
    physical = widths * scale
    counts = metrics["counts_per_bin"][:, j]
    valid = counts > 0
    coverage = metrics["coverage_per_bin"][:, j]
    low = metrics["bin_tolerance_normal"]["2sigma_low"][:, j]
    high = metrics["bin_tolerance_normal"]["2sigma_high"][:, j]
    lower_miss = float(np.mean(y[:, j] < result.lower[:, j]))
    upper_miss = float(np.mean(y[:, j] > result.upper[:, j]))
    row = {
        "global_coverage": float(metrics["global_coverage"][j]),
        "global_miscoverage": float(1 - metrics["global_coverage"][j]),
        "global_undercoverage_pvalue": float(metrics["global_undercoverage_pvalue"][j]),
        "global_width_iqr_phys": float(np.quantile(physical, .75) - np.quantile(physical, .25)),
        "min_coverage_per_bin": float(coverage[valid].min()),
        "max_coverage_per_bin": float(coverage[valid].max()),
        "max_undercoverage_gap": float(np.maximum(0, .9 - coverage[valid]).max()),
        "min_count_per_bin": int(counts[valid].min()),
        "max_count_per_bin": int(counts[valid].max()),
        "min_bin_2sigma_low": float(low[valid].min()),
        "max_bin_2sigma_high": float(high[valid].max()),
        "lower_miss_rate": lower_miss,
        "upper_miss_rate": upper_miss,
        "global_tail_miss_imbalance": abs(lower_miss - upper_miss),
    }
    for space, values in [("std", widths), ("phys", physical)]:
        for name, value in [("mean", np.mean(values)), ("median", np.median(values)),
                            ("q90", np.quantile(values, .9)), ("q95", np.quantile(values, .95))]:
            row[f"global_{name}_width_{space}"] = float(value)
    masks = {"outside": (coverage < low) | (coverage > high),
             "under": coverage < low, "over": coverage > high}
    for name, mask in masks.items():
        count = int((valid & mask).sum())
        row[f"n_bins_{name}_2sigma"] = count
        row[f"{name}_bin_fraction_2sigma"] = count / int(valid.sum())
    for key, values in metrics["global_tolerance_normal"].items():
        row[f"global_{key}"] = float(values[j])
    row["global_within_2sigma"] = bool(row["global_2sigma_low"] <= row["global_coverage"] <= row["global_2sigma_high"])
    # Historical compact reports use these aliases.
    row.update(coverage=row["global_coverage"], tail_miss_imbalance=row["global_tail_miss_imbalance"])
    for name in ("median", "q90", "q95"):
        row[f"{name}_width_phys"] = row[f"global_{name}_width_phys"]
    return row


def input_paths(model, root, data_root):
    train_name = ("train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json"
                  if model == "M10" else "train_m11_final_G1_500k.json")
    config_path = root / "configs/experiments" / train_name
    config = json.loads(config_path.read_text())
    dataset_id = config["dataset"]["dataset_id"]
    results = data_root / "results" / dataset_id
    pred_name = ("m10_inputzscore_500k_cal_test_predictions_embeddings.npz" if model == "M10"
                 else "m11_final_G1_500k_cal_test_predictions_embeddings.npz")
    prediction_path = results / pred_name
    if model == "M10" and not prediction_path.is_file():
        prediction_path = data_root / "results" / pred_name
    if model == "M10":
        directory = data_root / "results/mondrian_M10_final_baseline"
        selection_path = directory / "selected_configurations.csv"
        references = [selection_path, directory / "selected_systems_summary.csv"]
        bundle_name = "selected_calibrators_frozen.pkl"
    else:
        directory = results
        selection_path = directory / "mondrian_m11_selected_from_val.csv"
        references = [directory / "mondrian_m11_final_test_evaluation.csv",
                      directory / "mondrian_m11_test_selected_systems.csv"]
        bundle_name = "mondrian_m11_selected_calibrators_frozen.pkl"
    checkpoint_name = (f'{dataset_id}_SimpleCNN_ResidualDilated_'
                       f'{config["outputs"]["checkpoint_tag"]}_MSELoss_seed123_checkpoint.pt')
    checkpoint_candidates = [data_root / "models/checkpoints" / dataset_id / checkpoint_name,
                             data_root / "models/checkpoints" / checkpoint_name]
    checkpoint_path = next((p for p in checkpoint_candidates if p.is_file()), checkpoint_candidates[0])
    split_path, stats_path = [resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id,
        file_name=config["dataset"][key]) for key in ("split_file", "label_stats_file")]
    return dict(config=config, config_path=config_path, dataset_id=dataset_id,
                predictions=prediction_path, selection=selection_path, references=references,
                checkpoint=checkpoint_path, split=split_path, stats=stats_path,
                destination=directory / bundle_name)


def freeze(model, root, data_root, output_dir):
    paths = input_paths(model, root, data_root)
    target = output_dir / model / paths["destination"].name
    require(not target.exists(), f"Refusing to overwrite {target}")
    for path in [paths[k] for k in ("predictions", "selection", "checkpoint", "split", "stats")] + paths["references"]:
        require(path.is_file(), f"Missing input: {path}")
    selection = pd.read_csv(paths["selection"])
    require(not selection.duplicated(["final_policy", "label"]).any(), "Duplicate selected keys")
    require(set(zip(selection.final_policy, selection.label)) ==
            {(policy, label) for policy in ("conservative", "efficient") for label in LABELS}, "Selection rows")
    extra_sources = []
    if model == "M11":
        require(selection.selection_split.eq("val").all(), "M11 selection must be from val")
        require(set(selection.selection_status) <= {"selected", "no_strict_candidate"}, "Unknown status")
        selected = selection[selection.selection_status == "selected"].copy()
        provenance_path = paths["selection"].parent / "mondrian_m11_cal_val_test_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        require([provenance[k] for k in ("fit_split", "selection_split", "evaluation_split")]
                == ["cal", "val", "test"], "M11 split roles")
        require(provenance["confidence_level"] == .9 and provenance["n_neighbors"] == 5
                and provenance["min_samples_per_bin"] == 20 and provenance["apply_jitter"] is True,
                "M11 conformal contract")
        extra_sources.append(provenance_path)
    else:
        selected = selection.copy()
    references = [(p, pd.read_csv(p)) for p in paths["references"]]
    for path, frame in references:
        require(not frame.duplicated(["final_policy", "label"]).any(), f"Duplicate historical rows {path}")
        if model == "M11" and "evaluation_split" in frame:
            require(frame.evaluation_split.eq("test").all(), "Historical reference must be test")
        for row in selected.itertuples():
            match = frame[(frame.final_policy == row.final_policy) & (frame.label == row.label)]
            require(len(match) == 1, f"Missing historical reference {row.label}/{row.final_policy}")
            for key in ("taxonomy_mode", "interval_mode", "n_bins"):
                require(match.iloc[0][key] == getattr(row, key), f"Historical selection mismatch {key}")

    with np.load(paths["predictions"], allow_pickle=True) as z:
        require(z["label_names"].tolist() == LABELS, "Label order")
        require(z["model_config"].item()["dataset_id"] == paths["dataset_id"], "Dataset provenance")
        require(z["input_normalization"].item() == paths["config"]["input_normalization"], "Input contract")
        arrays = {k: np.asarray(z[k], dtype=float) for k in
                  ("pred_cal", "y_cal", "emb_cal", "pred_test", "y_test", "emb_test", "y_mean", "y_std")}
        indices = {s: np.asarray(z[f"idx_{s}"], dtype=np.int64) for s in ("cal", "test")}
        original_provenance = {k: z[k].item() for k in
                              ("checkpoint_file", "dataset_path", "split_path", "label_stats_path", "model_config")}
    for key, path_key in [("checkpoint_file", "checkpoint"), ("split_path", "split"), ("label_stats_path", "stats")]:
        require(Path(original_provenance[key]).name == paths[path_key].name, f"Artifact identity {key}")
    for s in ("cal", "test"):
        require(indices[s].shape == (30000,) and len(np.unique(indices[s])) == 30000, f"Indices {s}")
        for name, width in [("pred", 3), ("y", 3), ("emb", 64)]:
            a = arrays[f"{name}_{s}"]
            require(a.shape == (30000, width) and np.isfinite(a).all(), f"Invalid {name}_{s}")
    with np.load(paths["split"], allow_pickle=True) as z:
        split_indices = {s: z[f"{s}_idx"] for s in ("train", "val", "cal", "test")}
        for s in ("cal", "test"):
            # Batch prediction saves sorted physical IDs, unlike the split NPZ.
            # Preserve prediction/target/embedding order; compare membership only.
            require(np.array_equal(np.sort(indices[s]), np.sort(split_indices[s])),
                    f"Original split membership {s}")
        all_indices = np.concatenate(list(split_indices.values()))
        require(len(all_indices) == 500000 and len(np.unique(all_indices)) == 500000
                and all_indices.min() == 0 and all_indices.max() == 499999, "Disjoint original partition")
    with np.load(paths["stats"], allow_pickle=True) as z:
        require(np.array_equal(z["train_idx"], split_indices["train"]), "Scaler must use original train only")
        for key in ("y_mean", "y_std"):
            require(np.array_equal(z[key].astype(np.float32).astype(float), arrays[key]), f"Scaler {key}")
    require(arrays["y_mean"].shape == arrays["y_std"].shape == (3,)
            and np.isfinite(arrays["y_mean"]).all() and np.isfinite(arrays["y_std"]).all()
            and (arrays["y_std"] > 0).all(), "Scaler validity")
    print(f"{model}: input provenance, splits and train-only scalers validated", flush=True)
    systems = {}
    for _, row in selected.iterrows():
        print(f"{model}: fit cal only {row.final_policy}/{row.label}: "
              f"{row.taxonomy_mode}/{row.interval_mode}/{int(row.n_bins)}", flush=True)
        systems.update(fit_selected_calibrators(pd.DataFrame([row]), arrays["pred_cal"], arrays["y_cal"], LABELS,
            emb_cal=arrays["emb_cal"], confidence_level=.9, n_neighbors=5, min_samples_per_bin=20,
            apply_jitter=True, jitter_variation=1e-10))
    source_files = [paths[k] for k in ("config_path", "predictions", "selection", "checkpoint", "split", "stats")]
    source_files += paths["references"] + extra_sources + [Path(__file__).resolve()]
    source_files += sorted((root / "src/conformal").glob("*.py"))
    metadata = dict(model=model, label_names=LABELS, y_mean=arrays["y_mean"].tolist(),
        y_std=arrays["y_std"].tolist(), calibration_space="standardized", fit_split="cal",
        selection_split="test" if model == "M10" else "val", evaluation_split="test",
        confidence_level=.9, n_neighbors=5, min_samples_per_bin=20, apply_jitter=True,
        jitter_variation=1e-10, jitter_seed=None,
        reconstruction="one cal-only fit per frozen selection; no grid, reselection or retry",
        checkpoint_sha256=sha256(paths["checkpoint"]), checkpoint_path=str(paths["checkpoint"]),
        selection_sha256=sha256(paths["selection"]), selection_path=str(paths["selection"]),
        predictions_path=str(paths["predictions"]), predictions_sha256=sha256(paths["predictions"]),
        cal_split_path=str(paths["split"]), cal_split_sha256=sha256(paths["split"]),
        idx_cal_sha256=array_hash(indices["cal"]), idx_test_sha256=array_hash(indices["test"]),
        array_order="unaltered prediction NPZ order; split membership verified independently",
        original_prediction_provenance=original_provenance,
        source_sha256={str(p): sha256(p) for p in dict.fromkeys(source_files)},
        created_utc=datetime.now(timezone.utc).isoformat(),
        versions={"python": platform.python_version(), **{p: version(p) for p in ("numpy", "scipy", "scikit-learn", "pandas")}},
        historical_tolerance=dict(rtol=RTOL, atol=ATOL), round_trip_tolerance="exact array equality",
        intended_destination=str(paths["destination"]),
        selection_status=selection[["final_policy", "label"] + (["selection_status"] if model == "M11" else [])].to_dict("records"))
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target.with_suffix(".candidate.pkl")
    require(not candidate.exists(), f"Refusing to overwrite {candidate}")
    # Serialize before applying test: the artifact contains cal-derived state only.
    with candidate.open("xb") as f:
        pickle.dump(dict(metadata=metadata, calibrators=systems), f, protocol=4)
    with candidate.open("rb") as f:
        reloaded = pickle.load(f)
    reports, round_trip, summaries = [], [], []
    for key, system in systems.items():
        policy, label = key
        j = system.label_index
        print(f"{model}: round-trip test {policy}/{label}", flush=True)
        kwargs = dict(pred_target=arrays["pred_test"],
                      target_embedding=arrays["emb_test"] if system.taxonomy_mode == "difficulty" else None)
        before = apply_mondrian(system.fitted, **kwargs)
        after = apply_mondrian(reloaded["calibrators"][key].fitted, **kwargs)
        for field in ("lower", "upper", "bin_indices", "binning_scores"):
            equal = np.array_equal(getattr(before, field), getattr(after, field))
            round_trip.append(dict(model=model, final_policy=policy, label=label, field=field, exact_equal=equal))
            require(equal, f"Round-trip mismatch: {model}/{key}/{field}")
        measured = metrics_for_label(system.fitted, after, arrays["y_test"], arrays["y_std"][j], j)
        summaries.append(dict(model=model, final_policy=policy, label=label,
            taxonomy_mode=system.taxonomy_mode, interval_mode=system.interval_mode, n_bins=system.n_bins, **measured))
        for reference_path, reference in references:
            historical = reference[(reference.final_policy == policy) & (reference.label == label)].iloc[0]
            for metric, value in measured.items():
                if metric not in historical:
                    continue
                expected = float(historical[metric])
                passed = bool(np.isclose(value, expected, rtol=RTOL, atol=ATOL))
                reports.append(dict(model=model, final_policy=policy, label=label, metric=metric,
                    historical=expected, reloaded=float(value), absolute_difference=abs(value-expected),
                    passed=passed, reference_path=str(reference_path)))
    report = pd.DataFrame(reports)
    prefix = target.with_suffix("")
    report.to_csv(str(prefix) + "_validation.csv", index=False)
    pd.DataFrame(round_trip).to_csv(str(prefix) + "_round_trip.csv", index=False)
    pd.DataFrame(summaries).to_csv(str(prefix) + "_test_metrics.csv", index=False)
    accepted = bool(report.passed.all())
    manifest = dict(metadata=metadata, accepted=accepted, bundle_sha256=sha256(candidate),
                    systems=pd.DataFrame(summaries)[["final_policy", "label", "taxonomy_mode", "interval_mode", "n_bins"]].to_dict("records"),
                    historical_checks=len(report), failed_checks=int((~report.passed).sum()),
                    max_absolute_difference=float(report.absolute_difference.max()),
                    round_trip_all_exact=all(row["exact_equal"] for row in round_trip))
    Path(str(prefix) + "_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not accepted:
        print(report[~report.passed].to_string(index=False), flush=True)
        raise ValueError(f"Rejected {model}: historical mismatch. Candidate retained for diagnosis: {candidate}")
    candidate.rename(target)
    print(f"ACCEPTED {model}: {len(systems)} systems, {len(report)} historical checks; {sha256(target)}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="Staging directory; no automatic publication")
    parser.add_argument("--model", choices=("M10", "M11", "both"), default="both")
    args = parser.parse_args()
    root = resolve_project_root()
    config = json.loads((root / "configs/experiments/train_m11_final_G1_500k.json").read_text())
    data_root = resolve_data_root(cli_data_root=args.data_root, config_data_root=config["data_root"])
    for model in (["M10", "M11"] if args.model == "both" else [args.model]):
        freeze(model, root, data_root, args.output_dir)


if __name__ == "__main__":
    main()
