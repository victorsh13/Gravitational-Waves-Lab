#!/usr/bin/env python3
"""M13.5 (CPU): check the renoised G3 rows and predict them with the M13.3b/M13.4a final models.

a) Reproduction control: control_reproduce.h5 vs the G3 trainio HDF5 on its rows: max|dX| is
   reported; y, final_distance and file_group_id must be identical (abort otherwise).
b) Renoise shards (renoise_shardXXofNN.h5, all N present and complete): for rows_all12k,
   y, final_distance, target_network_snr, file_group_id and split identical to G3 (abort
   otherwise); per-row Pearson correlation of new vs G3 X is informative (should not be ~1).
c) Inference (eval, CPU, batches of 64) with <data_root>/models/checkpoints/M13_3/<tag>_final.pt,
   model built from the checkpoint model_config. X goes through float32 -> float16 -> float32,
   as the fp16 memmap used in training. Predicted: (i) rows with new noise, (ii) the same rows
   with the G3 noise from the trainio; (ii) is compared with the stored train40k / val npz
   predictions (max |d| in standardized units; warning if > 1e-3).
d) Writes m13_5_predictions.npz and m13_5_summary.json in m13_5_renoise/ (exclusive). Per model
   and label (standardized units): MSE on train10k / val2k with original and new noise, MSE on
   full val (40k, stored npz), f = (MSE_tr_new - MSE_tr_orig) / (MSE_val40k - MSE_tr_orig) and
   the val control ratio MSE_val_new / MSE_val_orig, each with a 95 % percentile bootstrap CI
   (2000 replicates, default_rng(135004); train10k and val2k rows resampled paired between
   original and new noise, val40k rows resampled independently; same draws for every model).
   Reading of the point estimate f: >= 0.7 RUIDO_FIJO, <= 0.3 ENTORNO, else MIXTO
   (SIN_BRECHA if MSE_val40k <= MSE_tr_orig). Control: VALIDO if the ratio CI contains 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir, resolve_data_root, resolve_processed_artifact
from src.models import network

DATASET_ID = "M13_G3_band1024_w16_500k"
TAGS = ("M13_3b_G3_s123", "M13_3b_G3_s124", "M13_4a_G3_s123")
LABELS = ["chirp_mass", "total_mass", "chi_eff"]
BATCH = 64
N_BOOT, SEED_BOOT = 2000, 135004
PRED_TOL = 1e-3
FP16_MAX = 65504.0
F_FIXED, F_ENV = 0.7, 0.3
META_EXACT = ("y", "final_distance", "target_network_snr", "file_group_id", "split")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(message):
    raise SystemExit(f"ABORT: {message}")


def read_column(h, name, rows):
    d = h[name]
    values = d.asstr()[:] if h5py.check_string_dtype(d.dtype) else d[:]
    return values[rows]


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")


def build_model(ck):
    mc = ck["model_config"]
    model = getattr(network, mc["class_name"])(**mc["model_kwargs"])
    model.load_state_dict(ck["model_state_dict"])
    return model.eval()


def fp16_round(X):
    if not (np.isfinite(X).all() and np.abs(X).max() <= FP16_MAX):
        fail("X not representable in float16")
    return X.astype(np.float16).astype(np.float32)


def row_corr(a, b):
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def check_control(work, trainio, rows_control):
    path = work / "control_reproduce.h5"
    with h5py.File(path, "r") as c:
        if c.attrs["m13_5_mode"] != "reproduce_original" or not bool((c["status"][:] == 1).all()):
            fail(f"{path} is not a complete reproduction control")
        rows = c["row_index"][:]
        if not np.array_equal(rows, rows_control):
            fail("control row_index differs from rows_control16.npy")
        X_c = c["X"][:]
        meta_c = {n: read_column(c, n, slice(None)) for n in ("y", "final_distance", "file_group_id")}
    X_g = trainio["X"][rows]
    result = {"path": str(path), "n": int(rows.size),
              "max_abs_dX": float(np.max(np.abs(X_c.astype(np.float64) - X_g)))}
    for name, values in meta_c.items():
        same = bool(np.array_equal(values, read_column(trainio, name, rows)))
        result[f"{name}_identical"] = same
        if not same:
            fail(f"control {name} differs from G3 trainio")
    print(f"[control] {result}", flush=True)
    if result["max_abs_dX"] > 1e-5:
        print(f"WARNING: control max|dX| = {result['max_abs_dX']:.3e} (expected ~1e-6 or less)", flush=True)
    return result


def open_renoise_shards(work, rows_all, rows_all_sha):
    paths = sorted(work.glob("renoise_shard*of*.h5"))
    if not paths:
        fail(f"No renoise shards in {work}")
    parsed = [re.fullmatch(r"renoise_shard(\d+)of(\d+)\.h5", p.name) for p in paths]
    if any(m is None for m in parsed):
        fail("Unexpected renoise shard file name")
    counts = {int(m.group(2)) for m in parsed}
    if len(counts) != 1 or sorted(int(m.group(1)) for m in parsed) != list(range(counts.pop())):
        fail(f"Incomplete or mixed renoise shard set: {[p.name for p in paths]}")
    shards, row_ids, meta, provenance = [], [], {n: [] for n in META_EXACT}, []
    seeds_orig, seeds_new = [], []
    for p in paths:
        h = h5py.File(p, "r")
        if h.attrs["m13_5_mode"] != "renoise" or h.attrs["rows_file_sha256"] != rows_all_sha:
            fail(f"{p.name}: not a renoise shard of rows_all12k.npy")
        if not bool((h["status"][:] == 1).all()):
            fail(f"{p.name}: incomplete")
        index = h["row_index"][:]
        if not np.all(np.diff(index) > 0):
            fail(f"{p.name}: row_index not ascending")
        shards.append(h)
        row_ids.append(index)
        for n in META_EXACT:
            meta[n].append(read_column(h, n, slice(None)))
        seeds_orig.append(h["noise_seed_original"][:])
        seeds_new.append(h["noise_seed_new"][:])
        provenance.append({"path": str(p), "size": p.stat().st_size, "rows": int(index.size),
                           "m13_5_generator_sha256": h.attrs["m13_5_generator_sha256"],
                           "noise_seed_rule": h.attrs["noise_seed_rule"]})
    shard_of = np.concatenate([np.full(r.size, i) for i, r in enumerate(row_ids)])
    pos_of = np.concatenate([np.arange(r.size) for r in row_ids])
    all_ids = np.concatenate(row_ids)
    order = np.argsort(all_ids)
    if not np.array_equal(all_ids[order], rows_all):
        fail("Renoise shards do not cover exactly rows_all12k.npy")
    meta = {n: np.concatenate(v)[order] for n, v in meta.items()}
    seeds = {"original": np.concatenate(seeds_orig)[order], "new": np.concatenate(seeds_new)[order]}
    return shards, shard_of[order], pos_of[order], meta, seeds, provenance


def read_new_X(shards, shard_of, pos_of, sel):
    out = np.empty((sel.size, 3, 16384), dtype=np.float32)
    for s in np.unique(shard_of[sel]):
        mask = shard_of[sel] == s
        out[mask] = shards[s]["X"][pos_of[sel][mask]]  # positions ascending within a shard
    return out


def bootstrap(err, rng):
    """err: dict of (n, M, 3) squared errors. Returns point and replicate f and val ratio."""
    def stats(m):
        f = (m["tr_new"] - m["tr_orig"]) / (m["val40k"] - m["tr_orig"])
        return f, m["va_new"] / m["va_orig"]

    point = stats({k: v.mean(axis=0) for k, v in err.items()})
    reps_f, reps_r = [], []
    n_tr, n_va, n_40 = len(err["tr_orig"]), len(err["va_orig"]), len(err["val40k"])
    for _ in range(N_BOOT):
        it = rng.integers(0, n_tr, n_tr)
        iv = rng.integers(0, n_va, n_va)
        i40 = rng.integers(0, n_40, n_40)
        m = {"tr_orig": err["tr_orig"][it].mean(0), "tr_new": err["tr_new"][it].mean(0),
             "va_orig": err["va_orig"][iv].mean(0), "va_new": err["va_new"][iv].mean(0),
             "val40k": err["val40k"][i40].mean(0)}
        f, r = stats(m)
        reps_f.append(f)
        reps_r.append(r)
    return point, np.stack(reps_f), np.stack(reps_r)


def ci(reps):
    finite = np.isfinite(reps)
    lo = np.where(finite, reps, np.nan)
    return np.nanpercentile(lo, 2.5, axis=0), np.nanpercentile(lo, 97.5, axis=0), (~finite).sum(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = resolve_data_root(cli_data_root=args.data_root)
    work = dataset_processed_dir(root, DATASET_ID) / "m13_5_renoise"
    p_pred, p_sum = work / "m13_5_predictions.npz", work / "m13_5_summary.json"
    for p in (p_pred, p_sum):
        if p.exists():
            fail(f"Output exists, refusing to overwrite: {p}")
    rows_files = {k: work / f"rows_{k}.npy" for k in ("train10k", "val2k", "all12k", "control16")}
    rows = {k: np.load(p).astype(np.int64) for k, p in rows_files.items()}
    if not np.array_equal(rows["all12k"], np.sort(np.concatenate([rows["train10k"], rows["val2k"]]))):
        fail("rows_all12k.npy is not the sorted union of train10k and val2k")

    # Checkpoints and the stored predictions of the same runs.
    ckpt_dir = root / "models" / "checkpoints" / "M13_3"
    res_dir = root / "results" / "M13_3"
    models, saved, provenance = {}, {}, {"checkpoints": {}, "stored_predictions": {}}
    y_mean = y_std = dataset_file = None
    for tag in TAGS:
        p_ck = ckpt_dir / f"{tag}_final.pt"
        ck = load_checkpoint(p_ck)
        dcfg = ck["config"]["dataset"]
        if dcfg["dataset_id"] != DATASET_ID or ck["checkpoint_tag"] != tag:
            fail(f"{p_ck}: wrong dataset_id or tag")
        m, s = ck["y_mean"].numpy().astype(np.float32), ck["y_std"].numpy().astype(np.float32)
        if y_mean is None:
            y_mean, y_std, dataset_file = m, s, dcfg["dataset_file"]
        elif not (np.array_equal(m, y_mean) and np.array_equal(s, y_std) and dcfg["dataset_file"] == dataset_file):
            fail(f"{tag}: label stats or dataset file differ between checkpoints")
        models[tag] = build_model(ck)
        provenance["checkpoints"][tag] = {"path": str(p_ck), "sha256": sha256(p_ck),
                                          "model_config": json.loads(json.dumps(ck["model_config"], default=str))}
        saved[tag] = {}
        for split, name in (("train", "train40k_predictions.npz"), ("val", "val_predictions_embeddings.npz")):
            p = res_dir / tag / name
            with np.load(p, allow_pickle=True) as z:
                if [str(x) for x in z["label_names"]] != LABELS:
                    fail(f"{p}: unexpected label order")
                if not (np.array_equal(z["y_mean"], y_mean) and np.array_equal(z["y_std"], y_std)):
                    fail(f"{p}: label stats differ from the checkpoint")
                saved[tag][split] = {k: z[f"{k}_{split}"] for k in ("idx", "pred", "y")}
            provenance["stored_predictions"][f"{tag}/{split}"] = {"path": str(p), "sha256": sha256(p)}
    val_idx = saved[TAGS[0]]["val"]["idx"]
    for tag in TAGS:
        if not np.array_equal(saved[tag]["val"]["idx"], val_idx):
            fail("Stored val idx differ between models")

    trainio_path = resolve_processed_artifact(data_root=root, dataset_id=DATASET_ID, file_name=dataset_file)
    trainio = h5py.File(trainio_path, "r")
    summary = {"dataset_id": DATASET_ID, "trainio": str(trainio_path), "labels": LABELS,
               "units": "train-only standardized labels, (y - y_mean) / (y_std + 1e-8)",
               "rows_files": {k: {"path": str(p), "sha256": sha256(p)} for k, p in rows_files.items()},
               "script_sha256": sha256(__file__), **provenance}

    # a) Reproduction control.
    summary["control_reproduce"] = check_control(work, trainio, rows["control16"])

    # b) Renoise metadata identity.
    rows_all = rows["all12k"]
    shards, shard_of, pos_of, meta, seeds, summary["renoise_shards"] = open_renoise_shards(
        work, rows_all, sha256(rows_files["all12k"]))
    for name in META_EXACT:
        if not np.array_equal(meta[name], read_column(trainio, name, rows_all)):
            fail(f"renoise {name} differs from G3 trainio")
    split = meta["split"]
    is_tr = np.isin(rows_all, rows["train10k"])
    if not (np.all(split[is_tr] == "train") and np.all(split[~is_tr] == "val")):
        fail("Split of selected rows does not match train10k/val2k")
    if "gaussian_seed" in trainio and not np.array_equal(seeds["original"],
                                                         read_column(trainio, "gaussian_seed", rows_all)):
        fail("noise_seed_original differs from G3 gaussian_seed")
    if np.any(seeds["new"] == seeds["original"]):
        fail("Some rows kept the original noise seed")
    y_std_units = ((meta["y"].astype(np.float32) - y_mean) / (y_std + 1e-8)).astype(np.float32)

    # c) Inference with new and original noise; X correlation computed on the fly.
    pred = {tag: {"new": np.empty((rows_all.size, 3), np.float32),
                  "orig": np.empty((rows_all.size, 3), np.float32)} for tag in TAGS}
    corr = np.empty(rows_all.size)
    with torch.no_grad():
        for start in range(0, rows_all.size, BATCH):
            sel = np.arange(start, min(start + BATCH, rows_all.size))
            X_new = read_new_X(shards, shard_of, pos_of, sel)
            X_orig = trainio["X"][rows_all[sel]]
            corr[sel] = row_corr(X_new, X_orig)
            for kind, X in (("new", X_new), ("orig", X_orig)):
                x = torch.from_numpy(fp16_round(X))
                for tag, model in models.items():
                    out = model(x)
                    out = out[0] if isinstance(out, tuple) else out
                    pred[tag][kind][sel] = out.numpy()
            if (start // BATCH) % 20 == 0:
                print(f"predicted {sel[-1] + 1}/{rows_all.size}", flush=True)
    q = np.quantile(corr, [0.0, 0.05, 0.5, 0.95, 1.0])
    summary["x_corr_new_vs_g3"] = dict(zip(["min", "q05", "median", "q95", "max"], map(float, q)))
    print(f"[renoise] per-row corr(X_new, X_G3): {summary['x_corr_new_vs_g3']}", flush=True)
    if q[-1] > 0.99:
        print("WARNING: some renoised rows are almost identical to G3", flush=True)

    # Stored-prediction consistency of the original-noise predictions.
    summary["stored_prediction_check"] = {}
    for tag in TAGS:
        res = {}
        for s, mask in (("train", is_tr), ("val", ~is_tr)):
            st = saved[tag][s]
            pos = np.searchsorted(st["idx"], rows_all[mask])
            if not np.array_equal(st["idx"][np.minimum(pos, st["idx"].size - 1)], rows_all[mask]):
                fail(f"{tag}: selected {s} rows missing from stored predictions")
            res[f"{s}_max_abs_dpred"] = float(np.max(np.abs(pred[tag]["orig"][mask] - st["pred"][pos])))
            res[f"{s}_y_identical"] = bool(np.array_equal(st["y"][pos], y_std_units[mask]))
        summary["stored_prediction_check"][tag] = res
        print(f"[stored check] {tag}: {res}", flush=True)
        if max(res["train_max_abs_dpred"], res["val_max_abs_dpred"]) > PRED_TOL:
            print(f"WARNING: {tag} original-noise predictions differ from stored ones by > {PRED_TOL}", flush=True)
        if not (res["train_y_identical"] and res["val_y_identical"]):
            print(f"WARNING: {tag} standardized y differs from the stored npz", flush=True)

    # d) Metrics and bootstrap.
    def sq(p, mask):
        return (p[mask].astype(np.float64) - y_std_units[mask]) ** 2

    err = {"tr_orig": np.stack([sq(pred[t]["orig"], is_tr) for t in TAGS], axis=1),
           "tr_new": np.stack([sq(pred[t]["new"], is_tr) for t in TAGS], axis=1),
           "va_orig": np.stack([sq(pred[t]["orig"], ~is_tr) for t in TAGS], axis=1),
           "va_new": np.stack([sq(pred[t]["new"], ~is_tr) for t in TAGS], axis=1),
           "val40k": np.stack([(saved[t]["val"]["pred"].astype(np.float64) - saved[t]["val"]["y"]) ** 2
                               for t in TAGS], axis=1)}
    (f_pt, r_pt), f_reps, r_reps = bootstrap(err, np.random.default_rng(SEED_BOOT))
    f_lo, f_hi, f_bad = ci(f_reps)
    r_lo, r_hi, r_bad = ci(r_reps)
    mse = {k: v.mean(axis=0) for k, v in err.items()}
    summary["bootstrap"] = {"replicates": N_BOOT, "seed": SEED_BOOT, "ci": "percentile 2.5/97.5",
                            "resampling": "train10k and val2k rows paired orig/new; val40k independent; "
                                          "same draws for all models"}
    summary["reading_rule"] = {"f_point >= 0.7": "RUIDO_FIJO", "f_point <= 0.3": "ENTORNO", "else": "MIXTO",
                               "MSE_val40k <= MSE_train_orig": "SIN_BRECHA",
                               "control": "VALIDO if val ratio CI contains 1"}
    summary["results"] = {}
    for m, tag in enumerate(TAGS):
        summary["results"][tag] = {}
        for j, label in enumerate(LABELS):
            gap = mse["val40k"][m, j] - mse["tr_orig"][m, j]
            f = float(f_pt[m, j])
            reading = ("SIN_BRECHA" if gap <= 0 else "RUIDO_FIJO" if f >= F_FIXED
                       else "ENTORNO" if f <= F_ENV else "MIXTO")
            entry = {
                "MSE_train_orig": float(mse["tr_orig"][m, j]), "MSE_train_new": float(mse["tr_new"][m, j]),
                "MSE_val2k_orig": float(mse["va_orig"][m, j]), "MSE_val2k_new": float(mse["va_new"][m, j]),
                "MSE_val40k": float(mse["val40k"][m, j]),
                "f": f, "f_ci95": [float(f_lo[m, j]), float(f_hi[m, j])],
                "f_nonfinite_replicates": int(f_bad[m, j]), "reading": reading,
                "val_ratio_new_over_orig": float(r_pt[m, j]),
                "val_ratio_ci95": [float(r_lo[m, j]), float(r_hi[m, j])],
                "val_ratio_nonfinite_replicates": int(r_bad[m, j]),
                "val_control": "VALIDO" if r_lo[m, j] <= 1.0 <= r_hi[m, j] else "NO_VALIDO",
            }
            summary["results"][tag][label] = entry
            print(f"{tag} {label}: f = {f:.3f} [{f_lo[m, j]:.3f}, {f_hi[m, j]:.3f}] {reading}; "
                  f"val ratio {r_pt[m, j]:.3f} [{r_lo[m, j]:.3f}, {r_hi[m, j]:.3f}] {entry['val_control']}",
                  flush=True)

    payload = {"label_names": np.array(LABELS), "y_mean": y_mean, "y_std": y_std, "models": np.array(TAGS)}
    for tag in TAGS:
        payload.update({f"{tag}__pred_new": pred[tag]["new"], f"{tag}__pred_orig": pred[tag]["orig"],
                        f"{tag}__y": y_std_units, f"{tag}__idx": rows_all,
                        f"{tag}__split": split.astype(str)})
    with open(p_pred, "xb") as fh:
        np.savez_compressed(fh, **payload)
    with open(p_sum, "x") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"saved {p_pred}\nsaved {p_sum}", flush=True)
    for h in shards:
        h.close()
    trainio.close()


if __name__ == "__main__":
    main()
