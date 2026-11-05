#!/usr/bin/env python3
"""M13.12 step 0: rebuild val rows with and without the start step (click), as fp16 memmaps for inference.

Protocol: docs/experiments/M13_12_click_dependence.md (pre-registered 4-11-2026). Local CPU.

Selection: val membership from the pinned split (g1_npe_contract.load_contract); y and
target_network_snr from the G3 trainio HDF5. tau30 = Newtonian tau of the true chirp mass,
decile edges from val quantiles (M13_8 cell 3, same code as m13_11_prepare_rows.select_rows).
default_rng(131201) draws 400 val rows per decile without replacement, D1 to D10 in that order.
--limit takes the first N rows interleaved over deciles; it does not change the draw.

Reconstruction: m13_11_prepare_rows.capture_row (build_g1_row with the G3 physics, g6.setup),
plus a tap on g1.build_gaussian_processing_noise that records the Gaussian noise n of the row.
  X_orig    = the X of build_g1_row (must equal the G3 trainio X, G-REC).
  X_noclick = zscore(PROCESSOR.process_network(inject(n, s_S), psd_proc)): same noise, PSD,
              processing and z-score (own per-detector std) as G3; s_S = projection of
              S = _opt_click_free_aligned(opt_args) (SEOBNRv4_opt from 20 Hz, taper to 28 Hz,
              aligned with A in phase and sub-sample time), same projector, params, t_c, window
              and injection as A.
Gates (any failure aborts; gates.json is written, manifests are not):
  G-REC   max|X_orig - X_G3| <= 1e-5 (pipeline X and reprocessed captured s + n).
  G-ALIGN residual phase < 0.01 rad and |SNR_click_aligned - SNR_click_phasemax| <=
          max(0.2 * SNR_click_phasemax, 0.3) (m13_11 diagnose_click, data part; amendment 1 of the
          pre-registration), and the S used here is the one diagnose_click evaluated (identical
          alignment diagnostics).
  G-DIFF  (raw_orig - L(s_A)) vs (raw_noclick - L(s_S)) before the z-score (L = process_clean):
          max|diff| / max|raw_orig - L(s_A)| <= 1e-6.

Output (exclusive, UTC stamp): <data_root>/results/M13_3/M13_12/inputs_<stamp>/
  selection.json, gates.json, rows.json (per row, ascending row id), build.log,
  split_m13_12_val.npz (val_idx = selected row ids, ascending),
  orig/ and noclick/: X_val.npy (float16, (N, 3, T)), y_val.npy (float32, unstandardized),
          row_ids_val.npy (int64, ascending), manifest.json   [format of build_fp16_memmap.py],
  predict_configs/predict_m13_12_<model>_<variant>.json for scripts/predict_cnn_hdf5.py
          (data_loading_mode fp16_memmap); predictions go to results/M13_3/M13_12/preds_<stamp>/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import scripts.generate_m11_final_G1 as g1
from scripts.generate_m11_final_G1 import (DETECTORS, LABELS, read_psd_intervals, require,
                                           resolve_cached_psd_files, sha256_file)
import scripts.m13.generate_m13_G3_whitening as g3
import scripts.m13.generate_m13_6_clean as g6
from scripts.m12.g1_npe_contract import load_contract
from scripts.m13.m13_1_diagnostics import tau_newtonian
from scripts.m13.m13_11_prepare_rows import (CLICK_ALIGN_PHASE_TOL, CLICK_ALIGNED_GATE_RTOL, DEFAULT_CONFIG,
                                             GREC_MAX, MODEL_TAGS, _opt_click_free_aligned, capture_row,
                                             crop_to_final, diagnose_click, final_start_of, process_clean,
                                             psd_at_final_resolution, write_json)

SELECTION_SEED = 131201
N_PER_DECILE = 400
N_DECILES = 10
# Analysis groups of the pre-registration (decile indices, 0 = D1).
TAU_GROUPS = {"D1-D3": (0, 1, 2), "D4-D7": (3, 4, 5, 6), "D8-D10": (7, 8, 9)}
GDIFF_RTOL = 1e-6
# Amendment 1: absolute floor of the G-ALIGN SNR comparison. With a nearly null click (truncated rows,
# whose step falls outside the final window) the relative tolerance is not applicable.
CLICK_ALIGNED_GATE_ATOL = 0.3
MAX_WORKERS = 4
SPLIT = "val"
VARIANTS = ("orig", "noclick")
# Template of the M13.7 fp16_memmap predictions; only split_file, memmap_dir, checkpoint and output change.
PREDICT_TEMPLATE = PROJECT_ROOT / "configs/prediction/predict_m13_4b_G3aug_s123_valcheck.json"
RESULTS_SUBDIR = ("M13_3", "M13_12")

_CTX = {}  # set in the parent before the worker pool is forked


class _NoiseTap:
    """Record the strains returned by g1.build_gaussian_processing_noise (module global) while active."""

    def __init__(self):
        self.noise = None

    def __enter__(self):
        self._fn = g1.build_gaussian_processing_noise

        def tap(**kwargs):
            out = self._fn(**kwargs)
            require(self.noise is None, "build_gaussian_processing_noise called twice in one row")
            self.noise = {ifo: out[ifo].copy() for ifo in DETECTORS}
            return out

        g1.build_gaussian_processing_noise = tap
        return self

    def __exit__(self, *exc):
        g1.build_gaussian_processing_noise = self._fn
        return False


def group_of(decile):
    return next(name for name, ds in TAU_GROUPS.items() if decile in ds)


def select_rows(root, trainio):
    _, splits, _, _, _ = load_contract(root)
    idx_val = splits["val"]
    with h5py.File(trainio, "r") as h:
        y_val = h["y"][:][idx_val].astype(np.float64)
        snr_val = h["target_network_snr"][:][idx_val].astype(np.float64)
    # M13_8 cell 3: Newtonian tau30 of the TRUE Mc, val quantile edges.
    tau30 = tau_newtonian(y_val[:, 0])
    tau_edges = np.quantile(tau30, np.linspace(0, 1, N_DECILES + 1))
    tdec = np.clip(np.searchsorted(tau_edges[1:-1], tau30, side="right"), 0, N_DECILES - 1)
    rng = np.random.default_rng(SELECTION_SEED)
    per_decile = []
    for d in range(N_DECILES):
        pool = np.flatnonzero(tdec == d)
        require(len(pool) >= N_PER_DECILE, f"Decile D{d + 1}: only {len(pool)} val rows")
        positions = np.sort(rng.choice(pool, N_PER_DECILE, replace=False))
        per_decile.append([dict(row_index=int(idx_val[p]), tau30_s=float(tau30[p]), tau30_decile=f"D{d + 1}",
                                tau_group=group_of(d), target_network_snr=float(snr_val[p]),
                                y_true=dict(zip(LABELS, map(float, y_val[p]))))
                           for p in positions])
    return per_decile, tau_edges, idx_val


def gate_failures(rec):
    fails = []
    if max(rec["G_REC"]["max_abs_dX_build"], rec["G_REC"]["max_abs_dX_reprocess"]) > GREC_MAX:
        fails.append("G-REC")
    a = rec["G_ALIGN"]
    if not (a["residual_ok"] and a["aligned_gate_ok"] and a["align_consistent"]):
        fails.append("G-ALIGN")
    if not rec["G_DIFF"]["rel_max_abs_dnoise"] <= GDIFF_RTOL:  # also catches NaN
        fails.append("G-DIFF")
    return fails


def rebuild_pair(row, index, *, long_strains, psd_cache, trainio_h, injector):
    t0 = time.perf_counter()
    with _NoiseTap() as nt:
        X_row, meta, cap, opt_args = capture_row(row, long_strains=long_strains, psd_cache=psd_cache)
    require(nt.noise is not None, f"Row {index}: noise not captured")
    require(np.array_equal(trainio_h["y"][index], row[list(LABELS)].to_numpy(dtype=np.float32)),
            f"Trainio y differs at row {index}")
    final_start, processing_start = final_start_of(row), float(row["processing_start"])
    s_A = cap.signal["projection"].strains

    # n + s_A with the generator's injection must be the captured s + n, bit for bit.
    inj_A = injector.inject_network(noises=nt.noise, signals=s_A)
    for ifo in DETECTORS:
        require(np.array_equal(inj_A[ifo].strain.numpy(), cap.strains[ifo].numpy()),
                f"Row {index} {ifo}: tapped noise + s_A differs from the captured s + n")

    # G-REC
    raw_orig, X_rep = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(
        strains={ifo: cap.strains[ifo].copy() for ifo in DETECTORS}, psds=cap.psds))
    X_g3 = np.asarray(trainio_h["X"][index], dtype=np.float64)
    X_orig = np.asarray(X_row, dtype=np.float32)

    # X_noclick: same noise, PSD, injection, processing and z-score; only the signal is S.
    s_hp, s_hc, align = _opt_click_free_aligned(opt_args)
    proj_S = g1.DETECTOR_PROJECTOR.project(h_plus=s_hp, h_cross=s_hc, parameters=cap.params,
                                           geocentric_coalescence_time=cap.geocentric_time)
    inj_S = injector.inject_network(noises=nt.noise, signals=proj_S.strains)
    raw_noclick, X_noclick = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(
        strains={ifo: inj_S[ifo].strain for ifo in DETECTORS}, psds=cap.psds))

    # G-DIFF: the processed noise must be the same in both inputs (L linear, g6.check_linear_processing).
    dn_orig = np.asarray(raw_orig, dtype=np.float64) - process_clean(
        s_A, processing_start=processing_start, psds=cap.psds, injector=injector)
    dn_noclick = np.asarray(raw_noclick, dtype=np.float64) - process_clean(
        proj_S.strains, processing_start=processing_start, psds=cap.psds, injector=injector)
    gdiff = float(np.abs(dn_orig - dn_noclick).max() / np.abs(dn_orig).max())

    # G-ALIGN: data part of the M13.11 click diagnostic (final 4 s window, psd_proc at 1/4 Hz).
    def crop(series):
        return crop_to_final(series, final_start, injector)

    psd4 = {ifo: psd_at_final_resolution(cap.psds[ifo]) for ifo in DETECTORS}
    s_opt = {ifo: crop(s_A[ifo]) for ifo in DETECTORS}
    click = diagnose_click(row, opt_args, cap, s_opt, psd4, crop)["data"]
    snr_pm = click["snr_click_phasemax_network"]
    aligned_tol = max(CLICK_ALIGNED_GATE_RTOL * snr_pm, CLICK_ALIGNED_GATE_ATOL)

    x_orig16 = g6.to_fp16(X_orig, f"X_orig row {index}")
    rec = dict(
        row_index=int(index), block_id=str(row["block_id"]),
        G_REC=dict(max_abs_dX_build=float(np.abs(X_orig.astype(np.float64) - X_g3).max()),
                   max_abs_dX_reprocess=float(np.abs(np.asarray(X_rep, dtype=np.float64) - X_g3).max()),
                   # Elements whose float16 differs from float16(X_G3), i.e. from the existing val memmap.
                   n_fp16_diff=int(np.count_nonzero(x_orig16 != X_g3.astype(np.float32).astype(np.float16)))),
        G_ALIGN=dict(residual_rad=float(align["residual_rad"]), residual_ok=bool(align["residual_ok"]),
                     aligned_rel_diff=float(click["aligned_rel_diff"]),
                     aligned_gate_ok=bool(abs(click["snr_click_aligned_network"] - snr_pm) <= aligned_tol),
                     align_consistent=bool(click["align"] == align)),
        G_DIFF=dict(rel_max_abs_dnoise=gdiff),
        snr_click_aligned_network=click["snr_click_aligned_network"],
        snr_click_phasemax_network=click["snr_click_phasemax_network"],
        snr_click_network=click["snr_click_network"],
        snr_click_aligned={ifo: v["snr_click_aligned"] for ifo, v in click["detectors"].items()},
        final_network_snr=float(meta["final_network_snr"]), is_truncated=bool(trainio_h["is_truncated"][index]),
        # Samples of the projected signal before the processing window (cut at its start by inject).
        n_clipped_before_A={ifo: int(inj_A[ifo].n_clipped_before) for ifo in DETECTORS},
        n_clipped_before_S={ifo: int(inj_S[ifo].n_clipped_before) for ifo in DETECTORS},
        align=align, s_taper_duration_s=float(click["s_taper_duration_s"]),
    )
    rec["time_s"] = time.perf_counter() - t0
    return rec, x_orig16, g6.to_fp16(np.asarray(X_noclick, dtype=np.float32), f"X_noclick row {index}")


def run_block(indices):
    """Rebuild the rows of one block_id; stops after the first row that fails a gate."""
    block_rows = _CTX["frame"].loc[list(indices)]
    long_strains = read_psd_intervals(block_rows.iloc[0], _CTX["psd_files"])
    psd_cache = {}
    # inject/build_zero_strain never draw from rng; it is set only to make that explicit.
    injector = g1.SignalInjector(config=g1.CONFIG, rng=np.random.default_rng(0))
    out = []
    with h5py.File(_CTX["trainio"], "r") as t:
        for index, row in block_rows.iterrows():
            out.append(rebuild_pair(row, index, long_strains=long_strains, psd_cache=psd_cache, trainio_h=t,
                                    injector=injector))
            if gate_failures(out[-1][0]):
                break
    return out


def predict_configs(out_dir, stamp, dataset_id):
    template = json.loads(PREDICT_TEMPLATE.read_text())
    require(template["dataset"]["dataset_id"] == dataset_id, "Prediction template dataset_id is not the G3 one")
    require(template["prediction"]["data_loading_mode"] == "fp16_memmap", "Template is not fp16_memmap")
    # memmap_dir and split_file resolve against processed/<dataset_id>/, output against results/<dataset_id>/.
    rel_inputs = "../../results/" + "/".join(RESULTS_SUBDIR) + f"/inputs_{stamp}"
    rel_preds = "../" + "/".join(RESULTS_SUBDIR) + f"/preds_{stamp}"
    cfg_dir = out_dir / "predict_configs"
    cfg_dir.mkdir()
    paths = []
    for key, tag in MODEL_TAGS.items():
        for variant in VARIANTS:
            cfg = json.loads(json.dumps(template))
            cfg["dataset"]["split_file"] = f"{rel_inputs}/split_m13_12_{SPLIT}.npz"
            cfg["prediction"].update(checkpoint_file=f"M13_3/{tag}_final.pt", splits=[SPLIT],
                                     memmap_dir=f"{rel_inputs}/{variant}")
            cfg["output"] = dict(file_name=f"{rel_preds}/{key}_{variant}_{SPLIT}_predictions_embeddings.npz",
                                 overwrite=False)
            p = cfg_dir / f"predict_m13_12_{key}_{variant}.json"
            with p.open("x") as f:
                json.dump(cfg, f, indent=2)
            paths.append(p.name)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="G3 generation config")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Override G1 manifest_path; still verified against G1 manifest_sha256.")
    parser.add_argument("--limit", type=int, help="Rebuild only the first N selected rows, interleaved over deciles")
    parser.add_argument("--workers", type=int, default=1, help=f"Worker processes (fork), at most {MAX_WORKERS}")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()
    require(args.limit is None or args.limit >= 1, "Invalid --limit")
    require(1 <= args.workers <= MAX_WORKERS, f"--workers must be in 1..{MAX_WORKERS}")
    args.mode = "m13_12_build_click_free"  # only recorded by g6 attrs_for

    root, spec_g1, df, trainio, _, attrs_for = g6.setup(args)
    per_decile, tau_edges, idx_val = select_rows(root, trainio)
    selected = [r for rows in per_decile for r in rows]
    interleaved = [r for tup in zip(*per_decile) for r in tup]
    todo = interleaved[:args.limit] if args.limit else interleaved
    row_ids = np.sort(np.array([r["row_index"] for r in todo], dtype=np.int64))
    pos = {int(i): p for p, i in enumerate(row_ids)}
    sel_of = {r["row_index"]: r for r in selected}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root.joinpath("results", *RESULTS_SUBDIR, f"inputs_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=False)
    log_file = (out_dir / "build.log").open("x")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    write_json(out_dir / "selection.json", dict(
        selection_seed=SELECTION_SEED, n_per_decile=N_PER_DECILE, tau_groups=TAU_GROUPS,
        tau30="tau_newtonian(true chirp_mass), f_ref 30 Hz; decile edges = val quantiles (M13_8 cell 3)",
        tau_edges_s=tau_edges.tolist(), n_val=int(len(idx_val)), limit=args.limit, rows=selected))
    log(f"Selected {len(selected)} rows; rebuilding {len(todo)} with {args.workers} worker(s) -> {out_dir}")

    frame = g3.adjust_processing_windows(df.loc[[r["row_index"] for r in todo]], spec_g1)
    psd_files, raw_psd_files = resolve_cached_psd_files(root, frame)
    blocks = [list(b.index) for _, b in frame.groupby("block_id", sort=True)]
    with h5py.File(trainio, "r") as t:
        require(t["X"].shape[0] == len(df), "G3 trainio row count differs from the manifest")
        x_shape = (len(row_ids),) + tuple(t["X"].shape[1:])
        y = t["y"][row_ids].astype(np.float32)
    require(x_shape[1] == len(DETECTORS), f"Unexpected trainio X shape {x_shape}")

    X_out = {}
    for variant in VARIANTS:
        (out_dir / variant).mkdir()
        X_out[variant] = np.lib.format.open_memmap(out_dir / variant / f"X_{SPLIT}.npy", mode="w+",
                                                   dtype=np.float16, shape=x_shape)

    _CTX.update(frame=frame, psd_files=psd_files, trainio=trainio)
    records, failed, error = [], None, None
    t_start = time.perf_counter()

    def consume(results):
        nonlocal failed
        for rec, x_orig, x_noclick in results:
            i = rec["row_index"]
            X_out["orig"][pos[i]] = x_orig
            X_out["noclick"][pos[i]] = x_noclick
            rec.update(position=pos[i], tau30_decile=sel_of[i]["tau30_decile"])
            records.append(rec)
            fails = gate_failures(rec)
            log(f"row {i} ({rec['tau30_decile']}) {rec['time_s']:.1f} s: max|dX| "
                f"{max(rec['G_REC']['max_abs_dX_build'], rec['G_REC']['max_abs_dX_reprocess']):.2e}, "
                f"residual {rec['G_ALIGN']['residual_rad']:+.1e} rad, aligned rel {rec['G_ALIGN']['aligned_rel_diff']:.3f}, "
                f"G-DIFF {rec['G_DIFF']['rel_max_abs_dnoise']:.1e}, SNR_click {rec['snr_click_aligned_network']:.2f}, "
                f"truncated {rec['is_truncated']}"
                + (f"  FAIL {fails}" if fails else "") + f"  [{len(records)}/{len(todo)}]")
            if fails and failed is None:
                failed = dict(row_index=i, gates=fails)

    try:
        if args.workers == 1:
            for b in blocks:
                consume(run_block(b))
                if failed:
                    break
        else:
            # fork: workers inherit the g1 globals set by g6.setup and _CTX.
            pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork"))
            try:
                futures = [pool.submit(run_block, b) for b in blocks]
                for fut in as_completed(futures):
                    consume(fut.result())
                    if failed:
                        break
            finally:
                # On a failure, blocks not yet started are cancelled; running ones finish and are discarded.
                pool.shutdown(wait=True, cancel_futures=True)
    except Exception as exc:  # recorded in gates.json, then re-raised
        error = f"{type(exc).__name__}: {exc}"
        log(f"ERROR {error}")

    def worst(gate, key):
        return max((r[gate][key] for r in records), default=None)

    complete = failed is None and error is None and len(records) == len(todo)
    provenance = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                  for k, v in attrs_for(len(todo), {"selection_policy": "m13_12_tau30_deciles",
                                                     "selection_seed": SELECTION_SEED}).items()}
    write_json(out_dir / "gates.json", dict(
        verdict="PASS" if complete else "FAIL", first_failure=failed, error=error,
        n_rows_done=len(records), n_rows_todo=len(todo), limit=args.limit, workers=args.workers,
        wall_time_s=time.perf_counter() - t_start,
        G_REC=dict(threshold=GREC_MAX, max_abs_dX_build=worst("G_REC", "max_abs_dX_build"),
                   max_abs_dX_reprocess=worst("G_REC", "max_abs_dX_reprocess"),
                   n_fp16_diff_total=sum(r["G_REC"]["n_fp16_diff"] for r in records),
                   comparison="X_orig (build_g1_row) and zscore(process_network(captured s + n)) vs G3 trainio X"),
        G_ALIGN=dict(residual_tol_rad=CLICK_ALIGN_PHASE_TOL, aligned_rel_tol=CLICK_ALIGNED_GATE_RTOL,
                     aligned_abs_tol=CLICK_ALIGNED_GATE_ATOL,
                     aligned_rule="|aligned - phasemax| <= max(aligned_rel_tol * phasemax, aligned_abs_tol)",
                     max_abs_residual_rad=max((abs(r["G_ALIGN"]["residual_rad"]) for r in records), default=None),
                     max_aligned_rel_diff=worst("G_ALIGN", "aligned_rel_diff"),
                     n_inconsistent=sum(not r["G_ALIGN"]["align_consistent"] for r in records)),
        G_DIFF=dict(threshold=GDIFF_RTOL, max_rel=worst("G_DIFF", "rel_max_abs_dnoise"),
                    comparison="max|(raw_orig - L(s_A)) - (raw_noclick - L(s_S))| / max|raw_orig - L(s_A)|, "
                               "raw = process_network before the z-score, L = m13_11 process_clean"),
        script_sha256=sha256_file(__file__), raw_psd_files=raw_psd_files, provenance=provenance,
        rows=sorted(records, key=lambda r: r["position"])))
    if not complete:
        log_file.close()
        raise SystemExit(f"M13.12 build aborted ({failed or error}); no manifests written; see {out_dir / 'gates.json'}")

    for variant in VARIANTS:
        X_out[variant].flush()
        d = out_dir / variant
        np.save(d / f"y_{SPLIT}.npy", y)
        np.save(d / f"row_ids_{SPLIT}.npy", row_ids)
        write_json(d / "manifest.json", dict(
            kind="m13_12_click_free_inputs", variant=variant, splits=[SPLIT], n_rows=int(len(row_ids)),
            x_shape=list(x_shape), x_dtype="float16", y_dtype="float32 unstandardized (G3 trainio y)",
            label_names=list(LABELS), row_ids="ascending G3 trainio row ids", created_utc=stamp,
            x=("X of build_g1_row (G3)" if variant == "orig" else
               "zscore(process_network(inject(n, projection of S aligned)), psd_proc); n, psd_proc, "
               "projection, t_c, window and processing of G3"),
            source_trainio=str(trainio), script_sha256=sha256_file(__file__)))
    np.savez(out_dir / f"split_m13_12_{SPLIT}.npz", **{f"{SPLIT}_idx": row_ids})
    rows = []
    for rec in sorted(records, key=lambda r: r["position"]):
        s = sel_of[rec["row_index"]]
        rows.append(dict(position=rec["position"], row_index=rec["row_index"], block_id=rec["block_id"],
                         tau30_s=s["tau30_s"], tau30_decile=s["tau30_decile"], tau_group=s["tau_group"],
                         target_network_snr=s["target_network_snr"], final_network_snr=rec["final_network_snr"],
                         snr_click_aligned_network=rec["snr_click_aligned_network"],
                         snr_click_phasemax_network=rec["snr_click_phasemax_network"],
                         snr_click_network=rec["snr_click_network"], snr_click_aligned=rec["snr_click_aligned"],
                         y_true=s["y_true"], is_truncated=rec["is_truncated"],
                         n_clipped_before_A=rec["n_clipped_before_A"], n_clipped_before_S=rec["n_clipped_before_S"],
                         time_s=rec["time_s"]))
    write_json(out_dir / "rows.json", dict(label_names=list(LABELS), order="ascending row_index (= X position)",
                                           rows=rows))
    cfgs = predict_configs(out_dir, stamp, trainio.parent.name)
    log(f"Gates PASS; wrote {len(row_ids)} rows x {VARIANTS} and {len(cfgs)} prediction configs to {out_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
