#!/usr/bin/env python3
"""M13.11, amendment 2: the 180 val rows with their exact noise and an SEOBNRv4_ROM signal. Local CPU.

Protocol: docs/experiments/M13_11_bayesian_reference.md (amendment 2, 6-11-2026).

The selection is m13_11_prepare_rows.select_rows (default_rng(131101)). Each row is rebuilt by build_g1_row
(m13_11.capture_row), which gives the noise of G3, psd_proc, the projection, t_c and the final distance. Only
the signal changes: SEOBNRv4_ROM from get_td_waveform with f_lower = 15 Hz, aligned by m13_11._align_to_A,
then the same projector, G-EDGE, injection, processing and z-score as the S15 of m13_13_build_g3t.noisy_row.

Alignment: _align_to_A fits against A, on the whole of A, and that is exactly how S15 was aligned; ROM and S15
therefore end up on the same sub-sample grid. The phase left between them is measured directly (analytic
signal of h+, which does not depend on the inclination) and reported as residual_vs_s15_rad.

Gates, per row (a failing row gets no .npz, the others go on, and the script exits with an error at the end):
  G-REC    the rebuild reproduces the G3 X, max|dX| <= 1e-5.
  G-NOISE  processed noise of the ROM path vs that of the G3 path, max|d| / max|n| <= 1e-6.
  G-ALIGN  |residual against A| and |residual_vs_s15_rad| <= 0.04 rad.

Class of the row (amendment 3; it is a label, the row is kept and goes through the PE): snr2_fraction_flat is the
fraction of the network SNR^2 of s_ROM inside the flat part of the Tukey window of the PE, from the whitened
30-1024 Hz signal in time on the processing segment. Below 0.999 the row is "truncated_or_edge", out of the
primary analysis and reported apart; otherwise "primary". snr2_fraction_final4s is the same inside the final 4 s.

Output, <data_root>/results/M13_3/M13_11/rom_rows_<UTC>/ (--resume-dir continues one):
  selection.json          same fields as rows/<stamp>/selection.json
  meta.json, rows.jsonl   run identity; gate record per finished row
  gates_FAIL.json         only if a row failed
  row_<index>.npz         final 4 s of s_ROM + n (strain, 3 x 16384, float64), s_ROM (signal), epoch, delta_t,
                          psd (psd_proc at 1/4 Hz), param_*, target SNR (param_target_network_snr), y_true;
                          for the PE, 8 s centred on that window: strain_8s, signal_8s, epoch_8s, psd_8s (1/8 Hz),
                          psd_delta_f_8s, pe_tukey_ramp_s, row_class, snr2_fraction_flat, snr2_fraction_final4s
  X_work.npy              float16, written row by row, sorted by row id
  fp16_memmap/            only with all rows done: X_val, y_val, row_ids_val (180 rows, format of fp16_memmap),
                          gates_val.json, manifest.json

--limit N takes the first N rows interleaved short/medium/long (as prepare_rows), plus the longest row of the
long regime, which the timing PE needs. The rest can be added later with --resume-dir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import h5py
import numpy as np
from scipy.signal import hilbert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import scripts.generate_m11_final_G1 as g1
from scripts.generate_m11_final_G1 import DETECTORS, LABELS, require, resolve_cached_psd_files, sha256_file
import scripts.m13.generate_m13_G3_whitening as g3
import scripts.m13.generate_m13_6_clean as g6
from scripts.m13.m13_12_build_click_free import GDIFF_RTOL, _NoiseTap
# Importing it also sets the S15 band in m13_11 (f_lower 15 Hz, taper to 25 Hz).
import scripts.m13.m13_13_build_g3t as g3t
m11 = g3t.m11

ROM_F_LOWER = 15.0
ALIGN_MAX = g3t.ALIGN_TOL  # 0.04 rad
PE_MARGIN_S = 2.0       # of the processing segment at each side of the final window
PE_TUKEY_RAMP_S = 0.5   # each ramp, inside the margin
FLAT_SNR2_MIN = 0.999
SPLIT = "val"


def rom_make(kwargs):
    """Waveform for m13_11._align_to_A. t_valid = 0: as for S15, the fit covers the whole of A (30 Hz up)."""
    from pycbc.waveform import get_td_waveform
    hp, hc = get_td_waveform(**{**kwargs, "approximant": m11.ROM_APPROXIMANT, "f_lower": ROM_F_LOWER})
    return hp, hc, 0.0


def residual_vs_s15(rom_hp, s15_hp, segment_s):
    """Phase S15 - ROM (rad) on the last segment_s of their overlap; both on the sub-sample grid of A."""
    dt = float(rom_hp.delta_t)
    shift = int(round((float(s15_hp.start_time) - float(rom_hp.start_time)) / dt))
    end = min(len(rom_hp), len(s15_hp) + shift)
    begin = max(end - int(round(segment_s / dt)), shift, 0)
    r = hilbert(rom_hp.numpy()[begin:end])
    s = hilbert(s15_hp.numpy()[begin - shift:end - shift])
    return float(np.angle(np.vdot(r, s)))


def psd_at_resolution(psd, delta_f, flength):
    out = g1.interpolate(psd, float(delta_f))
    require(len(out) >= flength, f"Interpolated PSD too short: {len(out)}")
    out = out[:flength]
    require(np.all(np.isfinite(out.numpy())), "Non-finite interpolated PSD")
    return out


def snr2_density(x, psd, dt):
    """Squared whitened 30-1024 Hz signal per sample: where the SNR^2 of x lies in time."""
    n = len(x)
    f = np.fft.rfftfreq(n, dt)
    band = (f >= m11.BAND_HZ[0]) & (f <= m11.BAND_HZ[1])
    X = np.fft.rfft(x)
    X[~band] = 0.0
    X[band] /= np.sqrt(psd_at_resolution(psd, 1.0 / (n * dt), f.size).numpy()[band])
    return np.fft.irfft(X, n) ** 2


def rom_row(row, index, ctx, trainio_h, selected):
    """Sequence of m13_13_build_g3t.noisy_row with the ROM in place of S15."""
    from pycbc.filter import sigmasq
    t0 = time.perf_counter()
    injector = ctx["injector"]
    with _NoiseTap() as nt:
        X_row, meta, cap, opt_args = m11.capture_row(row, long_strains=ctx["long_strains"], psd_cache=ctx["psd_cache"])
    require(nt.noise is not None, f"Row {index}: noise not captured")
    require(np.array_equal(trainio_h["y"][index], row[list(LABELS)].to_numpy(dtype=np.float32)),
            f"Trainio y differs at row {index}")
    processing_start, final_start = float(row["processing_start"]), m11.final_start_of(row)
    s_A = cap.signal["projection"].strains

    inj_A = injector.inject_network(noises=nt.noise, signals=s_A)
    for ifo in DETECTORS:
        require(np.array_equal(inj_A[ifo].strain.numpy(), cap.strains[ifo].numpy()),
                f"Row {index} {ifo}: tapped noise + s_A differs from the captured s + n")

    # G-REC
    raw_A, X_rep = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(
        strains={ifo: cap.strains[ifo].copy() for ifo in DETECTORS}, psds=cap.psds))
    X_g3 = np.asarray(trainio_h["X"][index], dtype=np.float64)
    drec = max(float(np.abs(np.asarray(X_row, dtype=np.float64) - X_g3).max()),
               float(np.abs(np.asarray(X_rep, dtype=np.float64) - X_g3).max()))

    # Same noise, PSD, injection, processing and z-score; only the signal is the ROM.
    hp, hc, align = m11._align_to_A(opt_args, rom_make)
    s15_hp, _, align_s15 = m11._opt_click_free_aligned(opt_args)
    direct = residual_vs_s15(hp, s15_hp, align["common_segment_s"])
    proj = g1.DETECTOR_PROJECTOR.project(h_plus=hp, h_cross=hc, parameters=cap.params,
                                         geocentric_coalescence_time=cap.geocentric_time)
    s_R, n_before = g3t.edge_taper(proj.strains, processing_start)
    inj_R = injector.inject_network(noises=nt.noise, signals=s_R)

    # Crop before processing, so that the stored window cannot be affected by process_network.
    dt, n = g1.CONFIG.delta_t, g1.CONFIG.length
    i0 = g1.CONFIG.processing_context_start_samples
    for ifo in DETECTORS:
        require(abs(float(inj_R[ifo].strain.start_time) + i0 * dt - final_start) < g3.TIME_TOL_S,
                f"Row {index} {ifo}: final window misaligned")
    strain = np.stack([np.array(inj_R[ifo].strain.numpy()[i0:i0 + n], dtype=np.float64) for ifo in DETECTORS])
    clean = {ifo: m11.crop_to_final(s_R[ifo], final_start, injector) for ifo in DETECTORS}
    psd4 = {ifo: m11.psd_at_final_resolution(cap.psds[ifo]) for ifo in DETECTORS}

    # 8 s of the PE: the same injected series, 2 s more at each side of the final window.
    margin, ramp = int(round(PE_MARGIN_S / dt)), int(round(PE_TUKEY_RAMP_S / dt))
    a, b = i0 - margin, i0 + n + margin
    require(a >= 0 and b <= g1.CONFIG.processing_length, f"Row {index}: 8 s segment outside the processing segment")
    strain_8s = np.stack([np.array(inj_R[ifo].strain.numpy()[a:b], dtype=np.float64) for ifo in DETECTORS])
    clean_full = {ifo: injector.inject(injector.build_zero_strain(start_time=processing_start,
                                                                  length=g1.CONFIG.processing_length),
                                       s_R[ifo]).strain.numpy() for ifo in DETECTORS}
    signal_8s = np.stack([np.array(clean_full[ifo][a:b], dtype=np.float64) for ifo in DETECTORS])
    delta_f_8s = 1.0 / ((b - a) * dt)
    psd8 = {ifo: psd_at_resolution(cap.psds[ifo], delta_f_8s, (b - a) // 2 + 1) for ifo in DETECTORS}
    density = sum(snr2_density(clean_full[ifo], cap.psds[ifo], dt) for ifo in DETECTORS)
    frac_flat = float(density[a + ramp:b - ramp].sum() / density.sum())
    frac_4s = float(density[i0:i0 + n].sum() / density.sum())
    row_class = "primary" if frac_flat >= FLAT_SNR2_MIN else "truncated_or_edge"

    raw_R, X_R = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(
        strains={ifo: inj_R[ifo].strain for ifo in DETECTORS}, psds=cap.psds))
    dn_A = np.asarray(raw_A, dtype=np.float64) - m11.process_clean(
        s_A, processing_start=processing_start, psds=cap.psds, injector=injector)
    dn_R = np.asarray(raw_R, dtype=np.float64) - m11.process_clean(
        s_R, processing_start=processing_start, psds=cap.psds, injector=injector)

    snr2 = {ifo: float(sigmasq(clean[ifo], psd=psd4[ifo], low_frequency_cutoff=m11.BAND_HZ[0],
                               high_frequency_cutoff=m11.BAND_HZ[1])) for ifo in DETECTORS}
    rec = dict(row_index=int(index), regime=selected["regime"], block_id=str(row["block_id"]), G_REC=drec,
               G_NOISE=float(np.abs(dn_A - dn_R).max() / np.abs(dn_A).max()),
               G_ALIGN=float(align["residual_rad"]), residual_vs_s15_rad=direct,
               s15_residual_rad=float(align_s15["residual_rad"]), rom_time_shift_s=float(align["time_shift_s"]),
               rom_coa_phase_shift_rad=float(align["coa_phase_shift_rad"]),
               edge_samples=n_before, edge=bool(max(n_before.values()) > 0),
               is_truncated=bool(meta["is_truncated"]), target_network_snr=float(row["target_network_snr"]),
               row_class=row_class, snr2_fraction_flat=frac_flat, snr2_fraction_final4s=frac_4s,
               snr_rom_network_4s=float(np.sqrt(sum(snr2.values()))), time_s=time.perf_counter() - t0)
    arrays = dict(
        row_index=int(index), regime=selected["regime"], tau30_s=selected["tau30_s"],
        tau30_decile=selected["tau30_decile"], detectors=np.array(DETECTORS), label_names=np.array(LABELS),
        y_true=row[list(LABELS)].to_numpy(dtype=np.float64), approximant=m11.ROM_APPROXIMANT,
        injection_f_lower=ROM_F_LOWER, f_lower=m11.BAND_HZ[0], f_upper=m11.BAND_HZ[1],
        epoch=np.full(len(DETECTORS), final_start), delta_t=float(dt), strain=strain,
        signal=np.stack([clean[ifo].numpy() for ifo in DETECTORS]).astype(np.float64),
        psd=np.stack([psd4[ifo].numpy() for ifo in DETECTORS]).astype(np.float64),
        psd_delta_f=float(psd4[DETECTORS[0]].delta_f),
        epoch_8s=np.full(len(DETECTORS), final_start - PE_MARGIN_S), strain_8s=strain_8s, signal_8s=signal_8s,
        psd_8s=np.stack([psd8[ifo].numpy() for ifo in DETECTORS]).astype(np.float64), psd_delta_f_8s=delta_f_8s,
        pe_tukey_ramp_s=PE_TUKEY_RAMP_S, row_class=row_class, snr2_fraction_flat=frac_flat,
        snr2_fraction_final4s=frac_4s,
        param_distance=float(meta["final_distance"]), param_geocentric_time=float(meta["actual_geocentric_time"]),
        param_final_network_snr=float(meta["final_network_snr"]),
        **{f"param_{c}": float(row[c]) for c in m11.PARAM_COLUMNS},
        **{f"param_{k}": v for k, v in m11.PYCBC_DEFAULTS.items()},
        param_rom_coa_phase_shift=float(align["coa_phase_shift_rad"]),
        param_is_truncated=bool(meta["is_truncated"]))
    return rec, arrays, g6.to_fp16(np.asarray(X_R, dtype=np.float32), f"X_ROM row {index}")


def gate_fails(rec):
    fails = []
    if not rec["G_REC"] <= m11.GREC_MAX:
        fails.append("G-REC")
    if not rec["G_NOISE"] <= GDIFF_RTOL:  # also catches NaN
        fails.append("G-NOISE")
    if not max(abs(rec["G_ALIGN"]), abs(rec["residual_vs_s15_rad"])) <= ALIGN_MAX:
        fails.append("G-ALIGN")
    return fails


def rom_chunk(indices):
    out = []
    frame = g3t._CTX["frame"].loc[indices]
    with h5py.File(g3t._CTX["trainio"], "r") as t:
        for index, row in frame.iterrows():
            out.append(rom_row(row, index, g3t.block_context(row), t, g3t._CTX["selected"][int(index)]))
    return out


def write_row(path, arrays):
    # The .npz is what marks a row as done, so it appears complete or not at all.
    tmp = path.with_suffix(f".tmp_{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def summary(records, fails):
    good = list(records.values())
    block = dict(n_rows=len(good), n_failed=len(fails), failed=fails,
                 G_REC=dict(threshold=m11.GREC_MAX, max=max((r["G_REC"] for r in good), default=None)),
                 G_NOISE=dict(threshold=GDIFF_RTOL, max=max((r["G_NOISE"] for r in good), default=None)),
                 G_ALIGN=dict(threshold_rad=ALIGN_MAX,
                              max_vs_A=max((abs(r["G_ALIGN"]) for r in good), default=None),
                              max_vs_s15=max((abs(r["residual_vs_s15_rad"]) for r in good), default=None)),
                 G_EDGE=dict(n_rows_tapered=sum(r["edge"] for r in good)),
                 flat_zone=dict(threshold=FLAT_SNR2_MIN,
                                truncated_or_edge=[r["row_index"] for r in good if r["row_class"] != "primary"],
                                is_truncated_in_g3=[r["row_index"] for r in good if r["is_truncated"]],
                                min_fraction_flat=min((r["snr2_fraction_flat"] for r in good), default=None),
                                min_fraction_final4s=min((r["snr2_fraction_final4s"] for r in good), default=None)),
                 time_s_per_row_mean=float(np.mean([r["time_s"] for r in good])) if good else None)
    return block


def publish(out_dir, rows, X_work, records, fails, src_dir, trainio, provenance):
    """All rows done: fp16_memmap/ with the 180 rows, in the format of the G3 fp16_memmap."""
    final = out_dir / "fp16_memmap"
    require(not final.exists(), f"Already published: {final}")
    row_ids = np.array(rows, dtype=np.int64)
    g3_ids = np.load(src_dir / f"row_ids_{SPLIT}.npy")
    pos = np.searchsorted(g3_ids, row_ids)
    require(np.array_equal(g3_ids[pos], row_ids), f"Rows outside row_ids_{SPLIT}.npy")
    y = np.array(np.load(src_dir / f"y_{SPLIT}.npy", mmap_mode="r")[pos])
    tmp = final.with_name(f"fp16_memmap.tmp_{os.getpid()}")
    tmp.mkdir()
    np.save(tmp / f"row_ids_{SPLIT}.npy", row_ids)
    np.save(tmp / f"y_{SPLIT}.npy", y)
    (tmp / f"gates_{SPLIT}.json").write_text(json.dumps(summary(records, fails), indent=2))
    (tmp / "manifest.json").write_text(json.dumps(dict(
        kind="m13_11_rom_rows", dataset_id=trainio.parent.name, x_dtype="float16", y_dtype="float32 (unstandardized)",
        x="zscore(process_network(inject(n, SEOBNRv4_ROM aligned with S15)), psd_proc); n, psd_proc, projection, "
          "t_c and processing of G3",
        rom_f_lower=ROM_F_LOWER, source_hdf5=dict(path=str(trainio)), provenance=provenance,
        splits={SPLIT: dict(n_rows=int(row_ids.size), shape_X=list(X_work.shape),
                            row_ids_sha256=sha256_file(tmp / f"row_ids_{SPLIT}.npy"),
                            script_sha256=sha256_file(__file__), contents="the 180 rows of M13.11, by row id",
                            gates=f"gates_{SPLIT}.json")}), indent=2))
    X_work.flush()
    os.rename(out_dir / "X_work.npy", tmp / f"X_{SPLIT}.npy")
    os.rename(tmp, final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=m11.DEFAULT_CONFIG, help="G3 generation config")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Override G1 manifest_path; still verified against G1 manifest_sha256.")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--limit", type=int, help="First N interleaved rows, plus the longest row of the long regime")
    parser.add_argument("--workers", type=int, default=1, help=f"Worker processes (fork), at most {g3t.MAX_WORKERS}")
    parser.add_argument("--resume-dir", type=Path, help="Existing rom_rows_<UTC> directory to continue")
    args = parser.parse_args()
    require(1 <= args.workers <= g3t.MAX_WORKERS, f"--workers must be in 1..{g3t.MAX_WORKERS}")
    require(args.limit is None or args.limit >= 1, "Invalid --limit")
    args.mode = "m13_11_build_rom_rows"  # only recorded by g6 attrs_for

    root, spec_g1, df, trainio, _, attrs_for = g6.setup(args)
    lal_data = Path(os.environ.setdefault("LAL_DATA_PATH", str(root / "external" / "lal_data")))
    require(lal_data.is_dir(), f"LAL_DATA_PATH is not a directory: {lal_data}")
    require(abs(g1.CONFIG.delta_t - 1.0 / 4096) < 1e-12, "delta_t differs from the G3 one")

    chosen, tau_edges, idx_val, _, _, _ = m11.select_rows(root, trainio, m11.N_PER_REGIME)
    selected_rows = [r for name in m11.REGIMES for r in chosen[name]]
    interleaved = [r for triple in zip(*(chosen[name] for name in m11.REGIMES)) for r in triple]
    longest = max(chosen["long"], key=lambda r: r["tau30_s"])
    todo = interleaved[:args.limit] if args.limit else interleaved
    if longest not in todo:
        todo = todo + [longest]
    row_ids = sorted(r["row_index"] for r in selected_rows)
    position = {i: p for p, i in enumerate(row_ids)}

    meta = dict(script_sha256=sha256_file(__file__), g3t_script_sha256=sha256_file(g3t.__file__),
                row_ids=row_ids, rom_f_lower=ROM_F_LOWER, approximant=m11.ROM_APPROXIMANT,
                lal_data_path=str(lal_data))
    if args.resume_dir:
        out_dir = args.resume_dir
        require((out_dir / "meta.json").exists(), f"Nothing to resume in {out_dir}")
        require(json.loads((out_dir / "meta.json").read_text()) == meta, "Resume with another script or selection")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = root / "results" / "M13_3" / "M13_11" / f"rom_rows_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=False)
        m11.write_json(out_dir / "meta.json", meta)
        m11.write_json(out_dir / "selection.json", dict(
            selection_seed=m11.SELECTION_SEED, n_per_regime=m11.N_PER_REGIME,
            regimes={k: [f"D{d + 1}" for d in v] for k, v in m11.REGIMES.items()},
            tau30="tau_newtonian(true chirp_mass), f_ref 30 Hz; decile edges = val quantiles (M13_8 cell 3)",
            tau_edges_s=tau_edges.tolist(), n_val=int(len(idx_val)),
            timing_row_longest_long_regime=longest, rows=selected_rows))

    with h5py.File(trainio, "r") as t:
        shape = (len(row_ids),) + tuple(t["X"].shape[1:])
    X_path = out_dir / "X_work.npy"
    if (out_dir / "fp16_memmap").exists():
        raise SystemExit(f"Already complete: {out_dir / 'fp16_memmap'}")
    X = np.lib.format.open_memmap(X_path, mode="r+" if X_path.exists() else "w+", dtype=np.float16, shape=shape)

    log = g3t.Log(out_dir / "build.log")
    jsonl = out_dir / "rows.jsonl"
    records = {}
    if jsonl.exists():  # a row can appear twice after a crash; the last record wins
        with jsonl.open() as f:
            for line in f:
                rec = json.loads(line)
                records[rec["row_index"]] = rec
    npz = {i: out_dir / f"row_{i:06d}.npz" for i in row_ids}
    records = {i: r for i, r in records.items() if npz[i].exists()}
    pending = [r for r in todo if not npz[r["row_index"]].exists()]
    log(f"M13.11 ROM rows: {len(todo)} requested, {len(pending)} pending, {args.workers} worker(s) -> {out_dir}; "
        f"LAL_DATA_PATH {lal_data}")
    fails = []

    if pending:
        frame = g3.adjust_processing_windows(df.loc[[r["row_index"] for r in pending]], spec_g1)
        psd_files, _ = resolve_cached_psd_files(root, frame)
        g3t._CTX.update(frame=frame, psd_files=psd_files, trainio=trainio,
                        selected={r["row_index"]: r for r in selected_rows})
        rows_file = jsonl.open("a")
        progress = g3t.Progress(len(todo), len(todo) - len(pending))

        def consume(results):
            for rec, arrays, x16 in results:
                index = rec["row_index"]
                failed = gate_fails(rec)
                if failed:
                    fails.append(dict(gates=failed, record=rec))
                    log(f"row {index} ({rec['regime']}) FAIL {failed}: G-REC {rec['G_REC']:.1e}, G-NOISE "
                        f"{rec['G_NOISE']:.1e}, residual {rec['G_ALIGN']:+.1e} / {rec['residual_vs_s15_rad']:+.1e} rad")
                    continue
                X[position[index]] = x16
                X.flush()
                rows_file.write(json.dumps(rec) + "\n")
                rows_file.flush()
                os.fsync(rows_file.fileno())
                write_row(npz[index], arrays)
                records[index] = rec
                log(f"row {index} ({rec['regime']}) {rec['time_s']:.1f} s: G-REC {rec['G_REC']:.1e}, G-NOISE "
                    f"{rec['G_NOISE']:.1e}, residual {rec['G_ALIGN']:+.1e} rad, vs S15 "
                    f"{rec['residual_vs_s15_rad']:+.1e} rad, SNR {rec['snr_rom_network_4s']:.2f} (target "
                    f"{rec['target_network_snr']:.2f}), edge {int(rec['edge'])}, flat "
                    f"{rec['snr2_fraction_flat']:.5f}, 4 s {rec['snr2_fraction_final4s']:.5f}, {rec['row_class']}  "
                    f"{progress.tick()}")

        g3t.run_tasks(rom_chunk, g3t.block_chunks(frame), args.workers, consume)
        rows_file.close()

    s = summary(records, fails)
    log(f"Gates: {s['n_rows']} rows OK, {s['n_failed']} failed; G-REC {s['G_REC']['max']}, G-NOISE "
        f"{s['G_NOISE']['max']}, G-ALIGN {s['G_ALIGN']['max_vs_A']} / {s['G_ALIGN']['max_vs_s15']} rad")
    log(f"Flat zone: {s['flat_zone']}")
    if fails:
        g3t.overwrite_json(out_dir / "gates_FAIL.json", dict(fails=fails))
        raise SystemExit(f"{len(fails)} rows failed a gate; see {out_dir / 'gates_FAIL.json'}")
    if all(npz[i].exists() for i in row_ids):
        provenance = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                      for k, v in attrs_for(len(row_ids), {"selection_policy": "m13_11_tau30_regimes",
                                                           "selection_seed": m11.SELECTION_SEED}).items()}
        final = publish(out_dir, row_ids, X, records, fails, trainio.parent / "fp16_memmap", trainio, provenance)
        log(f"Published {final}")
    else:
        log(f"{sum(p.exists() for p in npz.values())}/{len(row_ids)} rows done; continue with --resume-dir {out_dir}")


if __name__ == "__main__":
    main()
