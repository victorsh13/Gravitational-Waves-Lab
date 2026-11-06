#!/usr/bin/env python3
"""M13.11, amendment 2: the 180 val rows with their exact noise and an SEOBNRv4_ROM signal. Local CPU.

Protocol: docs/experiments/M13_11_bayesian_reference.md (amendment 2, 6-11-2026).

The selection is m13_11_prepare_rows.select_rows (default_rng(131101)). Each row is rebuilt by build_g1_row
(m13_11.capture_row), which gives the noise of G3, psd_proc, the projection, t_c and the final distance. Only
the signal changes: SEOBNRv4_ROM from get_td_waveform with f_lower = 15 Hz, aligned by align_rom, then the same
projector, G-EDGE, injection, processing and z-score as the S15 of m13_13_build_g3t.noisy_row.

Alignment (align_rom, local: m13_11._align_to_A is frozen and searches only +-32 samples, while the time-domain
ROM has its peak tens of ms away from SEOBNRv4_opt). Reference: S15 face-on, on the segment S15 was fitted on,
the length of A. (a) Coarse: FFT cross-correlation of h = h+ - i hx, ROM against S15, within +-0.25 s; a peak at
the edge fails the row, not the run. (b) Fine, the procedure of _align_to_A on the result of (a): sub-sample lag
by parabolic interpolation of the peak, then coa_phase += sign * delta / 2. The ROM is padded with zeros before
being shifted, so that a delay cannot cut its merger. The phase left after the real inclination is regenerated
is measured apart (analytic signal of h+, which does not depend on the inclination): residual_vs_s15_rad.
The mismatch ROM-S15 (30-1024 Hz, aLIGOZeroDetHighPower as in G-TAPER) is informative only.

Gates, per row (a failing row gets no .npz, the others go on, and the script exits with an error at the end):
  G-REC    the rebuild reproduces the G3 X, max|dX| <= 1e-5.
  G-NOISE  processed noise of the ROM path vs that of the G3 path, max|d| / max|n| <= 1e-6.
  G-ALIGN  |residual of the face-on fit| and |residual_vs_s15_rad| <= 0.04 rad; coarse peak inside +-0.25 s.

Class of the row (amendment 4; a label, every row goes through the PE): "primary" if is_truncated = False in G3,
i.e. the waveform from 30 Hz fits in the final 4 s; otherwise "truncated_or_edge", reported apart.
snr2_fraction_flat and snr2_fraction_final4s are information only: the fraction of the network SNR^2 of s_ROM in
the flat part of the Tukey window of the PE and in the final 4 s, from the whitened 30-1024 Hz signal in time. The
whitening filter spreads that energy out of the window, so ~0.998 is normal in a row that is not truncated.

G-EDGE: inject would cut a signal that starts before processing_start, so m13_13_build_g3t.edge_taper rolls it up
over the first 0.5 s of the processing segment ("edge" in the records, samples before it in edge_samples). The
ROM series can start long before that: the inspiral from 15 Hz, and the padding of its FD-to-TD conversion.
edge_snr2_removed_fraction is the fraction of the network SNR^2 in 30-1024 Hz of the projection removed by the
cut and the taper.

--reclassify DIR, for rows already built: writes DIR/row_classes.json with the class from is_truncated and
edge_snr2_removed_fraction, which needs the noiseless projection again (the taper is outside the stored 8 s): ROM,
aligned, projected and tapered from the .npz parameters, and required to reproduce the stored signal_8s. Nothing
of the rows changes (.npz, rows.jsonl, X); a stale row_class in an .npz is ignored, since run_pe takes the class
from is_truncated. It records this script in meta.json, so that --resume-dir can add rows to DIR.

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
from src.parameters import CBCParameters
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
COARSE_WINDOW_S = 0.25
PE_MARGIN_S = 2.0       # of the processing segment at each side of the final window
PE_TUKEY_RAMP_S = 0.5   # each ramp, inside the margin
RECONSTRUCTION_RTOL = 1e-6  # --reclassify: projection rebuilt vs stored signal_8s
SPLIT = "val"


class RomAlignmentError(Exception):
    """The ROM of this row cannot be aligned; the row fails, the run goes on."""


def rom_make(kwargs):
    from pycbc.waveform import get_td_waveform
    return get_td_waveform(**{**kwargs, "approximant": m11.ROM_APPROXIMANT, "f_lower": ROM_F_LOWER})


def rom_s15_mismatch(r_hp, r_hc, s_hp, s_hc):
    """Worst of h+ and hx: (time and phase maximised, phase maximised at the alignment found), 30-1024 Hz."""
    from pycbc.filter import match, overlap_cplx
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import TimeSeries
    dt = float(r_hp.delta_t)
    t_ref = min(float(r_hp.start_time), float(s_hp.start_time))
    k_r, k_s = (int(round((float(ts.start_time) - t_ref) / dt)) for ts in (r_hp, s_hp))
    n_fft = 2 * int(2 ** np.ceil(np.log2(max(k_r + len(r_hp), k_s + len(s_hp)))))
    psd = aLIGOZeroDetHighPower(n_fft // 2 + 1, 1.0 / (n_fft * dt), m11.INTRINSIC_PSD_F_LOW)

    def place(ts, k):
        x = np.zeros(n_fft)
        x[k:k + len(ts)] = ts.numpy()
        return TimeSeries(x, delta_t=dt, epoch=0.0)

    lo, hi = m11.BAND_HZ
    best, here = 0.0, 0.0
    for r, s_ in ((r_hp, s_hp), (r_hc, s_hc)):
        a, b = place(r, k_r), place(s_, k_s)
        best = max(best, 1.0 - float(match(a, b, psd=psd, low_frequency_cutoff=lo, high_frequency_cutoff=hi)[0]))
        here = max(here, 1.0 - abs(overlap_cplx(a, b, psd=psd, low_frequency_cutoff=lo, high_frequency_cutoff=hi,
                                                normalized=True)))
    return best, float(here)


def align_rom(opt_args):
    """(h+, hx, diagnostics) of the ROM aligned with S15; see the module docstring.

    Returns the ROM with the real kwargs. S15 and the ROM share a sub-sample grid, as S15 and A do in _align_to_A.
    """
    from pycbc.types import TimeSeries
    dt = float(opt_args["delta_t"])
    face = {**opt_args, "inclination": 0.0}
    coa0 = float(opt_args.get("coa_phase", 0.0))
    s_hp, s_hc, align_s = m11._opt_click_free_aligned(face)
    a = s_hp.numpy() - 1j * s_hc.numpy()
    n_seg = int(round(align_s["common_segment_s"] / dt))
    i_s = len(a) - n_seg
    require(i_s >= 0, "S15 is shorter than the segment it was fitted on")
    a_seg = a[i_s:]

    r_hp, r_hc = rom_make(face)
    start_r, n_r = float(r_hp.start_time), len(r_hp)
    x = r_hp.numpy() - 1j * r_hc.numpy()
    offset = (float(s_hp.start_time) - start_r) / dt
    k_e = int(round(offset))
    r0 = offset - k_e

    # (a) r[m] = sum_k x[k + m] conj(a_seg[k]), around m0 = the alignment given by the epochs.
    n_fft = int(2 ** np.ceil(np.log2(len(x) + n_seg)))
    r = np.fft.ifft(np.fft.fft(x, n_fft) * np.conj(np.fft.fft(a_seg, n_fft)))
    w = int(round(COARSE_WINDOW_S / dt))
    c = np.abs(r[(i_s + k_e + np.arange(-w, w + 1)) % n_fft])
    i = int(np.argmax(c))
    if i == 0 or i == 2 * w:
        raise RomAlignmentError(f"Coarse peak at the edge of +-{COARSE_WINDOW_S:g} s ({(i - w) * dt * 1e3:+.1f} ms)")
    coarse = i - w

    # (b) Sub-sample lag and phase, as _align_to_A. Zeros at both ends, since _advance keeps the input length.
    fine = 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2.0 * c[i] + c[i + 1])
    lag = coarse + fine
    epoch = start_r - w * dt + r0 * dt  # on the sub-sample grid of S15

    def aligned(hp, hc):
        if float(hp.start_time) != start_r or len(hp) != n_r:
            raise RomAlignmentError("Regenerated ROM is on another time grid")
        return tuple(m11._advance(TimeSeries(np.concatenate((np.zeros(w), ts.numpy(), np.zeros(w))), delta_t=dt,
                                             epoch=start_r - w * dt), lag, epoch) for ts in (hp, hc))

    def phase(hp, hc):
        h = hp.numpy() - 1j * hc.numpy()
        begin = i_s + k_e + w
        seg = h[begin:begin + n_seg]
        return float(np.angle(np.vdot(seg, a_seg[:len(seg)])))

    delta = phase(*aligned(r_hp, r_hc))
    best = None
    for sign in (1.0, -1.0):
        candidate = aligned(*rom_make({**face, "coa_phase": coa0 + sign * delta / 2}))
        residual = phase(*candidate)
        if best is None or abs(residual) < abs(best[1]):
            best = (sign, residual, candidate)
        if abs(residual) < m11.CLICK_ALIGN_PHASE_TOL:
            break
    sign, residual, face_aligned = best
    mismatch, mismatch_here = rom_s15_mismatch(*face_aligned, s_hp, s_hc)
    hp, hc = aligned(*rom_make({**opt_args, "coa_phase": coa0 + sign * delta / 2}))
    return hp, hc, dict(coarse_lag_ms=coarse * dt * 1e3, fine_lag_ms=fine * dt * 1e3, delta_rad=delta, sign=sign,
                        residual_rad=residual, coa_phase_shift_rad=sign * delta / 2,
                        time_shift_s=float((lag - r0) * dt), common_segment_s=n_seg * dt,
                        mismatch_rom_s15=mismatch, mismatch_rom_s15_at_alignment=mismatch_here)


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


def row_class_of(is_truncated):
    return "truncated_or_edge" if is_truncated else "primary"


def band_snr2(x, psd, dt, n_fft):
    """Optimal SNR^2 of x in 30-1024 Hz, up to a constant; psd interpolated linearly to the grid of n_fft."""
    f = np.fft.rfftfreq(n_fft, dt)
    band = (f >= m11.BAND_HZ[0]) & (f <= m11.BAND_HZ[1])
    S = np.interp(f[band], psd.sample_frequencies.numpy(), psd.numpy())
    return float(np.sum(np.abs(np.fft.rfft(x, n_fft)[band]) ** 2 / S))


def edge_snr2_removed(full, tapered, n_before, psds, dt):
    """Fraction of the network SNR^2 (30-1024 Hz) of the projection `full` lost in G-EDGE."""
    removed, total = 0.0, 0.0
    for ifo in DETECTORS:
        x = np.array(full[ifo].numpy(), dtype=np.float64)
        kept = np.array(tapered[ifo].numpy(), dtype=np.float64)
        kept[:n_before[ifo]] = 0.0  # what precedes processing_start never reaches the strain
        n_fft = int(2 ** np.ceil(np.log2(2 * len(x))))
        removed += band_snr2(x - kept, psds[ifo], dt, n_fft)
        total += band_snr2(x, psds[ifo], dt, n_fft)
    return removed / total


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
    try:
        hp, hc, align = align_rom(opt_args)
    except RomAlignmentError as exc:
        return dict(row_index=int(index), regime=selected["regime"], alignment_error=str(exc)), None, None
    s15_hp, _, align_s15 = m11._opt_click_free_aligned(opt_args)
    direct = residual_vs_s15(hp, s15_hp, align["common_segment_s"])
    proj = g1.DETECTOR_PROJECTOR.project(h_plus=hp, h_cross=hc, parameters=cap.params,
                                         geocentric_coalescence_time=cap.geocentric_time)
    s_R, n_before = g3t.edge_taper(proj.strains, processing_start)
    edge_removed = edge_snr2_removed(proj.strains, s_R, n_before, cap.psds, g1.CONFIG.delta_t)
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
    row_class = row_class_of(bool(meta["is_truncated"]))

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
               coarse_lag_ms=align["coarse_lag_ms"], fine_lag_ms=align["fine_lag_ms"],
               mismatch_rom_s15=align["mismatch_rom_s15"],
               mismatch_rom_s15_at_alignment=align["mismatch_rom_s15_at_alignment"],
               edge_samples=n_before, edge=bool(max(n_before.values()) > 0),
               edge_snr2_removed_fraction=edge_removed,
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
        snr2_fraction_final4s=frac_4s, edge_snr2_removed_fraction=edge_removed,
        param_distance=float(meta["final_distance"]), param_geocentric_time=float(meta["actual_geocentric_time"]),
        param_final_network_snr=float(meta["final_network_snr"]),
        **{f"param_{c}": float(row[c]) for c in m11.PARAM_COLUMNS},
        **{f"param_{k}": v for k, v in m11.PYCBC_DEFAULTS.items()},
        param_rom_coa_phase_shift=float(align["coa_phase_shift_rad"]),
        param_is_truncated=bool(meta["is_truncated"]))
    return rec, arrays, g6.to_fp16(np.asarray(X_R, dtype=np.float32), f"X_ROM row {index}")


def gate_fails(rec):
    if "alignment_error" in rec:
        return ["G-ALIGN"]
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
                              max_fit=max((abs(r["G_ALIGN"]) for r in good), default=None),
                              max_vs_s15=max((abs(r["residual_vs_s15_rad"]) for r in good), default=None)),
                 coarse_lag_ms=dict(min=min((r["coarse_lag_ms"] for r in good), default=None),
                                    max=max((r["coarse_lag_ms"] for r in good), default=None)),
                 mismatch_rom_s15_max=max((r["mismatch_rom_s15"] for r in good), default=None),
                 G_EDGE=dict(n_rows_tapered=sum(r["edge"] for r in good)),
                 flat_zone=dict(truncated_or_edge=[r["row_index"] for r in good if r["is_truncated"]],
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


def reclassify(rows_dir, meta):
    from pycbc.types import FrequencySeries
    old_meta = json.loads((rows_dir / "meta.json").read_text())
    require({k: v for k, v in old_meta.items() if k not in ("script_sha256", "previous_script_sha256")}
            == {k: v for k, v in meta.items() if k != "script_sha256"}, f"{rows_dir} has another selection or G3T")
    injector = g1.SignalInjector(config=g1.CONFIG, rng=np.random.default_rng(0))
    regime = {r["row_index"]: r["regime"] for r in json.loads((rows_dir / "selection.json").read_text())["rows"]}
    dt, n_proc = g1.CONFIG.delta_t, g1.CONFIG.processing_length
    print("row regime truncated class (was) flat final4s edge_samples removed_fraction reconstruction")
    out = []
    for index in sorted(regime):
        path = rows_dir / f"row_{index:06d}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as a:
            row = {k: a[k] for k in a.files}
        v = {k[len("param_"):]: row[k].item() for k in row if k.startswith("param_")}
        opt_args = dict(mass1=v["mass_1"], mass2=v["mass_2"], spin1z=v["spin_1z"], spin2z=v["spin_2z"],
                        inclination=v["inclination"], distance=v["distance"], delta_t=dt,
                        f_lower=g1.CONFIG.low_frequency_cutoff, approximant=g1.CONFIG.waveform_approximant)
        hp, hc, _ = align_rom(opt_args)
        params = CBCParameters(mass_1=v["mass_1"], mass_2=v["mass_2"], distance=v["distance"],
                               inclination=v["inclination"], ra=v["ra"], dec=v["dec"], spin_1z=v["spin_1z"],
                               spin_2z=v["spin_2z"], polarization_angle=v["polarization_angle"])
        proj = g1.DETECTOR_PROJECTOR.project(h_plus=hp, h_cross=hc, parameters=params,
                                             geocentric_coalescence_time=v["geocentric_time"])
        processing_start = float(row["epoch"][0]) - g1.CONFIG.processing_context_start_seconds
        s_R, n_before = g3t.edge_taper(proj.strains, processing_start)
        i0, margin = g1.CONFIG.processing_context_start_samples, int(round(PE_MARGIN_S / dt))
        a_i, b_i = i0 - margin, i0 + g1.CONFIG.length + margin
        rebuilt = np.stack([injector.inject(injector.build_zero_strain(start_time=processing_start, length=n_proc),
                                            s_R[ifo]).strain.numpy()[a_i:b_i] for ifo in DETECTORS])
        diff = float(np.abs(rebuilt - row["signal_8s"]).max() / np.abs(row["signal_8s"]).max())
        require(diff <= RECONSTRUCTION_RTOL, f"Row {index}: rebuilt projection differs from signal_8s ({diff:.1e})")
        psds = {ifo: FrequencySeries(row["psd_8s"][i], delta_f=row["psd_delta_f_8s"].item())
                for i, ifo in enumerate(DETECTORS)}
        rec = dict(row_index=index, regime=regime[index], is_truncated=bool(v["is_truncated"]),
                   row_class=row_class_of(bool(v["is_truncated"])), previous_row_class=row["row_class"].item(),
                   snr2_fraction_flat=row["snr2_fraction_flat"].item(),
                   snr2_fraction_final4s=row["snr2_fraction_final4s"].item(), edge=bool(max(n_before.values()) > 0),
                   edge_samples=n_before, edge_snr2_removed_fraction=edge_snr2_removed(proj.strains, s_R, n_before,
                                                                                       psds, dt),
                   reconstruction_rel_diff=diff)
        out.append(rec)
        print(f"{index} {rec['regime']} {rec['is_truncated']} {rec['row_class']} ({rec['previous_row_class']}) "
              f"{rec['snr2_fraction_flat']:.5f} {rec['snr2_fraction_final4s']:.5f} {list(n_before.values())} "
              f"{rec['edge_snr2_removed_fraction']:.3e} {diff:.1e}", flush=True)
    g3t.overwrite_json(rows_dir / "row_classes.json", dict(criterion="primary = is_truncated False (amendment 4)",
                                                           script_sha256=meta["script_sha256"], rows=out))
    if old_meta["script_sha256"] != meta["script_sha256"]:
        # Between these versions only the labels change, not how the rows are built.
        previous = old_meta.get("previous_script_sha256", []) + [old_meta["script_sha256"]]
        g3t.overwrite_json(rows_dir / "meta.json", {**meta, "previous_script_sha256": previous})
    print(f"{sum(r['row_class'] == 'primary' for r in out)}/{len(out)} rows primary -> {rows_dir / 'row_classes.json'}")


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
    parser.add_argument("--reclassify", type=Path, metavar="DIR",
                        help="Only write DIR/row_classes.json for the rows already built; see the docstring")
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
    if args.reclassify:
        reclassify(args.reclassify, meta)
        return
    if args.resume_dir:
        out_dir = args.resume_dir
        require((out_dir / "meta.json").exists(), f"Nothing to resume in {out_dir}")
        saved = json.loads((out_dir / "meta.json").read_text())
        saved.pop("previous_script_sha256", None)  # set by --reclassify
        require(saved == meta, "Resume with another script or selection")
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
                    log(f"row {index} ({rec['regime']}) FAIL {failed}: {rec.get('alignment_error', rec)}")
                    continue
                X[position[index]] = x16
                X.flush()
                rows_file.write(json.dumps(rec) + "\n")
                rows_file.flush()
                os.fsync(rows_file.fileno())
                write_row(npz[index], arrays)
                records[index] = rec
                log(f"row {index} ({rec['regime']}) {rec['time_s']:.1f} s: G-REC {rec['G_REC']:.1e}, G-NOISE "
                    f"{rec['G_NOISE']:.1e}, lag {rec['coarse_lag_ms']:+.2f} + {rec['fine_lag_ms']:+.3f} ms, residual "
                    f"{rec['G_ALIGN']:+.1e} rad, vs S15 {rec['residual_vs_s15_rad']:+.1e} rad, mismatch "
                    f"{rec['mismatch_rom_s15']:.1e}, SNR {rec['snr_rom_network_4s']:.2f} (target "
                    f"{rec['target_network_snr']:.2f}), edge {int(rec['edge'])}, flat "
                    f"{rec['snr2_fraction_flat']:.5f}, 4 s {rec['snr2_fraction_final4s']:.5f}, {rec['row_class']}  "
                    f"{progress.tick()}")

        g3t.run_tasks(rom_chunk, g3t.block_chunks(frame), args.workers, consume)
        rows_file.close()

    s = summary(records, fails)
    log(f"Gates: {s['n_rows']} rows OK, {s['n_failed']} failed; G-REC {s['G_REC']['max']}, G-NOISE "
        f"{s['G_NOISE']['max']}, G-ALIGN {s['G_ALIGN']['max_fit']} / {s['G_ALIGN']['max_vs_s15']} rad; "
        f"coarse lag {s['coarse_lag_ms']}, max mismatch ROM-S15 {s['mismatch_rom_s15_max']}")
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
