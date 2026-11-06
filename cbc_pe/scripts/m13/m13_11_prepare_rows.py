#!/usr/bin/env python3
"""M13.11 step 0: select val rows per tau30 regime, rebuild them exactly, run gates G-REC and G-MM.

Protocol: docs/experiments/M13_11_bayesian_reference.md (pre-registered 4-11-2026). Local CPU.

Selection: val membership from the pinned split (g1_npe_contract.load_contract); y and
target_network_snr from the G3 trainio HDF5. tau30 = Newtonian tau of the true chirp mass
(m13_1_diagnostics.tau_newtonian), decile edges from val quantiles: same code as M13_8 cell 3.
Regimes short = D1-D2, medium = D5-D6, long = D9-D10; default_rng(131101) draws --n-per-regime
val rows per regime without replacement, in that order. --limit does not change the draw.

Reconstruction: scripts.generate_m11_final_G1.build_g1_row with the G3 physics and
g3.adjust_processing_windows, set up by generate_m13_6_clean.setup as in its control mode;
neither generator is modified. RowCapture swaps two g1 module globals (looked up at call time)
during one row: PROCESSOR (proxy that records the coloured s + n and the PSDs handed to
process_network; psd_proc is also the PSD of the Gaussian noise) and
build_signal_network_for_params (records the final-distance projection s, its parameters and
geocentric time). Seeds: placement_seed and gaussian_seed of the manifest, consumed inside
build_g1_row exactly as in G3; the PSD is re-estimated from the cached GWOSC strain of the block.

Gates:
  G-REC  zscore(PROCESSOR.process_network(s + n, psd_proc)) vs G3 trainio X[row]:
         max|dX| <= 1e-5 in every row, otherwise abort (gates.json written, no .npz).
  G-MM   per detector, pycbc.filter.match of s (SEOBNRv4_opt) and the same projection with
         SEOBNRv4_ROM, both cropped to the final 4 s window, psd_proc interpolated to 1/4 Hz,
         30-1024 Hz. Mismatch <= 0.003 evaluated and reported per regime; never aborts.

Output (exclusive, UTC stamp): <data_root>/results/M13_3/M13_11/rows/<stamp>/
  selection.json, gates.json and, unless --only-gates, row_<index>.npz.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import resolve_processed_artifact
import scripts.generate_m11_final_G1 as g1
from scripts.generate_m11_final_G1 import (DETECTORS, LABELS, read_psd_intervals, require,
                                           resolve_cached_psd_files, sha256_file)
import scripts.m13.generate_m13_G3_whitening as g3
import scripts.m13.generate_m13_6_clean as g6
from scripts.m12.g1_npe_contract import load_contract
from scripts.m12.m12_5_3_posthoc_check import npz_to_physical
from scripts.m13.m13_1_diagnostics import tau_newtonian

DEFAULT_CONFIG = PROJECT_ROOT / "configs/generation/generate_m13_G3_band1024_w16.json"
SELECTION_SEED = 131101
N_PER_REGIME = 60
# tau30 decile indices (0 = D1) of each regime, in draw order.
REGIMES = {"short": (0, 1), "medium": (4, 5), "long": (8, 9)}
GREC_MAX = 1e-5
MISMATCH_MAX = 0.003
BAND_HZ = (30.0, 1024.0)
ROM_APPROXIMANT = "SEOBNRv4_ROM"
# Not passed by src/waveform.py, so G3 used the pycbc defaults (pycbc/waveform/parameters.py).
PYCBC_DEFAULTS = {"coa_phase": 0.0, "f_ref": 0.0}
# (key, results tag) as in MODEL_SPECS of M13_8; predictions on val, no new inference.
MODEL_TAGS = {"4b_aug_s123": "M13_4b_G3aug_s123",
              "4a_aug_s123": "M13_4a_G3aug_s123",
              "4a_aug_s124": "M13_4a_G3aug_s124"}
VAL_NPZ = "val_predictions_embeddings.npz"
# y_val of the npz vs G3 trainio y (physical); rtol absorbs an inverse standardisation in float32.
Y_RTOL, Y_ATOL = 1e-5, 1e-6
PARAM_COLUMNS = ("mass_1", "mass_2", "spin_1z", "spin_2z", "inclination", "ra", "dec",
                 "polarization_angle", "target_network_snr")
# --diag-mm (diagnostic of G-MM; does not change the gate).
DIAG_FLOW_TD = (30.0, 32.0, 35.0, 40.0)
DIAG_FLOW_FD = (30.0, 35.0)
DIAG_SNR_CUTS_HZ = (32.0, 35.0, 40.0)
DIAG_TUKEY_S = 0.1
# --diag-intrinsic: face-on, unit-distance polarisations, no detector, analytic PSD.
INTRINSIC_APPROXIMANTS = ("SEOBNRv4_opt", "SEOBNRv4_ROM", "SEOBNRv4")
INTRINSIC_DELTA_T = 1.0 / 4096
INTRINSIC_F_LOWER = 30.0
# Below the 30 Hz match band, so no PSD bin inside 30-1024 Hz is zeroed by the analytic cutoff.
INTRINSIC_PSD_F_LOW = 20.0
# --diag-click: S = SEOBNRv4_opt from 20 Hz, half-Kaiser roll-up (td_taper) from its start to the
# first sample where the instantaneous frequency reaches 28 Hz, i.e. no step at the start.
CLICK_F_LOWER = 20.0
CLICK_TAPER_END_HZ = 28.0
CLICK_BETA = 8
# A is generated with f_lower 30 Hz and the pycbc default f_ref = 0, which LAL takes as f_ref = f_lower.
# S starts at 20 Hz, so without an explicit f_ref its phase would be referenced at 20 Hz and the
# difference A - S would contain a constant phase offset that is not the start step under test.
CLICK_F_REF = 30.0
# Coherence flag: with the phase reference matched, frac_network above this suggests a residual phase
# (or time) offset rather than a start step; flagged as PHASE?, never aborts.
CLICK_PHASE_FLAG_FRAC = 0.2
# Explicit alignment of S (and E) with A, face-on, on the common segment after the taper.
CLICK_ALIGN_MAX_LAG = 32         # samples searched around the epoch alignment (t = 0 at the peak)
CLICK_ALIGN_PHASE_TOL = 0.01     # rad, residual phase after regenerating with coa_phase += sign * delta / 2
CLICK_ALIGNED_GATE_RTOL = 0.2    # |SNR_click_aligned - SNR_click_phasemax| / SNR_click_phasemax (network)
E_F_LOWER = 10.0                 # E: SEOBNRv4_opt from 10 Hz, no taper (notebook M13_11_click_visualization)


class _ProcessorTap:
    """Stand-in for g1.PROCESSOR: records the inputs of process_network, then delegates."""

    def __init__(self, inner, cap):
        self._inner = inner
        self._cap = cap

    def process_network(self, strains, psds=None):
        require(self._cap.strains is None, "process_network called twice in one row")
        # Copies: the recorded s + n must not alias anything the processor touches.
        self._cap.strains = {ifo: strains[ifo].copy() for ifo in DETECTORS}
        self._cap.psds = psds
        return self._inner.process_network(strains=strains, psds=psds)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _TdArgsTap:
    """Record the kwargs of every src.waveform.get_td_waveform call (module global) while active."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        import src.waveform as waveform_module
        self._module, self._fn = waveform_module, waveform_module.get_td_waveform

        def tap(*args, **kwargs):
            self.calls.append(dict(kwargs))
            return self._fn(*args, **kwargs)

        waveform_module.get_td_waveform = tap
        return self

    def __exit__(self, *exc):
        self._module.get_td_waveform = self._fn
        return False


class RowCapture:
    """Swap g1.PROCESSOR and g1.build_signal_network_for_params during one build_g1_row call."""

    def __init__(self):
        self.strains = self.psds = self.signal = self.params = self.geocentric_time = None

    def __enter__(self):
        self._processor = g1.PROCESSOR
        self._signal_fn = g1.build_signal_network_for_params
        g1.PROCESSOR = _ProcessorTap(self._processor, self)
        g1.build_signal_network_for_params = self._signal
        return self

    def __exit__(self, *exc):
        g1.PROCESSOR = self._processor
        g1.build_signal_network_for_params = self._signal_fn
        return False

    def _signal(self, params, *, geocentric_time, final_start, injector):
        require(self.signal is None, "build_signal_network_for_params called twice in one row")
        out = self._signal_fn(params, geocentric_time=geocentric_time, final_start=final_start,
                              injector=injector)
        self.signal, self.params, self.geocentric_time = out, params, float(geocentric_time)
        return out


def select_rows(root, trainio, n_per_regime):
    _, splits, c_mean, c_std, _ = load_contract(root)
    idx_val = splits["val"]
    with h5py.File(trainio, "r") as h:
        y_val = h["y"][:][idx_val].astype(np.float64)
        snr_val = h["target_network_snr"][:][idx_val].astype(np.float64)
    # M13_8 cell 3: Newtonian tau30 of the TRUE Mc, val quantile edges.
    tau30 = tau_newtonian(y_val[:, 0])
    tau_edges = np.quantile(tau30, np.linspace(0, 1, 11))
    tdec = np.clip(np.searchsorted(tau_edges[1:-1], tau30, side="right"), 0, 9)
    rng = np.random.default_rng(SELECTION_SEED)
    chosen = {}
    for name, deciles in REGIMES.items():
        pool = np.flatnonzero(np.isin(tdec, deciles))
        require(len(pool) >= n_per_regime, f"Regime {name}: only {len(pool)} val rows")
        positions = np.sort(rng.choice(pool, n_per_regime, replace=False))
        chosen[name] = [dict(row_index=int(idx_val[p]), regime=name, tau30_s=float(tau30[p]),
                             tau30_decile=f"D{int(tdec[p]) + 1}", target_network_snr=float(snr_val[p]),
                             y=dict(zip(LABELS, map(float, y_val[p]))))
                        for p in positions]
    return chosen, tau_edges, idx_val, y_val, c_mean, c_std


def load_predictions(root, trainio, idx_val, y_val, c_mean, c_std):
    preds, prov = {}, {}
    for key, tag in MODEL_TAGS.items():
        cfg = json.loads((PROJECT_ROOT / "configs/experiments/m13" / f"train_m{tag[1:]}.json").read_text())
        ds = cfg["dataset"]
        h5 = resolve_processed_artifact(data_root=root, dataset_id=ds["dataset_id"], file_name=ds["dataset_file"])
        require(Path(h5).resolve() == Path(trainio).resolve(), f"{key}: model dataset is not {trainio}")
        path = root / "results" / "M13_3" / tag / VAL_NPZ
        require(path.exists(), f"Missing val npz: {path}")
        # load_pred of M13_8 (cell 2): emb_val and object arrays are never read.
        with np.load(path, allow_pickle=False) as a:
            f, y = (a[k].astype(np.float64) for k in ("pred_val", "y_val"))
            idx = a["idx_val"].astype(np.int64)
            y_mean, y_std = a["y_mean"].astype(np.float64), a["y_std"].astype(np.float64)
            label_names = a["label_names"].tolist()
        require(label_names == list(LABELS), f"{key}: label order")
        require(np.allclose(y_mean, c_mean, rtol=1e-6) and np.allclose(y_std, c_std, rtol=1e-6, atol=1e-8),
                f"{key}: npz y_mean/y_std differ from the M11 train-only label stats")
        y, f, units = npz_to_physical(y, f, c_mean, c_std)
        o = np.argsort(idx)
        require(np.array_equal(idx[o], idx_val), f"{key}: idx_val differs from val membership")
        require(np.allclose(y[o], y_val, rtol=Y_RTOL, atol=Y_ATOL), f"{key}: y_val differs from G3 trainio y")
        preds[key] = f[o]
        prov[key] = dict(path=str(path), sha256=sha256_file(path), units=units)
    return preds, prov


def psd_at_final_resolution(psd):
    # psd_proc has delta_f = 1/processing_duration; the PE sees only the 4 s window.
    out = g1.interpolate(psd, float(g1.CONFIG.delta_f))
    require(len(out) >= g1.CONFIG.flength, f"Interpolated PSD too short: {len(out)}")
    out = out[:g1.CONFIG.flength]
    require(np.all(np.isfinite(out.numpy())), "Non-finite interpolated PSD")
    return out


def _mismatch(a, b, psd, f_low):
    from pycbc.filter import match
    return 1.0 - float(match(a, b, psd=psd, low_frequency_cutoff=f_low, high_frequency_cutoff=BAND_HZ[1])[0])


def _taper_start(series, seconds):
    """Rising half of a Tukey window over `seconds` from the first non-zero sample; the end is untouched."""
    from pycbc.types import TimeSeries
    x = np.array(series.numpy(), dtype=np.float64)
    nz = np.flatnonzero(x)
    if len(nz):
        k = int(round(seconds / float(series.delta_t)))
        i = int(nz[0])
        m = min(k, len(x) - i)
        x[i:i + m] *= 0.5 * (1.0 - np.cos(np.pi * np.arange(m) / k))
    return TimeSeries(x, delta_t=series.delta_t, epoch=series.start_time)


def _rom_fd_projection(params, geocentric_time, final_start):
    """SEOBNRv4_ROM from get_fd_waveform, projected in frequency (antenna pattern and delay at t_geo).

    Same waveform arguments as src/waveform.py, delta_f = 1/4 Hz. Unlike DetectorProjector
    (lal project_wave, time-dependent response), the antenna pattern is frozen at the geocentric
    time. The phase factor places the merger at its detector time within the final window; match
    maximises over cyclic time shifts anyway. At delta_f = 1/4 Hz the waveform is cyclic over 4 s
    and not truncated to the window, whereas the opt signal is.
    """
    from pycbc.types import FrequencySeries
    from pycbc.waveform import get_fd_waveform
    hp, hc = get_fd_waveform(mass1=params.mass_1, mass2=params.mass_2, distance=params.distance,
                             inclination=params.inclination, spin1z=params.spin_1z, spin2z=params.spin_2z,
                             f_lower=g1.CONFIG.low_frequency_cutoff, delta_f=g1.CONFIG.delta_f,
                             approximant=ROM_APPROXIMANT)
    require(abs(float(hp.delta_f) - g1.CONFIG.delta_f) < 1e-12, "ROM FD delta_f differs from 1/4 Hz")
    hp.resize(g1.CONFIG.flength)
    hc.resize(g1.CONFIG.flength)
    f = hp.sample_frequencies.numpy()
    out = {}
    for ifo in DETECTORS:
        det = g1.DETECTOR_PROJECTOR.detectors[ifo]
        fp, fc = det.antenna_pattern(params.ra, params.dec, params.polarization_angle, geocentric_time)
        t_det = geocentric_time + det.time_delay_from_earth_center(params.ra, params.dec, geocentric_time)
        h = (fp * hp.numpy() + fc * hc.numpy()) * np.exp(-2j * np.pi * f * (t_det - final_start))
        out[ifo] = FrequencySeries(h, delta_f=hp.delta_f, epoch=final_start)
    return out


def _snr2_fraction_below(s, psd):
    # Optimal SNR^2 density |h(f)|^2 / S(f) of the opt signal in the final window, 30-1024 Hz.
    h = s.to_frequencyseries()
    f = h.sample_frequencies.numpy()
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1])
    dens = np.zeros(len(f))
    dens[band] = np.abs(h.numpy()[band]) ** 2 / psd.numpy()[band]
    total = dens.sum()
    return {f"snr2_frac_below_{c:g}": float(dens[f < c].sum() / total) for c in DIAG_SNR_CUTS_HZ}


def diagnose_mm(cap, s_opt, s_rom, psd4, final_start):
    rom_fd = _rom_fd_projection(cap.params, cap.geocentric_time, final_start)
    out = {}
    for ifo in DETECTORS:
        v = {f"D1_rom_td_flow{f:g}": _mismatch(s_opt[ifo], s_rom[ifo], psd4[ifo], f) for f in DIAG_FLOW_TD}
        v["D2_opt_tukey_start_flow30"] = _mismatch(s_opt[ifo], _taper_start(s_opt[ifo], DIAG_TUKEY_S),
                                                   psd4[ifo], BAND_HZ[0])
        opt_fd = s_opt[ifo].to_frequencyseries()
        v.update({f"D3_rom_fd_flow{f:g}": _mismatch(opt_fd, rom_fd[ifo], psd4[ifo], f) for f in DIAG_FLOW_FD})
        v.update(_snr2_fraction_below(s_opt[ifo], psd4[ifo]))
        out[ifo] = v
    return out


def summarize_diag(diag_rows):
    variants = [k for k in diag_rows[0]["detectors"][DETECTORS[0]] if k.startswith("D")]
    out = {}
    for name in REGIMES:
        rs = [r for r in diag_rows if r["regime"] == name]
        if rs:
            out[name] = {}
            for k in variants:
                worst = [max(r["detectors"][ifo][k] for ifo in DETECTORS) for r in rs]
                out[name][k] = dict(n=len(rs), median=float(np.median(worst)), max=float(np.max(worst)))
    return out


def diagnose_intrinsic(row, opt_args, rom_args):
    from pycbc.filter import match
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.waveform import get_td_waveform
    keys = sorted(set(opt_args) | set(rom_args))
    args = {k: dict(opt=opt_args.get(k), rom=rom_args.get(k), differs=opt_args.get(k) != rom_args.get(k))
            for k in keys}
    common = dict(mass1=float(row["mass_1"]), mass2=float(row["mass_2"]), spin1z=float(row["spin_1z"]),
                  spin2z=float(row["spin_2z"]), inclination=0.0, coa_phase=0.0, distance=1.0,
                  delta_t=INTRINSIC_DELTA_T, f_lower=INTRINSIC_F_LOWER)
    waves, info = {}, {}
    for apx in INTRINSIC_APPROXIMANTS:
        try:
            hp, hc = get_td_waveform(approximant=apx, **common)
        except Exception as exc:
            if apx != "SEOBNRv4":
                raise
            info[apx] = dict(error=f"{type(exc).__name__}: {exc}")
            continue
        dt = float(hp.delta_t)
        h = hp.numpy() - 1j * hc.numpy()
        info[apx] = dict(n_samples=len(hp), duration_s=len(hp) * dt, start_time_s=float(hp.start_time),
                         peak_minus_end_s=(int(np.argmax(np.abs(h))) - len(hp)) * dt)
        waves[apx] = (hp, hc)
    # Zero-pad at the end to a common power of 2 holding every complete waveform.
    n_fft = int(2 ** np.ceil(np.log2(max(len(hp) for hp, _ in waves.values()))))
    delta_f = 1.0 / (n_fft * INTRINSIC_DELTA_T)
    psd = aLIGOZeroDetHighPower(n_fft // 2 + 1, delta_f, INTRINSIC_PSD_F_LOW)

    def pad(ts):
        out = ts.copy()
        out.resize(n_fft)
        return out

    matches = {}
    ref = waves[INTRINSIC_APPROXIMANTS[0]]
    for apx in INTRINSIC_APPROXIMANTS[1:]:
        if apx in waves:
            for i, pol in enumerate(("hp", "hc")):
                matches[f"opt_vs_{apx}_{pol}"] = float(match(
                    pad(ref[i]), pad(waves[apx][i]), psd=psd, low_frequency_cutoff=BAND_HZ[0],
                    high_frequency_cutoff=BAND_HZ[1])[0])
    return dict(args=args, differing_keys=[k for k in keys if args[k]["differs"]], common_args=common,
                n_fft=n_fft, delta_f=delta_f, waveforms=info, match=matches)


def _opt_click_free(kwargs):
    """S: kwargs with f_lower = 20 Hz, both polarisations rolled up from the start to the 28 Hz crossing."""
    from pycbc.waveform import get_td_waveform
    from pycbc.waveform.utils import frequency_from_polarizations, td_taper
    s_kwargs = {**kwargs, "f_lower": CLICK_F_LOWER, "f_ref": CLICK_F_REF}  # see CLICK_F_REF
    hp, hc = get_td_waveform(**s_kwargs)
    # The phase of frequency_from_polarizations is arctan2(hx, h+), with hx ~ cos(inclination): it runs
    # backwards for inclination > pi/2 and is ill-defined near pi/2. SEOBNRv4_opt is (2,2)-only, so the
    # phase evolution does not depend on inclination; the 28 Hz instant is taken from a face-on copy.
    if float(kwargs.get("inclination", 0.0)) == 0.0:
        fp, fc = hp, hc
    else:
        fp, fc = get_td_waveform(**{**s_kwargs, "inclination": 0.0})
        require(len(fp) == len(hp) and float(fp.start_time) == float(hp.start_time),
                "Face-on copy of S is on a different time grid")
    f = frequency_from_polarizations(fp, fc)
    above = np.flatnonzero(f.numpy() >= CLICK_TAPER_END_HZ)
    require(len(above) > 0, f"S never reaches {CLICK_TAPER_END_HZ:g} Hz")
    t0, t_end = float(hp.start_time), float(f.sample_times[above[0]])
    hp = td_taper(hp, t0, t_end, beta=CLICK_BETA)
    hc = td_taper(hc, t0, t_end, beta=CLICK_BETA)
    return hp, hc, t_end - t0


def _advance(ts, lag, epoch):
    """ts advanced by `lag` (fractional) samples, y[m] = x[m + lag], via FFT of a zero-padded copy."""
    from pycbc.types import TimeSeries
    x = ts.numpy()
    n2 = int(2 ** np.ceil(np.log2(2 * len(x))))
    k = np.fft.rfftfreq(n2)  # cycles per sample
    y = np.fft.irfft(np.fft.rfft(x, n2) * np.exp(2j * np.pi * k * lag), n2)[:len(x)]
    return TimeSeries(y, delta_t=ts.delta_t, epoch=epoch)


def _align_to_A(kwargs, make):
    """Waveform of make(kw) -> (hp, hc, t_valid_s) aligned in phase and sub-sample time with A.

    A = get_td_waveform(**kwargs). Both are generated face-on (inclination 0); on the common segment
    from t_valid_s after the start of make's waveform (end of the taper) to the end of A, with
    h = h+ - i hx:
      - lag: maximum of |sum h_A conj(h_X(lag))| over integer lags around the epoch alignment, refined by
        parabolic interpolation; X is advanced by lag samples via FFT and its epoch put on A's sub-sample
        grid (start_A - k0 dt), so A and X differ by an integer number of samples in any later rounding;
      - delta = arg sum h_A conj(h_X); X is regenerated with coa_phase += sign * delta / 2 ((2,2) mode),
        the sign being the one that leaves a residual phase < CLICK_ALIGN_PHASE_TOL (both are tried).
    Returns X with the real kwargs (real inclination) shifted by the same lag, plus the diagnostics.
    """
    from pycbc.waveform import get_td_waveform
    dt = float(kwargs["delta_t"])
    face = {**kwargs, "inclination": 0.0}
    a_hp, a_hc = get_td_waveform(**face)
    a = a_hp.numpy() - 1j * a_hc.numpy()
    start_a = float(a_hp.start_time)
    coa0 = float(kwargs.get("coa_phase", 0.0))
    x_hp, x_hc, t_valid = make(face)
    start_x = float(x_hp.start_time)
    offset = (start_a - start_x) / dt
    k0 = int(round(offset))
    r0 = offset - k0
    L = CLICK_ALIGN_MAX_LAG
    require(k0 >= L, "Aligned waveform does not start before A")
    j_lo = max(0, int(np.ceil((start_x + t_valid - start_a) / dt)))
    j_hi = min(len(a), len(x_hp) - k0 - L)
    require(j_hi - j_lo > 2 * L, "No common segment after the taper")
    a_seg = a[j_lo:j_hi]
    x = x_hp.numpy() - 1j * x_hc.numpy()
    c = np.array([abs(np.vdot(x[j_lo + k0 + l:j_hi + k0 + l], a_seg)) for l in range(-L, L + 1)])
    i = int(np.argmax(c))
    require(0 < i < 2 * L, "Cross-correlation peak at the edge of the lag search")
    lag = (i - L) + 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2.0 * c[i] + c[i + 1])
    epoch = start_x + r0 * dt  # = start_a - k0 dt

    def aligned(hp, hc):
        require(float(hp.start_time) == start_x and len(hp) == len(x_hp), "Regenerated waveform on another grid")
        return _advance(hp, lag, epoch), _advance(hc, lag, epoch)

    def phase(hp, hc):
        h = hp.numpy() - 1j * hc.numpy()
        return float(np.angle(np.vdot(h[j_lo + k0:j_hi + k0], a_seg)))

    delta = phase(*aligned(x_hp, x_hc))
    best = None
    for sign in (1.0, -1.0):
        hp_s, hc_s, _ = make({**face, "coa_phase": coa0 + sign * delta / 2})
        residual = phase(*aligned(hp_s, hc_s))
        if best is None or abs(residual) < abs(best[1]):
            best = (sign, residual)
        if abs(residual) < CLICK_ALIGN_PHASE_TOL:
            break
    sign, residual = best
    hp, hc, _ = make({**kwargs, "coa_phase": coa0 + sign * delta / 2})
    hp, hc = aligned(hp, hc)
    return hp, hc, dict(delta_rad=delta, sign=sign, coa_phase_shift_rad=sign * delta / 2,
                        residual_rad=residual, residual_ok=bool(abs(residual) < CLICK_ALIGN_PHASE_TOL),
                        lag_samples=float(lag), time_shift_s=float((lag - r0) * dt),
                        common_segment_s=(j_hi - j_lo) * dt)


def _opt_click_free_aligned(kwargs):
    """S (f_lower 20 Hz + taper to 28 Hz) aligned with A = get_td_waveform(**kwargs); see _align_to_A."""
    return _align_to_A(kwargs, _opt_click_free)


def opt_extended_aligned(kwargs, f_lower=E_F_LOWER):
    """E: kwargs from f_lower (no taper), aligned with A as S is."""
    from pycbc.waveform import get_td_waveform

    def make(kw):
        hp, hc = get_td_waveform(**{**kw, "f_lower": f_lower})
        return hp, hc, 0.0

    return _align_to_A(kwargs, make)


def _sigmasq_diff(x, y, psd):
    from pycbc.filter import sigmasq
    from pycbc.types import TimeSeries
    d = TimeSeries(x.numpy() - y.numpy(), delta_t=x.delta_t, epoch=x.start_time)
    return float(sigmasq(d, psd=psd, low_frequency_cutoff=BAND_HZ[0], high_frequency_cutoff=BAND_HZ[1]))


def diagnose_click(row, opt_args, cap, s_opt, psd4, crop):
    from pycbc.filter import match, overlap_cplx, sigmasq
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import TimeSeries
    from pycbc.waveform import get_td_waveform

    # 1. Intrinsic: face-on, unit distance, coa_phase 0.
    common = dict(mass1=float(row["mass_1"]), mass2=float(row["mass_2"]), spin1z=float(row["spin_1z"]),
                  spin2z=float(row["spin_2z"]), inclination=0.0, coa_phase=0.0, distance=1.0,
                  delta_t=INTRINSIC_DELTA_T)
    opt = g1.CONFIG.waveform_approximant
    waves = {"A": get_td_waveform(approximant=opt, f_lower=INTRINSIC_F_LOWER, **common)}
    s_hp, s_hc, taper_s = _opt_click_free(dict(approximant=opt, **common))
    waves["S"] = (s_hp, s_hc)
    waves["R"] = get_td_waveform(approximant=ROM_APPROXIMANT, f_lower=INTRINSIC_F_LOWER, **common)
    # Each waveform keeps its own epoch (t = 0 at the peak for both opt series); they are placed on a common
    # grid starting at the earliest epoch, so A and S overlap in time; zeros fill the rest up to 2^k.
    dt = INTRINSIC_DELTA_T
    t_ref = min(float(hp.start_time) for hp, _ in waves.values())
    offsets = {k: (float(hp.start_time) - t_ref) / dt for k, (hp, _) in waves.items()}
    shift = {k: int(round(v)) for k, v in offsets.items()}
    n_fft = int(2 ** np.ceil(np.log2(max(shift[k] + len(hp) for k, (hp, _) in waves.items()))))

    def place(ts, k):
        x = np.zeros(n_fft)
        x[shift[k]:shift[k] + len(ts)] = ts.numpy()
        return TimeSeries(x, delta_t=dt, epoch=t_ref)

    placed = {k: (place(hp, k), place(hc, k)) for k, (hp, hc) in waves.items()}
    psd = aLIGOZeroDetHighPower(n_fft // 2 + 1, 1.0 / (n_fft * dt), INTRINSIC_PSD_F_LOW)
    matches = {}
    for a, b in (("A", "S"), ("S", "R"), ("A", "R")):
        for i, pol in enumerate(("hp", "hc")):
            matches[f"{a}_vs_{b}_{pol}"] = float(match(placed[a][i], placed[b][i], psd=psd,
                                                       low_frequency_cutoff=BAND_HZ[0],
                                                       high_frequency_cutoff=BAND_HZ[1])[0])
    info = {}
    for k, (hp, hc) in waves.items():
        h = np.abs(hp.numpy() - 1j * hc.numpy())
        info[k] = dict(start_time_s=float(hp.start_time), duration_s=len(hp) * dt,
                       peak_time_s=float(hp.start_time) + int(np.argmax(h)) * dt,
                       subsample_offset=float(offsets[k] - shift[k]),
                       # |x(t0)| / max|x|: size of the step at the first sample.
                       step_hp=float(abs(hp[0]) / np.max(np.abs(hp.numpy()))),
                       step_hc=float(abs(hc[0]) / np.max(np.abs(hc.numpy()))))
    intrinsic = dict(waveforms=info, match=matches, n_fft=n_fft, s_taper_duration_s=taper_s)

    # 2. In data: S with the injected signal's kwargs (f_lower 20 Hz + taper), same projection, t_c and window.
    d_hp, d_hc, d_taper_s = _opt_click_free(opt_args)
    proj = g1.DETECTOR_PROJECTOR.project(h_plus=d_hp, h_cross=d_hc, parameters=cap.params,
                                         geocentric_coalescence_time=cap.geocentric_time)
    # S aligned with the injected A (phase and sub-sample time, face-on fit), same projection path.
    al_hp, al_hc, align = _opt_click_free_aligned(opt_args)
    proj_al = g1.DETECTOR_PROJECTOR.project(h_plus=al_hp, h_cross=al_hc, parameters=cap.params,
                                            geocentric_coalescence_time=cap.geocentric_time)
    lo, hi = BAND_HZ
    det, num, den, num_pm, num_al = {}, 0.0, 0.0, 0.0, 0.0
    for ifo in DETECTORS:
        s_s = crop(proj.strains[ifo])
        dd = _sigmasq_diff(s_opt[ifo], s_s, psd4[ifo])
        dd_al = _sigmasq_diff(s_opt[ifo], crop(proj_al.strains[ifo]), psd4[ifo])
        num_al += dd_al
        ss = float(sigmasq(s_opt[ifo], psd=psd4[ifo], low_frequency_cutoff=lo, high_frequency_cutoff=hi))
        # Control: phase-only maximisation (normalised complex overlap at zero time shift); assumes
        # <S,S> ~ <s,s>, so <D,D>_min ~ 2 <s,s> (1 - m).
        m = float(abs(overlap_cplx(s_opt[ifo], s_s, psd=psd4[ifo], low_frequency_cutoff=lo,
                                   high_frequency_cutoff=hi, normalized=True)))
        dd_pm = 2.0 * ss * max(1.0 - m, 0.0)
        det[ifo] = dict(snr_click=float(np.sqrt(dd)), frac=dd / ss, match_phase_only=m,
                        snr_click_phasemax=float(np.sqrt(dd_pm)), snr_click_aligned=float(np.sqrt(dd_al)))
        num, den, num_pm = num + dd, den + ss, num_pm + dd_pm
    frac_network = num / den
    snr_pm, snr_al = float(np.sqrt(num_pm)), float(np.sqrt(num_al))
    rel = abs(snr_al - snr_pm) / snr_pm if snr_pm > 0 else float("inf")
    data = dict(detectors=det, snr_click_network=float(np.sqrt(num)), frac_network=frac_network,
                snr_click_phasemax_network=snr_pm, snr_click_aligned_network=snr_al,
                aligned_rel_diff=rel, aligned_gate_ok=bool(rel <= CLICK_ALIGNED_GATE_RTOL), align=align,
                phase_flag=bool(frac_network > CLICK_PHASE_FLAG_FRAC),
                target_network_snr=float(row["target_network_snr"]), s_taper_duration_s=d_taper_s)
    return dict(intrinsic=intrinsic, data=data)


def print_click(index, regime, d):
    i, m = d["intrinsic"], d["intrinsic"]["match"]
    print(f"diag_click row {index} ({regime}): N = {i['n_fft']}, S taper {i['s_taper_duration_s']:.3f} s; "
          + ", ".join(f"{k} {v:.6f}" for k, v in m.items()), flush=True)
    for k, w in i["waveforms"].items():
        print(f"  {k} start {w['start_time_s']:+.4f} s, peak {w['peak_time_s']:+.5f} s, "
              f"sub-sample offset {w['subsample_offset']:+.3f}, step hp {w['step_hp']:.3e}, "
              f"hc {w['step_hc']:.3e}", flush=True)
    for ifo, v in d["data"]["detectors"].items():
        print(f"  {ifo} SNR_click {v['snr_click']:.4f}, frac {v['frac']:.3e}, phase-only match "
              f"{v['match_phase_only']:.6f}, SNR_click_phasemax {v['snr_click_phasemax']:.4f}, "
              f"SNR_click_aligned {v['snr_click_aligned']:.4f}", flush=True)
    x = d["data"]
    a = x["align"]
    print(f"  align: delta {a['delta_rad']:+.4f} rad, sign {a['sign']:+.0f}, residual {a['residual_rad']:+.2e} rad"
          f"{'' if a['residual_ok'] else ' (RESIDUAL > tol)'}, lag {a['lag_samples']:+.3f} samples, "
          f"dt {a['time_shift_s'] * 1e3:+.4f} ms", flush=True)
    print(f"  network SNR_click {x['snr_click_network']:.4f}, SNR_click_phasemax "
          f"{x['snr_click_phasemax_network']:.4f}, SNR_click_aligned {x['snr_click_aligned_network']:.4f} "
          f"(rel diff {x['aligned_rel_diff']:.3f}{'' if x['aligned_gate_ok'] else ' ALIGNED-GATE FAIL'}), "
          f"frac {x['frac_network']:.3e}, target SNR {x['target_network_snr']:.2f}"
          + ("  PHASE?" if x["phase_flag"] else ""), flush=True)


def summarize_click(click_rows):
    out = {}
    for name in REGIMES:
        rs = [r for r in click_rows if r["regime"] == name]
        if rs:
            out[name] = {key: dict(n=len(rs), median=float(np.median(x)), p90=float(np.quantile(x, 0.9)),
                                   max=float(np.max(x)))
                         for key in ("snr_click_network", "snr_click_phasemax_network",
                                     "snr_click_aligned_network", "frac_network", "target_network_snr")
                         for x in [np.array([r["data"][key] for r in rs])]}
            flagged = [r["row_index"] for r in rs if r["data"]["phase_flag"]]
            out[name]["phase_flag"] = dict(threshold_frac_network=CLICK_PHASE_FLAG_FRAC, n=len(flagged),
                                           rows=flagged)
            failed = [r["row_index"] for r in rs if not r["data"]["aligned_gate_ok"]]
            out[name]["aligned_gate"] = dict(rtol=CLICK_ALIGNED_GATE_RTOL, n_fail=len(failed), rows=failed)
            bad = [r["row_index"] for r in rs if not r["data"]["align"]["residual_ok"]]
            out[name]["align_residual"] = dict(tol_rad=CLICK_ALIGN_PHASE_TOL, n_fail=len(bad), rows=bad)
    return out


def print_intrinsic(index, d):
    print(f"diag_intrinsic row {index}: get_td_waveform kwargs (opt signal | ROM of G-MM; * = differs)", flush=True)
    for k, v in d["args"].items():
        print(f"  {'*' if v['differs'] else ' '} {k:<12} {v['opt']!r:<28} {v['rom']!r}", flush=True)
    for apx, w in d["waveforms"].items():
        text = (f"FAILED {w['error']}" if "error" in w else
                f"duration {w['duration_s']:.4f} s, peak(|h+ - i hx|) - end {w['peak_minus_end_s']:+.4f} s")
        print(f"  {apx:<13} {text}", flush=True)
    print(f"  match (N = {d['n_fft']}, aLIGOZeroDetHighPower, {BAND_HZ[0]:g}-{BAND_HZ[1]:g} Hz): "
          + ", ".join(f"{k} {v:.6f}" for k, v in d["match"].items()), flush=True)


def capture_row(row, *, long_strains, psd_cache):
    """build_g1_row under RowCapture and _TdArgsTap: (X_row, meta, cap, get_td_waveform kwargs of the signal)."""
    tap = _TdArgsTap()
    with tap, RowCapture() as cap:
        X_row, meta = g1.build_g1_row(row, long_strains=long_strains, psd_cache=psd_cache)
    require(cap.strains is not None and cap.signal is not None, f"Row {row.name}: capture incomplete")
    # build_g1_row generates the reference-distance waveform, then the final-distance signal.
    opt_args = tap.calls[-1]
    require(opt_args.get("distance") == cap.params.distance and
            opt_args.get("approximant") == g1.CONFIG.waveform_approximant,
            f"Row {row.name}: last get_td_waveform call is not the injected signal")
    return X_row, meta, cap, opt_args


def final_start_of(row):
    # row of g3.adjust_processing_windows; same expression as build_g1_row.
    return float(row["processing_start"]) + g1.CONFIG.processing_context_start_seconds


def crop_to_final(series, final_start, injector):
    """Projected signal cropped to the final 4 s window, with the generator's injection rounding."""
    zero = injector.build_zero_strain(start_time=final_start, length=g1.CONFIG.length)
    return injector.inject(zero, series).strain


def process_clean(strains, *, processing_start, psds, injector):
    """L(s) as in M13.6: noiseless projection in zero processing-length strain, G3 processor, no z-score."""
    zeros = {ifo: injector.build_zero_strain(start_time=processing_start, length=g1.CONFIG.processing_length)
             for ifo in DETECTORS}
    injected = {ifo: injector.inject(zeros[ifo], strains[ifo]).strain for ifo in DETECTORS}
    return g6.stack_raw(g1.PROCESSOR.process_network(strains=injected, psds=psds))


def rebuild_row(row, index, *, long_strains, psd_cache, trainio_h, rom_generator, crop_injector, diag=False,
                diag_intrinsic=False, diag_click=False):
    X_row, meta, cap, opt_args = capture_row(row, long_strains=long_strains, psd_cache=psd_cache)
    require(np.array_equal(trainio_h["y"][index], row[list(LABELS)].to_numpy(dtype=np.float32)),
            f"Trainio y differs at row {index}")
    dt, n = g1.CONFIG.delta_t, g1.CONFIG.length
    i0 = g1.CONFIG.processing_context_start_samples
    final_start = final_start_of(row)
    for ifo in DETECTORS:
        require(abs(float(cap.strains[ifo].start_time) + i0 * dt - final_start) < g3.TIME_TOL_S,
                f"Row {index} {ifo}: final window misaligned")
    # Crop before reprocessing, so the stored window cannot be affected by process_network.
    data = np.stack([np.array(cap.strains[ifo].numpy()[i0:i0 + n], dtype=np.float64) for ifo in DETECTORS])

    # G-REC: reprocess the captured s + n with the G3 processor and the same psd_proc.
    _, X_rec = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(strains=cap.strains, psds=cap.psds))
    X_g3 = np.asarray(trainio_h["X"][index], dtype=np.float64)

    # G-MM: same parameters, geocentric time and projection; only the approximant changes.
    def crop(series):
        return crop_to_final(series, final_start, crop_injector)

    rom_tap = _TdArgsTap()
    with rom_tap:
        wf = rom_generator.generate(cap.params)
    proj_rom = g1.DETECTOR_PROJECTOR.project(h_plus=wf.h_plus, h_cross=wf.h_cross, parameters=cap.params,
                                             geocentric_coalescence_time=cap.geocentric_time)
    s_opt = {ifo: crop(cap.signal["projection"].strains[ifo]) for ifo in DETECTORS}
    s_rom = {ifo: crop(proj_rom.strains[ifo]) for ifo in DETECTORS}
    psd4 = {ifo: psd_at_final_resolution(cap.psds[ifo]) for ifo in DETECTORS}
    from pycbc.filter import match
    mm = {ifo: 1.0 - float(match(s_opt[ifo], s_rom[ifo], psd=psd4[ifo], low_frequency_cutoff=BAND_HZ[0],
                                 high_frequency_cutoff=BAND_HZ[1])[0])
          for ifo in DETECTORS}
    w = np.array([meta[f"snr_{ifo}"] ** 2 for ifo in DETECTORS])
    record = dict(
        row_index=int(index), block_id=str(row["block_id"]),
        max_abs_dX_reprocess=float(np.abs(X_rec - X_g3).max()),
        max_abs_dX_build=float(np.abs(np.asarray(X_row, dtype=np.float64) - X_g3).max()),
        mismatch=mm, mismatch_max_detector=max(mm.values()),
        # Optimal-SNR^2 weighted mean over detectors (SNRs of build_g1_row, 30-1024 Hz).
        mismatch_snr2_weighted=float(np.dot(w, [mm[ifo] for ifo in DETECTORS]) / w.sum()),
    )
    if diag:
        record["diag_mm"] = diagnose_mm(cap, s_opt, s_rom, psd4, final_start)
    if diag_intrinsic:
        record["diag_intrinsic"] = diagnose_intrinsic(row, opt_args, rom_tap.calls[-1])
    if diag_click:
        record["diag_click"] = diagnose_click(row, opt_args, cap, s_opt, psd4, crop)
    arrays = dict(
        epoch=np.full(len(DETECTORS), final_start), delta_t=float(dt),
        strain=data, signal=np.stack([s_opt[ifo].numpy() for ifo in DETECTORS]).astype(np.float64),
        psd=np.stack([psd4[ifo].numpy() for ifo in DETECTORS]).astype(np.float64),
        psd_delta_f=float(psd4[DETECTORS[0]].delta_f),
        param_distance=float(meta["final_distance"]), param_geocentric_time=float(meta["actual_geocentric_time"]),
        param_final_network_snr=float(meta["final_network_snr"]),
        **{f"param_{c}": float(row[c]) for c in PARAM_COLUMNS},
        **{f"param_{k}": v for k, v in PYCBC_DEFAULTS.items()},
        **{f"param_snr_{ifo}": float(meta[f"snr_{ifo}"]) for ifo in DETECTORS},
        **{f"gaussian_seed_{ifo}": int(meta[f"gaussian_seed_{ifo}"]) for ifo in DETECTORS},
        param_is_truncated=bool(meta["is_truncated"]),
    )
    return record, arrays


def summarize_mm(records, regime_of):
    out = {}
    for name in REGIMES:
        rs = [r for r in records if regime_of[r["row_index"]] == name]
        if not rs:
            continue
        worst = np.array([r["mismatch_max_detector"] for r in rs])
        net = np.array([r["mismatch_snr2_weighted"] for r in rs])
        out[name] = dict(
            n=len(rs), n_max_detector_le_threshold=int((worst <= MISMATCH_MAX).sum()),
            max_detector=dict(median=float(np.median(worst)), p90=float(np.quantile(worst, 0.9)), max=float(worst.max())),
            snr2_weighted=dict(median=float(np.median(net)), p90=float(np.quantile(net, 0.9)), max=float(net.max())),
            per_detector_median={ifo: float(np.median([r["mismatch"][ifo] for r in rs])) for ifo in DETECTORS},
        )
    return out


def write_json(path, payload):
    with path.open("x") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="G3 generation config")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Override G1 manifest_path; still verified against G1 manifest_sha256.")
    parser.add_argument("--n-per-regime", type=int, default=N_PER_REGIME)
    parser.add_argument("--only-gates", action="store_true", help="Evaluate the gates; write no .npz")
    parser.add_argument("--limit", type=int,
                        help="Rebuild only the first N selected rows, interleaved short/medium/long")
    parser.add_argument("--diag-mm", action="store_true",
                        help="G-MM diagnostic (f_low, start taper, FD ROM, low-frequency SNR^2); needs --only-gates")
    parser.add_argument("--diag-intrinsic", action="store_true",
                        help="Intrinsic opt/ROM/SEOBNRv4 comparison and get_td_waveform kwargs; needs --only-gates")
    parser.add_argument("--diag-click", action="store_true",
                        help="Start-step hypothesis: opt vs tapered opt from 20 Hz (intrinsic and in data); "
                             "needs --only-gates")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args()
    require(not args.diag_mm or args.only_gates, "--diag-mm requires --only-gates")
    require(not args.diag_intrinsic or args.only_gates, "--diag-intrinsic requires --only-gates")
    require(not args.diag_click or args.only_gates, "--diag-click requires --only-gates")
    require(args.n_per_regime >= 1, "Invalid --n-per-regime")
    require(args.limit is None or args.limit >= 1, "Invalid --limit")
    args.mode = "m13_11_prepare_rows"  # only recorded by g6 attrs_for

    root, spec_g1, df, trainio, _, attrs_for = g6.setup(args)
    chosen, tau_edges, idx_val, y_val, c_mean, c_std = select_rows(root, trainio, args.n_per_regime)
    preds, pred_prov = load_predictions(root, trainio, idx_val, y_val, c_mean, c_std)
    selected_rows = [r for name in REGIMES for r in chosen[name]]
    interleaved = [r for triple in zip(*(chosen[name] for name in REGIMES)) for r in triple]
    todo = interleaved[:args.limit] if args.limit else interleaved
    regime_of = {r["row_index"]: r["regime"] for r in selected_rows}
    longest = max(chosen["long"], key=lambda r: r["tau30_s"])

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / "results" / "M13_3" / "M13_11" / "rows" / stamp
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "selection.json", dict(
        selection_seed=SELECTION_SEED, n_per_regime=args.n_per_regime,
        regimes={k: [f"D{d + 1}" for d in v] for k, v in REGIMES.items()},
        tau30="tau_newtonian(true chirp_mass), f_ref 30 Hz; decile edges = val quantiles (M13_8 cell 3)",
        tau_edges_s=tau_edges.tolist(), n_val=int(len(idx_val)),
        timing_row_longest_long_regime=longest, rows=selected_rows))
    print(f"Selected {len(selected_rows)} rows; rebuilding {len(todo)} -> {out_dir}", flush=True)

    frame = g3.adjust_processing_windows(df.loc[[r["row_index"] for r in todo]], spec_g1)
    psd_files, raw_psd_files = resolve_cached_psd_files(root, frame)
    from src.waveform import WaveformGenerator
    # Same CONFIG (f_lower, delta_t) as G3; only the approximant differs.
    rom_generator = WaveformGenerator(dataclasses.replace(g1.CONFIG, waveform_approximant=ROM_APPROXIMANT))
    # inject/build_zero_strain never draw from rng; it is set only to make that explicit.
    crop_injector = g1.SignalInjector(config=g1.CONFIG, rng=np.random.default_rng(0))
    records, arrays, diag_rows, intrinsic_rows, click_rows = [], {}, [], [], []
    with h5py.File(trainio, "r") as t:
        require(t["X"].shape[0] == len(df), "G3 trainio row count differs from the manifest")
        for _, block_rows in frame.groupby("block_id", sort=True):
            long_strains = read_psd_intervals(block_rows.iloc[0], psd_files)
            psd_cache = {}
            for index, row in block_rows.iterrows():
                rec, arr = rebuild_row(row, index, long_strains=long_strains, psd_cache=psd_cache, trainio_h=t,
                                       rom_generator=rom_generator, crop_injector=crop_injector,
                                       diag=args.diag_mm, diag_intrinsic=args.diag_intrinsic,
                                       diag_click=args.diag_click)
                rec["regime"] = regime_of[index]
                if args.diag_click:
                    d = rec.pop("diag_click")
                    print_click(index, rec["regime"], d)
                    click_rows.append(dict(row_index=int(index), regime=rec["regime"], **d))
                if args.diag_intrinsic:
                    d = rec.pop("diag_intrinsic")
                    print_intrinsic(index, d)
                    intrinsic_rows.append(dict(row_index=int(index), regime=rec["regime"], **d))
                if args.diag_mm:
                    diag_rows.append(dict(row_index=int(index), regime=rec["regime"],
                                          detectors=rec.pop("diag_mm")))
                records.append(rec)
                arrays[index] = arr
                print(f"row {index} ({rec['regime']}): max|dX| {rec['max_abs_dX_reprocess']:.3e}, "
                      f"mismatch max-detector {rec['mismatch_max_detector']:.2e}", flush=True)

    worst = max(r["max_abs_dX_reprocess"] for r in records)
    grec_pass = worst <= GREC_MAX
    mm_summary = summarize_mm(records, regime_of)
    diag = None
    if args.diag_mm:
        cols = list(diag_rows[0]["detectors"][DETECTORS[0]])
        print("diag_mm (mismatch = 1 - match; snr2_frac = fraction of optimal SNR^2 of opt, 30-1024 Hz)", flush=True)
        print("row regime ifo " + " ".join(cols), flush=True)
        for r in diag_rows:
            for ifo in DETECTORS:
                print(f"{r['row_index']} {r['regime']} {ifo} "
                      + " ".join(f"{r['detectors'][ifo][c]:.3e}" for c in cols), flush=True)
        diag = dict(
            settings=dict(f_high_hz=BAND_HZ[1], window="final 4 s", psd="psd_proc at delta_f = 1/4 Hz",
                          D1="opt used vs ROM (TD, as G-MM) per f_low_match",
                          D2=f"opt used vs opt with a {DIAG_TUKEY_S:g} s Tukey ramp at its first non-zero sample",
                          D3="FFT of opt used vs get_fd_waveform ROM (delta_f 1/4 Hz, f_lower 30) projected in "
                             "frequency (antenna pattern at t_geo); ROM cyclic over 4 s and not truncated",
                          snr2_frac_below=list(DIAG_SNR_CUTS_HZ)),
            per_regime_worst_detector=summarize_diag(diag_rows), rows=diag_rows)
        for name, s in diag["per_regime_worst_detector"].items():
            print(f"diag_mm {name} worst detector (median/max): "
                  + "; ".join(f"{k} {v['median']:.2e}/{v['max']:.2e}" for k, v in s.items()), flush=True)
    provenance = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                  for k, v in attrs_for(len(todo), {"selection_policy": "m13_11_tau30_regimes",
                                                     "selection_seed": SELECTION_SEED}).items()}
    write_json(out_dir / "gates.json", dict(
        G_REC=dict(verdict="PASS" if grec_pass else "FAIL", max_abs_dX=worst, threshold=GREC_MAX,
                   n_fail=sum(r["max_abs_dX_reprocess"] > GREC_MAX for r in records),
                   comparison="zscore(PROCESSOR.process_network(captured s + n, psd_proc)) vs G3 trainio X"),
        G_MM=dict(threshold=MISMATCH_MAX, criterion="max over detectors of 1 - match <= threshold (evaluated, "
                  "not aborting)", all_rows_pass=all(r["mismatch_max_detector"] <= MISMATCH_MAX for r in records),
                  approximants=[g1.CONFIG.waveform_approximant, ROM_APPROXIMANT], band_hz=list(BAND_HZ),
                  window="final 4 s", psd="psd_proc interpolated to delta_f = 1/4 Hz", per_regime=mm_summary),
        n_rows=len(records), limit=args.limit, only_gates=bool(args.only_gates), pycbc_defaults=PYCBC_DEFAULTS,
        script_sha256=sha256_file(__file__), predictions=pred_prov, raw_psd_files=raw_psd_files,
        provenance=provenance, rows=records, diag_mm=diag,
        diag_intrinsic=(dict(settings=dict(approximants=list(INTRINSIC_APPROXIMANTS), delta_t=INTRINSIC_DELTA_T,
                                           f_lower=INTRINSIC_F_LOWER, inclination=0.0, coa_phase=0.0, distance=1.0,
                                           psd=f"aLIGOZeroDetHighPower(low_freq_cutoff={INTRINSIC_PSD_F_LOW:g})",
                                           band_hz=list(BAND_HZ), padding="zeros at the end to a common 2^k"),
                             rows=intrinsic_rows) if args.diag_intrinsic else None),
        diag_click=(dict(settings=dict(A=f"{g1.CONFIG.waveform_approximant}, f_lower {INTRINSIC_F_LOWER:g} Hz, no taper",
                                       S=f"{g1.CONFIG.waveform_approximant}, f_lower {CLICK_F_LOWER:g} Hz, "
                                         f"f_ref {CLICK_F_REF:g} Hz, td_taper "
                                         f"beta {CLICK_BETA} from start to first f(t) >= {CLICK_TAPER_END_HZ:g} Hz",
                                       R=f"{ROM_APPROXIMANT}, f_lower {INTRINSIC_F_LOWER:g} Hz",
                                       intrinsic="inclination 0, coa_phase 0, distance 1, aLIGOZeroDetHighPower, "
                                                 "placed by epoch on a common 2^k grid",
                                       data="Delta = injected projection - projection of S (same kwargs, params, "
                                            "t_c, final 4 s window); psd_proc at 1/4 Hz; no time/phase "
                                            "maximisation; frac_network = sum <D,D> / sum <s,s>",
                                       aligned="S aligned with A (face-on fit after the taper: sub-sample lag by "
                                               "parabolic cross-correlation peak, coa_phase += sign*delta/2); "
                                               f"gate |aligned - phasemax| / phasemax <= {CLICK_ALIGNED_GATE_RTOL:g}",
                                       band_hz=list(BAND_HZ)),
                         per_regime=summarize_click(click_rows), rows=click_rows) if args.diag_click else None)))
    if args.diag_click:
        for name, s in summarize_click(click_rows).items():
            flag, gate, res = s["phase_flag"], s["aligned_gate"], s["align_residual"]
            print(f"diag_click {name}: " + "; ".join(f"{k} median {v['median']:.3e}, p90 {v['p90']:.3e}, "
                                                     f"max {v['max']:.3e}" for k, v in s.items() if "median" in v)
                  + (f"; PHASE? {flag['n']} rows {flag['rows']}" if flag["n"] else "; PHASE? none")
                  + f"; aligned gate (rtol {gate['rtol']:g}) fail {gate['n_fail']} {gate['rows']}"
                  + f"; align residual > {res['tol_rad']:g} rad: {res['n_fail']} {res['rows']}", flush=True)
    print(f"G-REC {'PASS' if grec_pass else 'FAIL'}: max|dX| {worst:.3e} (threshold {GREC_MAX:g})", flush=True)
    for name, s in mm_summary.items():
        print(f"G-MM {name}: {s['n_max_detector_le_threshold']}/{s['n']} <= {MISMATCH_MAX:g}; "
              f"median {s['max_detector']['median']:.2e}, max {s['max_detector']['max']:.2e}", flush=True)
    if not grec_pass:
        raise SystemExit(f"G-REC failed: max|dX| {worst:.3e} > {GREC_MAX:g}; see {out_dir / 'gates.json'}")
    if args.only_gates:
        return

    pos = {int(i): p for p, i in enumerate(idx_val)}
    for index, arr in arrays.items():
        p = pos[int(index)]
        sel = next(r for r in selected_rows if r["row_index"] == index)
        model_preds = {f"pred_{k}": v[p] for k, v in preds.items()}
        with (out_dir / f"row_{int(index):06d}.npz").open("xb") as f:
            np.savez(f, row_index=int(index), regime=sel["regime"], tau30_s=sel["tau30_s"],
                     tau30_decile=sel["tau30_decile"], detectors=np.array(DETECTORS), label_names=np.array(LABELS),
                     y_true=y_val[p], approximant=g1.CONFIG.waveform_approximant,
                     f_lower=BAND_HZ[0], f_upper=BAND_HZ[1], **arr, **model_preds,
                     pred_4a_aug_mean=0.5 * (preds["4a_aug_s123"][p] + preds["4a_aug_s124"][p]))
    print(f"Wrote {len(arrays)} rows to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
