"""Señal sin clic S (banda de arranque baja o alta), su alineación con A y el taper de borde.

Copiados del proyecto de origen (ver docs/PROCEDENCIA.md). Allí la banda de S se fija con constantes
de módulo reasignadas en tiempo de llamada; aquí se pasa como argumento (f_lower, taper_end_hz). Los
valores y el orden de operaciones son los mismos.
"""
from __future__ import annotations

import numpy as np

from .steps import DETECTORS, require

# Copiado de scripts/m13/m13_11_prepare_rows.py (constantes) @ 4b34677; CLICK_F_LOWER y
# CLICK_TAPER_END_HZ pasan a ser argumentos (los fija band_for en gwsim/recipe.py).
CLICK_BETA = 8
# A is generated with f_lower 30 Hz and the pycbc default f_ref = 0, which LAL takes as f_ref = f_lower.
# S starts at a lower frequency, so without an explicit f_ref its phase would be referenced there and the
# difference A - S would contain a constant phase offset that is not the start step under test.
CLICK_F_REF = 30.0
CLICK_ALIGN_MAX_LAG = 32         # samples searched around the epoch alignment (t = 0 at the peak)
CLICK_ALIGN_PHASE_TOL = 0.01     # rad, residual phase after regenerating with coa_phase += sign * delta / 2

# Copiado de scripts/m13/m13_13_build_g3t.py (constantes) @ 4b34677; cambios: ninguno.
EDGE_TAPER_S = 0.5
EDGE_BETA = 8
# Copiado de scripts/m13/m13_13_build_g3t.py (constante ALIGN_ABORT, enmienda 3) @ 4b34677; cambios: ninguno.
ALIGN_ABORT = 0.5  # a code failure, not a small phase error


class AlignmentError(ValueError):
    """Fila descartada porque S no se alineó con A; su mensaje se guarda tal cual en la columna `error`."""


# Copiado de scripts/m13/m13_13_build_g3t.py:align_abort @ 4b34677; cambios: ninguno
def align_abort(residual):
    return not abs(residual) <= ALIGN_ABORT  # also catches NaN


# Copiado de scripts/m13/m13_11_prepare_rows.py:_opt_click_free @ 4b34677; cambios: CLICK_F_LOWER y
# CLICK_TAPER_END_HZ (constantes de módulo) -> argumentos f_lower y taper_end_hz.
def _opt_click_free(kwargs, f_lower, taper_end_hz):
    """S: kwargs with f_lower, both polarisations rolled up from the start to the taper_end_hz crossing."""
    from pycbc.waveform import get_td_waveform
    from pycbc.waveform.utils import frequency_from_polarizations, td_taper
    s_kwargs = {**kwargs, "f_lower": f_lower, "f_ref": CLICK_F_REF}  # see CLICK_F_REF
    hp, hc = get_td_waveform(**s_kwargs)
    # The phase of frequency_from_polarizations is arctan2(hx, h+), with hx ~ cos(inclination): it runs
    # backwards for inclination > pi/2 and is ill-defined near pi/2. SEOBNRv4_opt is (2,2)-only, so the
    # phase evolution does not depend on inclination; the crossing instant is taken from a face-on copy.
    if float(kwargs.get("inclination", 0.0)) == 0.0:
        fp, fc = hp, hc
    else:
        fp, fc = get_td_waveform(**{**s_kwargs, "inclination": 0.0})
        require(len(fp) == len(hp) and float(fp.start_time) == float(hp.start_time),
                "Face-on copy of S is on a different time grid")
    f = frequency_from_polarizations(fp, fc)
    above = np.flatnonzero(f.numpy() >= taper_end_hz)
    require(len(above) > 0, f"S never reaches {taper_end_hz:g} Hz")
    t0, t_end = float(hp.start_time), float(f.sample_times[above[0]])
    hp = td_taper(hp, t0, t_end, beta=CLICK_BETA)
    hc = td_taper(hc, t0, t_end, beta=CLICK_BETA)
    return hp, hc, t_end - t0


# Copiado de scripts/m13/m13_11_prepare_rows.py:_advance @ 4b34677; cambios: ninguno
def _advance(ts, lag, epoch):
    """ts advanced by `lag` (fractional) samples, y[m] = x[m + lag], via FFT of a zero-padded copy."""
    from pycbc.types import TimeSeries
    x = ts.numpy()
    n2 = int(2 ** np.ceil(np.log2(2 * len(x))))
    k = np.fft.rfftfreq(n2)  # cycles per sample
    y = np.fft.irfft(np.fft.rfft(x, n2) * np.exp(2j * np.pi * k * lag), n2)[:len(x)]
    return TimeSeries(y, delta_t=ts.delta_t, epoch=epoch)


# Copiado de scripts/m13/m13_11_prepare_rows.py:_align_to_A @ 4b34677; cambios: ninguno
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


# Copiado de scripts/m13/m13_11_prepare_rows.py:_opt_click_free_aligned @ 4b34677; cambios: recibe la
# banda (f_lower, taper_end_hz) y la pasa a _opt_click_free.
def opt_click_free_aligned(kwargs, f_lower, taper_end_hz):
    """S (f_lower + taper to taper_end_hz) aligned with A = get_td_waveform(**kwargs); see _align_to_A."""
    return _align_to_A(kwargs, lambda kw: _opt_click_free(kw, f_lower, taper_end_hz))


# Copiado de scripts/m13/m13_13_build_g3t.py:edge_taper @ 4b34677; cambios: ninguno
def edge_taper(strains, processing_start):
    """Edge taper: taper the signal from processing_start if it starts earlier; (strains, samples cut per ifo)."""
    from pycbc.types import TimeSeries
    out, n_before = {}, {}
    for ifo in DETECTORS:
        ts = strains[ifo]
        dt = float(ts.delta_t)
        # Same rounding as SignalInjector.inject (signal_start_index).
        n = -int(round((float(ts.start_time) - processing_start) / dt))
        n_before[ifo] = max(n, 0)
        if n <= 0:
            out[ifo] = ts
            continue
        k = int(round(EDGE_TAPER_S / dt))
        require(n + k <= len(ts), f"{ifo}: signal ends inside the edge taper")
        x = np.array(ts.numpy(), dtype=np.float64)
        x[n:n + k] *= np.kaiser(2 * k, EDGE_BETA)[:k]
        out[ifo] = TimeSeries(x, delta_t=ts.delta_t, epoch=ts.start_time)
    return out, n_before


# Copiado de scripts/m13/m13_13_build_g3t.py:s15_strains @ 4b34677, con la banda de scripts/m15/m15_build_g4.py:s15_strains_g4; cambios: nombre -> clickfree_strains; g1.DETECTOR_PROJECTOR -> phys.projector; cap.params / cap.geocentric_time -> argumentos params / geocentric_time; la banda llega como argumento (en el origen la fija taper_band(*band_for(mass1 + mass2))).
def clickfree_strains(phys, opt_args, params, geocentric_time, processing_start, band):
    """Projection of S aligned with A, same projector, params and t_c as A, after the edge taper."""
    hp, hc, align = opt_click_free_aligned(opt_args, *band)
    proj = phys.projector.project(h_plus=hp, h_cross=hc, parameters=params,
                                  geocentric_coalescence_time=geocentric_time)
    strains, n_before = edge_taper(proj.strains, processing_start)
    return strains, n_before, align
