#!/usr/bin/env python3
"""M13.11: one PE per row of rom_rows_<UTC>/ with pycbc.inference (model "relative", sampler dynesty). Local CPU.

Protocol: docs/experiments/M13_11_bayesian_reference.md (amendments 2, 3). Input: m13_11_build_rom_rows.py.

Data and PSD. No .ini and no pycbc_inference: the data and the PSD of each row are already in memory, and the model
is built in Python. Each detector: FrequencySeries = FFT of w(t) (s_ROM + n) on the 8 s centred on the final window,
keeping its epoch; PSD = psd_proc at 1/8 Hz, the PSD of the noise, as a FrequencySeries of the same delta_f. Inner
products 30-1024 Hz.

Window (amendment 3). Tukey with 0.5 s ramps (alpha = 0.25), inside the 2 s margins. The signal stays in the flat
part, so the standard model "relative" applies with an unwindowed template. The noise in the ramps is reduced;
the PSD is not scaled by mean(w^2), because that would raise the SNR of a signal lying in the flat part.
Rows of class truncated_or_edge (is_truncated = True in G3, amendment 4) also get a PE; they are out of the primary
analysis and reported apart. The class comes from is_truncated, not from the row_class stored in the .npz, which
is stale in rows built before amendment 4. The SNR^2 fractions are copied to the .json as information.

relbin_check: 8 posterior samples, relative loglr against the exact one (template at every frequency bin).
Criterion of the row: max_abs_diff <= 0.2 (relbin_ok in the .json). primary = is_truncated False and relbin_ok.

Priors (pre-registration):
  - m1, m2 U[5, 90], m1 >= m2. Sampled as (mchirp, q) with the pycbc distributions that are uniform in m1, m2,
    plus the constraint 5 <= m2, m1 <= 90: the same prior, and far easier for the sampler at low mass.
  - spin1z, spin2z U[-1, 1]; isotropic sky and inclination; polarization U[0, 2 pi].
  - t_c U[+-0.1 s] around the true geocentric time.
  - coa_phase marginalized analytically. The pre-registration fixed it at 0, but the injected ROM carries the
    phase of the alignment (param_rom_coa_phase_shift). Exact for the (2,2) mode with aligned spins.
  - distance: p(d) ~ 1/d^2 (pre-registration; the prior induced by a target SNR U[10, 25]) on DISTANCE_RANGE_MPC,
    the same for every row. --distance-prior volume gives p(d) ~ d^2 and writes to another directory.

Fiducial = true parameters. epsilon = 0.03 (--epsilon): the default of the model, 0.5, is too coarse here.

Bank (stopping rule of the pre-registration, applied on 6-11-2026): the first --n-per-regime rows (default 40) of each
regime, in the order of selection.json, which is already random. Rows alternate short/medium/long, so the regimes
advance together; a new run with a larger --n-per-regime continues and does not repeat the rows with a .json.
--row can still take any row of selection.json.

loglr_truth_aligned: exact loglr at the true parameters as they were injected. align_rom gives y(t) = h_ROM(t + tau),
tau = rom_time_shift_s of rows.jsonl (coarse + fine lag - sub-sample offset of the epochs), so the merger of the
injected signal is at t_c - tau; coa_phase is param_coa_phase + param_rom_coa_phase_shift. Marginalized over the phase
as the posterior, hence independent of the coa_phase; loglr_truth_aligned_fixed_phase keeps Re<h|d> - <h|h>/2, the one
that the coa_phase changes. loglr_max_posterior is the exact loglr at the nested sample of highest likelihood.
truth_inconsistent: loglr_truth_aligned is more than TRUTH_MAX_DEFICIT below it (information only, not for the bank).
--check-truth adds these fields to the .json of finished rows, from their stored samples, without running the PE.

Output: <rows_dir>/pe_8s_nlive<N>_walks<W>_eps<E>_d<prior>/
  row_<index>.npz    posterior_<param> (equal weight), nested_<param>, nested_logwt, nested_logl, truth_<param>
  row_<index>.json   wall time, likelihood calls, logz, summaries, relbin_check, loglr_truth_aligned, provenance
                     (written last)
  row_<index>.running / .failed    lock of a PE in progress / traceback of a PE that crashed
Resumable: a row with its .json is skipped, a .failed row is tried again. Several copies can run at once.

--status prints done / running / pending of the bank per regime and the mean time (standard library only).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")

REGIME_ORDER = ("short", "medium", "long")
DEFAULT_N_PER_REGIME = 40


def bank_rows(selection, n_per_regime):
    """{regime: first n_per_regime row indices, in the order of selection.json}."""
    return {name: [r["row_index"] for r in selection["rows"] if r["regime"] == name][:n_per_regime]
            for name in REGIME_ORDER}


def lock_alive(path):
    """(alive, description) of a .running file. On another host the process cannot be checked."""
    try:
        lock = json.loads(path.read_text())
    except (OSError, ValueError):
        return True, "lock being written"
    if lock["host"] != socket.gethostname():
        return True, f"{lock['host']}:{lock['pid']}"
    try:
        os.kill(lock["pid"], 0)
    except ProcessLookupError:
        return False, f"pid {lock['pid']} gone"
    except PermissionError:
        pass
    return True, f"pid {lock['pid']}, since {time.strftime('%d-%m %H:%M', time.localtime(lock['start']))}"


def run_status(rows_dir, n_per_regime):
    selection = json.loads((rows_dir / "selection.json").read_text())
    bank = bank_rows(selection, n_per_regime)
    in_bank = {i for ids in bank.values() for i in ids}
    outside = [r["row_index"] for r in selection["rows"] if r["row_index"] not in in_bank]
    n_built = sum((rows_dir / f"row_{i:06d}.npz").exists() for i in in_bank)
    print(f"{rows_dir}: bank of {n_per_regime} per regime, {n_built}/{len(in_bank)} rows built")
    pe_dirs = sorted(rows_dir.glob("pe_*"))
    if not pe_dirs:
        print("No PE started.")
    for pe_dir in pe_dirs:
        print(pe_dir.name)
        print(f"  {'regime':<8}{'done':>6}{'running':>9}{'failed':>8}{'pending':>9}{'primary':>9}{'trunc/edge':>12}"
              f"{'relbin>max':>12}{'mean h':>9}{'mean calls':>12}")
        total_h, orphans, running_rows, inconsistent = [], [], [], []
        for regime in REGIME_ORDER:
            ids = bank[regime]
            done, running, failed = [], 0, 0
            for i in ids:
                stem = pe_dir / f"row_{i:06d}"
                if stem.with_suffix(".json").exists():
                    done.append(json.loads(stem.with_suffix(".json").read_text()))
                elif stem.with_suffix(".running").exists():
                    alive, text = lock_alive(stem.with_suffix(".running"))
                    if alive:
                        running += 1
                        running_rows.append(f"{i} ({regime}, {text})")
                    else:
                        orphans.append(i)
                elif stem.with_suffix(".failed").exists():
                    failed += 1
            inconsistent += [d["row_index"] for d in done if d.get("truth_inconsistent")]
            hours = [d["wall_time_s"] / 3600 for d in done]
            total_h += hours
            mean_h = f"{sum(hours) / len(hours):.2f}" if hours else "-"
            calls = f"{sum(d['n_likelihood_calls'] for d in done) / len(done):.3g}" if done else "-"
            print(f"  {regime:<8}{len(done):>6}{running:>9}{failed:>8}{len(ids) - len(done) - running - failed:>9}"
                  f"{sum(d['primary'] for d in done):>9}{sum(d['row_class'] != 'primary' for d in done):>12}"
                  f"{sum(not d['relbin_ok'] for d in done):>12}{mean_h:>9}{calls:>12}")
        if total_h:
            print(f"  total: {len(total_h)} done, mean {sum(total_h) / len(total_h):.2f} h per row")
        extra = [i for i in outside if (pe_dir / f"row_{i:06d}.json").exists()]
        if extra:
            print(f"  done outside the bank, not counted above: {extra}")
        if inconsistent:
            print(f"  truth_inconsistent (information only): {inconsistent}")
        for text in running_rows:
            print(f"  running: row {text}")
        if orphans:
            print(f"  rows with a dead lock, counted as pending (a new run takes them): {orphans}")


def _early_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--rows-dir", type=Path)
    parser.add_argument("--n-per-regime", type=int, default=DEFAULT_N_PER_REGIME)
    return parser.parse_known_args(argv)[0]


if __name__ == "__main__" and _early_args(sys.argv[1:]).status:
    _early = _early_args(sys.argv[1:])
    if _early.rows_dir is None:
        raise SystemExit("--status needs --rows-dir")
    run_status(_early.rows_dir, _early.n_per_regime)
    sys.exit(0)

import numpy as np
from pycbc.conversions import chi_eff, mass1_from_mchirp_q, mass2_from_mchirp_q, mchirp_from_mass1_mass2
from pycbc.distributions import (JointDistribution, MchirpfromUniformMass1Mass2, QfromUniformMass1Mass2, SinAngle,
                                 Uniform, UniformAngle, UniformPowerLaw, UniformSky)
from pycbc.inference.models.relbin import Relative
from pycbc.transforms import MchirpQToMass1Mass2
from pycbc.types import Array, FrequencySeries, TimeSeries
from pycbc.waveform import get_fd_waveform_sequence
from scipy.signal.windows import tukey

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import resolve_data_root

APPROXIMANT = "SEOBNRv4_ROM"
MASS_RANGE = (5.0, 90.0)
SPIN_MAX = 1.0
TC_HALF_WIDTH_S = 0.1
# Wide against any row of the dataset; the same for all rows, so the prior does not depend on the truth.
DISTANCE_RANGE_MPC = (5.0, 50000.0)
# dim of pycbc UniformPowerLaw: p(d) ~ d^(dim - 1).
DISTANCE_PRIORS = {"inverse_square": -1, "volume": 3}
DLOGZ = 0.1
PE_SEED = 131111
N_CHECK = 8
RELBIN_MAX_DIFF = 0.2
TRUTH_MAX_DEFICIT = 10.0
VARIABLE_PARAMS = ["mchirp", "q", "spin1z", "spin2z", "ra", "dec", "inclination", "polarization", "tc", "distance"]
WAVEFORM_PARAMS = ("mass1", "mass2", "spin1z", "spin2z", "distance", "inclination")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact_sh_hh(model, params):
    """<h|d> and <h|h> of the same data and PSD at every frequency bin of the band, without relative binning."""
    sh, hh = 0j, 0.0
    tc_ref = model.fid_params["tc"]  # antenna pattern and delay at the fiducial time, as the model
    for ifo in model.data:
        k0, k1 = model.edges[ifo][0], model.edges[ifo][-1]
        f = model.f[ifo][k0:k1]
        hp, hc = get_fd_waveform_sequence(sample_points=Array(f.astype(np.float64)), approximant=APPROXIMANT,
                                          **{k: params[k] for k in (*WAVEFORM_PARAMS, "coa_phase") if k in params})
        det = model.det[ifo]
        fp, fc = det.antenna_pattern(params["ra"], params["dec"], params["polarization"], tc_ref)
        t_det = params["tc"] + det.time_delay_from_earth_center(params["ra"], params["dec"], tc_ref)
        h = (fp * hp.numpy() + fc * hc.numpy()) * np.exp(-2j * np.pi * f * (t_det - model.end_time[ifo]))
        weight = 4.0 * model.df[ifo] / model.psds[ifo].numpy()[k0:k1]
        sh += np.sum(np.conj(model.data[ifo].numpy()[k0:k1]) * h * weight)
        hh += float(np.sum(np.abs(h) ** 2 * weight))
    return sh, hh


def exact_loglr(model, params):
    return float(model.marginalize_loglr(*exact_sh_hh(model, params)))


def truth_as_injected(row, rows_dir, truth):
    """True parameters with the time and phase of the ROM alignment; see the module docstring."""
    index = scalar(row, "row_index")
    records = {}
    with (rows_dir / "rows.jsonl").open() as f:
        for line in f:  # a row can appear twice after a crash; the last record wins
            rec = json.loads(line)
            records[rec["row_index"]] = rec
    require(index in records, f"Row {index} is not in rows.jsonl")
    m1, m2 = scalar(row, "param_mass_1"), scalar(row, "param_mass_2")
    return {**truth, "mass1": m1, "mass2": m2, "tc": truth["tc"] - records[index]["rom_time_shift_s"],
            "coa_phase": scalar(row, "param_coa_phase") + scalar(row, "param_rom_coa_phase_shift")}


def truth_check(model, injected, nested, nested_logl):
    """Fields of the .json: loglr at the truth as injected against the exact loglr at the best nested sample."""
    sh, hh = exact_sh_hh(model, injected)
    i = int(np.argmax(nested_logl))
    best = {k: float(nested[k][i]) for k in (*VARIABLE_PARAMS, "mass1", "mass2")}
    aligned, top = float(model.marginalize_loglr(sh, hh)), exact_loglr(model, best)
    return dict(loglr_truth_aligned=aligned, loglr_truth_aligned_fixed_phase=float(sh.real - 0.5 * hh),
                loglr_max_posterior=top, truth_alignment=dict(tc=injected["tc"], coa_phase=injected["coa_phase"]),
                truth_inconsistent=bool(not top - aligned <= TRUTH_MAX_DEFICIT))


def component_mass_limits(params):
    """Constraint of the (mchirp, q) rectangle down to m2 >= 5, m1 <= 90; q >= 1 orders the masses."""
    return ((mass2_from_mchirp_q(params["mchirp"], params["q"]) >= MASS_RANGE[0])
            & (mass1_from_mchirp_q(params["mchirp"], params["q"]) <= MASS_RANGE[1]))


def build_prior(tc, distance_prior):
    lo, hi = MASS_RANGE
    mchirp = (float(mchirp_from_mass1_mass2(lo, lo)), float(mchirp_from_mass1_mass2(hi, hi)))
    return JointDistribution(
        VARIABLE_PARAMS,
        MchirpfromUniformMass1Mass2(mchirp=mchirp), QfromUniformMass1Mass2(q=(1.0, hi / lo)),
        Uniform(spin1z=(-SPIN_MAX, SPIN_MAX), spin2z=(-SPIN_MAX, SPIN_MAX)),
        UniformSky(polar_angle="dec", azimuthal_angle="ra", azimuthal_cyclic_domain=True),
        SinAngle(inclination=None), UniformAngle(cyclic_domain=True, polarization=None),
        Uniform(tc=(tc - TC_HALF_WIDTH_S, tc + TC_HALF_WIDTH_S)),
        UniformPowerLaw(dim=DISTANCE_PRIORS[distance_prior], distance=DISTANCE_RANGE_MPC),
        constraints=[component_mass_limits])


def load_row(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def scalar(row, key):
    return row[key].item()


def build_model(row, args):
    require("strain_8s" in row, "Row without the 8 s segment: rebuild it with the current m13_11_build_rom_rows.py")
    detectors = [str(d) for d in row["detectors"]]
    epoch, delta_t = float(row["epoch_8s"][0]), scalar(row, "delta_t")
    n = row["strain_8s"].shape[1]
    window = tukey(n, 2.0 * scalar(row, "pe_tukey_ramp_s") / (n * delta_t))
    data, psds = {}, {}
    for i, ifo in enumerate(detectors):
        data[ifo] = TimeSeries(window * row["strain_8s"][i], delta_t=delta_t, epoch=epoch).to_frequencyseries()
        require(len(data[ifo]) == row["psd_8s"].shape[1]
                and abs(data[ifo].delta_f - scalar(row, "psd_delta_f_8s")) < 1e-12,
                "PSD grid differs from the FFT of the 8 s segment")
        psds[ifo] = FrequencySeries(row["psd_8s"][i], delta_f=data[ifo].delta_f, epoch=data[ifo].epoch)
    m1, m2 = scalar(row, "param_mass_1"), scalar(row, "param_mass_2")
    require(m1 >= m2, "Row with mass_1 < mass_2")
    truth = dict(mchirp=float(mchirp_from_mass1_mass2(m1, m2)), q=m1 / m2, spin1z=scalar(row, "param_spin_1z"),
                 spin2z=scalar(row, "param_spin_2z"), ra=scalar(row, "param_ra"), dec=scalar(row, "param_dec"),
                 inclination=scalar(row, "param_inclination"), polarization=scalar(row, "param_polarization_angle"),
                 tc=scalar(row, "param_geocentric_time"), distance=scalar(row, "param_distance"))
    require(2 * DISTANCE_RANGE_MPC[0] < truth["distance"] < DISTANCE_RANGE_MPC[1] / 2,
            f"True distance {truth['distance']:.1f} Mpc too close to the limits of the prior")
    fiducial = {k: v for k, v in truth.items() if k not in ("mchirp", "q")}
    fiducial.update(mass1=m1, mass2=m2)
    model = Relative(
        list(VARIABLE_PARAMS), data, {ifo: scalar(row, "f_lower") for ifo in detectors},
        fiducial_params=fiducial, epsilon=args.epsilon, marginalize_phase=True,
        psds=psds, high_frequency_cutoff={ifo: scalar(row, "f_upper") for ifo in detectors},
        static_params=dict(approximant=APPROXIMANT, f_lower=scalar(row, "f_lower")),
        prior=build_prior(truth["tc"], args.distance_prior), waveform_transforms=[MchirpQToMass1Mass2()],
        ignore_failed_waveforms=True, no_save_data=True)
    return model, truth


def derived(samples):
    """Detector-frame labels and component masses, for a dict of sample arrays."""
    out = dict(samples)
    out["mass1"] = mass1_from_mchirp_q(out["mchirp"], out["q"])
    out["mass2"] = mass2_from_mchirp_q(out["mchirp"], out["q"])
    out["chirp_mass"] = out["mchirp"]
    out["total_mass"] = out["mass1"] + out["mass2"]
    out["chi_eff"] = chi_eff(out["mass1"], out["mass2"], out["spin1z"], out["spin2z"])
    return out


def relbin_check(model, samples, n):
    params = [{k: float(samples[k][i]) for k in VARIABLE_PARAMS} for i in range(min(n, len(samples["mchirp"])))]
    rows = []
    for p in params:
        model.update(**p)
        full = {**p, "mass1": float(mass1_from_mchirp_q(p["mchirp"], p["q"])),
                "mass2": float(mass2_from_mchirp_q(p["mchirp"], p["q"]))}
        rows.append(dict(loglr_relative=float(model.loglr), loglr_exact=exact_loglr(model, full)))
    worst = float(np.max(np.abs([r["loglr_relative"] - r["loglr_exact"] for r in rows])))  # keeps a NaN, which fails
    return dict(rows=rows, max_abs_diff=worst, threshold=RELBIN_MAX_DIFF, relbin_ok=bool(worst <= RELBIN_MAX_DIFF))


def run_row(row_path, out_stem, args):
    from pycbc.inference.sampler.dynesty import DynestySampler
    import dynesty
    import pycbc

    t_start = time.perf_counter()
    row = load_row(row_path)
    index = scalar(row, "row_index")
    row_class = "truncated_or_edge" if scalar(row, "param_is_truncated") else "primary"
    model, truth = build_model(row, args)
    model.update(**truth)
    loglr_truth = float(model.loglr)
    injected = truth_as_injected(row, row_path.parent, truth)  # before the sampler: a missing record fails at once
    print(f"row {index} ({scalar(row, 'regime')}): loglr(truth) {loglr_truth:.2f}, loglr(truth aligned) "
          f"{exact_loglr(model, injected):.2f}, target SNR "
          f"{scalar(row, 'param_target_network_snr'):.2f}, bins {[len(e) for e in model.edges.values()]}, "
          f"{row_class}, SNR^2 in flat zone {scalar(row, 'snr2_fraction_flat'):.5f}", flush=True)

    rng = np.random.default_rng([PE_SEED, index])
    sampler = DynestySampler(
        model, args.nlive, nprocesses=args.workers,
        extra_kwds=dict(sample="rwalk", bound="multi", walks=args.walks, rstate=rng),
        run_kwds=dict(dlogz=DLOGZ, print_progress=False), internal_kwds={})
    t_sampler = time.perf_counter()
    try:
        sampler.run()
    finally:
        sampler.pool.close_pool()
    sampling_s = time.perf_counter() - t_sampler
    res = sampler._sampler.results
    # Columns of the sampler are model.sampling_params, which here are the variable params.
    names = list(model.sampling_params)
    require(set(names) == set(VARIABLE_PARAMS), f"Unexpected sampling params: {names}")
    nested = {name: np.asarray(res.samples)[:, i] for i, name in enumerate(names)}
    weights = np.exp(np.asarray(res.logwt) - float(res.logz[-1]))
    weights /= weights.sum()
    n_eff = int(1.0 / np.sum(weights ** 2))
    draw = rng.choice(weights.size, size=n_eff, p=weights)
    posterior = derived({k: v[draw] for k, v in nested.items()})
    nested = derived(nested)
    check = relbin_check(model, posterior, N_CHECK)
    truth_fields = truth_check(model, injected, nested, res.logl)

    labels = [str(x) for x in row["label_names"]]
    summary = {k: dict(mean=float(np.average(nested[k], weights=weights)),
                       std=float(np.sqrt(np.average((nested[k] - np.average(nested[k], weights=weights)) ** 2,
                                                    weights=weights))),
                       truth=float(row["y_true"][j]))
               for j, k in enumerate(labels)}
    arrays = dict(row_index=index, regime=row["regime"], label_names=row["label_names"], y_true=row["y_true"],
                  nested_logwt=np.asarray(res.logwt), nested_logl=np.asarray(res.logl),
                  **{f"posterior_{k}": v for k, v in posterior.items()},
                  **{f"nested_{k}": v for k, v in nested.items()},
                  **{f"truth_{k}": v for k, v in truth.items()})
    tmp = out_stem.with_suffix(f".tmp_{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, out_stem.with_suffix(".npz"))
    record = dict(
        row_index=index, regime=scalar(row, "regime"), tau30_s=scalar(row, "tau30_s"),
        wall_time_s=time.perf_counter() - t_start, sampling_time_s=sampling_s,
        n_likelihood_calls=int(np.sum(res.ncall)), n_iterations=int(res.niter), n_posterior_samples=n_eff,
        logz=float(res.logz[-1]), logz_err=float(res.logzerr[-1]), loglr_truth=loglr_truth, **truth_fields,
        target_network_snr=scalar(row, "param_target_network_snr"), labels=summary, relbin_check=check,
        row_class=row_class, snr2_fraction_flat=scalar(row, "snr2_fraction_flat"),
        snr2_fraction_final4s=scalar(row, "snr2_fraction_final4s"), is_truncated=scalar(row, "param_is_truncated"),
        relbin_ok=check["relbin_ok"], primary=bool(row_class == "primary" and check["relbin_ok"]),
        settings=dict(nlive=args.nlive, walks=args.walks, epsilon=args.epsilon, dlogz=DLOGZ, workers=args.workers,
                      distance_prior=args.distance_prior, distance_range_mpc=list(DISTANCE_RANGE_MPC),
                      tc_half_width_s=TC_HALF_WIDTH_S, segment_s=row["strain_8s"].shape[1] * scalar(row, "delta_t"),
                      tukey_ramp_s=scalar(row, "pe_tukey_ramp_s"), seed=[PE_SEED, index],
                      approximant=APPROXIMANT, band_hz=[scalar(row, "f_lower"), scalar(row, "f_upper")]),
        provenance=dict(rows_npz=str(row_path), lal_data_path=os.environ["LAL_DATA_PATH"],
                        pycbc=pycbc.__version__, dynesty=dynesty.__version__, host=socket.gethostname()))
    # Last: the .json is what marks the row as done.
    tmp_json = out_stem.with_suffix(f".jsontmp_{os.getpid()}")
    tmp_json.write_text(json.dumps(record, indent=2))
    os.replace(tmp_json, out_stem.with_suffix(".json"))
    print(f"row {index}: {record['wall_time_s'] / 3600:.2f} h, {record['n_likelihood_calls']} calls, "
          f"{n_eff} samples, relbin max|dloglr| {check['max_abs_diff']:.3f} "
          f"({'OK' if check['relbin_ok'] else 'ABOVE LIMIT'}), primary {record['primary']}, "
          f"{truth_summary(truth_fields)}, "
          + ", ".join(f"{k} {s['mean']:.3f} +- {s['std']:.3f} (truth {s['truth']:.3f})" for k, s in summary.items()),
          flush=True)


def truth_summary(fields):
    return (f"loglr truth aligned {fields['loglr_truth_aligned']:.2f} (fixed phase "
            f"{fields['loglr_truth_aligned_fixed_phase']:.2f}), max posterior {fields['loglr_max_posterior']:.2f}"
            + (", TRUTH INCONSISTENT" if fields["truth_inconsistent"] else ""))


def check_truth_row(row_path, out_stem, args):
    """Adds the fields of truth_check to a finished row, from its stored samples; no PE."""
    record = json.loads(out_stem.with_suffix(".json").read_text())
    if "loglr_truth_aligned" in record:
        print(f"row {record['row_index']}: already has loglr_truth_aligned, unchanged", flush=True)
        return
    row = load_row(row_path)
    model, truth = build_model(row, args)
    model.update(**truth)
    with np.load(out_stem.with_suffix(".npz"), allow_pickle=False) as a:
        nested = {k: a[f"nested_{k}"] for k in (*VARIABLE_PARAMS, "mass1", "mass2")}
        nested_logl = a["nested_logl"]
    fields = truth_check(model, truth_as_injected(row, row_path.parent, truth), nested, nested_logl)
    tmp_json = out_stem.with_suffix(f".jsontmp_{os.getpid()}")
    tmp_json.write_text(json.dumps({**record, **fields}, indent=2))
    os.replace(tmp_json, out_stem.with_suffix(".json"))
    print(f"row {record['row_index']}: loglr_truth {record['loglr_truth']:.2f}, {truth_summary(fields)}", flush=True)


def acquire(lock):
    if lock.exists():
        alive, _ = lock_alive(lock)
        if alive:
            return False
        lock.unlink()
    try:
        with lock.open("x") as f:
            json.dump(dict(host=socket.gethostname(), pid=os.getpid(), start=time.time()), f)
    except FileExistsError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows-dir", type=Path, required=True, help="rom_rows_<UTC> of m13_11_build_rom_rows.py")
    parser.add_argument("--row", help="Row index, or 'longest' (timing row of selection.json); default: all rows")
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--walks", type=int, default=100, help="Minimum steps of the dynesty random walk")
    parser.add_argument("--epsilon", type=float, default=0.03, help="Tuning of the relative binning")
    parser.add_argument("--distance-prior", choices=sorted(DISTANCE_PRIORS), default="inverse_square")
    parser.add_argument("--workers", type=int, default=1, help="Processes inside one PE")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--n-per-regime", type=int, default=DEFAULT_N_PER_REGIME,
                        help="Bank: first N rows of each regime in selection.json; also read by --status")
    parser.add_argument("--status", action="store_true", help="Print the state of the bank and exit")
    parser.add_argument("--check-truth", action="store_true",
                        help="Add loglr_truth_aligned to the finished rows of the bank (or --row), without PE")
    args = parser.parse_args()
    require(args.workers >= 1, "Invalid --workers")
    require(args.n_per_regime >= 1, "Invalid --n-per-regime")

    root = resolve_data_root(cli_data_root=args.data_root)
    lal_data = Path(os.environ.setdefault("LAL_DATA_PATH", str(Path(root) / "external" / "lal_data")))
    require(lal_data.is_dir(), f"LAL_DATA_PATH is not a directory: {lal_data}")
    print(f"LAL_DATA_PATH {lal_data}", flush=True)
    try:
        import dynesty  # noqa: F401
    except ImportError:
        raise SystemExit("dynesty is not installed in this environment")

    selection = json.loads((args.rows_dir / "selection.json").read_text())
    ordered = [r["row_index"] for r in selection["rows"]]
    if args.row == "longest":
        todo = [selection["timing_row_longest_long_regime"]["row_index"]]
    elif args.row is not None:
        todo = [int(args.row)]
        require(todo[0] in ordered, f"Row {args.row} is not in selection.json")
    else:
        # Interleaved, so that a partial bank covers the three regimes evenly.
        bank = bank_rows(selection, args.n_per_regime)
        todo = [i for triple in zip(*(bank[name] for name in REGIME_ORDER)) for i in triple]

    pe_dir = args.rows_dir / f"pe_8s_nlive{args.nlive}_walks{args.walks}_eps{args.epsilon:g}_d{args.distance_prior}"
    if args.check_truth:
        require(pe_dir.is_dir(), f"No PE directory: {pe_dir}")
        for index in todo:
            stem = pe_dir / f"row_{index:06d}"
            if stem.with_suffix(".json").exists():
                check_truth_row(args.rows_dir / f"row_{index:06d}.npz", stem, args)
        return
    pe_dir.mkdir(exist_ok=True)
    n_done = n_failed = 0
    for index in todo:
        row_path = args.rows_dir / f"row_{index:06d}.npz"
        stem = pe_dir / f"row_{index:06d}"
        if stem.with_suffix(".json").exists() or not row_path.exists():
            continue
        lock = stem.with_suffix(".running")
        if not acquire(lock):
            continue
        try:
            stem.with_suffix(".failed").unlink(missing_ok=True)
            run_row(row_path, stem, args)
            n_done += 1
        except Exception:
            n_failed += 1
            stem.with_suffix(".failed").write_text(traceback.format_exc())
            print(f"row {index} FAILED, see {stem.with_suffix('.failed')}", flush=True)
        finally:
            lock.unlink(missing_ok=True)
    print(f"{n_done} PE done, {n_failed} failed in this run -> {pe_dir}", flush=True)
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
