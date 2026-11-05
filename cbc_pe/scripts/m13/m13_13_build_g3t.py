#!/usr/bin/env python3
"""M13.13 step 0: build G3T, i.e. G3 with the tapered waveform S15. Local CPU.

Protocol: docs/experiments/M13_13_G3T_taper.md (pre-registered 5-11-2026).

S15 = SEOBNRv4_opt from 15 Hz, half-Kaiser (beta = 8) roll-up from its start to the first sample where
the instantaneous frequency reaches 25 Hz, aligned in phase and sub-sample time with the G3 waveform A.
It is m13_11._opt_click_free_aligned with CLICK_F_LOWER/CLICK_TAPER_END_HZ set to these values below.
Source, placement, final distance (from the SNR of A), PSD, noise, processing, z-score, split and labels
are those of G3: the row is rebuilt by build_g1_row (m13_11.capture_row) and only the signal is replaced.

G-EDGE: if S15 starts before processing_start, inject would cut it there with a step. The projected
strain is then tapered from processing_start (half Kaiser, beta = 8, 0.5 s) and the row is counted.

Subcommands (outputs under <data_root>/processed/<dataset_id>/m13_13_g3t/; nothing is overwritten):
  gate-taper   G-TAPER (amendment 1), intrinsic (face-on, unit distance): S15 against Sref = same model
               from 10 Hz with taper to 14 Hz, aligned as S15; the mismatch with SEOBNRv4_ROM is
               informative only. gate_taper_<stamp>/gate_taper.json. --f-lower/--f-taper-end change the
               band of S15 in this subcommand only; --rows-from evaluates the rows that failed in a
               previous report, and its verdict (PASS_SUBSET) does not authorize generation.
               signal and noisy need --gate-json: a PASS report of the 2000 rows (not --rows-from) with
               the band of the generation (see require_gate_passed).
  signal       train L(s_S15) before z-score, noise = 0: signal/signal_shardIIofNN.h5, format of
               m13_6_aug/signal plus per-row align_residual_rad, edge_tapered, time_s (--resume).
  merge        memmap/{X_clean_train,psd_key_train}.npy, manifest.json, and relative symlinks
               bank_X.npy, bank_psd_key.npy to m13_6_aug/memmap/ (the bank is not copied).
  noisy        X = zscore(L(s_S15 + n)), n = noise of the G3 row, float16, for --split val, cal, test
               or train40k. Built in work_noisy_<split>/ (X.npy, done.npy, rows.jsonl; --resume),
               published on completion as fp16_memmap/ (val, train40k) or fp16_memmap_heldout/ (cal,
               test) in the format of build_fp16_memmap.py: X/y/row_ids_<split>.npy, gates_<split>.json
               and manifest.json. y and row_ids are byte copies of the G3 memmap.
  consistency  64 monitor rows, consistency_<stamp>/consistency_report.json.
  status       read only, without the heavy imports: per step of scripts/m13/run_m13_13.sh, done, running
               (k/N, %, ETA) or pending, from logs/m13/m13_13_*.{log,done} and the output directories.

train40k is published as X_train.npy with the shape of the whole G3 train split: train_m13_3.py reads the
monitor at positions sort(default_rng(133400).choice(400000, 40000)) of ds["train"]. Only those positions
are written; the others are holes of a sparse file and read as zeros. With noise augmentation the trainer
reads no other row of this file, but it must never be used for training without noise_augmentation.

The trainer has <processed>/<dataset_id>/fp16_memmap hard-coded, so the config uses the nested dataset_id
M13_G3_band1024_w16_500k/m13_13_g3t. dataset_file then resolves to a path that does not exist, and the
trainer skips its memmap-vs-HDF5 check (which would rightly fail: the HDF5 is G3).

G-ALIGN (amendment 3): a row above ALIGN_TOL is recorded and the run goes on. It stops if a row exceeds
ALIGN_ABORT (at once) or if more than ALIGN_MAX_FRACTION of a split or shard exceeds ALIGN_TOL (on closing
it). The 0.01 rad of m13_11 stays inside _align_to_A: there it only decides whether the other sign is tried.

--limit N takes N evenly spaced rows and writes to timing_<stamp>/, which nothing else reads.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import sys
import time

OUTPUT_SUBDIR = "m13_13_g3t"
NUM_SHARDS_RUN = 4  # of scripts/m13/run_m13_13.sh
# (step of run_m13_13.sh, its output under OUTPUT_SUBDIR); order of the chain.
STATUS_STEPS = (("noisy_val", "fp16_memmap/X_val.npy"), ("noisy_train40k", "fp16_memmap/X_train.npy"),
                *((f"signal_shard{i}", f"signal/signal_shard{i:02d}of{NUM_SHARDS_RUN:02d}.h5")
                  for i in range(NUM_SHARDS_RUN)),
                ("merge", "memmap/manifest.json"), ("consistency", "consistency_*/consistency_report.json"),
                ("noisy_cal", "fp16_memmap_heldout/X_cal.npy"), ("noisy_test", "fp16_memmap_heldout/X_test.npy"))
STATUS_IDLE_S = 900  # a step without log activity for longer than this is shown as stopped


def run_status(data_root):
    """Standard library only, defined before the project imports so that it answers at once."""
    data_root = data_root or os.environ.get("CBC_PE_DATA_ROOT")
    if not data_root:
        raise SystemExit("Use --data-root or set CBC_PE_DATA_ROOT")
    config = Path(__file__).resolve().parents[2] / "configs/generation/generate_m13_G3_band1024_w16.json"
    base = Path(data_root) / "processed" / json.loads(config.read_text())["dataset_id"] / OUTPUT_SUBDIR
    logs = Path(data_root) / "logs" / "m13"
    for step, output in STATUS_STEPS:
        log = logs / f"m13_13_{step}.log"
        if (logs / f"m13_13_{step}.done").exists():
            state = "hecho" if list(base.glob(output)) else f"hecho, PERO FALTA {output}"
        elif not log.exists():
            state = "pendiente"
        else:
            with log.open("rb") as f:
                f.seek(max(log.stat().st_size - 8192, 0))
                lines = f.read().decode(errors="replace").splitlines()
            found = next(filter(None, (re.search(r"\[(\d+)/(\d+)\] ETA (.+)$", x) for x in reversed(lines))), None)
            if found:
                k, n = int(found[1]), int(found[2])
                state = f"{k}/{n} ({100 * k / n:.1f} %), ETA {found[3]}"
            else:
                state = lines[-1][:100] if lines else "sin salida"
            mtime = log.stat().st_mtime
            if time.time() - mtime > STATUS_IDLE_S:
                last = time.strftime("%d-%m %H:%M", time.localtime(mtime))
                state = f"PARADO, sin actividad desde {last}: {lines[-1][:100] if lines else ''}"
            else:
                state = f"en curso: {state}"
        print(f"{step:<16} {state}")


if __name__ == "__main__" and sys.argv[1:2] == ["status"]:
    _status = argparse.ArgumentParser(prog="m13_13_build_g3t.py status")
    _status.add_argument("--data-root")
    run_status(_status.parse_args(sys.argv[2:]).data_root)
    sys.exit(0)

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir, resolve_data_root
import scripts.generate_m11_final_G1 as g1
from scripts.generate_m11_final_G1 import (DETECTORS, LABELS, M10_INPUT_ZSCORE_EPS, read_psd_intervals, require,
                                           resolve_cached_psd_files, sha256_file)
import scripts.m13.generate_m13_G3_whitening as g3
import scripts.m13.generate_m13_6_clean as g6
import scripts.m13.m13_11_prepare_rows as m11
from scripts.m13.m13_11_prepare_rows import BAND_HZ, DEFAULT_CONFIG, GREC_MAX, ROM_APPROXIMANT, write_json
from scripts.m13.m13_12_build_click_free import GDIFF_RTOL, _NoiseTap
from scripts.m13.merge_m13_6_clean import Stats, check_shard, shard_paths, stat_entry

# Band of the waveform of G3T: the only place signal, noisy and consistency take it from. Each of them
# requires a 2000-row PASS report of gate-taper with exactly these values, and records them in its output.
S15_F_LOWER = 15.0
S15_TAPER_END_HZ = 25.0
# _opt_click_free reads these two module constants at call time; S15 is the same construction as the S of
# M13.11/M13.12 with other frequencies. CLICK_F_REF stays at 30 Hz: it is the phase reference of A.
m11.CLICK_F_LOWER, m11.CLICK_TAPER_END_HZ = S15_F_LOWER, S15_TAPER_END_HZ
require(m11.CLICK_BETA == 8 and m11.CLICK_F_REF == 30.0, "Unexpected M13.11 taper constants")

EDGE_TAPER_S = 0.5
EDGE_BETA = 8
GATE_SEED = 131301
GATE_ROWS = 2000
GATE_TOP_MASS_ROWS = 20
GATE_STRATA = 10
# Amendment 1 of the pre-registration: reference Sref, and the ROM only as information. Amendment 2: criteria.
SREF_F_LOWER = 10.0
SREF_TAPER_END_HZ = 14.0
# Every row: in-band difference with SNR <= 1 at the top of the SNR prior, sqrt(2 m) * 25 <= 1.
GATE_MISMATCH_MAX = 8e-4
# At most this fraction of rows above GATE_MISMATCH_SOFT.
GATE_MISMATCH_SOFT = 1e-4
GATE_SOFT_FRACTION_MAX = 0.02
GATE_STEP_MAX = 2e-3  # edge value of the Kaiser window
GATE_MASS_BINS = (10.0, 25.0, 40.0, 60.0, 80.0, 100.0, 180.0)
# Amendment 3. Band difference with SNR ~ SNR * delta <= 1 at SNR 25, as amendment 2.
ALIGN_TOL = 0.04
ALIGN_ABORT = 0.5  # a code failure, not a small phase error
ALIGN_MAX_FRACTION = 1e-3  # of a split or shard above ALIGN_TOL
ETA_WINDOW_ROWS = 200
CONSISTENCY_SEED = 131303  # 131302 is the bootstrap seed of the pre-registration
CONSISTENCY_ROWS = 64
CONSISTENCY_MAX = 1e-5
# Monitor of train_m13_3.py; checked against its source in monitor_positions.
TRAIN_MONITOR_N = 40000
TRAIN_MONITOR_SEED = 133400
TRAINER = PROJECT_ROOT / "scripts/m13/train_m13_3.py"
MAX_WORKERS = 3
CHUNK_ROWS = 32
# split -> (split name in the memmap, memmap directory).
NOISY_SPLITS = {"val": ("val", "fp16_memmap"), "train40k": ("train", "fp16_memmap"),
                "cal": ("cal", "fp16_memmap_heldout"), "test": ("test", "fp16_memmap_heldout")}

_CTX = {}    # set in the parent before the workers are forked
_BLOCK = {}  # per worker: strain and PSD cache of the last block seen


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Log:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a")

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


class Progress:
    """"[k/N] ETA <local end time>", moving mean of the last ETA_WINDOW_ROWS rows."""

    def __init__(self, total, done):
        self.total, self.k = total, done
        self.times = deque(maxlen=ETA_WINDOW_ROWS + 1)

    def tick(self):
        self.k += 1
        self.times.append(time.time())
        elapsed, eta = self.times[-1] - self.times[0], "?"
        if elapsed > 0:
            left = (self.total - self.k) * elapsed / (len(self.times) - 1)
            eta = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.times[-1] + left))
        return f"[{self.k}/{self.total}] ETA {eta}"


def overwrite_json(path, payload):
    """For the summaries a resume computes again; everything else goes through the exclusive write_json."""
    path.write_text(json.dumps(payload, indent=2))


def align_abort(residual):
    return not abs(residual) <= ALIGN_ABORT  # also catches NaN


def align_summary(row_ids, residuals):
    """G-ALIGN of a split or shard. stop is the reason to stop the run, or None."""
    a = np.abs(np.asarray(residuals, dtype=np.float64))
    over = ~(a <= ALIGN_TOL)
    fraction = float(over.mean())
    stop = None
    if align_abort(a.max()):
        stop = f"max|residual| = {a.max():.3g} rad > {ALIGN_ABORT:g}"
    elif fraction > ALIGN_MAX_FRACTION:
        stop = f"{fraction:.2%} of the rows above {ALIGN_TOL:g} rad"
    return dict(tol_rad=ALIGN_TOL, abort_rad=ALIGN_ABORT, max_fraction=ALIGN_MAX_FRACTION, n_rows=int(a.size),
                n_over_tol=int(over.sum()), fraction_over_tol=fraction, max_abs_residual_rad=float(a.max()),
                p99_9_abs_residual_rad=float(np.quantile(a, 0.999)), stop=stop,
                rows_over_tol=[[int(i), float(r)] for i, r in zip(np.asarray(row_ids)[over], a[over])])


def align_line(g):
    return (f"G-ALIGN: {g['n_over_tol']}/{g['n_rows']} rows above {g['tol_rad']:g} rad, max "
            f"{g['max_abs_residual_rad']:.2e}, p99.9 {g['p99_9_abs_residual_rad']:.2e}")


def align_flag(residual):
    return "" if abs(residual) <= ALIGN_TOL else "  ALIGN>tol"


def thin(n, limit):
    """Positions of `limit` evenly spaced rows out of n (--limit: representative rows, not the first ones)."""
    return np.unique(np.linspace(0, n - 1, limit).astype(np.int64))


def run_tasks(worker, tasks, workers, consume):
    """consume(worker(task)) in the parent; stops submitting when consume returns False.

    At most 2 * workers tasks are in flight, so results never pile up in memory (X is ~98 kB per row).
    """
    tasks = iter(tasks)
    if workers == 1:
        for task in tasks:
            if consume(worker(task)) is False:
                return
        return
    stop, pending = False, set()
    # fork: workers inherit the g1 globals set by g6.setup and _CTX.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        while True:
            while not stop and len(pending) < 2 * workers:
                task = next(tasks, None)
                if task is None:
                    break
                pending.add(pool.submit(worker, task))
            if not pending:
                return
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                if consume(fut.result()) is False:
                    stop = True


def block_chunks(frame):
    """Row index lists of at most CHUNK_ROWS, block by block, so that a worker reloads the PSD rarely."""
    chunks = []
    for _, block_rows in frame.groupby("block_id", sort=True):
        index = list(block_rows.index)
        chunks.extend(index[i:i + CHUNK_ROWS] for i in range(0, len(index), CHUNK_ROWS))
    return chunks


def block_context(row):
    block = str(row["block_id"])
    if _BLOCK.get("id") != block:
        _BLOCK.update(id=block, long_strains=read_psd_intervals(row, _CTX["psd_files"]), psd_cache={},
                      # inject/build_zero_strain never draw from rng; it is set only to make that explicit.
                      injector=g1.SignalInjector(config=g1.CONFIG, rng=np.random.default_rng(0)))
    return _BLOCK


def edge_taper(strains, processing_start):
    """G-EDGE: taper the signal from processing_start if it starts earlier; (strains, samples cut per ifo)."""
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


def s15_strains(opt_args, cap, processing_start):
    """Projection of S15 aligned with A, same projector, params and t_c as A, after G-EDGE."""
    hp, hc, align = m11._opt_click_free_aligned(opt_args)
    proj = g1.DETECTOR_PROJECTOR.project(h_plus=hp, h_cross=hc, parameters=cap.params,
                                         geocentric_coalescence_time=cap.geocentric_time)
    strains, n_before = edge_taper(proj.strains, processing_start)
    return strains, n_before, align


# ---------------------------------------------------------------- gate-taper


def gate_rows(train_df, n_rows):
    """The GATE_TOP_MASS_ROWS heaviest train rows plus equal draws per total-mass quantile of the rest."""
    require(n_rows >= GATE_TOP_MASS_ROWS + GATE_STRATA, "Too few gate rows")
    mass = train_df["total_mass"].to_numpy(dtype=np.float64)
    ids = train_df.index.to_numpy(dtype=np.int64)
    order = np.argsort(mass, kind="stable")
    top, rest = order[-GATE_TOP_MASS_ROWS:], order[:-GATE_TOP_MASS_ROWS]
    edges = np.quantile(mass[rest], np.linspace(0, 1, GATE_STRATA + 1))
    strata = np.clip(np.searchsorted(edges[1:-1], mass[rest], side="right"), 0, GATE_STRATA - 1)
    n_draw = n_rows - GATE_TOP_MASS_ROWS
    counts = [n_draw // GATE_STRATA + (s < n_draw % GATE_STRATA) for s in range(GATE_STRATA)]
    rng = np.random.default_rng(GATE_SEED)
    drawn = [rng.choice(rest[strata == s], counts[s], replace=False) for s in range(GATE_STRATA)]
    return np.sort(ids[np.concatenate([top, *drawn])]), edges


@contextmanager
def taper_band(f_lower, taper_end_hz):
    """Other band for m13_11._opt_click_free during the block (module constants, read at call time)."""
    saved = m11.CLICK_F_LOWER, m11.CLICK_TAPER_END_HZ
    m11.CLICK_F_LOWER, m11.CLICK_TAPER_END_HZ = f_lower, taper_end_hz
    try:
        yield
    finally:
        m11.CLICK_F_LOWER, m11.CLICK_TAPER_END_HZ = saved


def gate_row(index):
    from pycbc.filter import match
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.waveform import get_td_waveform
    t0 = time.perf_counter()
    row = _CTX["frame"].loc[index]
    rec = dict(row_index=int(index), total_mass=float(row["total_mass"]), chi_eff=float(row["chi_eff"]))
    # Face-on, unit distance, as the intrinsic part of m13_11.diagnose_click.
    common = dict(mass1=float(row["mass_1"]), mass2=float(row["mass_2"]), spin1z=float(row["spin_1z"]),
                  spin2z=float(row["spin_2z"]), inclination=0.0, coa_phase=0.0, distance=1.0,
                  delta_t=m11.INTRINSIC_DELTA_T)
    try:
        kwargs = dict(approximant=g1.CONFIG.waveform_approximant, f_lower=m11.INTRINSIC_F_LOWER, **common)
        with taper_band(*_CTX["band"]):
            hp, hc, align = m11._opt_click_free_aligned(kwargs)
        with taper_band(SREF_F_LOWER, SREF_TAPER_END_HZ):
            sp, sc, align_ref = m11._opt_click_free_aligned(kwargs)
        dt = float(hp.delta_t)
        # Informative: a failure of the ROM must not decide the gate. It starts as S15: only the 30-1024 Hz
        # band is compared, and with f_lower = 30 Hz the conditioning of its time-domain version would sit
        # at the edge of that band.
        try:
            rp, rc = get_td_waveform(approximant=ROM_APPROXIMANT, f_lower=_CTX["band"][0], **common)
        except Exception as exc:
            rp, rc = None, None
            rec["rom_error"] = f"{type(exc).__name__}: {exc}"
        h = np.abs(hp.numpy() - 1j * hc.numpy())
        n_max = max(len(hp), len(sp), 0 if rp is None else len(rp))
        n_fft = 2 * int(2 ** np.ceil(np.log2(n_max)))  # 2x: no wrap of the cyclic shift
        psd = aLIGOZeroDetHighPower(n_fft // 2 + 1, 1.0 / (n_fft * dt), m11.INTRINSIC_PSD_F_LOW)

        def mismatch(x, y):
            x, y = x.copy(), y.copy()
            x.resize(n_fft)
            y.resize(n_fft)
            return 1.0 - float(match(x, y, psd=psd, low_frequency_cutoff=BAND_HZ[0],
                                     high_frequency_cutoff=BAND_HZ[1])[0])

        rec.update(mismatch_sref_hp=mismatch(hp, sp), mismatch_sref_hc=mismatch(hc, sc),
                   mismatch_rom_hp=None if rp is None else mismatch(hp, rp),
                   mismatch_rom_hc=None if rp is None else mismatch(hc, rc),
                   step=float(h[0] / h.max()), duration_s=len(hp) * dt, sref_duration_s=len(sp) * dt,
                   residual_rad=float(align["residual_rad"]), residual_ok=bool(align["residual_ok"]),
                   sref_residual_rad=float(align_ref["residual_rad"]), sref_residual_ok=bool(align_ref["residual_ok"]))
    except Exception as exc:  # a row that cannot be evaluated counts as a failure of the gate
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["time_s"] = time.perf_counter() - t0
    return rec


def gate_ok(rec):
    # Separate comparisons, so that a NaN fails.
    return ("error" not in rec and rec["mismatch_sref_hp"] <= GATE_MISMATCH_MAX
            and rec["mismatch_sref_hc"] <= GATE_MISMATCH_MAX and rec["step"] <= GATE_STEP_MAX)


def above_soft(rec):
    return "error" not in rec and not worst_mismatch(rec, "sref") <= GATE_MISMATCH_SOFT


def worst_mismatch(rec, kind):
    values = [rec[f"mismatch_{kind}_{pol}"] for pol in ("hp", "hc")]
    return None if values[0] is None else max(values)


def gate_mass_summary(records):
    """Per total-mass bin; a row with an error counts in n and n_failed but has no values."""
    edges = np.array(GATE_MASS_BINS)
    mass = np.array([r["total_mass"] for r in records])
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rs = [r for r, m in zip(records, mass) if lo <= m < hi or (hi == edges[-1] and m == hi)]
        good = [r for r in rs if "error" not in r]
        sref = [worst_mismatch(r, "sref") for r in good]
        rom = [v for v in (worst_mismatch(r, "rom") for r in good) if v is not None]
        out.append(dict(mass_bin=f"{lo:g}-{hi:g}", n=len(rs), n_failed=sum(not gate_ok(r) for r in rs),
                        n_above_soft=sum(above_soft(r) for r in rs),
                        mismatch_sref_median=float(np.median(sref)) if sref else None,
                        mismatch_sref_max=float(np.max(sref)) if sref else None,
                        mismatch_rom_median=float(np.median(rom)) if rom else None,
                        mismatch_rom_max=float(np.max(rom)) if rom else None,
                        step_max=max((r["step"] for r in good), default=None),
                        sref_duration_max_s=max((r["sref_duration_s"] for r in good), default=None)))
    return out, int(np.sum((mass < edges[0]) | (mass > edges[-1])))


def run_gate_taper(args, df, base):
    require(os.environ.get("LAL_DATA_PATH"), "LAL_DATA_PATH is not set (needed by SEOBNRv4_ROM)")
    require(abs(g1.CONFIG.delta_t - m11.INTRINSIC_DELTA_T) < 1e-12, "delta_t differs from the G3 one")
    require(args.f_lower > SREF_F_LOWER, f"--f-lower must be above the {SREF_F_LOWER:g} Hz of Sref")
    require(args.f_taper_end > args.f_lower, "--f-taper-end must be above --f-lower")
    train_df = df.loc[df.split == "train"]
    if args.rows_from:
        ids = np.array(sorted(json.loads(args.rows_from.read_text())["failed_rows"]), dtype=np.int64)
        require(ids.size > 0, f"No failed rows in {args.rows_from}")
        require(bool(np.isin(ids, train_df.index.to_numpy()).all()), "--rows-from has rows outside train")
        edges = None
    else:
        ids, edges = gate_rows(train_df, args.rows)
    out_dir = base / f"gate_taper_{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    log = Log(out_dir / "gate_taper.log")
    log(f"G-TAPER: {len(ids)} train rows" + (f" from {args.rows_from}" if args.rows_from else "")
        + f", S15 {args.f_lower:g} Hz, taper to {args.f_taper_end:g} Hz, {args.workers} worker(s) -> {out_dir}")
    _CTX.update(frame=train_df, band=(args.f_lower, args.f_taper_end))
    records = []

    def consume(rec):
        records.append(rec)
        if "error" in rec:
            log(f"row {rec['row_index']} M {rec['total_mass']:.2f} {rec['time_s']:.1f} s ERROR {rec['error']}")
        else:
            rom = worst_mismatch(rec, "rom")
            log(f"row {rec['row_index']} M {rec['total_mass']:.2f} {rec['time_s']:.1f} s: mismatch Sref "
                f"{worst_mismatch(rec, 'sref'):.2e}, mismatch ROM {'n/a' if rom is None else f'{rom:.2e}'}, "
                f"step {rec['step']:.2e}, Sref {rec['sref_duration_s']:.1f} s"
                + ("" if gate_ok(rec) else "  FAIL") + ("  >soft" if above_soft(rec) else "")
                + f"  [{len(records)}/{len(ids)}]")

    t0 = time.perf_counter()
    run_tasks(gate_row, [int(i) for i in ids], args.workers, consume)
    good = [r for r in records if "error" not in r]
    failed = [r for r in records if not gate_ok(r)]
    passed = not failed and len(records) == len(ids)
    n_soft = sum(above_soft(r) for r in records)
    soft_fraction = n_soft / max(len(records), 1)
    # A subset never authorizes generation, and its fraction above the soft limit means nothing (the rows
    # were chosen for failing): only the per-row criteria decide PASS_SUBSET.
    if args.rows_from:
        verdict = "PASS_SUBSET" if passed else "FAIL"
    else:
        verdict = ("PASS" if passed and soft_fraction <= GATE_SOFT_FRACTION_MAX and len(ids) == GATE_ROWS
                   else "FAIL")
    worst = [worst_mismatch(r, "sref") for r in good]
    worst_rom = [v for v in (worst_mismatch(r, "rom") for r in good) if v is not None]
    steps = [r["step"] for r in good]
    by_mass, n_outside = gate_mass_summary(records)
    report = dict(verdict=verdict, n_rows=len(records), n_failed=len(failed), n_errors=len(records) - len(good),
                  s15_f_lower=args.f_lower, s15_taper_end_hz=args.f_taper_end, seed=GATE_SEED,
                  rows_from=str(args.rows_from) if args.rows_from else None,
                  sref_f_lower=SREF_F_LOWER, sref_taper_end_hz=SREF_TAPER_END_HZ,
                  decides="mismatch(S15, Sref) and step; mismatch with the ROM is informative (amendment 1)",
                  mismatch_max_allowed=GATE_MISMATCH_MAX, step_max_allowed=GATE_STEP_MAX,
                  mismatch_soft=GATE_MISMATCH_SOFT, soft_fraction_max=GATE_SOFT_FRACTION_MAX,
                  n_above_soft=n_soft, fraction_above_soft=soft_fraction,
                  rows_above_soft=[r["row_index"] for r in records if above_soft(r)],
                  band_hz=list(BAND_HZ), psd="aLIGOZeroDetHighPower", rom_f_lower=args.f_lower,
                  step="|h(t0)| / max|h|, h = h+ - i hx",
                  total_mass_quantile_edges=None if edges is None else edges.tolist(),
                  mismatch_worst=dict(median=float(np.median(worst)), max=float(np.max(worst))) if good else None,
                  mismatch_rom_informative=(dict(median=float(np.median(worst_rom)), max=float(np.max(worst_rom)),
                                                 n_above_1e_3=sum(v > 1e-3 for v in worst_rom))
                                            if worst_rom else None),
                  n_rom_errors=sum("rom_error" in r for r in good), by_total_mass=by_mass,
                  n_outside_mass_bins=n_outside,
                  step_worst=dict(median=float(np.median(steps)), max=float(np.max(steps))) if good else None,
                  failed_rows=[r["row_index"] for r in failed], wall_time_s=time.perf_counter() - t0,
                  script_sha256=sha256_file(__file__), rows=records)
    write_json(out_dir / "gate_taper.json", report)
    for b in by_mass:
        log(f"M {b['mass_bin']}: n {b['n']}, failed {b['n_failed']}, mismatch Sref max {b['mismatch_sref_max']}, "
            f"mismatch ROM max {b['mismatch_rom_max']}, step max {b['step_max']}, "
            f"Sref max {b['sref_duration_max_s']} s")
    log(f"G-TAPER {verdict}: {len(failed)} failed of {len(records)}; max mismatch {report['mismatch_worst']} "
        f"(limit {GATE_MISMATCH_MAX:g}); above {GATE_MISMATCH_SOFT:g}: {n_soft}/{len(records)} = {soft_fraction:.4f} "
        f"(limit {GATE_SOFT_FRACTION_MAX:g}); step {report['step_worst']} (limit {GATE_STEP_MAX:g}) "
        f"-> {out_dir / 'gate_taper.json'}")
    if verdict == "FAIL":
        raise SystemExit("G-TAPER did not pass; stop (pre-registered).")


def require_gate_passed(path):
    """Generation only with a PASS of the 2000 rows, under the current criteria and the band of the generation."""
    path = Path(path).resolve()
    require(path.exists(), f"Missing G-TAPER report: {path}")
    r = json.loads(path.read_text())
    checks = {"verdict PASS": r["verdict"] == "PASS",
              f"{GATE_ROWS} rows": r["n_rows"] == GATE_ROWS,
              "not --rows-from": r.get("rows_from") is None,
              f"f_lower {S15_F_LOWER:g} Hz": r["s15_f_lower"] == S15_F_LOWER,
              f"taper end {S15_TAPER_END_HZ:g} Hz": r["s15_taper_end_hz"] == S15_TAPER_END_HZ,
              "reference Sref": r.get("sref_f_lower") == SREF_F_LOWER,
              "criteria of amendment 2": (r["mismatch_max_allowed"] == GATE_MISMATCH_MAX
                                          and r.get("mismatch_soft") == GATE_MISMATCH_SOFT
                                          and r.get("soft_fraction_max") == GATE_SOFT_FRACTION_MAX
                                          and r["step_max_allowed"] == GATE_STEP_MAX)}
    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        raise SystemExit(f"G-TAPER report not valid for generation ({path}): fails {missing}")
    return path


def s15_record(gate_path=None):
    return dict(f_lower=S15_F_LOWER, taper_end_hz=S15_TAPER_END_HZ, gate_taper_report=str(gate_path))


# ---------------------------------------------------------------- signal


def signal_chunk(indices):
    out = []
    frame = _CTX["frame"].loc[indices]
    for index, row in frame.iterrows():
        t0 = time.perf_counter()
        ctx = block_context(row)
        try:
            # Zero noise as in M13.6: the capture only needs params, t_c, PSDs and the waveform kwargs.
            with g6.RowCapture(keep_noise=False):
                _, _, cap, opt_args = m11.capture_row(row, long_strains=ctx["long_strains"],
                                                      psd_cache=ctx["psd_cache"])
            processing_start = float(row["processing_start"])
            strains, n_before, align = s15_strains(opt_args, cap, processing_start)
            L_s = m11.process_clean(strains, processing_start=processing_start, psds=cap.psds,
                                    injector=ctx["injector"])
            out.append(dict(row_index=int(index), x=g6.to_fp16(L_s, f"L(s) of row {index}"),
                            residual_rad=float(align["residual_rad"]),
                            edge=bool(max(n_before.values()) > 0), time_s=time.perf_counter() - t0))
        except Exception as exc:
            raise RuntimeError(f"Signal failed at {row.source_id} (row {index})") from exc
        if align_abort(out[-1]["residual_rad"]):
            break
    return out


def run_signal(args, root, spec_g1, df, base, attrs_for):
    gate_path = require_gate_passed(args.gate_json)
    train_df = df.loc[df.split == "train"]
    _, blocks = g6.shard_blocks(train_df, args)
    rows = g3.adjust_processing_windows(train_df.loc[train_df.block_id.astype(str).isin(blocks)], spec_g1)
    directory = base
    if args.limit:
        rows = rows.iloc[thin(len(rows), args.limit)]
        directory = base / f"timing_{utc_stamp()}"
    n = len(rows)
    attrs = attrs_for(n, {"selection_policy": "m13_6_train_rows_sorted_block_id_k_mod_num_shards",
                          "shard_index": args.shard, "num_shards": args.num_shards,
                          "shard_block_count": len(blocks), "limit": args.limit or 0})
    psd_files, attrs["raw_psd_files"] = resolve_cached_psd_files(root, rows)
    attrs.update({"m13_13_generator_sha256": sha256_file(__file__), "s15_f_lower": S15_F_LOWER,
                  "s15_taper_end_hz": S15_TAPER_END_HZ, "edge_taper_s": EDGE_TAPER_S,
                  "gate_taper_report": str(gate_path)})
    output = directory / "signal" / f"signal_shard{args.shard:02d}of{args.num_shards:02d}.h5"
    fields = {"X_clean": ((n, 3, g1.CONFIG.length), np.dtype("float16")),
              "row_index": ((n,), np.dtype("int64")),
              "y": ((n, len(LABELS)), np.dtype("float32")),
              "psd_key": ((n,), g6.STR),
              "status": ((n,), np.dtype("int8")),
              "align_residual_rad": ((n,), np.dtype("float32")),
              "edge_tapered": ((n,), np.dtype("int8")),
              "time_s": ((n,), np.dtype("float32"))}
    fixed = {"row_index": rows.index.to_numpy(dtype=np.int64),
             "y": rows[list(LABELS)].to_numpy(dtype=np.float32),
             "psd_key": rows.block_id.astype(str).to_numpy(dtype=object)}
    log = Log(output.with_suffix(".log"))
    log(f"M13.13 signal: {n} train rows, shard {args.shard}/{args.num_shards}, {args.workers} worker(s) -> {output}")

    h = g6.open_output(output, fields, fixed, attrs, args.resume)
    try:
        local = dict(zip(rows.index, range(n)))
        pending = rows.loc[h["status"][:] == 0]
        completed, generated, failed = n - len(pending), 0, None
        progress = Progress(n, completed)
        t0 = time.monotonic()

        def consume(results):
            nonlocal completed, generated, failed
            ok_rows = []
            for rec in results:
                i = local[rec["row_index"]]
                if align_abort(rec["residual_rad"]):
                    failed = rec["row_index"]
                    log(f"row {rec['row_index']} G-ALIGN FAIL: residual {rec['residual_rad']:+.2e} rad")
                    continue
                h["X_clean"][i] = rec["x"]
                h["align_residual_rad"][i] = rec["residual_rad"]
                h["edge_tapered"][i] = rec["edge"]
                h["time_s"][i] = rec["time_s"]
                ok_rows.append(i)
                log(f"row {rec['row_index']} {rec['time_s']:.2f} s: residual {rec['residual_rad']:+.1e} rad, "
                    f"edge {int(rec['edge'])}{align_flag(rec['residual_rad'])}  {progress.tick()}")
            h.flush()
            if ok_rows:
                h["status"][sorted(ok_rows)] = 1
            h.flush()
            completed += len(ok_rows)
            generated += len(ok_rows)
            return failed is None

        if pending.empty:
            log("All rows already written; no samples rewritten.")
        else:
            _CTX.update(frame=rows, psd_files=psd_files)
            run_tasks(signal_chunk, block_chunks(pending), args.workers, consume)
        if failed is not None:
            raise SystemExit(f"G-ALIGN: residual above {ALIGN_ABORT:g} rad at row {failed}; stop (amendment 3).")
        # Also when nothing was pending: a resume must not get past a shard that failed on closing.
        g_align = align_summary(fixed["row_index"], h["align_residual_rad"][:])
        overwrite_json(output.with_suffix(".align.json"), g_align)
        log(align_line(g_align))
        if g_align["stop"]:
            raise SystemExit(f"G-ALIGN: {g_align['stop']}; stop (amendment 3).")
        log(f"Complete: {output}; {(time.monotonic() - t0) / max(generated, 1):.3f} s/row wall ({generated} rows); "
            f"G-EDGE {int(h['edge_tapered'][:].sum())}/{n} rows")
    finally:
        h.close()


# ---------------------------------------------------------------- merge


def run_merge(args):
    t_start = time.time()
    dataset_id = json.loads(args.config.read_text())["dataset_id"]
    processed = dataset_processed_dir(resolve_data_root(cli_data_root=args.data_root), dataset_id)
    base, g3_mm, aug6 = processed / OUTPUT_SUBDIR, processed / "fp16_memmap", processed / "m13_6_aug" / "memmap"
    out_dir = base / "memmap"
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists, refusing to overwrite: {out_dir}")
    paths = shard_paths(base / "signal", "signal", args.num_shards)
    row_ids = np.load(g3_mm / "row_ids_train.npy")
    y_train = np.load(g3_mm / "y_train.npy", mmap_mode="r")
    if not (np.diff(row_ids) > 0).all():
        raise SystemExit("G3 memmap row_ids_train is not strictly increasing")
    man6 = json.loads((aug6 / "manifest.json").read_text())
    # m13_6_aug/memmap holds no row ids: its order is the G3 row_ids_train of its manifest.
    row_ids_sha = sha256_file(g3_mm / "row_ids_train.npy")
    if man6["g3_memmap"]["row_ids_train_sha256"] != row_ids_sha:
        raise SystemExit("m13_6_aug was merged against another row_ids_train.npy")
    tmp_dir = out_dir.with_name(f"memmap.tmp_{os.getpid()}")
    tmp_dir.mkdir(parents=False, exist_ok=False)

    reference, seen = {}, np.zeros(row_ids.size, dtype=bool)
    keys, generators, bands = np.empty(row_ids.size, dtype=object), set(), set()
    with h5py.File(paths[0], "r") as h:
        _, C, T = h["X_clean"].shape
    X = np.lib.format.open_memmap(tmp_dir / "X_clean_train.npy", mode="w+", dtype=np.float16,
                                  shape=(row_ids.size, C, T))
    stats = Stats(C)
    residuals, merged_ids, n_edge = [], [], 0
    for path in paths:
        with h5py.File(path, "r") as h:
            reference = check_shard(h, path, "m13_13_signal", reference)
            generators.add(h.attrs["m13_13_generator_sha256"])
            bands.add((float(h.attrs["s15_f_lower"]), float(h.attrs["s15_taper_end_hz"]),
                       str(h.attrs["gate_taper_report"])))
            ri = h["row_index"][:]
            pos = np.searchsorted(row_ids, ri)
            if (pos >= row_ids.size).any() or not np.array_equal(row_ids[np.minimum(pos, row_ids.size - 1)], ri):
                raise SystemExit(f"{path}: row_index not in G3 row_ids_train")
            if seen[pos].any() or not (np.diff(pos) > 0).all():
                raise SystemExit(f"{path}: duplicated or unordered rows")
            if not np.array_equal(h["y"][:], y_train[pos]):
                raise SystemExit(f"{path}: y differs from G3 memmap y_train")
            seen[pos] = True
            keys[pos] = h["psd_key"].asstr()[:]
            for lo in range(0, ri.size, 1024):
                x = h["X_clean"][lo:lo + 1024]
                stats.add(x)
                X[pos[lo:lo + 1024]] = x
            residuals.append(h["align_residual_rad"][:])
            merged_ids.append(ri)
            stop = align_summary(ri, residuals[-1])["stop"]
            if stop:
                raise SystemExit(f"{path}: G-ALIGN: {stop}")
            n_edge += int(h["edge_tapered"][:].sum())
        print(f"  merged {path.name}: {ri.size} rows | {time.time() - t_start:.0f} s", flush=True)
    if not seen.all():
        raise SystemExit(f"{int((~seen).sum())} G3 train rows missing from the signal shards")
    if len(generators) != 1:
        raise SystemExit("Signal shards come from different versions of this script")
    if len(bands) != 1 or next(iter(bands))[:2] != (S15_F_LOWER, S15_TAPER_END_HZ):
        raise SystemExit(f"Signal shards have another S15 band: {bands}")
    g_align = align_summary(np.concatenate(merged_ids), np.concatenate(residuals))
    X.flush()
    del X
    keys = keys.astype(str)
    if not np.array_equal(keys, np.load(aug6 / "psd_key_train.npy")):
        raise SystemExit("psd_key per row differs from m13_6_aug: rows are not in its order")
    np.save(tmp_dir / "psd_key_train.npy", keys)
    # Relative links, valid after the rename and if the data root moves. The trainer only np.load()s them.
    for name in ("bank_X.npy", "bank_psd_key.npy"):
        os.symlink(os.path.relpath(aug6 / name, out_dir), tmp_dir / name)

    manifest = {
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "script_sha256": sha256_file(__file__),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "dataset_id": dataset_id,
        "content": "L(s_S15) before the per-sample per-detector z-score; noise bank of M13.6 by symlink",
        "s15": s15_record(next(iter(bands))[2]),
        "shard_provenance": reference,
        "signal_shards": [stat_entry(p) for p in paths],
        # Read by the trainer (load_noise_aug): compared with m13_13_g3t/fp16_memmap/row_ids_train.npy.
        "g3_memmap": {"dir": str(g3_mm), "row_ids_train_sha256": row_ids_sha},
        "x_dtype": "float16",
        "X_clean_train": {"shape": [int(row_ids.size), int(C), int(T)],
                          "row_order": "G3 fp16_memmap row_ids_train (= m13_6_aug/memmap, psd_key checked)",
                          "per_detector": stats.report()},
        "bank_X": {**man6["bank_X"], "source": str(aug6 / "bank_X.npy"), "linked": True},
        "G_ALIGN": g_align,
        "G_EDGE": {"taper_s": EDGE_TAPER_S, "beta": EDGE_BETA, "split": "train", "n_rows_tapered": n_edge},
        "outputs_sha256": {name: sha256_file(tmp_dir / name) for name in ("X_clean_train.npy", "psd_key_train.npy")},
        "bank_outputs_sha256": {k: v for k, v in man6.get("outputs_sha256", {}).items() if k.startswith("bank_")},
    }
    manifest["total_time_s"] = time.time() - t_start
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if out_dir.exists():
        raise SystemExit(f"Output directory appeared during merge; result left in {tmp_dir}")
    os.rename(tmp_dir, out_dir)
    print(f"done in {manifest['total_time_s']:.0f} s -> {out_dir}; G-EDGE {n_edge}/{row_ids.size} rows; "
          f"{align_line(g_align)}", flush=True)


# ---------------------------------------------------------------- noisy


def monitor_positions(n_train):
    text = TRAINER.read_text()
    require(f"TRAIN_MONITOR_N = {TRAIN_MONITOR_N}\n" in text and f"TRAIN_MONITOR_SEED = {TRAIN_MONITOR_SEED}\n" in text,
            "Monitor constants differ from train_m13_3.py")
    return np.sort(np.random.default_rng(TRAIN_MONITOR_SEED).choice(n_train, TRAIN_MONITOR_N, replace=False))


def noisy_row(row, index, ctx, trainio_h):
    """X of G3T for one row, with G-REC, G-NOISE and G-ALIGN (sequence of m13_12.rebuild_pair)."""
    t0 = time.perf_counter()
    injector = ctx["injector"]
    with _NoiseTap() as nt:
        X_row, _, cap, opt_args = m11.capture_row(row, long_strains=ctx["long_strains"], psd_cache=ctx["psd_cache"])
    require(nt.noise is not None, f"Row {index}: noise not captured")
    require(np.array_equal(trainio_h["y"][index], row[list(LABELS)].to_numpy(dtype=np.float32)),
            f"Trainio y differs at row {index}")
    processing_start = float(row["processing_start"])
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

    # X_T: same noise, PSD, injection, processing and z-score; only the signal is S15.
    s_T, n_before, align = s15_strains(opt_args, cap, processing_start)
    inj_T = injector.inject_network(noises=nt.noise, signals=s_T)
    raw_T, X_T = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(
        strains={ifo: inj_T[ifo].strain for ifo in DETECTORS}, psds=cap.psds))

    # G-NOISE: the processed noise is the same in both inputs (L linear, g6.check_linear_processing).
    dn_A = np.asarray(raw_A, dtype=np.float64) - m11.process_clean(
        s_A, processing_start=processing_start, psds=cap.psds, injector=injector)
    dn_T = np.asarray(raw_T, dtype=np.float64) - m11.process_clean(
        s_T, processing_start=processing_start, psds=cap.psds, injector=injector)
    x16 = g6.to_fp16(np.asarray(X_T, dtype=np.float32), f"X_T row {index}")
    rec = dict(row_index=int(index), block_id=str(row["block_id"]), G_REC=drec,
               G_NOISE=float(np.abs(dn_A - dn_T).max() / np.abs(dn_A).max()),
               G_ALIGN=float(align["residual_rad"]),
               edge_samples=n_before, edge=bool(max(n_before.values()) > 0),
               is_truncated=bool(trainio_h["is_truncated"][index]),
               max_abs_x=np.abs(np.asarray(X_T, dtype=np.float64)).max(axis=1).tolist(),
               time_s=time.perf_counter() - t0)
    return rec, x16


def noisy_fails(rec):
    fails = []
    if not rec["G_REC"] <= GREC_MAX:
        fails.append("G-REC")
    if align_abort(rec["G_ALIGN"]):
        fails.append("G-ALIGN")
    if not rec["G_NOISE"] <= GDIFF_RTOL:  # also catches NaN
        fails.append("G-NOISE")
    return fails


def noisy_chunk(indices):
    out = []
    frame = _CTX["frame"].loc[indices]
    with h5py.File(_CTX["trainio"], "r") as t:
        for index, row in frame.iterrows():
            rec, x16 = noisy_row(row, index, block_context(row), t)
            out.append((rec, x16))
            if noisy_fails(rec):
                break
    return out


def noisy_summary(records, split, wall_s):
    times = [r["time_s"] for r in records]
    return dict(split=split, n_rows=len(records), wall_time_s=wall_s,
                time_s_per_row=dict(median=float(np.median(times)), mean=float(np.mean(times)),
                                    max=float(np.max(times))),
                G_REC=dict(threshold=GREC_MAX, max=max(r["G_REC"] for r in records)),
                G_NOISE=dict(threshold=GDIFF_RTOL, max=max(r["G_NOISE"] for r in records)),
                G_ALIGN=align_summary([r["row_index"] for r in records], [r["G_ALIGN"] for r in records]),
                G_EDGE=dict(taper_s=EDGE_TAPER_S, beta=EDGE_BETA, n_rows_tapered=sum(r["edge"] for r in records),
                            n_rows_truncated_in_g3=sum(r["is_truncated"] for r in records)))


def publish_noisy(split, name, row_ids, positions, work, out_dir, src_dir, x_shape, trainio, summary):
    """Move X.npy to its final name and add the split to manifest.json (run the splits one after another)."""
    out_dir.mkdir(exist_ok=True)
    for stem in ("row_ids", "y"):
        shutil.copyfile(src_dir / f"{stem}_{name}.npy", out_dir / f"{stem}_{name}.npy")
    overwrite_json(out_dir / f"gates_{split}.json", summary)
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else dict(
        kind="m13_13_g3t_inputs", dataset_id=trainio.parent.name, x_dtype="float16", y_dtype="float32 (unstandardized)",
        x="zscore(process_network(inject(n, S15)), psd_proc); n, psd_proc, projection, t_c and processing of G3",
        source_hdf5=dict(path=str(trainio)), splits={})
    manifest["splits"][name] = dict(
        n_rows=int(row_ids.size), shape_X=list(x_shape), n_rows_written=int(positions.size),
        row_ids_sha256=sha256_file(out_dir / f"row_ids_{name}.npy"), script_sha256=sha256_file(__file__),
        s15=summary["s15"],
        contents=("monitor rows only (train_m13_3.py); all other rows are zeros (sparse file)"
                  if split == "train40k" else "all rows"), gates=f"gates_{split}.json")
    tmp = path.with_name(f"manifest.json.tmp_{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, path)
    # Last: X_<split>.npy existing means the split is fully published. Everything above can be repeated.
    os.rename(work / "X.npy", out_dir / f"X_{name}.npy")


def run_noisy(args, root, spec_g1, df, trainio, base, attrs_for):
    gate_path = require_gate_passed(args.gate_json)
    name, mm_name = NOISY_SPLITS[args.split]
    src_dir, out_dir = trainio.parent / mm_name, base / mm_name
    row_ids = np.load(src_dir / f"row_ids_{name}.npy")
    require(bool(df.loc[row_ids, "split"].eq(name).all()), f"row_ids_{name}.npy has rows of other splits")
    positions = monitor_positions(row_ids.size) if args.split == "train40k" else np.arange(row_ids.size)
    final_x = out_dir / f"X_{name}.npy"
    require(not final_x.exists(), f"Already published, refusing to overwrite: {final_x}")
    if args.limit:
        require(not args.resume, "--limit does not resume")
        positions = positions[thin(positions.size, args.limit)]
        work = base / f"timing_{utc_stamp()}" / f"noisy_{args.split}"
    else:
        work = base / f"work_noisy_{args.split}"
    with h5py.File(trainio, "r") as t:
        x_shape = (row_ids.size,) + tuple(t["X"].shape[1:])

    meta = dict(split=args.split, script_sha256=sha256_file(__file__), x_shape=list(x_shape),
                n_positions=int(positions.size), limit=args.limit, s15=s15_record(gate_path))
    if args.resume:
        require((work / "meta.json").exists(), f"Nothing to resume in {work}")
        require(json.loads((work / "meta.json").read_text()) == meta, "Resume with another script or selection")
        X = np.lib.format.open_memmap(work / "X.npy", mode="r+")
        done = np.lib.format.open_memmap(work / "done.npy", mode="r+")
    else:
        require(not work.exists(), f"{work} exists; use --resume")
        work.mkdir(parents=True)
        write_json(work / "meta.json", meta)
        # w+ extends the file without writing data: a sparse file for train40k.
        X = np.lib.format.open_memmap(work / "X.npy", mode="w+", dtype=np.float16, shape=x_shape)
        done = np.lib.format.open_memmap(work / "done.npy", mode="w+", dtype=np.uint8, shape=(row_ids.size,))

    log = Log(work / "build.log")
    pending = positions[done[positions] == 0]
    log(f"M13.13 noisy {args.split}: {positions.size} rows, {pending.size} pending, {args.workers} worker(s) -> {work}")
    pos = {int(row_ids[p]): int(p) for p in positions}
    records = {}
    jsonl = work / "rows.jsonl"
    if jsonl.exists():  # a row can appear twice after a crash; the last record wins
        with jsonl.open() as f:
            for line in f:
                rec = json.loads(line)
                records[rec["row_index"]] = rec
    t0 = time.perf_counter()
    failed = None
    progress = Progress(positions.size, positions.size - pending.size)

    if pending.size:  # a resume after the last row has nothing to build, only the publication
        frame = g3.adjust_processing_windows(df.loc[[int(row_ids[p]) for p in pending]], spec_g1)
        psd_files, _ = resolve_cached_psd_files(root, frame)
        _CTX.update(frame=frame, psd_files=psd_files, trainio=trainio)
        rows_file = jsonl.open("a")
        generated = 0

        def consume(results):
            # Order on disk: X, then its gate record, then done. An interrupted chunk is simply redone.
            nonlocal failed, generated
            ok = [(rec, x16) for rec, x16 in results if not noisy_fails(rec)]
            for rec, x16 in ok:
                X[pos[rec["row_index"]]] = x16
            X.flush()
            for rec, _ in ok:
                records[rec["row_index"]] = rec
                rows_file.write(json.dumps(rec) + "\n")
                log(f"row {rec['row_index']} {rec['time_s']:.2f} s: G-REC {rec['G_REC']:.1e}, G-NOISE "
                    f"{rec['G_NOISE']:.1e}, residual {rec['G_ALIGN']:+.1e} rad, edge {int(rec['edge'])}"
                    f"{align_flag(rec['G_ALIGN'])}  {progress.tick()}")
            rows_file.flush()
            os.fsync(rows_file.fileno())
            done[[pos[rec["row_index"]] for rec, _ in ok]] = 1
            done.flush()
            generated += len(ok)
            for rec, _ in results:
                fails = noisy_fails(rec)
                if fails:
                    failed = dict(row_index=rec["row_index"], gates=fails, record=rec)
                    log(f"row {rec['row_index']} FAIL {fails}")
            return failed is None

        run_tasks(noisy_chunk, block_chunks(frame), args.workers, consume)
        rows_file.close()

    X.flush()
    if failed is not None:
        overwrite_json(work / "gates_FAIL.json", failed)
        raise SystemExit(f"Gate failed at row {failed['row_index']}: {failed['gates']}; stop (pre-registered).")
    require(bool(done[positions].all()), "Rows left unwritten")
    selected = [records[int(row_ids[p])] for p in positions]
    summary = noisy_summary(selected, args.split, time.perf_counter() - t0)
    log(align_line(summary["G_ALIGN"]))
    if summary["G_ALIGN"]["stop"]:
        overwrite_json(work / "gates_FAIL.json", summary["G_ALIGN"])
        raise SystemExit(f"G-ALIGN: {summary['G_ALIGN']['stop']}; stop (amendment 3).")
    summary["s15"] = s15_record(gate_path)
    summary["provenance"] = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                             for k, v in attrs_for(positions.size, {"selection_policy": f"m13_13_noisy_{args.split}",
                                                                    "limit": args.limit or 0}).items()}
    log(f"Gates PASS {args.split}: G-REC {summary['G_REC']['max']:.1e}, G-NOISE {summary['G_NOISE']['max']:.1e}, "
        f"G-EDGE {summary['G_EDGE']['n_rows_tapered']}/{positions.size}, "
        f"{summary['time_s_per_row']['mean']:.2f} s/row (mean, one worker)")
    del X, done
    if args.limit:
        write_json(work / f"gates_{args.split}.json", summary)
    else:
        publish_noisy(args.split, name, row_ids, positions, work, out_dir, src_dir, x_shape, trainio, summary)
        log(f"Published {final_x}")


# ---------------------------------------------------------------- consistency


def run_consistency(args, root, spec_g1, df, trainio, base):
    X_clean = np.load(base / "memmap" / "X_clean_train.npy", mmap_mode="r")
    X_noisy = np.load(base / "fp16_memmap" / "X_train.npy", mmap_mode="r")
    row_ids = np.load(base / "fp16_memmap" / "row_ids_train.npy")
    pick = np.sort(np.random.default_rng(CONSISTENCY_SEED).choice(monitor_positions(row_ids.size), CONSISTENCY_ROWS,
                                                                  replace=False))
    out_dir = base / f"consistency_{utc_stamp()}"
    out_dir.mkdir(exist_ok=False)
    pos = {int(row_ids[p]): int(p) for p in pick}
    rows = g3.adjust_processing_windows(df.loc[[int(row_ids[p]) for p in pick]], spec_g1)
    psd_files, _ = resolve_cached_psd_files(root, rows)
    _CTX.update(psd_files=psd_files)
    normalize = g1.normalize_input_per_sample_per_detector_zscore

    def fp16_tol(x16):
        # Recomputed vs stored float16: one float16 step, for FFT differences between machines.
        return np.spacing(np.abs(x16)).astype(np.float64) + 1e-7

    report = []
    for index, row in rows.sort_values("block_id", kind="stable").iterrows():
        ctx = block_context(row)
        with _NoiseTap() as nt:
            _, _, cap, opt_args = m11.capture_row(row, long_strains=ctx["long_strains"], psd_cache=ctx["psd_cache"])
        processing_start = float(row["processing_start"])
        s_T, _, _ = s15_strains(opt_args, cap, processing_start)
        L_s = m11.process_clean(s_T, processing_start=processing_start, psds=cap.psds, injector=ctx["injector"])
        L_n = g6.stack_raw(g1.PROCESSOR.process_network(
            strains={ifo: nt.noise[ifo].copy() for ifo in DETECTORS}, psds=cap.psds))
        inj_T = ctx["injector"].inject_network(noises=nt.noise, signals=s_T)
        _, X_T = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(
            strains={ifo: inj_T[ifo].strain for ifo in DETECTORS}, psds=cap.psds))
        p = pos[int(index)]
        stored_clean = np.asarray(X_clean[p], dtype=np.float16)
        stored_x = np.asarray(X_noisy[p], dtype=np.float16)
        X_sum = normalize(L_s + L_n, eps=M10_INPUT_ZSCORE_EPS)
        X_sum16 = normalize(stored_clean.astype(np.float64) + L_n.astype(np.float16).astype(np.float64),
                            eps=M10_INPUT_ZSCORE_EPS)
        rec = dict(row_index=int(index), position=p,
                   max_abs_dX=float(np.abs(X_sum - np.asarray(X_T, dtype=np.float64)).max()),
                   stored_signal_ok=bool(np.all(np.abs(L_s - stored_clean.astype(np.float64)) <= fp16_tol(stored_clean))),
                   stored_noisy_ok=bool(np.all(np.abs(np.asarray(X_T, dtype=np.float64) - stored_x.astype(np.float64))
                                               <= fp16_tol(stored_x))),
                   max_abs_dX_fp16_informative=float(np.abs(X_sum16 - stored_x.astype(np.float64)).max()))
        report.append(rec)
        print(f"row {index}: max|dX| {rec['max_abs_dX']:.2e}, stored signal {rec['stored_signal_ok']}, stored X "
              f"{rec['stored_noisy_ok']}, fp16 path {rec['max_abs_dX_fp16_informative']:.2e}", flush=True)
    worst = max(r["max_abs_dX"] for r in report)
    ok = worst <= CONSISTENCY_MAX and all(r["stored_signal_ok"] and r["stored_noisy_ok"] for r in report)
    write_json(out_dir / "consistency_report.json", dict(
        verdict="PASS" if ok else "FAIL", max_abs_dX=worst, threshold=CONSISTENCY_MAX, seed=CONSISTENCY_SEED,
        s15=s15_record(require_gate_passed(json.loads((base / "memmap" / "manifest.json").read_text())
                                            ["s15"]["gate_taper_report"])),
        comparison="float64: zscore(L(s_S15) + L(n of G3)) vs the X of the noisy path, plus recomputed L(s_S15) and X "
                   "equal to the stored float16 within one float16 step. The threshold cannot apply to the stored "
                   "float16 themselves (step ~1e-3 at |x| ~ 1); that path is reported as max_abs_dX_fp16_informative.",
        max_abs_dX_fp16_informative=max(r["max_abs_dX_fp16_informative"] for r in report),
        script_sha256=sha256_file(__file__), rows=report))
    print(f"Consistency {'PASS' if ok else 'FAIL'}: max|dX| {worst:.2e} -> {out_dir}", flush=True)
    if not ok:
        raise SystemExit("Consistency check failed; stop (pre-registered).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="G3 generation config")
    common.add_argument("--data-root", type=Path)
    common.add_argument("--manifest", type=Path,
                        help="Override G1 manifest_path; still verified against G1 manifest_sha256.")
    common.add_argument("--allow-version-mismatch", action="store_true")
    workers = argparse.ArgumentParser(add_help=False)
    workers.add_argument("--workers", type=int, default=1, help=f"Worker processes (fork), at most {MAX_WORKERS}")
    run = argparse.ArgumentParser(add_help=False, parents=[common, workers])
    run.add_argument("--limit", type=int, help="Only N evenly spaced rows, written to timing_<stamp>/")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--gate-json", type=Path, required=True, help="gate_taper.json with PASS (2000 rows, same band)")

    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate-taper", parents=[common, workers])
    gate.add_argument("--rows", type=int, default=GATE_ROWS, help="Fewer rows than the default never give PASS")
    gate.add_argument("--f-lower", type=float, default=S15_F_LOWER, help="S15 start frequency (Hz), above 10")
    gate.add_argument("--f-taper-end", type=float, default=S15_TAPER_END_HZ, help="S15 taper end frequency (Hz)")
    gate.add_argument("--rows-from", type=Path, help="gate_taper.json: evaluate only its failed_rows")
    signal = sub.add_parser("signal", parents=[run])
    signal.add_argument("--shard", type=int, default=0)
    signal.add_argument("--num-shards", type=int, default=1)
    merge = sub.add_parser("merge")
    merge.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    merge.add_argument("--data-root", type=Path)
    merge.add_argument("--num-shards", type=int, required=True)
    noisy = sub.add_parser("noisy", parents=[run])
    noisy.add_argument("--split", choices=sorted(NOISY_SPLITS), required=True)
    sub.add_parser("consistency", parents=[common])
    sub.add_parser("status", help="Handled before the project imports; only takes --data-root")
    args = parser.parse_args()

    if args.command == "merge":
        run_merge(args)
        return
    require(1 <= getattr(args, "workers", 1) <= MAX_WORKERS, f"--workers must be in 1..{MAX_WORKERS}")
    require(getattr(args, "limit", None) is None or args.limit >= 1, "Invalid --limit")
    if args.command == "signal":
        require(args.num_shards >= 1 and 0 <= args.shard < args.num_shards, "Invalid shard index")
        require(not (args.limit and args.resume), "--limit does not resume")
    args.mode = f"m13_13_{args.command}"  # recorded by g6 attrs_for; read by merge as m13_6_mode

    root, spec_g1, df, trainio, directory, attrs_for = g6.setup(args)
    base = directory.parent / OUTPUT_SUBDIR
    if args.command == "gate-taper":
        run_gate_taper(args, df, base)
    elif args.command == "signal":
        run_signal(args, root, spec_g1, df, base, attrs_for)
    elif args.command == "noisy":
        run_noisy(args, root, spec_g1, df, trainio, base, attrs_for)
    else:
        run_consistency(args, root, spec_g1, df, trainio, base)


if __name__ == "__main__":
    main()
