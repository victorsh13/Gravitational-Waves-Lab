#!/usr/bin/env python3
"""M15 G4: proposal, gates and training inputs of the targets T_R and T_F. Local CPU.

Protocol: docs/experiments/M15_G4_prior.md (pre-registered 20-11-2026), sections 2 and 3.

Rows are a new manifest with the schema of the M11-final G1 manifest (scripts/prepare_m11_final_manifest.py),
so the G3T path is reused unchanged: build_g1_row with the G3 physics (g6.setup, as M13.13), capture_row,
s15_strains, process_clean, gate_row, signal_chunk, run_signal and run_consistency of m13_13_build_g3t.py.
Only the sources change. Masses: section 2.2 (T_R) and 2.3 (T_F); everything else as G1, in own rngs.

Proposal per split (seed 151101..151104 for train, val, cal, test; streams SeedSequence([seed, k])):
  masses (k=1)     n uniforms "component is F" (< 0.5), then n uniforms w1, then n uniforms w2.
                   R: m1, m2 = sorted(5 + 120 w1, 5 + 120 w2). F: Mc = Mc_min + (Mc_max - Mc_min) w1 and q by
                   numerical inversion (Newton on a cubic-Hermite table) of the CDF of p(q | Mc) ∝ m1(Mc, q)^2 on
                   [q_min(Mc), 1] at w2; q strictly inside the support, 5 <= m2 <= m1 <= 125 checked.
  acceptance (k=2) one u per row, common to both targets: accepted by T_R if u < a_R, by T_F if u < a_F = 1 - a_R,
                   a_R = p_R / (p_R + p_F) of the mixture (never the component label). Cut: the first N accepted
                   rows of each target in proposal order (800k/40k/30k/30k).
  parameters (k=3) ParameterSampler with the G1 PriorConfig, one sample_one() per row: same rng draw order as G1;
                   its two masses are drawn and discarded, spins, angles and the 1000 Mpc distance are kept.
  snr (k=4)        uniform [10, 25), as G1 (uniform(*range, size=n)).
  assignment (k=5) sample_assignments of prepare_m11_final_manifest.py (environment, block, crop) on the frozen bank
                   of the split; in train only blocks present in m13_6_aug/memmap/bank_psd_key.npy.
Row id = source_index = index in the concatenated proposal (train, val, cal, test), so a row shared by both targets
has the same id. placement_seed = 150000000 + source_index, gaussian_seed = 160000000 + source_index.

Subcommands (outputs under <data_root>/processed/M15_G4/; nothing is overwritten):
  sample       proposal/{train,val,cal,test}.npz (theta of every proposal row; full manifest columns only for rows
               in at least one cut), proposal/prior_grid.npz, proposal/sample_report.json with G-PRIOR (2.5.1).
               Stops with an error if G-PRIOR fails (the report is kept).
  gate-taper   G-TAPER-G4 (2.5.2): m13_13.gate_row and gate_ok on 2000 train rows of the union, stratified
               (default_rng(151105)): 600 with M_tot in (180, 250] (300 per bin 180-220 and 220-250, including the
               20 heaviest, at least 75 per bin with chi_eff < -0.5) and 1400 with M_tot <= 180 in 10 quantile strata.
               gate_taper/gate_taper.json: the fields m13_13.require_gate_passed reads, plus per-bin summaries
               (bins of M13.13 plus 180-220 and 220-250), overall and for chi_eff < -0.5, with failed rows and errors.
  signal       m13_13.run_signal unchanged on the train rows of the union: signal/signal_shardIIofNN.h5.
  noisy        X = zscore(L(s_S15 + n)), float16, for --split val, cal, test (union of both cuts) or train40k (union of
               the monitor rows monitor_positions(800000) of each target): work_noisy_<split>/, published as
               noisy_union/{X,y,row_ids}_<split>.npy, rows_<split>.jsonl, gates_<split>.json. Variant of
               m13_13.noisy_row without G-REC (there is no G3 row to rebuild); G-NOISE, G-ALIGN and G-EDGE kept.
  materialize  --target R|F: <target>/ in the format train_m13_3.py reads (phase A, P1): fp16_memmap/ (X_train
               sparse with the monitor rows only, val), fp16_memmap_heldout/ (cal, test), memmap/ (X_clean_train,
               psd_key_train, manifest.json, relative links bank_X.npy, bank_psd_key.npy to the M13.6 bank), the split
               npz; rows are copied. The G1 label stats of model 5a are linked (not recomputed) at M15_G4/<name>.
  consistency  --target R|F: m13_13.run_consistency unchanged on <target>/ (64 monitor rows).
  status       read only, without the heavy imports.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

DATASET_ID = "M15_G4"
TARGETS = ("R", "F")
NUM_SHARDS_RUN = 4  # of scripts/m15/run_m15_g4.sh
STATUS_STEPS = (("sample", "proposal/sample_report.json"), ("gate_taper", "gate_taper/gate_taper.json"),
                ("noisy_val", "noisy_union/X_val.npy"), ("noisy_train40k", "noisy_union/X_train40k.npy"),
                *((f"signal_shard{i}", f"signal/signal_shard{i:02d}of{NUM_SHARDS_RUN:02d}.h5")
                  for i in range(NUM_SHARDS_RUN)),
                ("noisy_cal", "noisy_union/X_cal.npy"), ("noisy_test", "noisy_union/X_test.npy"),
                *((f"{step}_{t}", out.format(t=t)) for t in TARGETS
                  for step, out in (("materialize", "{t}/materialize_manifest.json"),
                                    ("consistency", "{t}/consistency_*/consistency_report.json"))))
STATUS_IDLE_S = 900


def run_status(data_root):
    """Standard library only, as m13_13 status; logs in <data_root>/logs/m15/m15_g4_<step>.{log,done}."""
    data_root = data_root or os.environ.get("CBC_PE_DATA_ROOT")
    if not data_root:
        raise SystemExit("Use --data-root or set CBC_PE_DATA_ROOT")
    base = Path(data_root) / "processed" / DATASET_ID
    logs = Path(data_root) / "logs" / "m15"
    for step, output in STATUS_STEPS:
        log = logs / f"m15_g4_{step}.log"
        if (logs / f"m15_g4_{step}.done").exists():
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
    _status = argparse.ArgumentParser(prog="m15_build_g4.py status")
    _status.add_argument("--data-root")
    run_status(_status.parse_args(sys.argv[2:]).data_root)
    sys.exit(0)

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.parameters import CBCParameters  # noqa: E402
from src.paths import dataset_processed_dir, resolve_data_root, resolve_processed_artifact  # noqa: E402
from src.sampling import ParameterSampler, PriorConfig  # noqa: E402
import scripts.generate_m11_final_G1 as g1  # noqa: E402
from scripts.generate_m11_final_G1 import DETECTORS, LABELS, require, resolve_cached_psd_files, sha256_file  # noqa: E402
import scripts.m13.generate_m13_G3_whitening as g3  # noqa: E402
import scripts.m13.generate_m13_6_clean as g6  # noqa: E402
import scripts.m13.m13_11_prepare_rows as m11  # noqa: E402
# Importing m13_13 also sets the S15 band (15 Hz, taper to 25 Hz) in m13_11, as when M13.13 runs.
import scripts.m13.m13_13_build_g3t as g3t  # noqa: E402
from scripts.m13.generate_m13_G2_band import config_sha256  # noqa: E402
from scripts.m13.merge_m13_6_clean import Stats, check_shard, shard_paths, stat_entry  # noqa: E402
from scripts.prepare_m11_final_manifest import PARAMETERS, load_bank, sample_assignments  # noqa: E402

GENERATOR = "m15_build_g4"
PREREG = PROJECT_ROOT / "docs/experiments/M15_G4_prior.md"
G1_CONFIG = PROJECT_ROOT / "configs/generation/generate_m11_final_G1_500k.json"
G3_CONFIG = m11.DEFAULT_CONFIG  # configs/generation/generate_m13_G3_band1024_w16.json, as M13.13
REFERENCE_TRAIN_CONFIG = PROJECT_ROOT / "configs/experiments/m14/train_m14_5a_4aG3T_fullcov_s123.json"

SPLITS = ("train", "val", "cal", "test")
N_PROPOSAL = {"train": 1620000, "val": 82000, "cal": 62000, "test": 62000}
N_TARGET = {"train": 800000, "val": 40000, "cal": 30000, "test": 30000}
SPLIT_SEED = {"train": 151101, "val": 151102, "cal": 151103, "test": 151104}
STREAMS = {"masses": 1, "acceptance": 2, "parameters": 3, "snr": 4, "assignment": 5}
GATE_SEED_G4 = 151105
# Integer seeds of build_g1_row: base + source_index. G1 (and G3, G3T, which reuse its manifest) use
# [11000000, 11500000) and [21000000, 21500000); these ranges hold 1826000 rows and are disjoint from them
# and from each other (checked in sample). The M13.6 bank uses SeedSequence([key_hash, 136101, k]) instead.
PLACEMENT_BASE = 150000000
GAUSSIAN_BASE = 160000000
SOURCE_ID_PREFIX = "M15G4_SRC_"
EXPERIMENT_ID = "M15_G4"
SNR_RANGE = (10.0, 25.0)
REFERENCE_DISTANCE_MPC = 1000.0
EXTRINSIC = ("distance", "inclination", "ra", "dec", "spin_1z", "spin_2z", "polarization_angle")

M_LO, M_HI = 5.0, 125.0
G3T_M_HI = 90.0
Q_FLOOR = M_LO / M_HI  # smallest q of the triangle 5 <= m2 <= m1 <= 125
P_R = 2.0 / (M_HI - M_LO) ** 2
GRID_Q_CELLS = 1 << 16  # cells of the table of F in s = ln q over [ln 0.04, 0]
GRID_GL_NODES = 8       # Gauss-Legendre nodes per cell
GRID_MC_NODES = 40001   # trapezoid nodes in Mc for the numerical CDFs (the corner Mc(125, 5) is a node)
GRID_X_NODES = 4001     # nodes of each numerical CDF, interpolated linearly at the samples
BISECTION_STEPS = 64
NEWTON_STEPS = 8
NEWTON_RESIDUAL_MAX = 1e-10
CDF_SELF_CHECK_MAX = 1e-5  # implementation check of the numerical CDFs against the analytic ones of T_R
KS_P_MIN = 1e-3
ACCEPT_SIGMAS = 5.0
SQUARE_FRACTION = (0.500, 0.506)

GATE_HIGH_M = 180.0
GATE_HIGH_BINS = ((180.0, 220.0), (220.0, 250.0))
GATE_HIGH_PER_BIN = 300
GATE_NEG_CHI = -0.5
GATE_NEG_PER_BIN = 75
GATE_LOW_STRATA = 10
GATE_MIN_HIGH = 600
GATE_MIN_NEG_HIGH = 150
GATE_MASS_BINS_G4 = tuple(g3t.GATE_MASS_BINS) + (220.0, 250.0)

NOISY_DIR = "noisy_union"
NOISY_SPLITS_G4 = ("val", "cal", "test", "train40k")
HELDOUT_MEMMAP = {"val": "fp16_memmap", "cal": "fp16_memmap_heldout", "test": "fp16_memmap_heldout"}
COPY_CHUNK = 1024


def chirp_mass(m1, m2):
    # Same expression as CBCParameters.chirp_mass.
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


MC_MIN, MC_MAX, MC_CORNER = chirp_mass(M_LO, M_LO), chirp_mass(M_HI, M_HI), chirp_mass(M_HI, M_LO)
U_MC = 1.0 / (MC_MAX - MC_MIN)  # density u of T_F in Mc

SPEC = {
    "experiment_id": EXPERIMENT_ID, "dataset_id": DATASET_ID, "pre_registration": str(PREREG.relative_to(PROJECT_ROOT)),
    "proposal": "0.5 p_R + 0.5 p_F; component drawn per row", "n_proposal": N_PROPOSAL, "n_target": N_TARGET,
    "split_seeds": SPLIT_SEED, "streams": STREAMS, "rng": "default_rng(SeedSequence([split_seed, stream]))",
    "row_id": "index in the concatenated proposal train, val, cal, test = source_index",
    "placement_base": PLACEMENT_BASE, "gaussian_base": GAUSSIAN_BASE, "source_id_prefix": SOURCE_ID_PREFIX,
    "component_mass_range": [M_LO, M_HI], "chirp_mass_range": [MC_MIN, MC_MAX],
    "T_R": "m1, m2 iid U[5, 125], sorted", "T_F": "p_R(m1, m2) u(Mc) / p_R,Mc(Mc)",
    "snr_range": list(SNR_RANGE), "reference_distance_mpc": REFERENCE_DISTANCE_MPC,
    "grid": {"q_cells": GRID_Q_CELLS, "gl_nodes": GRID_GL_NODES, "mc_nodes": GRID_MC_NODES, "x_nodes": GRID_X_NODES,
             "bisection_steps": BISECTION_STEPS, "newton_steps": NEWTON_STEPS},
    "g_prior": {"ks_p_min": KS_P_MIN, "accept_sigmas": ACCEPT_SIGMAS, "square_fraction": list(SQUARE_FRACTION)},
    "gate_taper": {"seed": GATE_SEED_G4, "rows": g3t.GATE_ROWS, "high_bins": [list(b) for b in GATE_HIGH_BINS],
                   "high_per_bin": GATE_HIGH_PER_BIN, "neg_chi_eff": GATE_NEG_CHI, "neg_per_bin": GATE_NEG_PER_BIN,
                   "top_mass_rows": g3t.GATE_TOP_MASS_ROWS, "low_strata": GATE_LOW_STRATA,
                   "mass_bins": list(GATE_MASS_BINS_G4)},
    "g1_config": str(G1_CONFIG.relative_to(PROJECT_ROOT)), "g3_config": str(Path(G3_CONFIG).relative_to(PROJECT_ROOT)),
    "reference_train_config": str(REFERENCE_TRAIN_CONFIG.relative_to(PROJECT_ROOT)),
}


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def write_json(path, payload):
    m11.write_json(path, jsonable(payload))  # exclusive


def sha_bytes(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------- prior (sections 2.2-2.4)


def log_f(s):
    """ln(m1 / Mc) at q = e^s: m1 = Mc (1 + q)^(1/5) q^(-3/5); decreasing in s."""
    return 0.2 * np.log1p(np.exp(s)) - 0.6 * s


def log_g(s):
    """ln(m2 / Mc) = ln(q m1 / Mc); increasing in s."""
    return 0.2 * np.log1p(np.exp(s)) + 0.4 * s


def log_h(s):
    """ln(M_tot / Mc) = (6/5) ln(1 + q) - (3/5) ln q; decreasing in s."""
    return 1.2 * np.log1p(np.exp(s)) - 0.6 * s


def dF_ds(s):
    # |d(m1, m2)/d(Mc, q)| = m1^2 / Mc, so with uniform masses p(q | Mc) ∝ m1^2 ∝ (1+q)^(2/5) q^(-6/5);
    # in s = ln q the integrand gets the factor q.
    q = np.exp(s)
    return (1.0 + q) ** 0.4 * q ** -0.2


class PriorGrid:
    """F(s) = int_{ln Q_FLOOR}^{s} dF_ds: composite Gauss-Legendre table, cubic Hermite (exact derivative) between nodes."""

    def __init__(self):
        self.s0 = float(np.log(Q_FLOOR))
        self.h = -self.s0 / GRID_Q_CELLS
        self.s = self.s0 + self.h * np.arange(GRID_Q_CELLS + 1)
        self.s[-1] = 0.0
        x, w = np.polynomial.legendre.leggauss(GRID_GL_NODES)
        mid = self.s[:-1] + 0.5 * self.h
        cell = (dF_ds(mid[:, None] + 0.5 * self.h * x[None, :]) * w[None, :]).sum(axis=1) * 0.5 * self.h
        self.F = np.concatenate([[0.0], np.cumsum(cell)])
        self.d = dF_ds(self.s)
        self.F1 = float(self.F[-1])
        self.lnf, self.lnh = log_f(self.s), log_h(self.s)

    def F_at(self, s):
        s = np.asarray(s, dtype=np.float64)
        i = np.clip(np.floor((s - self.s0) / self.h).astype(np.int64), 0, GRID_Q_CELLS - 1)
        t = (s - self.s[i]) / self.h
        h00, h10 = (1 + 2 * t) * (1 - t) ** 2, t * (1 - t) ** 2
        h01, h11 = t * t * (3 - 2 * t), t * t * (t - 1)
        return h00 * self.F[i] + h10 * self.h * self.d[i] + h01 * self.F[i + 1] + h11 * self.h * self.d[i + 1]

    def s_min(self, mc):
        """ln q_min(Mc) = max(root of m1 = 125, root of m2 = 5), by bisection on [ln 0.04, 0] (upper bracket ends)."""
        mc = np.asarray(mc, dtype=np.float64)
        ya, yb = np.log(M_HI / mc), np.log(M_LO / mc)
        lo_a, hi_a = np.full(mc.shape, self.s0), np.zeros(mc.shape)
        lo_b, hi_b = lo_a.copy(), hi_a.copy()
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo_a + hi_a)
            up = log_f(mid) > ya  # m1 > 125 at mid: the root is above
            lo_a, hi_a = np.where(up, mid, lo_a), np.where(up, hi_a, mid)
            mid = 0.5 * (lo_b + hi_b)
            up = log_g(mid) < yb  # m2 < 5 at mid: the root is above
            lo_b, hi_b = np.where(up, mid, lo_b), np.where(up, hi_b, mid)
        return np.maximum(hi_a, hi_b)

    def p_r_mc(self, mc):
        """Marginal of Mc under T_R: (2/120^2) int_{q_min}^1 m1^2 / Mc dq = P_R Mc (F(0) - F(s_min))."""
        mc = np.asarray(mc, dtype=np.float64)
        return P_R * mc * (self.F1 - self.F_at(self.s_min(mc)))

    def a_r(self, mc):
        # p_F / p_R = u(Mc) / p_R,Mc(Mc), so a_R = p_R / (p_R + p_F) depends on Mc only.
        p = self.p_r_mc(mc)
        return p / (p + U_MC)

    def sample_q(self, mc, v):
        """q of T_F given Mc at the uniform v: F(s) = F(s_min) + v (F(0) - F(s_min)); Newton from table inversion."""
        smin = self.s_min(mc)
        fmin = self.F_at(smin)
        target = fmin + v * (self.F1 - fmin)
        s = np.clip(np.interp(target, self.F, self.s), smin, 0.0)
        for _ in range(NEWTON_STEPS):
            s = np.clip(s - (self.F_at(s) - target) / dF_ds(s), smin, 0.0)
        resid = float(np.abs(self.F_at(s) - target).max()) if s.size else 0.0
        require(resid <= NEWTON_RESIDUAL_MAX, f"q inversion did not converge (residual {resid:.3g})")
        require(bool(((s > smin) & (s < 0.0)).all()), "q of T_F not strictly inside (q_min(Mc), 1)")
        return s

    def masses_f(self, mc, v):
        s = self.sample_q(mc, v)
        m1 = mc * np.exp(log_f(s))
        return m1, np.exp(s) * m1


def mc_nodes():
    """Trapezoid nodes and weights in Mc, with the kink Mc(125, 5) of q_min(Mc) as a node."""
    k1 = int(round((MC_CORNER - MC_MIN) / (MC_MAX - MC_MIN) * (GRID_MC_NODES - 1))) + 1
    x = np.concatenate([np.linspace(MC_MIN, MC_CORNER, k1), np.linspace(MC_CORNER, MC_MAX, GRID_MC_NODES - k1 + 1)[1:]])
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return x, w


def numerical_cdfs(grid):
    """CDFs of q, m1 and M_tot under T_R and T_F: int w_T(Mc) P(. <= x | Mc) dMc, with p(q | Mc) shared (2.3)."""
    mc, wt = mc_nodes()
    smin = grid.s_min(mc)
    fmin = grid.F_at(smin)
    den = grid.F1 - fmin
    degenerate = den <= 1e-14  # Mc = Mc_min: q = 1 only
    den = np.where(degenerate, 1.0, den)
    weights = {"R": wt * grid.p_r_mc(mc), "F": wt * U_MC}
    xs = {"q": np.linspace(Q_FLOOR, 1.0, GRID_X_NODES), "mass_1": np.linspace(M_LO, M_HI, GRID_X_NODES),
          "total_mass": np.linspace(2 * M_LO, 2 * M_HI, GRID_X_NODES)}
    out = {t: {} for t in TARGETS}
    for name, x in xs.items():
        vals = {t: np.empty(x.size) for t in TARGETS}
        for lo in range(0, x.size, 50):
            xc = x[lo:lo + 50]
            if name == "q":
                sx = np.log(xc)[:, None]
                P = (grid.F_at(np.clip(sx, smin[None, :], 0.0)) - fmin[None, :]) / den[None, :]
                P = np.where(degenerate[None, :], (sx >= smin[None, :]).astype(float), P)
            else:
                table, at_one = (grid.lnf, log_f(0.0)) if name == "mass_1" else (grid.lnh, log_h(0.0))
                y = np.log(xc[:, None] / mc[None, :])
                # m1 (or M_tot) <= x  <=>  q >= q*(Mc, x), the function being decreasing in q.
                s_star = np.interp(y.ravel(), table[::-1], grid.s[::-1]).reshape(y.shape)
                P = (grid.F1 - grid.F_at(np.clip(s_star, smin[None, :], 0.0))) / den[None, :]
                P = np.where(degenerate[None, :], (y >= at_one).astype(float), P)
            for t in TARGETS:
                vals[t][lo:lo + 50] = P @ weights[t]
        for t in TARGETS:
            out[t][name] = (x, vals[t])
    cum = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(mc) * (grid.p_r_mc(mc)[1:] + grid.p_r_mc(mc)[:-1]))])
    out["R"]["chirp_mass"] = (mc, cum)
    return out


def analytic_cdf_r(name, x):
    d = M_HI - M_LO
    if name == "mass_1":
        return ((x - M_LO) / d) ** 2
    if name == "total_mass":
        s = x - 2 * M_LO
        return np.where(s <= d, s ** 2 / (2 * d * d), 1.0 - (2 * d - s) ** 2 / (2 * d * d))
    if name == "q":  # area of {m2 <= x m1} in the triangle, x >= 5/125
        return P_R * (x * M_HI ** 2 / 2 + M_LO ** 2 / (2 * x) - M_LO * M_HI)
    raise ValueError(name)


def ks_test(sample, cdf):
    from scipy.stats import kstwo
    x = np.sort(np.asarray(sample, dtype=np.float64))
    n = x.size
    c = cdf(x)
    i = np.arange(1, n + 1, dtype=np.float64)
    d = float(max((i / n - c).max(), (c - (i - 1) / n).max()))
    return {"n": int(n), "D": d, "p": float(kstwo.sf(d, n))}


# ---------------------------------------------------------------- sample


def g1_reference(root):
    """G1 config (sources, bank, prior) checked against the G1 HDF5, its manifest blocks per split, and the bank keys."""
    spec = json.loads(G1_CONFIG.read_text())
    m13 = g3.load_m13_config(G3_CONFIG)
    g1_path = root / "processed" / m13["base_g1_h5"]
    with h5py.File(g1_path, "r") as h:
        attrs = {k: h.attrs[k] for k in ("config_sha256", "manifest_path", "manifest_sha256")}
    require(config_sha256(spec) == attrs["config_sha256"], f"{G1_CONFIG} is not the config of {g1_path}")
    manifest = Path(str(attrs["manifest_path"]))
    require(sha256_file(manifest) == attrs["manifest_sha256"], f"G1 manifest hash differs: {manifest}")
    blocks = pd.read_csv(manifest, usecols=["block_id", "split"])
    g1_blocks = {s: set(blocks.loc[blocks.split == s, "block_id"].astype(str)) for s in SPLITS}
    bank_path = root / "processed" / m13["dataset_id"] / g6.OUTPUT_SUBDIR / "memmap" / "bank_psd_key.npy"
    bank_keys = np.unique(np.load(bank_path))
    return spec, m13, g1_path, manifest, g1_blocks, bank_path, bank_keys


def cut_mask(accepted, n):
    idx = np.flatnonzero(accepted)
    require(idx.size >= n, f"only {idx.size} accepted rows for a cut of {n}")
    mask = np.zeros(accepted.size, dtype=bool)
    mask[idx[:n]] = True
    return mask


def propose_split(split, offset, grid, prior, crops):
    n, seed = N_PROPOSAL[split], SPLIT_SEED[split]
    rng = {k: np.random.default_rng(np.random.SeedSequence([seed, v])) for k, v in STREAMS.items()}
    comp_f = rng["masses"].random(n) < 0.5
    w1, w2 = rng["masses"].random(n), rng["masses"].random(n)
    m1, m2 = np.empty(n), np.empty(n)
    r = ~comp_f
    ma, mb = M_LO + (M_HI - M_LO) * w1[r], M_LO + (M_HI - M_LO) * w2[r]
    m1[r], m2[r] = np.maximum(ma, mb), np.minimum(ma, mb)
    m1[comp_f], m2[comp_f] = grid.masses_f(MC_MIN + (MC_MAX - MC_MIN) * w1[comp_f], w2[comp_f])
    require(bool(((M_LO <= m2) & (m2 <= m1) & (m1 <= M_HI)).all()), f"{split}: masses outside 5 <= m2 <= m1 <= 125")
    mc = chirp_mass(m1, m2)
    a_r = grid.a_r(mc)
    u = rng["acceptance"].random(n)
    acc = {"R": u < a_r, "F": u < 1.0 - a_r}
    cut = {t: cut_mask(acc[t], N_TARGET[split]) for t in TARGETS}

    sampler = ParameterSampler(prior_config=prior, rng=rng["parameters"])
    ext = {k: np.empty(n) for k in EXTRINSIC}
    for i in range(n):
        p = sampler.sample_one()  # same draws as G1; its masses are discarded
        for k in EXTRINSIC:
            ext[k][i] = getattr(p, k)
    require(bool((ext["distance"] == REFERENCE_DISTANCE_MPC).all()), "Reference distance is not 1000 Mpc")
    snr = rng["snr"].uniform(*SNR_RANGE, size=n)
    counts = {s: (n if s == split else 0) for s in SPLITS}
    assign = sample_assignments(crops, counts, np.random.SeedSequence([seed, STREAMS["assignment"]]))
    require(bool(assign["split"].eq(split).all()), f"{split}: crop of another split")

    ids = offset + np.arange(n, dtype=np.int64)
    theta = {"row_id": ids, "component_F": comp_f, "mass_1": m1, "mass_2": m2, "chirp_mass": mc, "q": m2 / m1,
             "total_mass": m1 + m2, "a_R": a_r, "u": u, "acc_R": acc["R"], "acc_F": acc["F"],
             "in_R": cut["R"], "in_F": cut["F"]}
    sel = np.flatnonzero(cut["R"] | cut["F"])
    # Preallocated float64 columns instead of a list of per-row dicts (memory); same values and column order.
    cols = {name: np.empty(sel.size, dtype=np.float64) for name in (*PARAMETERS, "reference_distance_mpc")}
    for j, i in enumerate(sel):
        p = CBCParameters(mass_1=float(m1[i]), mass_2=float(m2[i]),
                          **{k: float(ext[k][i]) for k in EXTRINSIC})
        require(p.mass_1 == m1[i] and p.mass_2 == m2[i], f"row {ids[i]}: masses reordered")
        for name in PARAMETERS:
            cols[name][j] = float(getattr(p, name))
        cols["reference_distance_mpc"][j] = float(p.distance)
    frame = pd.DataFrame(cols)
    del cols
    frame["source_index"] = ids[sel]
    frame["manifest_index"] = ids[sel]
    frame["source_id"] = [f"{SOURCE_ID_PREFIX}{i:07d}" for i in ids[sel]]
    frame["target_network_snr"] = snr[sel]
    frame = pd.concat([frame, assign.iloc[sel].reset_index(drop=True)], axis=1)
    frame["reference_geocentric_time"] = frame["center_gps"]
    frame["geocentric_time_policy"] = "m11_6_random_contained_reproject"
    frame["placement_seed"] = PLACEMENT_BASE + frame.source_index
    frame["gaussian_seed"] = GAUSSIAN_BASE + frame.source_index
    frame["experiment_id"] = EXPERIMENT_ID
    frame["config_sha256"] = config_sha256(SPEC)
    frame["component_F"] = comp_f[sel]
    frame["u"], frame["a_R"] = u[sel], a_r[sel]
    frame["in_R"], frame["in_F"] = cut["R"][sel], cut["F"][sel]
    require(bool(frame.target_network_snr.between(*SNR_RANGE).all()), "Invalid SNR target")
    return theta, frame


def save_split(path, theta, frame):
    payload = {f"theta__{k}": np.asarray(v) for k, v in theta.items()}
    for c in frame.columns:
        a = frame[c].to_numpy()
        payload[f"rows__{c}"] = a.astype(str) if a.dtype == object else a
    with open(path, "xb") as fh:
        np.savez(fh, **payload)


def noise_seed_audit(frame, spec_g1):
    """Informative: the integer seeds are disjoint; the 32-bit detector seeds drawn from them can still coincide."""
    def detector_seeds(seeds):
        out = np.empty((len(seeds), len(DETECTORS)), dtype=np.uint64)
        for k, s in enumerate(seeds):
            r = np.random.default_rng(int(s))
            out[k] = [int(r.integers(0, 2 ** 32 - 1)) for _ in DETECTORS]  # as build_g1_row
        return out
    g4 = detector_seeds(frame.gaussian_seed.to_numpy())
    gen = spec_g1["generation"]
    ref = detector_seeds(gen["seeds"]["gaussian_base"] + np.arange(gen["num_samples"]))
    intra = int(sum(len(set(r)) < len(DETECTORS) for r in g4))
    _, counts = np.unique(g4.ravel(), return_counts=True)
    return {"rows": int(len(frame)), "rows_with_equal_detector_seeds": intra,
            "repeated_detector_seed_values_within_g4": int((counts > 1).sum()),
            "detector_seed_values_shared_with_g1": int(np.intersect1d(g4.ravel(), ref.ravel()).size),
            "note": "32-bit seeds of noise_from_psd; a row with equal detector seeds stops build_g1_row"}


def run_sample(args):
    root = resolve_data_root(cli_data_root=args.data_root)
    base = dataset_processed_dir(root, DATASET_ID)
    out = base / "proposal"
    require(not out.exists(), f"Output exists, refusing to overwrite: {out}")
    t0 = time.time()
    spec_g1, m13, g1_path, g1_manifest, g1_blocks, bank_path, bank_keys = g1_reference(root)
    gen = spec_g1["generation"]
    n_total = sum(N_PROPOSAL.values())
    ranges = {"g1_placement": (gen["seeds"]["placement_base"], gen["seeds"]["placement_base"] + gen["num_samples"]),
              "g1_gaussian": (gen["seeds"]["gaussian_base"], gen["seeds"]["gaussian_base"] + gen["num_samples"]),
              "g4_placement": (PLACEMENT_BASE, PLACEMENT_BASE + n_total),
              "g4_gaussian": (GAUSSIAN_BASE, GAUSSIAN_BASE + n_total)}
    names = sorted(ranges)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            require(ranges[a][1] <= ranges[b][0] or ranges[b][1] <= ranges[a][0], f"Seed ranges {a}, {b} overlap")

    crops = load_bank(spec_g1, root)
    train_blocks = set(crops.loc[crops.split == "train", "block_id"].astype(str))
    require(set(bank_keys.astype(str)) <= train_blocks, "bank_psd_key has keys that are not train blocks")
    allowed = crops.loc[(crops.split != "train") | crops.block_id.astype(str).isin(set(bank_keys.astype(str)))]
    bank_restriction = {
        "train_blocks_in_bank": int(len(set(bank_keys.astype(str)))), "train_blocks_in_crop_bank": len(train_blocks),
        "train_environments_before": int(crops.loc[crops.split == "train", "file_group_id"].nunique()),
        "train_environments_after": int(allowed.loc[allowed.split == "train", "file_group_id"].nunique()),
        "train_crops_before": int((crops.split == "train").sum()),
        "train_crops_after": int((allowed.split == "train").sum())}
    prior = PriorConfig.from_dict(spec_g1["parameter_sampler"])
    grid = PriorGrid()
    print(f"G4 sample: grid F(1) = {grid.F1:.12g}, Mc in [{MC_MIN:.6g}, {MC_MAX:.6g}]", flush=True)

    tmp = base / f"proposal.tmp_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=False)
    thetas, frames, offset = {}, {}, 0
    for split in SPLITS:
        ts = time.time()
        thetas[split], frames[split] = propose_split(split, offset, grid, prior, allowed)
        used = set(frames[split].block_id.astype(str))
        require(used <= g1_blocks[split], f"{split}: blocks not used by G1 (no cached PSD strain): "
                                          f"{sorted(used - g1_blocks[split])[:5]}")
        save_split(tmp / f"{split}.npz", thetas[split], frames[split])
        offset += N_PROPOSAL[split]
        print(f"  {split}: {N_PROPOSAL[split]} proposed, {len(frames[split])} rows in a cut, "
              f"{time.time() - ts:.0f} s", flush=True)

    # G-PRIOR (2.5.1)
    cdfs = numerical_cdfs(grid)
    self_check = {name: float(np.abs(cdfs["R"][name][1] - analytic_cdf_r(name, cdfs["R"][name][0])).max())
                  for name in ("q", "mass_1", "total_mass")}
    self_check["chirp_mass_normalization"] = float(abs(cdfs["R"]["chirp_mass"][1][-1] - 1.0))
    require(max(self_check.values()) <= CDF_SELF_CHECK_MAX, f"Numerical CDF self-check failed: {self_check}")

    def cdf_of(target, name):
        if target == "F" and name == "chirp_mass":
            return lambda x: (x - MC_MIN) / (MC_MAX - MC_MIN)
        if target == "R" and name != "chirp_mass":
            return lambda x: analytic_cdf_r(name, x)
        gx, gy = cdfs[target][name]
        return lambda x: np.interp(x, gx, gy)

    th = thetas["train"]
    ks = {t: {name: ks_test(th[name][th[f"in_{t}"]], cdf_of(t, name))
              for name in ("chirp_mass", "q", "total_mass", "mass_1")} for t in TARGETS}
    acceptance = {}
    for split in SPLITS:
        n = N_PROPOSAL[split]
        sigma = float(np.sqrt(0.25 / n))
        for t in TARGETS:
            rate = float(thetas[split][f"acc_{t}"].mean())
            acceptance[f"{t}_{split}"] = {"rate": rate, "sigma": sigma, "z": (rate - 0.5) / sigma,
                                          "ok": abs(rate - 0.5) <= ACCEPT_SIGMAS * sigma}
    in_r = th["in_R"]
    square = float(((th["mass_1"][in_r] <= G3T_M_HI) & (th["mass_2"][in_r] <= G3T_M_HI)).mean())
    checks = {"ks": all(v["p"] > KS_P_MIN for t in TARGETS for v in ks[t].values()),
              "acceptance": all(v["ok"] for v in acceptance.values()),
              "square_fraction_T_R": SQUARE_FRACTION[0] <= square <= SQUARE_FRACTION[1]}
    verdict = "PASS" if all(checks.values()) else "FAIL"

    grid_path = tmp / "prior_grid.npz"
    with open(grid_path, "xb") as fh:
        np.savez(fh, s=grid.s, F=grid.F, mc_nodes=mc_nodes()[0],
                 **{f"cdf_{t}_{k}_x": v[0] for t in TARGETS for k, v in cdfs[t].items()},
                 **{f"cdf_{t}_{k}_y": v[1] for t in TARGETS for k, v in cdfs[t].items()})
    union = pd.concat([frames[s] for s in SPLITS], ignore_index=True)
    shared = {s: {"R_and_F": int((thetas[s]["in_R"] & thetas[s]["in_F"]).sum()),
                  "union": int((thetas[s]["in_R"] | thetas[s]["in_F"]).sum())} for s in SPLITS}
    report = {
        "generator": GENERATOR, "script_sha256": sha256_file(__file__), "spec": SPEC,
        "spec_sha256": config_sha256(SPEC), "pre_registration_sha256": sha256_file(PREREG),
        "inputs": {"g1_config": str(G1_CONFIG), "g1_config_sha256": sha256_file(G1_CONFIG),
                   "g1_hdf5": str(g1_path), "g1_manifest": str(g1_manifest),
                   "g1_manifest_sha256": sha256_file(g1_manifest), "crop_bank_sha256": spec_g1["bank"]["sha256"],
                   "bank_psd_key": str(bank_path), "bank_psd_key_sha256": sha256_file(bank_path),
                   "sampling_sha256": sha256_file(PROJECT_ROOT / "src/sampling.py"),
                   "prepare_manifest_sha256": sha256_file(PROJECT_ROOT / "scripts/prepare_m11_final_manifest.py")},
        "prior_config_g1": spec_g1["parameter_sampler"], "seed_ranges": ranges,
        "bank_restriction": bank_restriction, "grid": {"F1": grid.F1, "mc_corner": MC_CORNER,
                                                       "sha256": sha256_file(grid_path)},
        "row_id_offsets": {s: int(sum(N_PROPOSAL[x] for x in SPLITS[:SPLITS.index(s)])) for s in SPLITS},
        "rows_in_cut": shared, "noise_seed_audit": noise_seed_audit(union, spec_g1),
        "G_PRIOR": {"verdict": verdict, "checks": checks, "ks_train": ks, "acceptance": acceptance,
                    "square_fraction_T_R_train": square, "cdf_self_check": self_check},
        "outputs_sha256": {f"{s}.npz": sha256_file(tmp / f"{s}.npz") for s in SPLITS},
        "total_time_s": time.time() - t0,
    }
    write_json(tmp / "sample_report.json", report)
    os.rename(tmp, out)
    print(f"G-PRIOR {verdict}: {checks}; square {square:.4f}; "
          f"KS p min {min(v['p'] for t in TARGETS for v in ks[t].values()):.3g} -> {out}", flush=True)
    if verdict != "PASS":
        raise SystemExit("G-PRIOR did not pass; stop (pre-registered).")


# ---------------------------------------------------------------- common setup of the generation subcommands


class Context:
    pass


def load_proposal(base):
    report_path = base / "proposal" / "sample_report.json"
    require(report_path.exists(), f"Missing {report_path}; run sample first")
    report = json.loads(report_path.read_text())
    require(report["G_PRIOR"]["verdict"] == "PASS", "G-PRIOR did not pass")
    thetas, frames = {}, []
    for split in SPLITS:
        path = base / "proposal" / f"{split}.npz"
        require(sha256_file(path) == report["outputs_sha256"][f"{split}.npz"], f"{path} changed after sample")
        with np.load(path) as z:
            thetas[split] = {k[len("theta__"):]: z[k] for k in z.files if k.startswith("theta__")}
            frames.append(pd.DataFrame({k[len("rows__"):]: z[k] for k in z.files if k.startswith("rows__")}))
    frame = pd.concat(frames, ignore_index=True)
    frame.index = frame.source_index.to_numpy(dtype=np.int64)
    require(frame.index.is_unique and bool((np.diff(frame.index) > 0).all()), "Row ids not unique and ascending")
    return report, sha256_file(report_path), thetas, frame


def cut_ids(ctx, target, split):
    th = ctx.thetas[split]
    return th["row_id"][th[f"in_{target}"]].astype(np.int64)  # ascending (proposal order)


def monitor_ids(ctx, target):
    return cut_ids(ctx, target, "train")[g3t.monitor_positions(N_TARGET["train"])]


def setup(args):
    """G3 physics exactly as M13.13 (g6.setup on the G3 config), then the G4 proposal."""
    args.config, args.manifest = G3_CONFIG, None
    args.mode = f"m15_{args.command}"  # recorded by g6 attrs_for as m13_6_mode
    root, spec_g1, g1_frame, trainio, _, attrs_for = g6.setup(args)
    # The G1 manifest is only validated by g6.setup; attrs_for does not close over it and g6 keeps no global copy.
    del g1_frame
    ctx = Context()
    ctx.root, ctx.spec_g1, ctx.trainio, ctx.attrs_for_g3 = root, spec_g1, trainio, attrs_for
    ctx.base = dataset_processed_dir(root, DATASET_ID)
    ctx.report, ctx.report_sha, ctx.thetas, ctx.frame = load_proposal(ctx.base)
    g6.check_psd_key_per_block(ctx.frame)
    # Memory before the fork (workers inherit the parent): keep only the rows the subcommand reads, in the same
    # (ascending id) order. materialize and consistency keep every row. The constant columns below are not read
    # by build_g1_row, capture_row, signal_chunk, run_signal, run_consistency, attrs_for or this script.
    if args.command in ("gate-taper", "signal"):
        keep = (ctx.frame.split == "train").to_numpy()
    elif args.command == "noisy":
        keep = ctx.frame.index.isin(noisy_row_ids(ctx, args.split))
    else:
        keep = None
    if keep is not None:
        ctx.frame = ctx.frame.loc[keep]
    ctx.frame = ctx.frame.drop(columns=["experiment_id", "config_sha256", "geocentric_time_policy"])
    gc.collect()
    return ctx


def log_memory(ctx):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: kB
    print(f"M15 G4 {len(ctx.frame)} rows in ctx.frame; parent max RSS {rss_mb:.0f} MB", flush=True)


def require_g4_gate(path, ctx):
    """m13_13 criteria (PASS of 2000 rows, S15 band, amendment 2) plus: this script, this proposal, this location."""
    path = g3t.require_gate_passed(path)
    r = json.loads(path.read_text())
    require(r.get("generator") == GENERATOR, f"{path} is not a G-TAPER-G4 report")
    require(r.get("proposal_report_sha256") == ctx.report_sha, f"{path} was made for another proposal")
    require(path == (ctx.base / "gate_taper" / "gate_taper.json").resolve(), f"{path} is not the M15_G4 gate")
    return path


def attrs_for_g4(ctx, gate_path):
    def attrs_for(n_rows, selection):
        attrs = ctx.attrs_for_g3(n_rows, selection)
        attrs.update({"m15_generator_sha256": sha256_file(__file__), "m15_proposal_report_sha256": ctx.report_sha,
                      "m15_dataset_id": DATASET_ID, "m15_gate_taper_report": str(gate_path),
                      "m15_note": "G1/G3 attrs describe the reused physics; the rows are the M15_G4 proposal"})
        return attrs
    return attrs_for


# ---------------------------------------------------------------- gate-taper (2.5.2)


def gate_rows_g4(train):
    mass = train.total_mass.to_numpy(dtype=np.float64)
    neg = train.chi_eff.to_numpy(dtype=np.float64) < GATE_NEG_CHI
    ids = train.index.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(GATE_SEED_G4)
    taken = np.zeros(ids.size, dtype=bool)
    taken[np.argsort(mass, kind="stable")[-g3t.GATE_TOP_MASS_ROWS:]] = True
    bins = [(mass > lo) & (mass < hi) if lo == GATE_HIGH_M else (mass >= lo) & (mass <= hi)
            for lo, hi in GATE_HIGH_BINS]
    def draw(pool, k):
        require(k >= 0 and int(pool.sum()) >= k, "Not enough rows for a stratum of the gate")
        taken[rng.choice(np.flatnonzero(pool), k, replace=False)] = True

    for in_bin in bins:  # first the chi_eff < -0.5 rows of the bin, then the rest of the bin
        draw(in_bin & neg & ~taken, max(GATE_NEG_PER_BIN - int((taken & in_bin & neg).sum()), 0))
        draw(in_bin & ~taken, GATE_HIGH_PER_BIN - int((taken & in_bin).sum()))
    low = (mass <= GATE_HIGH_M) & ~taken
    n_low = g3t.GATE_ROWS - int(taken.sum())
    edges = np.quantile(mass[low], np.linspace(0, 1, GATE_LOW_STRATA + 1))
    strata = np.clip(np.searchsorted(edges[1:-1], mass, side="right"), 0, GATE_LOW_STRATA - 1)
    for s in range(GATE_LOW_STRATA):
        draw(low & (strata == s), n_low // GATE_LOW_STRATA + int(s < n_low % GATE_LOW_STRATA))
    high = mass > GATE_HIGH_M
    info = {"n_rows": int(taken.sum()), "n_total_mass_above_180": int((taken & high).sum()),
            "n_chi_eff_below_-0.5_and_total_mass_above_180": int((taken & high & neg).sum()),
            "per_high_bin": [int((taken & b).sum()) for b in bins], "low_quantile_edges": edges.tolist()}
    require(info["n_rows"] == g3t.GATE_ROWS and info["n_total_mass_above_180"] >= GATE_MIN_HIGH
            and info["n_chi_eff_below_-0.5_and_total_mass_above_180"] >= GATE_MIN_NEG_HIGH,
            f"Gate selection does not meet 2.5.2: {info}")
    return np.sort(ids[taken]), info


def gate_bins(records, keep):
    edges = GATE_MASS_BINS_G4
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rs = [r for r in records if keep(r) and (lo <= r["total_mass"] < hi or (hi == edges[-1] and r["total_mass"] == hi))]
        good = [r for r in rs if "error" not in r]
        sref = [g3t.worst_mismatch(r, "sref") for r in good]
        bad = [r for r in rs if not g3t.gate_ok(r)]
        out.append(dict(mass_bin=f"{lo:g}-{hi:g}", n=len(rs), n_failed=len(bad), n_errors=len(rs) - len(good),
                        n_above_soft=sum(g3t.above_soft(r) for r in rs),
                        mismatch_sref_max=float(np.max(sref)) if sref else None,
                        step_max=max((r["step"] for r in good), default=None),
                        failed=[dict(row_index=r["row_index"], total_mass=r["total_mass"], chi_eff=r["chi_eff"],
                                     error=r.get("error")) for r in bad]))
    return out


def run_gate_g4(args, ctx):
    require(os.environ.get("LAL_DATA_PATH"), "LAL_DATA_PATH is not set (needed by SEOBNRv4_ROM)")
    require(abs(g1.CONFIG.delta_t - m11.INTRINSIC_DELTA_T) < 1e-12, "delta_t differs from the G3 one")
    out_dir = ctx.base / "gate_taper"
    out_dir.mkdir(exist_ok=False)
    train = ctx.frame.loc[ctx.frame.split == "train"]
    ids, selection = gate_rows_g4(train)
    log = g3t.Log(out_dir / "gate_taper.log")
    band = (g3t.S15_F_LOWER, g3t.S15_TAPER_END_HZ)
    log(f"G-TAPER-G4: {len(ids)} train rows {selection}, S15 {band[0]:g} Hz, taper to {band[1]:g} Hz, "
        f"{args.workers} worker(s) -> {out_dir}")
    g3t._CTX.update(frame=train, band=band)
    records = []

    def consume(rec):
        records.append(rec)
        status = f"ERROR {rec['error']}" if "error" in rec else (
            f"mismatch Sref {g3t.worst_mismatch(rec, 'sref'):.2e}, step {rec['step']:.2e}"
            + ("" if g3t.gate_ok(rec) else "  FAIL"))
        log(f"row {rec['row_index']} M {rec['total_mass']:.2f} chi_eff {rec['chi_eff']:+.2f} {rec['time_s']:.1f} s: "
            f"{status}  [{len(records)}/{len(ids)}]")

    t0 = time.perf_counter()
    g3t.run_tasks(g3t.gate_row, [int(i) for i in ids], args.workers, consume)
    good = [r for r in records if "error" not in r]
    failed = [r for r in records if not g3t.gate_ok(r)]
    n_soft = sum(g3t.above_soft(r) for r in records)
    soft_fraction = n_soft / max(len(records), 1)
    passed = not failed and len(records) == len(ids) == g3t.GATE_ROWS
    verdict = "PASS" if passed and soft_fraction <= g3t.GATE_SOFT_FRACTION_MAX else "FAIL"
    worst = [g3t.worst_mismatch(r, "sref") for r in good]
    steps = [r["step"] for r in good]
    report = dict(
        verdict=verdict, generator=GENERATOR, proposal_report_sha256=ctx.report_sha, n_rows=len(records),
        n_failed=len(failed), n_errors=len(records) - len(good), rows_from=None,
        s15_f_lower=band[0], s15_taper_end_hz=band[1], seed=GATE_SEED_G4, selection=selection,
        sref_f_lower=g3t.SREF_F_LOWER, sref_taper_end_hz=g3t.SREF_TAPER_END_HZ,
        decides="mismatch(S15, Sref) and step (m13_13.gate_ok); mismatch with the ROM is informative",
        generates="per row A (SEOBNRv4_opt from 30 Hz, phase and time reference), S15 and Sref (m13_13.gate_row)",
        mismatch_max_allowed=g3t.GATE_MISMATCH_MAX, step_max_allowed=g3t.GATE_STEP_MAX,
        mismatch_soft=g3t.GATE_MISMATCH_SOFT, soft_fraction_max=g3t.GATE_SOFT_FRACTION_MAX,
        n_above_soft=n_soft, fraction_above_soft=soft_fraction,
        mismatch_worst=dict(median=float(np.median(worst)), max=float(np.max(worst))) if good else None,
        step_worst=dict(median=float(np.median(steps)), max=float(np.max(steps))) if good else None,
        by_total_mass=gate_bins(records, lambda r: True),
        by_total_mass_chi_eff_below=dict(threshold=GATE_NEG_CHI,
                                         bins=gate_bins(records, lambda r: r["chi_eff"] < GATE_NEG_CHI)),
        failed_rows=[r["row_index"] for r in failed], wall_time_s=time.perf_counter() - t0,
        script_sha256=sha256_file(__file__), m13_13_sha256=sha256_file(g3t.__file__), rows=records)
    write_json(out_dir / "gate_taper.json", report)
    for b in report["by_total_mass"]:
        log(f"M {b['mass_bin']}: n {b['n']}, failed {b['n_failed']} (errors {b['n_errors']}), "
            f"mismatch Sref max {b['mismatch_sref_max']}, step max {b['step_max']}")
    log(f"G-TAPER-G4 {verdict}: {len(failed)} failed of {len(records)}; above soft {n_soft} ({soft_fraction:.4f})")
    if verdict == "FAIL":
        raise SystemExit("G-TAPER-G4 did not pass; stop (pre-registered).")


# ---------------------------------------------------------------- noisy (variant of m13_13 noisy without G-REC)


def noisy_row_ids(ctx, split):
    if split == "train40k":
        return np.union1d(monitor_ids(ctx, "R"), monitor_ids(ctx, "F")).astype(np.int64)
    return np.union1d(cut_ids(ctx, "R", split), cut_ids(ctx, "F", split)).astype(np.int64)


def noisy_row_g4(row, index, ctx):
    """m13_13.noisy_row without the G3 trainio: no G-REC; y comes from the manifest; is_truncated from build_g1_row."""
    t0 = time.perf_counter()
    injector = ctx["injector"]
    with g3t._NoiseTap() as nt:
        _, meta, cap, opt_args = m11.capture_row(row, long_strains=ctx["long_strains"], psd_cache=ctx["psd_cache"])
    require(nt.noise is not None, f"Row {index}: noise not captured")
    processing_start = float(row["processing_start"])
    s_A = cap.signal["projection"].strains
    inj_A = injector.inject_network(noises=nt.noise, signals=s_A)
    for ifo in DETECTORS:
        require(np.array_equal(inj_A[ifo].strain.numpy(), cap.strains[ifo].numpy()),
                f"Row {index} {ifo}: tapped noise + s_A differs from the captured s + n")
    raw_A, _ = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(
        strains={ifo: cap.strains[ifo].copy() for ifo in DETECTORS}, psds=cap.psds))
    s_T, n_before, align = g3t.s15_strains(opt_args, cap, processing_start)
    inj_T = injector.inject_network(noises=nt.noise, signals=s_T)
    raw_T, X_T = g1.stack_and_normalize_network(g1.PROCESSOR.process_network(
        strains={ifo: inj_T[ifo].strain for ifo in DETECTORS}, psds=cap.psds))
    dn_A = np.asarray(raw_A, dtype=np.float64) - m11.process_clean(
        s_A, processing_start=processing_start, psds=cap.psds, injector=injector)
    dn_T = np.asarray(raw_T, dtype=np.float64) - m11.process_clean(
        s_T, processing_start=processing_start, psds=cap.psds, injector=injector)
    x16 = g6.to_fp16(np.asarray(X_T, dtype=np.float32), f"X_T row {index}")
    rec = dict(row_index=int(index), block_id=str(row["block_id"]),
               G_NOISE=float(np.abs(dn_A - dn_T).max() / np.abs(dn_A).max()),
               G_ALIGN=float(align["residual_rad"]), edge_samples=n_before, edge=bool(max(n_before.values()) > 0),
               is_truncated=bool(meta["is_truncated"]), meta=jsonable({k: v for k, v in meta.items()}),
               max_abs_x=np.abs(np.asarray(X_T, dtype=np.float64)).max(axis=1).tolist(),
               time_s=time.perf_counter() - t0)
    return rec, x16


def noisy_fails_g4(rec):
    fails = []
    if g3t.align_abort(rec["G_ALIGN"]):
        fails.append("G-ALIGN")
    if not rec["G_NOISE"] <= g3t.GDIFF_RTOL:  # also catches NaN
        fails.append("G-NOISE")
    return fails


def noisy_chunk_g4(indices):
    out = []
    frame = g3t._CTX["frame"].loc[indices]
    for index, row in frame.iterrows():
        rec, x16 = noisy_row_g4(row, index, g3t.block_context(row))
        out.append((rec, x16))
        if noisy_fails_g4(rec):
            break
    return out


def noisy_summary_g4(records, split, wall_s):
    times = [r["time_s"] for r in records]
    return dict(split=split, n_rows=len(records), wall_time_s=wall_s,
                time_s_per_row=dict(median=float(np.median(times)), mean=float(np.mean(times)),
                                    max=float(np.max(times))),
                G_REC="not applicable: G4 rows have no G3 row to rebuild",
                G_NOISE=dict(threshold=g3t.GDIFF_RTOL, max=max(r["G_NOISE"] for r in records)),
                G_ALIGN=g3t.align_summary([r["row_index"] for r in records], [r["G_ALIGN"] for r in records]),
                G_EDGE=dict(taper_s=g3t.EDGE_TAPER_S, beta=g3t.EDGE_BETA,
                            n_rows_tapered=sum(r["edge"] for r in records),
                            n_rows_truncated=sum(r["is_truncated"] for r in records)))


def run_noisy_g4(args, ctx):
    gate_path = require_g4_gate(args.gate_json, ctx)
    split = args.split
    row_ids = noisy_row_ids(ctx, split)
    out_dir = ctx.base / NOISY_DIR
    final_x = out_dir / f"X_{split}.npy"
    require(not final_x.exists(), f"Already published, refusing to overwrite: {final_x}")
    positions = np.arange(row_ids.size)
    if args.limit:
        require(not args.resume, "--limit does not resume")
        positions = positions[g3t.thin(positions.size, args.limit)]
        work = ctx.base / f"timing_{g3t.utc_stamp()}" / f"noisy_{split}"
    else:
        work = ctx.base / f"work_noisy_{split}"
    x_shape = (row_ids.size, len(DETECTORS), g1.CONFIG.length)
    meta = dict(split=split, script_sha256=sha256_file(__file__), m13_13_sha256=sha256_file(g3t.__file__),
                proposal_report_sha256=ctx.report_sha, row_ids_sha256=sha_bytes(row_ids), x_shape=list(x_shape),
                n_positions=int(positions.size), limit=args.limit, s15=g3t.s15_record(gate_path))
    if args.resume:
        require((work / "meta.json").exists(), f"Nothing to resume in {work}")
        require(json.loads((work / "meta.json").read_text()) == jsonable(meta), "Resume with another script or selection")
        X = np.lib.format.open_memmap(work / "X.npy", mode="r+")
        done = np.lib.format.open_memmap(work / "done.npy", mode="r+")
    else:
        require(not work.exists(), f"{work} exists; use --resume")
        work.mkdir(parents=True)
        write_json(work / "meta.json", meta)
        X = np.lib.format.open_memmap(work / "X.npy", mode="w+", dtype=np.float16, shape=x_shape)
        done = np.lib.format.open_memmap(work / "done.npy", mode="w+", dtype=np.uint8, shape=(row_ids.size,))

    log = g3t.Log(work / "build.log")
    pending = positions[done[positions] == 0]
    log(f"M15 G4 noisy {split}: {positions.size} rows, {pending.size} pending, {args.workers} worker(s) -> {work}")
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
    progress = g3t.Progress(positions.size, positions.size - pending.size)
    if pending.size:
        frame = g3.adjust_processing_windows(ctx.frame.loc[row_ids[pending]], ctx.spec_g1)
        psd_files, _ = resolve_cached_psd_files(ctx.root, frame)
        g3t._CTX.update(frame=frame, psd_files=psd_files)
        rows_file = jsonl.open("a")

        def consume(results):
            # Order on disk as m13_13: X, then its gate record, then done. An interrupted chunk is redone.
            nonlocal failed
            ok = [(rec, x16) for rec, x16 in results if not noisy_fails_g4(rec)]
            for rec, x16 in ok:
                X[pos[rec["row_index"]]] = x16
            X.flush()
            for rec, _ in ok:
                records[rec["row_index"]] = rec
                rows_file.write(json.dumps(rec) + "\n")
                log(f"row {rec['row_index']} {rec['time_s']:.2f} s: G-NOISE {rec['G_NOISE']:.1e}, residual "
                    f"{rec['G_ALIGN']:+.1e} rad, edge {int(rec['edge'])}{g3t.align_flag(rec['G_ALIGN'])}  "
                    f"{progress.tick()}")
            rows_file.flush()
            os.fsync(rows_file.fileno())
            done[[pos[rec["row_index"]] for rec, _ in ok]] = 1
            done.flush()
            for rec, _ in results:
                fails = noisy_fails_g4(rec)
                if fails:
                    failed = dict(row_index=rec["row_index"], gates=fails, record=rec)
                    log(f"row {rec['row_index']} FAIL {fails}")
            return failed is None

        g3t.run_tasks(noisy_chunk_g4, g3t.block_chunks(frame), args.workers, consume)
        rows_file.close()
    X.flush()
    if failed is not None:
        g3t.overwrite_json(work / "gates_FAIL.json", jsonable(failed))
        raise SystemExit(f"Gate failed at row {failed['row_index']}: {failed['gates']}; stop (pre-registered).")
    require(bool(done[positions].all()), "Rows left unwritten")
    selected = [records[int(row_ids[p])] for p in positions]
    summary = noisy_summary_g4(selected, split, time.perf_counter() - t0)
    log(g3t.align_line(summary["G_ALIGN"]))
    if summary["G_ALIGN"]["stop"]:
        g3t.overwrite_json(work / "gates_FAIL.json", jsonable(summary["G_ALIGN"]))
        raise SystemExit(f"G-ALIGN: {summary['G_ALIGN']['stop']}; stop (amendment 3).")
    summary["s15"] = g3t.s15_record(gate_path)
    summary["provenance"] = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in
                             attrs_for_g4(ctx, gate_path)(positions.size, {"selection_policy": f"m15_noisy_{split}",
                                                                           "limit": args.limit or 0}).items()}
    log(f"Gates PASS {split}: G-NOISE {summary['G_NOISE']['max']:.1e}, G-EDGE {summary['G_EDGE']['n_rows_tapered']}"
        f"/{positions.size}, {summary['time_s_per_row']['mean']:.2f} s/row (mean, one worker)")
    del X, done
    if args.limit:
        write_json(work / f"gates_{split}.json", summary)
        return
    # Publication: everything before the final rename can be repeated; X_<split>.npy existing means published.
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / f"row_ids_{split}.npy", row_ids)
    np.save(out_dir / f"y_{split}.npy", ctx.frame.loc[row_ids, list(LABELS)].to_numpy(dtype=np.float32))
    with (out_dir / f"rows_{split}.jsonl").open("w") as f:
        for rec in selected:
            f.write(json.dumps(rec) + "\n")
    g3t.overwrite_json(out_dir / f"gates_{split}.json", jsonable(summary))
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else dict(
        kind="m15_g4_noisy_union", dataset_id=DATASET_ID, x_dtype="float16", y_dtype="float32 (unstandardized)",
        x="zscore(process_network(inject(n, S15)), psd_proc); n = Gaussian noise of the G4 row (gaussian_seed)",
        splits={})
    manifest["splits"][split] = dict(n_rows=int(row_ids.size), shape_X=list(x_shape),
                                     row_ids_sha256=sha256_file(out_dir / f"row_ids_{split}.npy"),
                                     script_sha256=sha256_file(__file__), proposal_report_sha256=ctx.report_sha,
                                     s15=summary["s15"], gates=f"gates_{split}.json", rows=f"rows_{split}.jsonl")
    g3t.overwrite_json(path, jsonable(manifest))
    os.rename(work / "X.npy", final_x)
    log(f"Published {final_x}")


# ---------------------------------------------------------------- materialize


def split_file_name(target):
    return (f"m15_G4_{target}_splits_train{N_TARGET['train']}_val{N_TARGET['val']}"
            f"_cal{N_TARGET['cal']}_test{N_TARGET['test']}.npz")


def link_label_stats(ctx):
    """The G1 train-only label stats of model 5a, by relative symlink where the copied config key resolves them."""
    ref = json.loads(REFERENCE_TRAIN_CONFIG.read_text())["dataset"]
    rel = Path(ref["label_stats_file"])
    require(rel.parts[:1] == ("..",) and len(rel.parts) == 2, f"Unexpected label_stats_file {rel}")
    src = Path(os.path.normpath(resolve_processed_artifact(data_root=ctx.root, dataset_id=ref["dataset_id"],
                                                           file_name=rel)))
    require(src.exists(), f"Missing G1 label stats {src}")
    link = ctx.base / rel.name  # processed/M15_G4/<target>/../<name>
    if link.is_symlink() or link.exists():
        require(os.path.realpath(link) == os.path.realpath(src), f"{link} is not the label stats of 5a")
    else:
        os.symlink(os.path.relpath(src, link.parent), link)
    return {"config_value": str(rel), "link": str(link), "source": str(src), "sha256": sha256_file(src)}


def copy_rows(dst_X, dst_pos, src_X, src_pos):
    for lo in range(0, dst_pos.size, COPY_CHUNK):
        dst_X[dst_pos[lo:lo + COPY_CHUNK]] = src_X[src_pos[lo:lo + COPY_CHUNK]]


def lookup(src_ids, ids, what):
    j = np.searchsorted(src_ids, ids)
    require(bool((j < src_ids.size).all()) and np.array_equal(src_ids[np.minimum(j, src_ids.size - 1)], ids),
            f"{what}: rows missing from the source")
    return j


def write_memmap_split(mm_dir, name, ids, y, src_split, base, positions=None):
    """X/y/row_ids_<name>.npy; positions: write only those rows (sparse X, train40k monitor)."""
    noisy = base / NOISY_DIR
    src_ids = np.load(noisy / f"row_ids_{src_split}.npy")
    src_X = np.load(noisy / f"X_{src_split}.npy", mmap_mode="r")
    src_y = np.load(noisy / f"y_{src_split}.npy")
    dst_pos = np.arange(ids.size) if positions is None else np.asarray(positions, dtype=np.int64)
    j = lookup(src_ids, ids[dst_pos], f"{name} from noisy {src_split}")
    require(np.array_equal(src_y[j], y[dst_pos]), f"{name}: y differs from noisy {src_split}")
    mm_dir.mkdir(exist_ok=True)
    X = np.lib.format.open_memmap(mm_dir / f"X_{name}.npy", mode="w+", dtype=np.float16,
                                  shape=(ids.size,) + tuple(src_X.shape[1:]))
    copy_rows(X, dst_pos, src_X, j)
    X.flush()
    del X
    np.save(mm_dir / f"row_ids_{name}.npy", ids)
    np.save(mm_dir / f"y_{name}.npy", y)
    return dict(n_rows=int(ids.size), n_rows_written=int(dst_pos.size), shape_X=[int(ids.size), *src_X.shape[1:]],
                row_ids_sha256=sha256_file(mm_dir / f"row_ids_{name}.npy"), source=stat_entry(noisy / f"X_{src_split}.npy"),
                contents=("monitor rows only (train_m13_3.py); all other rows are zeros (sparse file)"
                          if positions is not None else "all rows"))


def run_materialize(args, ctx):
    gate_path = require_g4_gate(args.gate_json, ctx)
    t_start = time.time()
    target = args.target
    out = ctx.base / target
    require(not out.exists(), f"Output exists, refusing to overwrite: {out}")
    for s in NOISY_SPLITS_G4:
        require((ctx.base / NOISY_DIR / f"X_{s}.npy").exists(), f"noisy {s} not published")
    m13 = g3.load_m13_config(G3_CONFIG)
    aug6 = ctx.root / "processed" / m13["dataset_id"] / g6.OUTPUT_SUBDIR / "memmap"
    man6 = json.loads((aug6 / "manifest.json").read_text())
    stats_link = link_label_stats(ctx)
    tmp = ctx.base / f"{target}.tmp_{os.getpid()}"
    tmp.mkdir(exist_ok=False)
    ids = {s: cut_ids(ctx, target, s) for s in SPLITS}
    for s in SPLITS:
        require(ids[s].size == N_TARGET[s], f"{target} {s}: {ids[s].size} rows")
    y = {s: ctx.frame.loc[ids[s], list(LABELS)].to_numpy(dtype=np.float32) for s in SPLITS}
    with open(tmp / split_file_name(target), "xb") as fh:
        np.savez(fh, **{f"{s}_idx": ids[s] for s in SPLITS})

    # fp16_memmap (train: monitor rows only; val) and fp16_memmap_heldout (cal, test)
    mm_info = {"fp16_memmap": {}, "fp16_memmap_heldout": {}}
    monitor = g3t.monitor_positions(N_TARGET["train"])
    mm_info["fp16_memmap"]["train"] = write_memmap_split(tmp / "fp16_memmap", "train", ids["train"], y["train"],
                                                         "train40k", ctx.base, positions=monitor)
    for s in ("val", "cal", "test"):
        mm_info[HELDOUT_MEMMAP[s]][s] = write_memmap_split(tmp / HELDOUT_MEMMAP[s], s, ids[s], y[s], s, ctx.base)
    for mm, splits in mm_info.items():
        g3t.overwrite_json(tmp / mm / "manifest.json", jsonable(dict(
            kind="m15_g4_inputs", dataset_id=f"{DATASET_ID}/{target}", x_dtype="float16",
            y_dtype="float32 (unstandardized)", detector_order=list(DETECTORS),
            x="zscore(process_network(inject(n, S15)), psd_proc), copied from noisy_union",
            script_sha256=sha256_file(__file__), proposal_report_sha256=ctx.report_sha, splits=splits)))
    print(f"[{target}] fp16 memmaps written | {time.time() - t_start:.0f} s", flush=True)

    # memmap: X_clean_train from the signal shards (rows copied), psd_key_train, links to the M13.6 bank
    mm = tmp / "memmap"
    mm.mkdir()
    paths = shard_paths(ctx.base / "signal", "signal", args.num_shards)
    train_ids = ids["train"]
    seen = np.zeros(train_ids.size, dtype=bool)
    reference, versions, residuals, merged, n_edge = {}, set(), [], [], 0
    with h5py.File(paths[0], "r") as h:
        _, C, T = h["X_clean"].shape
    X = np.lib.format.open_memmap(mm / "X_clean_train.npy", mode="w+", dtype=np.float16, shape=(train_ids.size, C, T))
    stats = Stats(C)
    for path in paths:
        with h5py.File(path, "r") as h:
            reference = check_shard(h, path, "m15_signal", reference)
            versions.add((h.attrs["m15_generator_sha256"], h.attrs["m15_proposal_report_sha256"],
                          h.attrs["m15_gate_taper_report"], float(h.attrs["s15_f_lower"]),
                          float(h.attrs["s15_taper_end_hz"])))
            ri = h["row_index"][:]
            pos = np.searchsorted(train_ids, ri)
            hit = (pos < train_ids.size) & (train_ids[np.minimum(pos, train_ids.size - 1)] == ri)
            require(not seen[pos[hit]].any(), f"{path}: rows seen twice")
            require(np.array_equal(h["y"][:][hit], y["train"][pos[hit]]), f"{path}: y differs from the proposal")
            keys = ctx.frame.loc[ri[hit], "block_id"].astype(str).to_numpy()
            require(np.array_equal(h["psd_key"].asstr()[:][hit], keys), f"{path}: psd_key differs")
            for lo in range(0, ri.size, COPY_CHUNK):
                sel = hit[lo:lo + COPY_CHUNK]
                if sel.any():
                    x = h["X_clean"][lo:lo + COPY_CHUNK][sel]
                    stats.add(x)
                    X[pos[lo:lo + COPY_CHUNK][sel]] = x
            seen[pos[hit]] = True
            residuals.append(h["align_residual_rad"][:][hit])
            merged.append(ri[hit])
            n_edge += int(h["edge_tapered"][:][hit].sum())
        print(f"  [{target}] {path.name}: {int(hit.sum())} rows | {time.time() - t_start:.0f} s", flush=True)
    require(bool(seen.all()), f"{int((~seen).sum())} {target} train rows missing from the signal shards")
    require(len(versions) == 1, f"Signal shards from different versions or gates: {versions}")
    version = next(iter(versions))
    require(version[1] == ctx.report_sha and Path(version[2]).resolve() == gate_path
            and version[3:] == (g3t.S15_F_LOWER, g3t.S15_TAPER_END_HZ), f"Shards made for another setup: {version}")
    X.flush()
    del X
    g_align = g3t.align_summary(np.concatenate(merged), np.concatenate(residuals))
    require(not g_align["stop"], f"G-ALIGN: {g_align['stop']}")
    keys = ctx.frame.loc[train_ids, "block_id"].astype(str).to_numpy().astype(str)
    bank_keys = np.unique(np.load(aug6 / "bank_psd_key.npy"))
    require(bool(np.isin(keys, bank_keys).all()), "Some train psd_key has no entries in the noise bank")
    np.save(mm / "psd_key_train.npy", keys)
    for name in ("bank_X.npy", "bank_psd_key.npy"):
        # Relative to the final location <target>/memmap (same depth as the temporary directory).
        os.symlink(os.path.relpath(aug6 / name, out / "memmap"), mm / name)
    manifest = {
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)), "script_sha256": sha256_file(__file__),
        "config": str(Path(G3_CONFIG)), "config_sha256": sha256_file(G3_CONFIG), "dataset_id": f"{DATASET_ID}/{target}",
        "content": "L(s_S15) before the per-sample per-detector z-score; noise bank of M13.6 by symlink",
        "s15": g3t.s15_record(gate_path), "proposal_report_sha256": ctx.report_sha,
        "shard_provenance": reference, "signal_generator_sha256": version[0],
        "signal_shards": [stat_entry(p) for p in paths],
        # Read by train_m13_3.load_noise_aug: compared with <target>/fp16_memmap/row_ids_train.npy.
        "g3_memmap": {"dir": str(out / "fp16_memmap"),
                      "row_ids_train_sha256": sha256_file(tmp / "fp16_memmap" / "row_ids_train.npy")},
        "x_dtype": "float16",
        "X_clean_train": {"shape": [int(train_ids.size), int(C), int(T)],
                          "row_order": "fp16_memmap row_ids_train (ascending row id)", "per_detector": stats.report()},
        "bank_X": {**man6["bank_X"], "source": str(aug6 / "bank_X.npy"), "linked": True},
        "G_ALIGN": g_align, "G_EDGE": {"taper_s": g3t.EDGE_TAPER_S, "beta": g3t.EDGE_BETA, "split": "train",
                                       "n_rows_tapered": n_edge},
        "outputs_sha256": {n: sha256_file(mm / n) for n in ("X_clean_train.npy", "psd_key_train.npy")},
        "bank_outputs_sha256": {k: v for k, v in man6.get("outputs_sha256", {}).items() if k.startswith("bank_")},
    }
    g3t.overwrite_json(mm / "manifest.json", jsonable(manifest))
    write_json(tmp / "materialize_manifest.json", {
        "generator": GENERATOR, "script_sha256": sha256_file(__file__), "target": target,
        "proposal_report_sha256": ctx.report_sha, "gate_taper_report": str(gate_path),
        "split_file": {"name": split_file_name(target), "sha256": sha256_file(tmp / split_file_name(target))},
        "label_stats": stats_link, "n_rows": {s: int(ids[s].size) for s in SPLITS},
        "monitor": {"n": int(monitor.size), "rule": "m13_13.monitor_positions(800000) on row_ids_train"},
        "shared_rows_with_other_target": {s: int(np.intersect1d(ids[s], cut_ids(ctx, 'F' if target == 'R' else 'R', s)).size)
                                          for s in SPLITS},
        "total_time_s": time.time() - t_start})
    require(not out.exists(), f"Output appeared during materialize; result left in {tmp}")
    os.rename(tmp, out)
    print(f"[{target}] done in {time.time() - t_start:.0f} s -> {out}; G-EDGE {n_edge}; {g3t.align_line(g_align)}",
          flush=True)


# ---------------------------------------------------------------- consistency


def run_consistency_g4(args, ctx):
    base_t = ctx.base / args.target
    require((base_t / "materialize_manifest.json").exists(), f"{base_t} is not materialized")
    before = set(base_t.glob("consistency_*"))
    g3t.run_consistency(args, ctx.root, ctx.spec_g1, ctx.frame, ctx.trainio, base_t)
    new = sorted(set(base_t.glob("consistency_*")) - before)
    require(len(new) == 1, f"Expected one new consistency directory, got {new}")
    write_json(new[0] / "m15_provenance.json", {
        "generator": GENERATOR, "script_sha256": sha256_file(__file__), "m13_13_sha256": sha256_file(g3t.__file__),
        "target": args.target, "proposal_report_sha256": ctx.report_sha,
        "materialize_manifest_sha256": sha256_file(base_t / "materialize_manifest.json"),
        "note": "consistency_report.json is written by m13_13.run_consistency on the materialized target"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path)
    physics = argparse.ArgumentParser(add_help=False, parents=[common])
    physics.add_argument("--allow-version-mismatch", action="store_true")
    workers = argparse.ArgumentParser(add_help=False)
    workers.add_argument("--workers", type=int, default=1, help=f"Worker processes (fork), at most {g3t.MAX_WORKERS}")
    gated = argparse.ArgumentParser(add_help=False)
    gated.add_argument("--gate-json", type=Path, required=True, help="M15_G4/gate_taper/gate_taper.json with PASS")
    run = argparse.ArgumentParser(add_help=False, parents=[physics, workers, gated])
    run.add_argument("--limit", type=int, help="Only N evenly spaced rows, written to timing_<stamp>/")
    run.add_argument("--resume", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample", parents=[common])
    sub.add_parser("gate-taper", parents=[physics, workers])
    signal = sub.add_parser("signal", parents=[run])
    signal.add_argument("--shard", type=int, default=0)
    signal.add_argument("--num-shards", type=int, default=1)
    noisy = sub.add_parser("noisy", parents=[run])
    noisy.add_argument("--split", choices=NOISY_SPLITS_G4, required=True)
    mat = sub.add_parser("materialize", parents=[physics, gated])
    mat.add_argument("--target", choices=TARGETS, required=True)
    mat.add_argument("--num-shards", type=int, required=True)
    cons = sub.add_parser("consistency", parents=[physics])
    cons.add_argument("--target", choices=TARGETS, required=True)
    sub.add_parser("status", help="Handled before the project imports; only takes --data-root")
    args = parser.parse_args()

    if args.command == "sample":
        run_sample(args)
        return
    require(1 <= getattr(args, "workers", 1) <= g3t.MAX_WORKERS, f"--workers must be in 1..{g3t.MAX_WORKERS}")
    require(getattr(args, "limit", None) is None or args.limit >= 1, "Invalid --limit")
    if args.command == "signal":
        require(args.num_shards >= 1 and 0 <= args.shard < args.num_shards, "Invalid shard index")
        require(not (args.limit and args.resume), "--limit does not resume")
    ctx = setup(args)
    if args.command in ("gate-taper", "signal", "noisy"):
        log_memory(ctx)
    if args.command == "gate-taper":
        run_gate_g4(args, ctx)
    elif args.command == "signal":
        gate_path = require_g4_gate(args.gate_json, ctx)
        # m13_13.run_signal unchanged: it takes the train rows of the frame it is given (here the G4 union).
        g3t.run_signal(args, ctx.root, ctx.spec_g1, ctx.frame, ctx.base, attrs_for_g4(ctx, gate_path))
    elif args.command == "noisy":
        run_noisy_g4(args, ctx)
    elif args.command == "materialize":
        run_materialize(args, ctx)
    else:
        run_consistency_g4(args, ctx)


if __name__ == "__main__":
    main()
