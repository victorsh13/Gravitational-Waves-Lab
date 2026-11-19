"""M14.6: P512 bank of whitened SEOBNRv4_opt (S15 of G3T) templates per val PSD environment and detector (local, CPU).

Templates (pre-registration docs/experiments/M14_6_bank_probes.md):
  - parameters: default_rng(20261119).choice(train_idx, K, replace=False) over the frozen train split; each row is
    a draw of the training prior; (mass_1, mass_2, spin_1z, spin_2z) of the manifest, m1 >= m2. P32 / P128 are the
    first 32 / 128 in draw order (nested);
  - waveform: S15 of G3T (m13_11._opt_click_free with the m13_13_build_g3t band: f_lower 15 Hz, f_ref 30 Hz, Kaiser
    beta 8 taper up to the 25 Hz crossing), face-on, distance 1, coa_phase 0; G-EDGE edge taper as in G3T;
  - whitening: the very function that produced L(s) of G3T, m13_11.process_clean (zero strain of processing length
    + g1.PROCESSOR.process_network with the processing PSD of the block), after g6.setup (G3 physics);
  - one whitened h+ per (template, val block, detector), peak of h+ at window sample p = N - 1 - MERGER_MARGIN,
    normalized to unit L2 norm. The quadrature partner is the Hilbert transform of the whitened h+ and is not
    stored (computed in m14_6_bank_features.py).

Output: <data_root>/results/M13_3/M14_6/templates_<UTC>.npz (fp32; never overwrites). The (K, n_blocks, 3, N) array
is filled in a work .npy memmap next to the output (RAM-friendly) and removed after the npz is written.

Usage (from cbc_pe, local, CPU):
    OMP_NUM_THREADS=1 python scripts/m14/m14_6_make_templates.py [--n-templates 512] [--allow-version-mismatch]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing the G3T builder sets m13_11.CLICK_F_LOWER / CLICK_TAPER_END_HZ to the S15 band (15 / 25 Hz).
import scripts.m13.m13_13_build_g3t as g3t
from scripts.m13.m13_13_build_g3t import (DETECTORS, S15_F_LOWER, S15_TAPER_END_HZ, edge_taper, g1, g6, m11,
                                          read_psd_intervals, resolve_cached_psd_files)

SEED = 20261119
N_TEMPLATES = 512
MERGER_MARGIN = 512          # samples after the h+ peak kept in the window (0.125 s)
SYNTH_PROCESSING_START = 1.0e9  # synthetic GPS start of the processing segment (processing is time invariant)
TRAIN_CFG = PROJECT_ROOT / "configs/experiments/m14/train_m14_5b_4aG3T_tbank_fullcov_s123.json"
OUT_SUBDIR = "results/M13_3/M14_6"


def require(condition, message):
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=g3t.DEFAULT_CONFIG, help="G3 generation config")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--manifest", type=Path, help="Override G1 manifest_path; still verified by hash")
    p.add_argument("--allow-version-mismatch", action="store_true")
    p.add_argument("--n-templates", type=int, default=N_TEMPLATES,
                   help="First K of the seed-20261119 draw (512 is the pre-registered bank; fewer only for a test)")
    return p.parse_args()


def template_kwargs(row):
    """get_td_waveform kwargs of WaveformGenerator, face-on at unit distance (m1 >= m2, as CBCParameters)."""
    m1, m2, s1, s2 = (float(row[k]) for k in ("mass_1", "mass_2", "spin_1z", "spin_2z"))
    if m1 < m2:
        m1, m2, s1, s2 = m2, m1, s2, s1
    return dict(mass1=m1, mass2=m2, spin1z=s1, spin2z=s2, distance=1.0, inclination=0.0, coa_phase=0.0,
                f_lower=g1.CONFIG.low_frequency_cutoff, delta_t=g1.CONFIG.delta_t,
                approximant=g1.CONFIG.waveform_approximant)


def placed_strains(kwargs, peak_time):
    """S15 h+ (f_lower / f_ref / taper overridden by _opt_click_free) with its peak at peak_time, G-EDGE applied."""
    hp, _, taper_s = m11._opt_click_free(kwargs)
    hp.start_time = float(hp.start_time) + peak_time  # SEOBNRv4_opt: t = 0 at the peak
    strains, n_before = edge_taper({ifo: hp for ifo in DETECTORS}, SYNTH_PROCESSING_START)
    return strains, float(taper_s), int(max(n_before.values()))


def main():
    args = parse_args()
    require(1 <= args.n_templates <= N_TEMPLATES, f"--n-templates must be in 1..{N_TEMPLATES}")
    require(os.environ.get("OMP_NUM_THREADS") is not None, "Set OMP_NUM_THREADS (e.g. 1) before running")
    require((m11.CLICK_F_LOWER, m11.CLICK_TAPER_END_HZ) == (S15_F_LOWER, S15_TAPER_END_HZ),
            "S15 band not set by m13_13_build_g3t")
    args.mode = "m14_6_templates"  # recorded by g6.attrs_for (not used here)
    root, spec_g1, df, _trainio, directory, _ = g6.setup(args)
    N = int(g1.CONFIG.length)
    dt = float(g1.CONFIG.delta_t)
    p_peak = N - 1 - MERGER_MARGIN

    # Val rows and PSD environments (blocks), in fp16_memmap row order.
    memmap_dir = directory / "fp16_memmap"
    val_ids = np.load(memmap_dir / "row_ids_val.npy").astype(np.int64)
    val = df.loc[val_ids]
    require((val["split"] == "val").all(), "row_ids_val are not all val rows of the manifest")
    block_ids = sorted(val["block_id"].astype(str).unique())
    val_block_index = np.searchsorted(np.array(block_ids), val["block_id"].astype(str).to_numpy()).astype(np.int16)
    reps = val.groupby(val["block_id"].astype(str), sort=True).head(1)
    reps = reps.set_index(reps["block_id"].astype(str)).loc[block_ids]
    print(f"val: {len(val_ids)} rows, {len(block_ids)} PSD blocks", flush=True)

    # Template parameters from the frozen train split.
    train_cfg = json.loads(TRAIN_CFG.read_text())
    from src.paths import resolve_processed_artifact
    split_path = resolve_processed_artifact(data_root=root, dataset_id=train_cfg["dataset"]["dataset_id"],
                                            file_name=train_cfg["dataset"]["split_file"])
    with np.load(split_path) as z:
        train_idx = np.asarray(z["train_idx"], dtype=np.int64)
    require((df.loc[train_idx, "split"] == "train").all(), "split file train_idx != manifest train rows")
    pick = np.random.default_rng(SEED).choice(train_idx, size=N_TEMPLATES, replace=False)[:args.n_templates]
    K = len(pick)

    # Processing PSDs per block (same estimator and cache as the generator).
    psd_files, _ = resolve_cached_psd_files(root, reps)
    psds = {}
    for b in block_ids:
        row = reps.loc[b]
        psds[b] = g1.get_block_psds(row, long_strains=read_psd_intervals(row, psd_files), psd_cache={})["proc"]
    injector = g1.SignalInjector(config=g1.CONFIG, rng=np.random.default_rng(0))  # never draws
    peak_time = SYNTH_PROCESSING_START + g1.CONFIG.processing_context_start_seconds + p_peak * dt

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"templates_{stamp}.npz"
    work_path = out_dir / f"templates_{stamp}.work.npy"
    require(not out_path.exists() and not work_path.exists(), f"{out_path} exists")
    T = np.lib.format.open_memmap(work_path, mode="w+", dtype=np.float32, shape=(K, len(block_ids), 3, N))
    peak_index = np.zeros((K, len(block_ids), 3), dtype=np.int32)  # argmax |analytic(template)|, diagnostic
    taper_s = np.zeros(K)
    edge_samples = np.zeros(K, dtype=np.int64)

    t0 = time.perf_counter()
    for k, src in enumerate(pick):
        strains, taper_s[k], edge_samples[k] = placed_strains(template_kwargs(df.loc[src]), peak_time)
        for bi, b in enumerate(block_ids):
            L = m11.process_clean(strains, processing_start=SYNTH_PROCESSING_START, psds=psds[b], injector=injector)
            require(L.shape == (3, N) and np.isfinite(L).all(), f"template {k} block {b}: bad L {L.shape}")
            norm = np.linalg.norm(L, axis=1, keepdims=True)
            require((norm > 0).all(), f"template {k} block {b}: zero norm")
            L = L / norm
            spec = np.fft.rfft(L, axis=1)
            spec[:, 1:-1] *= 2.0
            peak_index[k, bi] = np.abs(np.fft.ifft(np.concatenate([spec, np.zeros((3, N - spec.shape[1]))], 1),
                                                   axis=1)).argmax(1)
            T[k, bi] = L.astype(np.float32)
        if k == 0 or (k + 1) % 16 == 0:
            el = time.perf_counter() - t0
            print(f"template {k + 1}/{K}: {el / 60:.1f} min, ETA {el / (k + 1) * (K - k - 1) / 60:.1f} min", flush=True)
    T.flush()

    rows = df.loc[pick]
    params = np.array([[template_kwargs(r)[c] for c in ("mass1", "mass2", "spin1z", "spin2z")]
                       for _, r in rows.iterrows()], dtype=np.float64)
    meta = dict(
        experiment="M14.6", created_utc=stamp, seed=SEED, n_templates=K, nested="P32/P128 = first 32/128",
        parameter_source="default_rng(seed).choice(train_idx, 512, replace=False)[:K], manifest rows",
        split_path=str(split_path), waveform=dict(approximant=g1.CONFIG.waveform_approximant,
                                                  f_lower=S15_F_LOWER, taper_end_hz=S15_TAPER_END_HZ,
                                                  f_ref=m11.CLICK_F_REF, taper_beta=m11.CLICK_BETA,
                                                  inclination=0.0, coa_phase=0.0, distance=1.0,
                                                  edge_taper_s=g3t.EDGE_TAPER_S),
        whitening="m13_11_prepare_rows.process_clean with g1.PROCESSOR (G3 spec) and the block processing PSD",
        processor=g1.PROCESSOR.metadata(), layout=dict(n_samples=N, peak_sample=p_peak, merger_margin=MERGER_MARGIN,
                                                       synthetic_processing_start=SYNTH_PROCESSING_START),
        normalization="unit L2 norm per (template, block, detector) of the whitened h+; quadrature = Hilbert",
        detectors=list(DETECTORS), memmap_dir=str(memmap_dir), g3_config=str(args.config.resolve()),
        script_sha256=sha256_file(__file__), elapsed_s=time.perf_counter() - t0)
    with out_path.open("xb") as f:
        np.savez(f, templates=T, block_ids=np.array(block_ids), detectors=np.array(DETECTORS),
                 source_row=pick, params_m1_m2_s1z_s2z=params,
                 labels_chirp_total_chieff=rows[["chirp_mass", "total_mass", "chi_eff"]].to_numpy(np.float64),
                 taper_s=taper_s, edge_samples=edge_samples, peak_index=peak_index, peak_sample=np.int64(p_peak),
                 val_row_ids=val_ids, val_block_index=val_block_index, metadata_json=np.array(json.dumps(meta)))
    del T
    with np.load(out_path) as z:  # the npz is complete before the work file is removed
        require(z["templates"].shape == (K, len(block_ids), 3, N), "npz templates shape")
    work_path.unlink()
    d = np.abs(peak_index - p_peak)
    print(f"|argmax envelope - peak sample|: median {np.median(d):.0f}, max {d.max()} samples", flush=True)
    print("Saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
