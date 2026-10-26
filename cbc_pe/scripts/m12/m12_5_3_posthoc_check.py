"""M12.5.3: post-hoc check of a necessary optimality condition for M11 on validation.

If M11 were E[theta|X], no function of (f(X), emb(X)) could lower its MSE.
Four pre-declared correctors are fitted leave-one-environment-out (LOEO) on the
val split only; cal/test are never read and X is never read. Exploratory: the
result must not be used to select a final model.
Protocol: docs/experiments/M12_5_3_posthoc_optimality_check.md
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_data_root  # noqa: E402
from scripts.m12.g1_npe_contract import (  # noqa: E402  (read-only reuse; file is hash-pinned)
    DATASET, LABELS, digest, load_contract, require)

NPZ_NAME = 'm11_final_G1_500k_val_predictions_embeddings.npz'
OUT_NAME = 'M12_5_3_posthoc'
CORRECTORS = ['C1_affine', 'C2_binned', 'C3_ridge', 'C4_knn']
N_BINS = 50
ALPHAS = [10.0**p for p in range(-2, 5)]
KS = [16, 64, 256]
# Bonferroni over 4 correctors x 3 labels -> two-sided 99.5833 % marginal intervals.
FAMILY_ALPHA = 0.05 / (len(CORRECTORS) * len(LABELS))
MARGIN_FRAC = 0.01
MIN_REGION_N = 200
SNR_BINS = [(10.0, 15.0, False), (15.0, 20.0, False), (20.0, 25.0, True)]  # last bin closed
KNN_BLOCK = 512
BOOT_CHUNK = 50


INVERSE_STANDARDIZE = 'physical = a * y_std + y_mean'


def npz_to_physical(y, f, y_mean, y_std):
    """Return (y, f, units) with y/f in physical units; f shares the units of y.

    'physical' if |mean(y) - y_mean| < 0.1 y_std per label; 'standardized' (train-only
    scaler) if |mean(y)| < 0.1 and |std(y) - 1| < 0.1 per label, inverted with the
    historical inverse_standardize formula; otherwise abort (no other conversion).
    """
    if np.all(np.abs(y.mean(0) - y_mean) < 0.1 * y_std):
        return y, f, 'physical'
    require(np.all(np.abs(y.mean(0)) < 0.1) and np.all(np.abs(y.std(0) - 1.0) < 0.1),
            'y_val is neither physical nor standardized with the train-only scaler')
    return y * y_std + y_mean, f * y_std + y_mean, 'standardized'


def standardize(train, *others):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [(a - mu) / sd for a in (train,) + others]


def fit_affine(f_tr, y_tr, f_te):
    """C1: theta ~ W f + b, full 3x3 W, least squares."""
    coef = np.linalg.lstsq(np.column_stack([f_tr, np.ones(len(f_tr))]), y_tr, rcond=None)[0]
    return np.column_stack([f_te, np.ones(len(f_te))]) @ coef


def fit_binned(f_tr, y_tr, f_te):
    """C2: per-label mean of theta_j in 50 equal-quantile bins of f_j (train envs).

    Linear interpolation between bin centres (mean f_j in bin); np.interp holds the
    end-bin value outside the range. Bin means are not forced to be monotone.
    """
    out = np.empty_like(f_te)
    for j in range(f_tr.shape[1]):
        edges = np.quantile(f_tr[:, j], np.linspace(0, 1, N_BINS + 1))
        b = np.clip(np.searchsorted(edges[1:-1], f_tr[:, j], side='right'), 0, N_BINS - 1)
        n = np.bincount(b, minlength=N_BINS)
        keep = n > 0
        centers = np.bincount(b, f_tr[:, j], N_BINS)[keep] / n[keep]
        means = np.bincount(b, y_tr[:, j], N_BINS)[keep] / n[keep]
        out[:, j] = np.interp(f_te[:, j], centers, means)
    return out


def ridge_paths(z_tr, r_tr, z_te):
    """C3 residual predictions for every alpha, shape (n_alpha, n_te, 3); intercept unpenalised."""
    a_tr, a_te = standardize(z_tr, z_te)
    r_mean = r_tr.mean(0)
    evals, vecs = np.linalg.eigh(a_tr.T @ a_tr)
    proj = vecs.T @ (a_tr.T @ (r_tr - r_mean))
    te = a_te @ vecs
    return np.stack([r_mean + te @ (proj / (evals + alpha)[:, None]) for alpha in ALPHAS])


def knn_paths(e_tr, r_tr, e_te):
    """C4 residual predictions for every k, shape (n_k, n_te, 3); Euclidean on standardised emb."""
    s_tr, s_te = standardize(e_tr, e_te)
    kmax = max(KS)
    require(len(s_tr) >= kmax, 'Too few training rows for kNN')
    sq_tr = (s_tr**2).sum(1)
    out = np.empty((len(KS), len(s_te), r_tr.shape[1]))
    for start in range(0, len(s_te), KNN_BLOCK):
        q = s_te[start:start + KNN_BLOCK]
        d2 = (q**2).sum(1)[:, None] - 2.0 * (q @ s_tr.T) + sq_tr[None, :]
        nn = np.argpartition(d2, kmax - 1, axis=1)[:, :kmax]
        order = np.argsort(np.take_along_axis(d2, nn, 1), axis=1, kind='stable')
        csum = np.cumsum(r_tr[np.take_along_axis(nn, order, 1)], axis=1)
        for i, k in enumerate(KS):
            out[i, start:start + len(q)] = csum[:, k - 1] / k
    return out


def select_inner(paths_fn, x, r, env):
    """Inner LOEO within the outer-training envs; per-label argmin of pooled MSE."""
    sse = 0.0
    for e in np.unique(env):
        te = env == e
        sse = sse + ((paths_fn(x[~te], r[~te], x[te]) - r[te])**2).sum(1)
    mse = sse / len(r)
    return mse.argmin(0), mse


def loeo_predictions(f, y, emb, env):
    """Out-of-fold corrected predictions (physical units) and per-fold hyperparameters."""
    preds = {c: np.empty_like(y) for c in CORRECTORS}
    z, r = np.hstack([emb, f]), y - f
    fold_rows = []
    for e in np.unique(env):
        te, tr = env == e, env != e
        preds['C1_affine'][te] = fit_affine(f[tr], y[tr], f[te])
        preds['C2_binned'][te] = fit_binned(f[tr], y[tr], f[te])
        for name, fn, x, grid, pname in (('C3_ridge', ridge_paths, z, ALPHAS, 'alpha'),
                                         ('C4_knn', knn_paths, emb, KS, 'k')):
            best, inner = select_inner(fn, x[tr], r[tr], env[tr])
            outer = fn(x[tr], r[tr], x[te])
            preds[name][te] = f[te] + np.stack([outer[best[j], :, j] for j in range(len(LABELS))], 1)
            for j, lab in enumerate(LABELS):
                for g, value in enumerate(grid):
                    fold_rows.append(dict(heldout_env=str(e), n_train=int(tr.sum()),
                                          n_heldout=int(te.sum()), corrector=name, label=lab,
                                          param=pname, value=value, inner_loeo_mse=inner[g, j],
                                          selected=bool(g == best[j])))
        print(f'fold env={e} done', flush=True)
    return preds, fold_rows


def bootstrap_means(d, env, masks, n_boot, seed):
    """Paired bootstrap of region means of d; rows resampled within each env, env counts fixed."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(env == e) for e in np.unique(env)]
    out = {k: np.empty((n_boot, d.shape[1])) for k in masks}
    for start in range(0, n_boot, BOOT_CHUNK):
        b = min(BOOT_CHUNK, n_boot - start)
        w = np.zeros((b, len(d)))
        for g in groups:
            w[:, g] = rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)), size=b)
        for k, m in masks.items():
            wm = w[:, m]
            with np.errstate(invalid='ignore', divide='ignore'):
                out[k][start:start + b] = (wm @ d[m]) / wm.sum(1)[:, None]
    return out


def decide(lo, hi, margin):
    if lo > margin:
        return 'AVOIDABLE_DEMONSTRATED'
    if lo >= -margin and hi <= margin:
        return 'NO_EVIDENCE_WITHIN_MARGIN'
    return 'UNRESOLVED'


def slope_row(p, t):
    """OLS pred-on-truth and truth-on-pred; R2 = 1 - SSE/SST (calibrated: slope_pred_on_truth ~ R2)."""
    tc, pc = t - t.mean(), p - p.mean()
    cov = tc @ pc
    b_pt, b_tp = cov / (tc @ tc), cov / (pc @ pc)
    r2 = 1.0 - ((p - t)**2).sum() / (tc @ tc)
    return dict(slope_pred_on_truth=b_pt, intercept_pred_on_truth=p.mean() - b_pt * t.mean(),
                r2=r2, slope_minus_r2=b_pt - r2, slope_truth_on_pred=b_tp,
                intercept_truth_on_pred=t.mean() - b_tp * p.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default=None)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=125301)
    args = ap.parse_args()
    cfg = json.loads((ROOT/'configs/experiments/train_m11_final_G1_500k.json').read_text())
    data_root = resolve_data_root(config_data_root=cfg.get('data_root'))
    results = data_root/'results'/DATASET
    out = Path(args.output) if args.output else results/OUT_NAME
    require(not out.exists(), f'Output already exists: {out}')
    require(args.bootstrap > 0, 'bootstrap must be positive')

    checks = []

    def check(ok, name):
        require(ok, 'Check failed: ' + name)
        checks.append(name)

    npz = results/NPZ_NAME
    # model_config / input_normalization are object arrays and are never accessed.
    with np.load(npz, allow_pickle=False) as a:
        f, y, emb = (a[k].astype(np.float64) for k in ('pred_val', 'y_val', 'emb_val'))
        idx = a['idx_val'].astype(np.int64)
        y_mean, y_std = a['y_mean'].astype(np.float64), a['y_std'].astype(np.float64)
        label_names = a['label_names'].tolist()
        split_name = Path(str(a['split_path'])).name
    n = len(idx)
    check(f.shape == y.shape == (n, 3) and emb.shape == (n, 64) and idx.shape == (n,),
          'npz shapes')
    check(label_names == LABELS, 'label order [chirp_mass, total_mass, chi_eff]')

    paths, splits, c_mean, c_std, identity = load_contract(data_root)
    checks.append('frozen G1 contract (g1_npe_contract.load_contract)')
    check(split_name == paths['split_file'].name, 'npz split_path is the pinned split file')
    # atol=1e-8 on y_std explicitly tolerates the historical +1e-8 added to the scaler
    # denominator (std + 1e-8); any larger difference aborts.
    check(np.allclose(y_mean, c_mean, rtol=1e-6) and
          np.allclose(y_std, c_std, rtol=1e-6, atol=1e-8),
          'npz y_mean/y_std equal train-only label stats (std up to +1e-8)')
    check(len(np.unique(idx)) == n, 'idx_val has no duplicates')
    check(np.array_equal(np.sort(idx), splits['val']), 'idx_val equals val membership of split')
    check(all(np.isfinite(v).all() for v in (f, y, emb, y_mean, y_std)), 'all values finite')
    # Everything below works in physical units.
    y, f, npz_units = npz_to_physical(y, f, y_mean, y_std)
    checks.append(f'npz labels recognised as {npz_units}; converted to physical'
                  if npz_units == 'standardized' else 'npz labels already physical')
    # No std check on pred: shrinkage legitimately lowers it.
    check(bool(np.all(np.abs(f.mean(0) - y.mean(0)) < 0.1 * y_std)),
          'pred_val in units of y_val (|mean(pred) - mean(y)| < 0.1 y_std)')

    # Only compact per-row metadata is read; X is never touched.
    with h5py.File(paths['dataset_file'], 'r') as h:
        env = h['file_group_id'][:][idx]
        y_h5 = h['y'][:][idx].astype(np.float64)
        snr_source = 'target_network_snr'
        snr = h[snr_source][:][idx] if snr_source in h else None
    check(np.allclose(y, y_h5, rtol=1e-4, atol=1e-4 * float(y_std.min())),
          'y_val equals HDF5 y[idx_val]')
    envs = np.unique(env)
    check(len(envs) >= 3, 'at least 3 environments in val')
    if snr is not None:
        check(bool(np.isfinite(snr).all()), 'target SNR finite')

    preds, fold_rows = loeo_predictions(f, y, emb, env)
    err_m11 = (f - y)**2
    err = {c: (preds[c] - y)**2 for c in CORRECTORS}
    d = np.hstack([err_m11 - err[c] for c in CORRECTORS])  # col = c*3 + j; >0 favours corrector

    masks = {('global', 'all'): np.ones(n, bool)}
    for e in envs:
        masks[('environment', str(e))] = env == e
    for j, lab in enumerate(LABELS):
        # Regions from M11 predictions (functions of X), never from the truth.
        edges = np.quantile(f[:, j], np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(edges[1:-1], f[:, j], side='right'), 0, 9)
        for q in range(10):
            masks[('pred_decile_' + lab, f'D{q + 1}')] = dec == q
    snr_info = dict(available=snr is not None)
    if snr is not None:
        in_bin = np.zeros(n, bool)
        for lo, hi, closed in SNR_BINS:
            m = (snr >= lo) & ((snr <= hi) if closed else (snr < hi))
            in_bin |= m
            masks[('target_snr', f'[{lo:g},{hi:g}' + (']' if closed else ')'))] = m
        snr_info.update(source='HDF5 ' + snr_source + '[idx_val]',
                        n_outside_bins=int((~in_bin).sum()))
    else:
        snr_info.update(reason=snr_source + ' absent from HDF5; SNR regions omitted (X not read)')
    masks = {k: m for k, m in masks.items() if m.any()}

    boot = bootstrap_means(d, env, masks, args.bootstrap, args.seed)
    q_lo, q_hi = FAMILY_ALPHA / 2, 1 - FAMILY_ALPHA / 2
    risk_rows, region_rows = [], []
    for key, m in masks.items():
        n_r = int(m.sum())
        for ci, c in enumerate(CORRECTORS):
            for j, lab in enumerate(LABELS):
                col = ci * len(LABELS) + j
                mse_m, mse_c = err_m11[m, j].mean(), err[c][m, j].mean()
                lo, hi = np.quantile(boot[key][:, col], [q_lo, q_hi])
                margin = MARGIN_FRAC * mse_m
                row = dict(region_type=key[0], region=key[1], n=n_r, corrector=c, label=lab,
                           mse_m11=mse_m, mse_corrector=mse_c, d_mean=d[m, col].mean(),
                           d_rel=d[m, col].mean() / mse_m, ci_lo=lo, ci_hi=hi, margin=margin)
                if key[0] == 'global':
                    row['decision'] = decide(lo, hi, margin)
                    risk_rows.append(row)
                else:
                    # Descriptive only: no multiplicity control across regions.
                    row['sparse'] = n_r < MIN_REGION_N
                    row['descriptive_label'] = ('SPARSE_NO_DECISION' if n_r < MIN_REGION_N
                                                else decide(lo, hi, margin))
                    region_rows.append(row)

    slope_rows = []
    for model, p in [('M11', f)] + [(c, preds[c]) for c in CORRECTORS]:
        for key, m in masks.items():
            if key[0] not in ('global', 'environment'):
                continue
            for j, lab in enumerate(LABELS):
                slope_rows.append(dict(model=model, region_type=key[0], region=key[1],
                                       n=int(m.sum()), label=lab, **slope_row(p[m, j], y[m, j])))

    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(risk_rows).to_csv(out/'risk.csv', index=False)
    pd.DataFrame(slope_rows).to_csv(out/'slopes.csv', index=False)
    pd.DataFrame(region_rows).to_csv(out/'regions.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out/'folds.csv', index=False)
    analysis = dict(
        experiment='M12.5.3 post-hoc optimality check (validation only, exploratory)',
        npz=str(npz), npz_sha256=digest(npz), script_sha256=digest(Path(__file__).resolve()),
        contract_identity=identity, labels=LABELS, npz_label_units=npz_units,
        npz_unit_conversion=INVERSE_STANDARDIZE if npz_units == 'standardized' else 'none',
        n_rows=n, n_environments=len(envs),
        rows_per_environment={str(e): int((env == e).sum()) for e in envs},
        environment_source='HDF5 file_group_id[idx_val]', snr=snr_info, checks_passed=checks,
        settings=dict(bootstrap=args.bootstrap, seed=args.seed, interval_level=1 - FAMILY_ALPHA,
                      margin_frac_of_m11_mse=MARGIN_FRAC, min_region_n=MIN_REGION_N,
                      c2_bins=N_BINS, c3_alphas=ALPHAS, c4_ks=KS, cv='LOEO, inner LOEO for C3/C4'),
        decisions={c: {r['label']: r['decision'] for r in risk_rows if r['corrector'] == c}
                   for c in CORRECTORS},
        global_results=risk_rows)
    with open(out/'analysis.json', 'x') as fh:
        json.dump(analysis, fh, indent=2, allow_nan=False,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(pd.DataFrame(risk_rows)[['corrector', 'label', 'mse_m11', 'mse_corrector', 'd_mean',
                                    'ci_lo', 'ci_hi', 'margin', 'decision']].to_string(index=False))
    print('Wrote', out)


if __name__ == '__main__':
    main()
