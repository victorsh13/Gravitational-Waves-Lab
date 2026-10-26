"""M12.5.4: origin of M11's excess contraction (training vs environment shift) by head refit on train.

The M11 head is linear on the 64-d embedding h (checked numerically below). The head is
refitted by OLS on all train rows (HR) and transferred to val; its gain is compared with
the out-of-environment C3 corrector of M12.5.3, which is fitted on val itself. M11 is not
modified (HR coefficients are stored separately); cal/test and X are never read.
Exploratory, single run. Protocol: docs/experiments/M12_5_4_head_refit_train.md
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
from scripts.m12.m12_5_3_posthoc_check import (  # noqa: E402  (read-only reuse; file is hash-pinned)
    ALPHAS, DATASET, LABELS, MIN_REGION_N, SNR_BINS, bootstrap_means, digest, load_contract,
    npz_to_physical, ridge_paths, require, select_inner, slope_row)
from scripts.m12.m12_5_3_posthoc_check import OUT_NAME as M53_OUT_NAME  # noqa: E402

NPZ_NAMES = dict(train='m11_final_G1_500k_train_predictions_embeddings.npz',
                 val='m11_final_G1_500k_val_predictions_embeddings.npz')
VAL_NPZ_SHA256 = '586b95f2cb2cfaf150fbdd05760f50f47a95a555451fcb46d89492b7d00ed88b'
OUT_NAME = 'M12_5_4_head_refit'
N_TRAIN = 400000
EMB_DIM = 64
HEAD_RESID_TOL = 1e-4   # max |f - [h,1] coef|, standardized units
HEAD_XFER_TOL = 1e-4    # max |[h_train,1] coef_val - f_train| per label, standardized units
REPRO_RTOL = 1e-6
# Bonferroni over 3 labels -> two-sided 98.333 % marginal intervals.
FAMILY_ALPHA = 0.05 / len(LABELS)
# Decision thresholds (point estimates), pre-registered in the protocol.
S_TR_TRAINING = 1.03
R_TRAINING = 0.75
S_TR_ENV_TOL = 0.01
R_ENV = 0.25
OVERFIT_RATIO = 0.9
MODELS = ['M11', 'HR', 'C3']
# Bootstrap column blocks in d: [d_HR | d_C3 | d_C3 - d_HR | e_M11^2], 3 labels each.
COL_HR, COL_C3, COL_DIFF, COL_M11 = 0, 3, 6, 9


def with_ones(h):
    return np.column_stack([h, np.ones(len(h))])


def load_split(results, split, paths, splits, c_mean, c_std, check):
    """M12.5.3 npz checks for one split; returns physical (f, y), emb, idx and units."""
    npz = results/NPZ_NAMES[split]
    # model_config / input_normalization are object arrays and are never accessed.
    with np.load(npz, allow_pickle=False) as a:
        f, y, emb = (a[f'{k}_{split}'].astype(np.float64) for k in ('pred', 'y', 'emb'))
        idx = a[f'idx_{split}'].astype(np.int64)
        y_mean, y_std = a['y_mean'].astype(np.float64), a['y_std'].astype(np.float64)
        label_names = a['label_names'].tolist()
        split_name = Path(str(a['split_path'])).name
    n = len(idx)
    check(f.shape == y.shape == (n, 3) and emb.shape == (n, EMB_DIM) and idx.shape == (n,),
          f'{split}: npz shapes (n,3), emb (n,{EMB_DIM})')
    check(label_names == LABELS, f'{split}: label order [chirp_mass, total_mass, chi_eff]')
    check(split_name == paths['split_file'].name, f'{split}: npz split_path is the pinned split file')
    # Same tolerance as M12.5.3 (historical +1e-8 in the scaler denominator).
    check(np.allclose(y_mean, c_mean, rtol=1e-6) and
          np.allclose(y_std, c_std, rtol=1e-6, atol=1e-8),
          f'{split}: npz y_mean/y_std equal train-only label stats (std up to +1e-8)')
    check(len(np.unique(idx)) == n, f'{split}: idx has no duplicates')
    check(np.array_equal(np.sort(idx), splits[split]), f'{split}: idx equals {split} membership of split')
    check(all(np.isfinite(v).all() for v in (f, y, emb, y_mean, y_std)), f'{split}: all values finite')
    y, f, units = npz_to_physical(y, f, y_mean, y_std)
    check(bool(np.all(np.abs(f.mean(0) - y.mean(0)) < 0.1 * y_std)),
          f'{split}: pred in units of y (|mean(pred) - mean(y)| < 0.1 y_std)')
    return dict(npz=npz, f=f, y=y, emb=emb, idx=idx, units=units, y_mean=y_mean, y_std=y_std)


def c3_loeo(f, y, emb, env):
    """C3 out-of-environment predictions, identical to the C3 branch of M12.5.3 loeo_predictions."""
    z, r = np.hstack([emb, f]), y - f
    out, alphas = np.empty_like(y), {}
    for e in np.unique(env):
        te, tr = env == e, env != e
        best, _ = select_inner(ridge_paths, z[tr], r[tr], env[tr])
        outer = ridge_paths(z[tr], r[tr], z[te])
        out[te] = f[te] + np.stack([outer[best[j], :, j] for j in range(len(LABELS))], 1)
        alphas[str(e)] = {lab: ALPHAS[best[j]] for j, lab in enumerate(LABELS)}
        print(f'C3 fold env={e} done', flush=True)
    return out, alphas


def region_masks(f, env, snr, with_snr=True):
    """Same regions as M12.5.3: global, environment, signed decile of M11 pred, target SNR."""
    n = len(f)
    masks = {('global', 'all'): np.ones(n, bool)}
    for e in np.unique(env):
        masks[('environment', str(e))] = env == e
    for j, lab in enumerate(LABELS):
        # Signed deciles of the M11 prediction (function of X), never of the truth.
        edges = np.quantile(f[:, j], np.linspace(0, 1, 11))
        dec = np.clip(np.searchsorted(edges[1:-1], f[:, j], side='right'), 0, 9)
        for q in range(10):
            masks[('pred_decile_' + lab, f'D{q + 1}')] = dec == q
    if with_snr and snr is not None:
        for lo, hi, closed in SNR_BINS:
            m = (snr >= lo) & ((snr <= hi) if closed else (snr < hi))
            masks[('target_snr', f'[{lo:g},{hi:g}' + (']' if closed else ')'))] = m
    return {k: m for k, m in masks.items() if m.any()}


def calib_rows(split, model, p, t, env):
    rows = []
    groups = [('global', 'all', np.ones(len(t), bool))]
    groups += [('environment', str(e), env == e) for e in np.unique(env)]
    for rt, rg, m in groups:
        for j, lab in enumerate(LABELS):
            rows.append(dict(split=split, model=model, region_type=rt, region=rg, n=int(m.sum()),
                             label=lab, mse=((p[m, j] - t[m, j])**2).mean(),
                             **slope_row(p[m, j], t[m, j])))
    return rows


def interval(x, q_lo, q_hi):
    with np.errstate(invalid='ignore', divide='ignore'):
        return tuple(np.nanquantile(x, [q_lo, q_hi]))


def decide(s_tr, r, c3_gain_positive):
    # R is only interpretable as a transferred fraction when the C3 gain is positive.
    if c3_gain_positive and s_tr >= S_TR_TRAINING and r >= R_TRAINING:
        return 'TRAINING_INDUCED'
    if c3_gain_positive and abs(s_tr - 1.0) < S_TR_ENV_TOL and r <= R_ENV:
        return 'ENV_SHIFT'
    return 'MIXED'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default=None)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=125401)
    args = ap.parse_args()
    cfg = json.loads((ROOT/'configs/experiments/train_m11_final_G1_500k.json').read_text())
    data_root = resolve_data_root(config_data_root=cfg.get('data_root'))
    results = data_root/'results'/DATASET
    out = Path(args.output) if args.output else results/OUT_NAME
    require(not out.exists(), f'Output already exists: {out}')
    require(args.bootstrap > 0, 'bootstrap must be positive')
    risk_path = results/M53_OUT_NAME/'risk.csv'
    require(risk_path.exists(), f'M12.5.3 risk.csv not found: {risk_path}')

    checks = []

    def check(ok, name):
        require(ok, 'Check failed: ' + name)
        checks.append(name)

    sha = {s: digest(results/NPZ_NAMES[s]) for s in ('train', 'val')}
    check(sha['val'] == VAL_NPZ_SHA256, 'val npz SHA256 equals pinned value')
    paths, splits, c_mean, c_std, identity = load_contract(data_root)
    checks.append('frozen G1 contract (g1_npe_contract.load_contract)')
    S = {s: load_split(results, s, paths, splits, c_mean, c_std, check) for s in ('train', 'val')}
    check(len(S['train']['idx']) == N_TRAIN, f'n_train = {N_TRAIN}')

    # Only compact per-row metadata is read; X is never touched.
    snr_source = 'target_network_snr'
    with h5py.File(paths['dataset_file'], 'r') as h:
        env_all, y_all = h['file_group_id'][:], h['y'][:].astype(np.float64)
        snr_all = h[snr_source][:] if snr_source in h else None
    for s, d in S.items():
        d['env'] = env_all[d['idx']]
        d['snr'] = snr_all[d['idx']] if snr_all is not None else None
        check(np.allclose(d['y'], y_all[d['idx']], rtol=1e-4, atol=1e-4 * float(d['y_std'].min())),
              f'{s}: y equals HDF5 y[idx]')
        if d['snr'] is not None:
            check(bool(np.isfinite(d['snr']).all()), f'{s}: target SNR finite')
    del env_all, y_all, snr_all
    check(len(np.unique(S['val']['env'])) >= 3, 'at least 3 environments in val')

    # Linear head: f is an affine function of h in each split, with the same head.
    head = {}
    for s, d in S.items():
        f_std = (d['f'] - d['y_mean']) / d['y_std']
        a = with_ones(d['emb'])
        coef, _, rank, _ = np.linalg.lstsq(a, f_std, rcond=None)
        resid = float(np.abs(a @ coef - f_std).max())
        del a
        head[s] = dict(coef=coef, max_abs_resid_std=resid, design_rank=int(rank))
        check(resid < HEAD_RESID_TOL, f'{s}: f linear in h (max |resid| {resid:.2e} < {HEAD_RESID_TOL:g}, std units)')
    # Compared in prediction space, not coefficients: coefficients are not unique if h is
    # rank-deficient, predictions are. The val head must reproduce the train predictions.
    f_tr_std = (S['train']['f'] - S['train']['y_mean']) / S['train']['y_std']
    xfer = np.abs(with_ones(S['train']['emb']) @ head['val']['coef'] - f_tr_std).max(0)
    del f_tr_std
    check(bool(np.all(xfer < HEAD_XFER_TOL)),
          f'same head in train and val (max |[h_train,1] coef_val - f_train| {xfer.max():.2e} '
          f'< {HEAD_XFER_TOL:g} per label, std units)')

    tr, va = S['train'], S['val']
    # B. Head refit: per-label OLS of theta on [h, 1] over all train rows, physical units, no penalty.
    a_tr = with_ones(tr['emb'])
    hr_coef, _, hr_rank, _ = np.linalg.lstsq(a_tr, tr['y'], rcond=None)  # (65, 3)
    f_hr_tr = a_tr @ hr_coef
    del a_tr
    f_hr = with_ones(va['emb']) @ hr_coef

    # C3 recomputed on val with the M12.5.3 settings, then checked against its risk.csv.
    f_c3, c3_alphas = c3_loeo(va['f'], va['y'], va['emb'], va['env'])
    e_m11, e_hr, e_c3 = ((p - va['y'])**2 for p in (va['f'], f_hr, f_c3))
    risk = pd.read_csv(risk_path).query("corrector == 'C3_ridge'").set_index('label').loc[LABELS]
    check(np.allclose(e_m11.mean(0), risk.mse_m11.values, rtol=REPRO_RTOL, atol=0)
          and np.allclose(e_c3.mean(0), risk.mse_corrector.values, rtol=REPRO_RTOL, atol=0),
          f'M11 and C3 val MSE reproduce M12.5.3 risk.csv (rtol {REPRO_RTOL:g})')

    # A + HR slopes.
    calib = (calib_rows('train', 'M11', tr['f'], tr['y'], tr['env'])
             + calib_rows('val', 'M11', va['f'], va['y'], va['env'])
             + calib_rows('val', 'HR', f_hr, va['y'], va['env']))

    # C. Paired bootstrap on val: identical resampled rows for every column.
    d_hr, d_c3 = e_m11 - e_hr, e_m11 - e_c3   # >0 favours the corrector
    d = np.hstack([d_hr, d_c3, d_c3 - d_hr, e_m11])
    masks = region_masks(va['f'], va['env'], va['snr'])
    boot = bootstrap_means(d, va['env'], masks, args.bootstrap, args.seed)
    q_lo, q_hi = FAMILY_ALPHA / 2, 1 - FAMILY_ALPHA / 2

    g_boot = boot[('global', 'all')]
    s_tr = {r['label']: r['slope_truth_on_pred'] for r in calib
            if (r['split'], r['model'], r['region_type']) == ('train', 'M11', 'global')}
    s_va = {r['label']: r['slope_truth_on_pred'] for r in calib
            if (r['split'], r['model'], r['region_type']) == ('val', 'M11', 'global')}
    summary = []
    for j, lab in enumerate(LABELS):
        mse_tr, mse_va = ((tr['f'][:, j] - tr['y'][:, j])**2).mean(), e_m11[:, j].mean()
        dh, dc = d_hr[:, j].mean(), d_c3[:, j].mean()
        r_pt = dh / dc
        with np.errstate(invalid='ignore', divide='ignore'):
            rel_hr_b = g_boot[:, COL_HR + j] / g_boot[:, COL_M11 + j]
            rel_c3_b = g_boot[:, COL_C3 + j] / g_boot[:, COL_M11 + j]
            r_b = g_boot[:, COL_HR + j] / g_boot[:, COL_C3 + j]
        row = dict(label=lab, n_train=len(tr['y']), n_val=len(va['y']),
                   mse_m11_train=mse_tr, mse_m11_val=mse_va,
                   mse_hr_train=((f_hr_tr[:, j] - tr['y'][:, j])**2).mean(),
                   mse_hr_val=e_hr[:, j].mean(), mse_c3_val=e_c3[:, j].mean(),
                   s_tr=s_tr[lab], s_val=s_va[lab],
                   d_hr_mean=dh, d_c3_mean=dc,
                   gain_rel_hr=dh / mse_va, gain_rel_c3=dc / mse_va, R=r_pt,
                   diff_c3_minus_hr=d[:, COL_DIFF + j].mean())
        for key, b in (('gain_rel_hr', rel_hr_b), ('gain_rel_c3', rel_c3_b), ('R', r_b),
                       ('diff_c3_minus_hr', g_boot[:, COL_DIFF + j])):
            row[key + '_ci_lo'], row[key + '_ci_hi'] = interval(b, q_lo, q_hi)
        row['c3_gain_positive'] = bool(dc > 0)
        row['decision'] = decide(row['s_tr'], r_pt, row['c3_gain_positive'])
        row['overfit_confound'] = bool(mse_tr < OVERFIT_RATIO * mse_va)
        summary.append(row)

    # E. Descriptive regions (no multiplicity control across regions).
    region_rows = []
    preds = dict(M11=va['f'], HR=f_hr, C3=f_c3)
    gain_col = dict(HR=COL_HR, C3=COL_C3)
    tot = dict(HR=d_hr.sum(0), C3=d_c3.sum(0))
    for key, m in masks.items():
        n_r = int(m.sum())
        for j, lab in enumerate(LABELS):
            mse_m = e_m11[m, j].mean()
            for mdl in MODELS:
                res = va['y'][m, j] - preds[mdl][m, j]
                row = dict(split='val', region_type=key[0], region=key[1], n=n_r, model=mdl,
                           label=lab, bias=res.mean(), mse=(res**2).mean(), mse_m11=mse_m,
                           sparse=n_r < MIN_REGION_N)
                if mdl in gain_col:
                    c = gain_col[mdl] + j
                    row['gain_rel'] = d[m, c].mean() / mse_m
                    row['gain_frac_of_global'] = d[m, c].sum() / tot[mdl][j]
                    with np.errstate(invalid='ignore', divide='ignore'):
                        rb = boot[key][:, c] / boot[key][:, COL_M11 + j]
                    row['gain_rel_ci_lo'], row['gain_rel_ci_hi'] = interval(rb, q_lo, q_hi)
                region_rows.append(row)
    # Train, M11 only: bias and MSE by signed decile of the train predictions.
    for key, m in region_masks(tr['f'], tr['env'], None, with_snr=False).items():
        if not key[0].startswith('pred_decile_'):
            continue
        for j, lab in enumerate(LABELS):
            res = tr['y'][m, j] - tr['f'][m, j]
            region_rows.append(dict(split='train', region_type=key[0], region=key[1], n=int(m.sum()),
                                    model='M11', label=lab, bias=res.mean(), mse=(res**2).mean(),
                                    mse_m11=(res**2).mean(), sparse=int(m.sum()) < MIN_REGION_N))

    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv', index=False)
    pd.DataFrame(calib).to_csv(out/'calib.csv', index=False)
    pd.DataFrame(region_rows).to_csv(out/'regions.csv', index=False)
    # theta_phys = [h, 1] @ coef; rows 0..63 multiply h, row 64 is the intercept.
    with open(out/'head_refit_coefs.npz', 'xb') as fh:
        np.savez(fh, coef=hr_coef, label_names=np.array(LABELS), units=np.array('physical'),
                 design=np.array('[emb_0..emb_63, 1]'), fit_split=np.array('train'))
    envs = {s: np.unique(S[s]['env']) for s in S}
    analysis = dict(
        experiment='M12.5.4 head refit on train (exploratory, single run)',
        npz={s: str(S[s]['npz']) for s in S}, npz_sha256=sha,
        script_sha256=digest(Path(__file__).resolve()), m12_5_3_risk_csv=str(risk_path),
        contract_identity=identity, labels=LABELS,
        npz_label_units={s: S[s]['units'] for s in S},
        n_rows={s: len(S[s]['idx']) for s in S},
        rows_per_environment={s: {str(e): int((S[s]['env'] == e).sum()) for e in envs[s]} for s in S},
        environment_source='HDF5 file_group_id[idx]',
        snr=dict(available=va['snr'] is not None, source='HDF5 ' + snr_source + '[idx]'),
        checks_passed=checks,
        linear_head=dict(max_abs_resid_std={s: head[s]['max_abs_resid_std'] for s in S},
                         design_rank={s: head[s]['design_rank'] for s in S},
                         max_abs_train_pred_from_val_head_std=dict(zip(LABELS, xfer.tolist()))),
        head_refit=dict(fit='per-label OLS theta ~ [h, 1], all train rows, no regularisation',
                        design_rank=int(hr_rank), n_params=int(hr_coef.shape[0])),
        c3=dict(settings='M12.5.3 C3_ridge: LOEO on val, inner LOEO alpha per label',
                alphas_selected=c3_alphas),
        reproduction=dict(mse_m11_val=e_m11.mean(0).tolist(), mse_c3_val=e_c3.mean(0).tolist(),
                          risk_csv_mse_m11=risk.mse_m11.tolist(),
                          risk_csv_mse_c3=risk.mse_corrector.tolist()),
        settings=dict(bootstrap=args.bootstrap, seed=args.seed, interval_level=1 - FAMILY_ALPHA,
                      bootstrap_scheme='paired, rows resampled within each val environment',
                      thresholds=dict(s_tr_training=S_TR_TRAINING, r_training=R_TRAINING,
                                      s_tr_env_tol=S_TR_ENV_TOL, r_env=R_ENV,
                                      overfit_ratio=OVERFIT_RATIO),
                      min_region_n=MIN_REGION_N),
        decisions={r['label']: r['decision'] for r in summary},
        overfit_confound={r['label']: r['overfit_confound'] for r in summary},
        # Non-finite bootstrap ratios (e.g. a replicate with zero C3 gain) are stored as null.
        global_results=[{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                         for k, v in r.items()} for r in summary])
    with open(out/'analysis.json', 'x') as fh:
        json.dump(analysis, fh, indent=2, allow_nan=False,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(pd.DataFrame(summary)[['label', 's_tr', 'mse_m11_train', 'mse_m11_val', 'gain_rel_hr',
                                 'gain_rel_c3', 'R', 'R_ci_lo', 'R_ci_hi', 'decision',
                                 'overfit_confound']].to_string(index=False))
    print('Wrote', out)


if __name__ == '__main__':
    main()
