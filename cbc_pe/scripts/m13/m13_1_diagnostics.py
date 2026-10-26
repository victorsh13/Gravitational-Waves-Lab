"""M13.1: CPU diagnostics of M11 on validation (descriptive).

(a) per-label bias/RMSE conditioned on tau30 deciles, f220 class, q and SNR (+ crosses);
(b) correlation between chi_eff and eta residuals (degeneracy);
(c) band-energy fractions of the strided-conv kernels of the M11 checkpoint.
Validation only: X is never read; cal/test are never touched. Exploratory.
Protocol: docs/experiments/M13_1_diagnostics.md
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_data_root  # noqa: E402
from scripts.m12.m12_5_3_posthoc_check import (  # noqa: E402  (read-only reuse)
    DATASET, LABELS, digest, load_contract, npz_to_physical, require)

NPZ_NAME = 'm11_final_G1_500k_val_predictions_embeddings.npz'
NPZ_SHA256 = '586b95f2cb2cfaf150fbdd05760f50f47a95a555451fcb46d89492b7d00ed88b'
CKPT_NAME = ('M11_final_G1_500k_SimpleCNN_ResidualDilated_M11_final_G1_inputzscore_resdilated_'
             'emb64_d124_train400k_MSELoss_seed123_checkpoint.pt')
OUT_NAME = 'M13_1_diagnostics'
META = ['mass_1', 'mass_2', 'spin_1z', 'spin_2z', 'target_network_snr', 'file_group_id']

N_BOOT = 1000
SEED = 131001
BOOT_CHUNK = 50
MIN_GROUP_N = 200
CHECK_RTOL = 1e-5
T_SUN = 4.925490947e-6  # G Msun / c^3 [s]
F_REF = 30.0  # Hz, reference frequency of tau30
SNR_BINS = [(10.0, 15.0, False), (15.0, 20.0, False), (20.0, 25.0, True)]  # last bin closed
Q_BINS = [(1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, np.inf)]
F220_CLASSES = [('<256', 0.0, 256.0), ('256-512', 256.0, 512.0), ('>512', 512.0, np.inf)]
CROSS_LABELS = ['chi_eff', 'total_mass']
# Final-spin fit coefficients, as fixed in the protocol.
S4, S5, T0, T2, T3 = -0.1229, 0.4537, -2.8904, -3.5171, 2.5763
A_MAX = 0.998  # Thorne limit applied to a_f and a_e
# Input sampling rate of each strided conv: every ConvBlock has stride 2 (input at 4096 Hz).
CONV_LAYERS = [('block1.conv', 4096.0, (16, 3, 16)), ('block2.conv', 2048.0, (32, 16, 16)),
               ('block3.conv', 1024.0, (64, 32, 16))]
N_FFT = 2048
HIST_BINS = 60
HIST_PCT = (0.5, 99.5)


def chirp_mass(m1, m2):
    return (m1 * m2)**0.6 / (m1 + m2)**0.2


def tau_newtonian(mc, f_ref=F_REF):
    """Leading-order (Newtonian) time to coalescence from f_ref [s]; mc in Msun."""
    return 5.0 / 256.0 * (T_SUN * mc)**(-5.0 / 3.0) * (np.pi * f_ref)**(-8.0 / 3.0)


def r_isco(a):
    """Bardeen-Press-Teukolsky ISCO radius in units of M; a < 0 is retrograde."""
    z1 = 1.0 + np.cbrt(1.0 - a**2) * (np.cbrt(1.0 + a) + np.cbrt(1.0 - a))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    return 3.0 + z2 - np.sign(a) * np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def f220_hz(m_big, m_small, chi_big, chi_small):
    """(f220 [Hz], a_f, M_f, n_af_clipped, n_ae_clipped) from the protocol fits (few-percent accuracy).

    The Barausse-Rezzolla fit can give |a_f| >= 1 at extreme q with near-maximal spin; a_f and
    a_e are clipped to the Thorne limit [-A_MAX, A_MAX] before E_isco and f220.
    """
    qr = m_small / m_big
    nu = qr / (1.0 + qr)**2
    a_t = (chi_big + chi_small * qr**2) / (1.0 + qr**2)
    a_f = (a_t + a_t * nu * (S4 * a_t + S5 * nu + T0)
           + nu * (2.0 * np.sqrt(3.0) + T2 * nu + T3 * nu**2))
    a_e = (chi_big + chi_small * qr**2) / (1.0 + qr)**2
    n_af, n_ae = int((np.abs(a_f) > A_MAX).sum()), int((np.abs(a_e) > A_MAX).sum())
    a_f, a_e = np.clip(a_f, -A_MAX, A_MAX), np.clip(a_e, -A_MAX, A_MAX)
    e_isco = np.sqrt(1.0 - 2.0 / (3.0 * r_isco(a_e)))
    e_rad = ((1.0 - e_isco) * nu
             + 4.0 * nu**2 * (4.0 * 0.04827 + 16.0 * 0.01707 * a_e * (a_e + 1.0) + e_isco - 1.0))
    m_f = (m_big + m_small) * (1.0 - e_rad)
    f = (1.5251 - 1.1568 * (1.0 - a_f)**0.1292) / (2.0 * np.pi * m_f * T_SUN)
    return f, a_f, m_f, n_af, n_ae


def snr_bins(snr):
    out = []
    for lo, hi, closed in SNR_BINS:
        m = (snr >= lo) & ((snr <= hi) if closed else (snr < hi))
        out.append((f'[{lo:g},{hi:g}' + (']' if closed else ')'), lo, hi, m))
    return out


def bootstrap_bias_rmse(r, env, G, n_boot, seed):
    """Bootstrap (n_boot, K, n_labels) of group bias and RMSE of residuals r.

    Rows are resampled with replacement within each environment (env counts fixed),
    the same scheme as M12.5.3; group sizes vary between replicates.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(env == e) for e in np.unique(env)]
    bias = np.empty((n_boot, G.shape[1], r.shape[1]))
    rmse = np.empty_like(bias)
    for start in range(0, n_boot, BOOT_CHUNK):
        b = min(BOOT_CHUNK, n_boot - start)
        w = np.zeros((b, len(r)))
        for g in groups:
            w[:, g] = rng.multinomial(len(g), np.full(len(g), 1.0 / len(g)), size=b)
        cnt = w @ G
        with np.errstate(invalid='ignore', divide='ignore'):
            for j in range(r.shape[1]):
                bias[start:start + b, :, j] = (w @ (G * r[:, [j]])) / cnt
                rmse[start:start + b, :, j] = np.sqrt((w @ (G * r[:, [j]]**2)) / cnt)
    return bias, rmse


def rank(a):
    return pd.Series(a).rank(method='average').to_numpy()


def corr_row(x, y):
    """Pearson/Spearman of (x, y) and OLS of y on x (R2 = 1 - SSE/SST)."""
    n = len(x)
    if n < 3:
        return dict(n=n, pearson=np.nan, spearman=np.nan, slope=np.nan, intercept=np.nan,
                    r2=np.nan)
    xc, yc = x - x.mean(), y - y.mean()
    slope = (xc @ yc) / (xc @ xc)
    sse = ((yc - slope * xc)**2).sum()
    return dict(n=n, pearson=np.corrcoef(x, y)[0, 1], spearman=np.corrcoef(rank(x), rank(y))[0, 1],
                slope=slope, intercept=y.mean() - slope * x.mean(), r2=1.0 - sse / (yc @ yc))


def load_state_dict(path):
    import torch  # only needed for part (c)
    try:
        obj = torch.load(str(path), map_location='cpu', weights_only=True)
        weights_only = True
    except TypeError:  # PyTorch < 1.13 (VM: 1.9.1) has no weights_only
        obj = torch.load(str(path), map_location='cpu')
        weights_only = False
    except pickle.UnpicklingError:  # project's own checkpoint holding non-tensor objects
        obj = torch.load(str(path), map_location='cpu', weights_only=False)
        weights_only = False
    if isinstance(obj, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
    state = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in obj.items()}
    return state, weights_only


def band_rows(layer, w, fs):
    """Per 1-D kernel (out, in): one-sided |H(f)|^2 (N_FFT zero-padded rfft) energy fractions.

    Interior rfft bins are weighted x2 (DC and Nyquist x1) so the fractions follow Parseval.
    The top band is [512 Hz, Nyquist]; for block3 (fs = 1024 Hz) it is the Nyquist bin only.
    """
    n_out, n_in, _ = w.shape
    p = np.abs(np.fft.rfft(w.reshape(n_out * n_in, -1), n=N_FFT, axis=1))**2
    freq = np.fft.rfftfreq(N_FFT, d=1.0 / fs)
    p[:, 1:-1] *= 2.0
    total = p.sum(1)
    bands = {'frac_0_256': freq < 256.0,
             'frac_256_512': (freq >= 256.0) & (freq < 512.0),
             'frac_512_nyq': freq >= 512.0}
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = {k: p[:, m].sum(1) / total for k, m in bands.items()}
    rows = []
    for i in range(n_out * n_in):
        rows.append(dict(layer=layer, fs_in_hz=fs, nyquist_hz=fs / 2, out_channel=i // n_in,
                         in_channel=i % n_in, energy_total=total[i],
                         **{k: v[i] for k, v in frac.items()},
                         frac_above_256=frac['frac_256_512'][i] + frac['frac_512_nyq'][i],
                         peak_freq_hz=freq[p[i].argmax()]))
    return rows


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = json.loads((ROOT/'configs/experiments/train_m11_final_G1_500k.json').read_text())
    data_root = resolve_data_root(config_data_root=cfg.get('data_root'))
    results = data_root/'results'/DATASET
    out = results/OUT_NAME
    require(not out.exists(), f'Output already exists: {out}')

    checks = []

    def check(ok, name):
        require(ok, 'Check failed: ' + name)
        checks.append(name)

    npz = results/NPZ_NAME
    npz_sha = digest(npz)
    check(npz_sha == NPZ_SHA256, 'npz SHA256 equals pinned value')
    # model_config / input_normalization are object arrays and are never accessed.
    with np.load(npz, allow_pickle=False) as a:
        f, y = (a[k].astype(np.float64) for k in ('pred_val', 'y_val'))
        idx = a['idx_val'].astype(np.int64)
        y_mean, y_std = a['y_mean'].astype(np.float64), a['y_std'].astype(np.float64)
        label_names = a['label_names'].tolist()
        split_name = Path(str(a['split_path'])).name
    n = len(idx)
    check(f.shape == y.shape == (n, 3) and idx.shape == (n,), 'npz shapes')
    check(label_names == LABELS, 'label order [chirp_mass, total_mass, chi_eff]')

    paths, splits, c_mean, c_std, identity = load_contract(data_root)
    checks.append('frozen G1 contract (g1_npe_contract.load_contract)')
    check(split_name == paths['split_file'].name, 'npz split_path is the pinned split file')
    check(np.allclose(y_mean, c_mean, rtol=1e-6) and
          np.allclose(y_std, c_std, rtol=1e-6, atol=1e-8),
          'npz y_mean/y_std equal train-only label stats (std up to +1e-8)')
    check(len(np.unique(idx)) == n, 'idx_val has no duplicates')
    check(np.array_equal(np.sort(idx), splits['val']), 'idx_val equals val membership of split')
    check(all(np.isfinite(v).all() for v in (f, y, y_mean, y_std)), 'npz values finite')
    y, f, npz_units = npz_to_physical(y, f, y_mean, y_std)
    checks.append(f'npz labels recognised as {npz_units}; converted to physical'
                  if npz_units == 'standardized' else 'npz labels already physical')
    check(bool(np.all(np.abs(f.mean(0) - y.mean(0)) < 0.1 * y_std)),
          'pred_val in units of y_val (|mean(pred) - mean(y)| < 0.1 y_std)')

    # Only compact per-row metadata is read; X is never touched.
    with h5py.File(paths['dataset_file'], 'r') as h:
        missing = [k for k in META if k not in h]
        require(not missing, f'HDF5 lacks metadata fields: {missing}')
        meta = {k: h[k][:][idx] for k in META}
    m1, m2, s1, s2, snr = (meta[k].astype(np.float64) for k in META[:5])
    env = meta['file_group_id']
    check(all(np.isfinite(v).all() for v in (m1, m2, s1, s2, snr)), 'HDF5 metadata finite')
    check(bool(np.all((m1 > 0) & (m2 > 0))), 'component masses positive')
    mtot = m1 + m2
    y_rec = np.column_stack([chirp_mass(m1, m2), mtot, (m1 * s1 + m2 * s2) / mtot])
    # Relative tolerance per label; the tiny absolute floor (1e-5 y_std) only covers chi_eff ~ 0.
    rec_err = np.abs(y_rec - y)
    check(bool(np.all(rec_err <= CHECK_RTOL * np.abs(y) + CHECK_RTOL * y_std)),
          f'(Mc, M, chi_eff) from mass_1/mass_2/spin_1z/spin_2z equal y_val (rtol {CHECK_RTOL:g})')
    envs = np.unique(env)
    check(len(envs) >= 2, 'at least 2 environments in val')

    # Derived variables from the true labels; chi1 is the spin of the heavier component.
    big = m1 >= m2
    m_big, m_small = np.where(big, m1, m2), np.where(big, m2, m1)
    chi_big, chi_small = np.where(big, s1, s2), np.where(big, s2, s1)
    q = m_big / m_small
    eta = m1 * m2 / mtot**2
    tau30 = tau_newtonian(y[:, 0])
    f220, a_f, m_f, n_af_clipped, n_ae_clipped = f220_hz(m_big, m_small, chi_big, chi_small)
    check(bool(np.all(np.isfinite(f220) & (f220 > 0) & np.isfinite(a_f) & (m_f > 0))),
          f'f220 finite and positive, a_f finite, M_f > 0 (after clipping a_f, a_e to +-{A_MAX})')
    ratio = f[:, 0] / f[:, 1]
    check(bool(np.all(ratio > 0)), 'Mc_pred / M_pred > 0')
    eta_pred = np.minimum(ratio**(5.0 / 3.0), 0.25)

    # ---------- (a) conditioned bias / RMSE ----------
    tau_edges = np.quantile(tau30, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(tau_edges[1:-1], tau30, side='right'), 0, 9)
    tau_groups = [(f'D{k + 1}', tau_edges[k], tau_edges[k + 1], dec == k) for k in range(10)]
    f_groups = [(name, lo, hi, (f220 >= lo) & ((f220 <= hi) if name == '256-512' else (f220 < hi)))
                for name, lo, hi in F220_CLASSES]
    q_groups = [(f'[{lo:g},{hi:g})', lo, hi, (q >= lo) & (q < hi)) for lo, hi in Q_BINS]
    s_groups = snr_bins(snr)

    groups = []  # (grouping, group, group2, lo, hi, mask, labels)
    for grouping, gs in (('tau30_decile', tau_groups), ('f220_class', f_groups),
                         ('q_bin', q_groups), ('snr_bin', s_groups)):
        groups += [(grouping, g, '', lo, hi, m, LABELS) for g, lo, hi, m in gs]
    for grouping, gs in (('tau30_decile_x_snr', tau_groups), ('f220_class_x_snr', f_groups)):
        groups += [(grouping, g, sg, lo, hi, m & sm, CROSS_LABELS)
                   for g, lo, hi, m in gs for sg, _, _, sm in s_groups]

    r = f - y
    G = np.column_stack([g[5] for g in groups]).astype(np.float64)
    b_bias, b_rmse = bootstrap_bias_rmse(r, env, G, N_BOOT, SEED)
    bins_rows, cross_rows = [], []
    for k, (grouping, g, g2, lo, hi, m, labs) in enumerate(groups):
        n_g = int(m.sum())
        for lab in labs:
            j = LABELS.index(lab)
            rj = r[m, j]
            with np.errstate(invalid='ignore'):
                bl, bh = np.nanquantile(b_bias[:, k, j], [0.025, 0.975]) if n_g else (np.nan,) * 2
                rl, rh = np.nanquantile(b_rmse[:, k, j], [0.025, 0.975]) if n_g else (np.nan,) * 2
            row = dict(grouping=grouping, group=g, lo=lo, hi=hi, label=lab, n=n_g,
                       sparse=n_g < MIN_GROUP_N,
                       bias=rj.mean() if n_g else np.nan, bias_ci_lo=bl, bias_ci_hi=bh,
                       rmse=np.sqrt((rj**2).mean()) if n_g else np.nan,
                       rmse_ci_lo=rl, rmse_ci_hi=rh)
            if g2:
                row = dict(row, snr_bin=g2)
                cross_rows.append(row)
            else:
                bins_rows.append(row)

    # ---------- (b) chi_eff / eta residual degeneracy ----------
    r_chi, r_eta = f[:, 2] - y[:, 2], eta_pred - eta
    mc_edges = np.quantile(y[:, 0], [0, 1 / 3, 2 / 3, 1])
    ter = np.clip(np.searchsorted(mc_edges[1:-1], y[:, 0], side='right'), 0, 2)
    b_groups = [('global', 'all', np.nan, np.nan, np.ones(n, bool))]
    b_groups += [('mc_tercile', f'T{k + 1}', mc_edges[k], mc_edges[k + 1], ter == k)
                 for k in range(3)]
    b_groups += [('f220_class', g, lo, hi, m) for g, lo, hi, m in f_groups]
    corr_rows = []
    for grouping, g, lo, hi, m in b_groups:
        c = corr_row(r_eta[m], r_chi[m])
        corr_rows.append(dict(grouping=grouping, group=g, lo=lo, hi=hi, sparse=c['n'] < MIN_GROUP_N,
                              **c))
    h_range = [np.percentile(r_eta, HIST_PCT).tolist(), np.percentile(r_chi, HIST_PCT).tolist()]
    counts, e_eta, e_chi = np.histogram2d(r_eta, r_chi, bins=HIST_BINS, range=h_range)
    sp = {row['group']: row['spearman'] for row in corr_rows}
    t_abs = [abs(sp[f'T{k + 1}']) for k in range(3)]
    degeneracy = dict(spearman_global=sp['all'], spearman_mc_terciles=[sp[f'T{k + 1}'] for k in range(3)],
                      abs_global_ge_0_3=bool(abs(sp['all']) >= 0.3),
                      low_mc_abs_exceeds_global=bool(t_abs[0] > abs(sp['all'])),
                      low_mc_abs_largest_tercile=bool(t_abs[0] == max(t_abs)))

    # ---------- (c) conv kernel band energy ----------
    ckpt = data_root/'models'/'checkpoints'/DATASET/CKPT_NAME
    c_info = dict(checkpoint=str(ckpt))
    filt_rows = []
    if not ckpt.exists():
        c_info.update(status='skipped', reason='checkpoint not found')
        print('WARNING: checkpoint not found; part (c) skipped:', ckpt, flush=True)
    else:
        state, weights_only = load_state_dict(ckpt)
        for layer, fs, shape in CONV_LAYERS:
            key = layer + '.weight'
            require(key in state, f'{key} not in checkpoint state_dict')
            w = state[key].detach().cpu().numpy().astype(np.float64)
            check(w.shape == shape, f'{key} shape {shape}')
            filt_rows += band_rows(layer, w, fs)
        b1 = pd.DataFrame([row for row in filt_rows if row['layer'] == 'block1.conv'])
        n_hi = int((b1['frac_above_256'] > 0.3).sum())
        c_info.update(status='done', checkpoint_sha256=digest(ckpt), torch_weights_only=weights_only,
                      kernel_definition='1-D kernel per (out_channel, in_channel)',
                      n_fft=N_FFT, input_fs_hz={name: fs for name, fs, _ in CONV_LAYERS},
                      block1_readout=dict(n_kernels=len(b1), n_frac_above_256_gt_0_3=n_hi,
                                          fraction=n_hi / len(b1),
                                          fraction_lt_0_10=bool(n_hi / len(b1) < 0.10)))

    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(bins_rows).to_csv(out/'a_bins.csv', index=False)
    pd.DataFrame(cross_rows).to_csv(out/'a_cross.csv', index=False)
    pd.DataFrame(corr_rows).to_csv(out/'b_corr.csv', index=False)
    if filt_rows:
        pd.DataFrame(filt_rows).to_csv(out/'c_filters.csv', index=False)
    f_cls_n = {g: int(m.sum()) for g, _, _, m in f_groups}
    analysis = dict(
        experiment='M13.1 CPU diagnostics of M11 on validation (descriptive, exploratory)',
        npz=str(npz), npz_sha256=npz_sha, script_sha256=digest(Path(__file__).resolve()),
        contract_identity=identity, labels=LABELS, npz_label_units=npz_units,
        n_rows=n, n_environments=len(envs), environment_source='HDF5 file_group_id[idx_val]',
        metadata_source={k: f'HDF5 {k}[idx_val]' for k in META}, checks_passed=checks,
        reconstruction_max_abs_err={lab: float(rec_err[:, j].max()) for j, lab in enumerate(LABELS)},
        n_mass1_lt_mass2=int((~big).sum()),
        n_eta_pred_clipped=int((ratio**(5.0 / 3.0) > 0.25).sum()),
        n_af_clipped=n_af_clipped, n_ae_clipped=n_ae_clipped, spin_clip_limit=A_MAX,
        n_outside_snr_bins=int((~np.any([m for _, _, _, m in s_groups], axis=0)).sum()),
        n_per_f220_class=f_cls_n,
        f220_percentiles_hz={str(p): float(np.percentile(f220, p)) for p in (0, 1, 50, 99, 100)},
        settings=dict(bootstrap=N_BOOT, seed=SEED, ci='percentile 95 %, within-environment resampling',
                      min_group_n=MIN_GROUP_N, f_ref_hz=F_REF, t_sun_s=T_SUN,
                      tau30_decile_edges_s=tau_edges.tolist(), mc_tercile_edges=mc_edges.tolist(),
                      q_bins=[[lo, str(hi)] for lo, hi in Q_BINS],
                      snr_bins=[g for g, _, _, _ in s_groups],
                      f220_classes='<256: f<256; 256-512: 256<=f<=512; >512: f>512',
                      groups_defined_on='true labels (val quantiles)',
                      eta_pred='(Mc_pred/M_pred)^(5/3) clipped to <= 0.25'),
        degeneracy_readout=degeneracy,
        b_hist2d=dict(x='r_eta', y='r_chi', range_percentiles=list(HIST_PCT),
                      x_edges=e_eta.tolist(), y_edges=e_chi.tolist(),
                      counts=counts.astype(int).tolist(), n_outside=int(n - counts.sum())),
        part_c=c_info)
    with open(out/'analysis.json', 'x') as fh:
        json.dump(analysis, fh, indent=2, allow_nan=False,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(pd.DataFrame(corr_rows)[['grouping', 'group', 'n', 'pearson', 'spearman', 'r2']]
          .to_string(index=False))
    print('Wrote', out)


if __name__ == '__main__':
    main()
