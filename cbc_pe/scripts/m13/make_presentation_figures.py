"""Presentation figures for M13 (read-only; descriptive).

Every plotted value is read from an existing closed artifact; nothing is re-estimated:
  - M13.1 CSVs (M11 diagnostics by f220 class x SNR)                          -> F1
  - history.npz of the M13.3a / M13.3b / M13.4a / M13.6 runs                   -> F2, F5, T1
  - M13_3b_4a_evaluation_summary.json (deciles, bands, SNR for M11 and G1)     -> F3, F4, F7
  - m13_5_summary.json (re-noise MSEs)                                         -> F6
  - latest M13_6_evaluation_summary_*.json (deciles, tau30, bands, SNR, rho)   -> F7-F11
Bins (SNR, tau30 deciles, f220 / M_total classes) and the paired row bootstrap are therefore exactly those of the
notebooks that wrote the summaries. Optional models follow the M13_6_evaluation rule (included if their final val npz
exists); summary-based figures can only show models present in that summary.

Output: <DATA_ROOT>/results/M13_presentation/figures/ (PNG 300 dpi + PDF) and figures_manifest.json.
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_data_root  # noqa: E402
from scripts.m12.m12_5_3_posthoc_check import LABELS, digest, require  # noqa: E402

# ---- Models (same specs and availability rule as M13_6_evaluation, cell 1) -------------------------------------
MODEL_SPECS = [
    ('3b_s123', 'M13_3b_G3_s123', False, True),
    ('3b_s124', 'M13_3b_G3_s124', False, True),
    ('4a_s123', 'M13_4a_G3_s123', False, True),
    ('6a_aug_s123', 'M13_6a_G3aug_s123', True, True),
    ('6a_aug_s124', 'M13_6a_G3aug_s124', True, False),
    ('4a_aug_s123', 'M13_4a_G3aug_s123', True, True),
    ('4a_aug_s124', 'M13_4a_G3aug_s124', True, False),
    ('4b_aug_s123', 'M13_4b_G3aug_s123', True, False),
]
G1_HIST_SPECS = [('R1', 'M13_3a_R1_s123'), ('R2', 'M13_3a_R2_s123'), ('R3', 'M13_3a_R3_s124')]
VAL_NPZ, HIST_NPZ = 'val_predictions_embeddings.npz', 'history.npz'
RENOISE_TAGS = {'3b_s123': 'M13_3b_G3_s123', '3b_s124': 'M13_3b_G3_s124', '4a_s123': 'M13_4a_G3_s123'}
SUMMARY_RTOL = 1e-9  # identical quantities written by two notebooks must agree to float round-off

# ---- Presentation style ----------------------------------------------------------------------------------------
GRID, REF = '#E1E6EC', '#8A98A6'
# Set key -> (legend name, colour, line style). Set keys follow the summaries: 'G1' = mean R2/R3, '3b' = mean of the
# 3b seeds ('G3' in the M13.3b/4a summary), '6a' / '4a_aug' / '4b_aug' = mean of the available seeds of that set.
STYLE = {
    'M11': ('Reference CNN (old data)', '#8A98A6', '-'),
    'G1': ('Corrected training (old data)', '#B8B7D9', '-'),
    '3b': ('New data: 30–1024 Hz, 16 s whitening', '#AFCBE3', '-'),
    '6a': ('+ noise augmentation', '#6F91B6', '-'),
    '4a_aug': ('+ transformer aggregation', '#243B53', '-'),
    '4a_s123': ('Transformer, no augmentation', '#C98B8B', '--'),
    '4b_aug': ('Dense multi-scale + transformer', '#5E8F6C', '-'),
    'R1': ('Dropout active throughout (old data)', '#D4A373', '-'),
}
SET_OF = {'R1': 'R1', 'R2': 'G1', 'R3': 'G1', '3b_s123': '3b', '3b_s124': '3b', '4a_s123': '4a_s123',
          '6a_aug_s123': '6a', '6a_aug_s124': '6a', '4a_aug_s123': '4a_aug', '4a_aug_s124': '4a_aug',
          '4b_aug_s123': '4b_aug'}
VAR = {'chirp_mass': r'$\mathcal{M}_c$ [M$_\odot$]', 'total_mass': r'$M$ [M$_\odot$]', 'chi_eff': r'$\chi_{\rm eff}$'}
FIG13, FIG1 = (13, 4.2), (6.5, 4.5)
SEQ3 = ['#AFCBE3', '#6F91B6', '#243B53']  # ordered categories (SNR bins, re-noise bars, rho regimes)


class Skip(Exception):
    """A figure cannot be produced from the available inputs (reported, not fatal)."""


def set_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
        'axes.labelsize': 13, 'axes.titlesize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 11, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'errorbar.capsize': 3, 'lines.linewidth': 1.6, 'lines.markersize': 5,
    })


def refline(ax, value):
    ax.axhline(value, color=REF, lw=0.9, zorder=1)


def errbars(ax, x, d, lo, hi, color, label=None):
    d, lo, hi = (np.asarray(v, float) for v in (d, lo, hi))
    ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt='o', color=color, ecolor=color, capsize=3, label=label)


def legend_below(fig, ncol=3):
    seen = {}
    for ax in fig.axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    if seen:
        fig.legend(list(seen.values()), list(seen), loc='upper center', bbox_to_anchor=(0.5, 0.0),
                   ncol=min(ncol, len(seen)))


def pretty_cls(c):
    """'[10,30)' -> '10–30'; '256-512' -> '256–512' (tick labels only; class definitions are unchanged)."""
    if c.startswith('['):
        a, b = c[1:-1].split(',')
        return f'{a}–{b}'
    return c.replace('-', '–')


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def col(rows, key):
    return np.array([r[key] for r in rows], float)


# ---- Context: inputs, checks and cached loads ------------------------------------------------------------------
class Ctx:
    def __init__(self, data_root):
        self.data_root = data_root
        self.m13_dir = data_root / 'results' / 'M13_3'
        self.m13_1_dir = data_root / 'results' / 'M11_final_G1_500k' / 'M13_1_diagnostics'
        g3_cfg = json.loads((ROOT / 'configs/experiments/m13/train_m13_3b_G3_s123.json').read_text())
        self.g3_dataset_id = g3_cfg['dataset']['dataset_id']
        self.m13_5_path = data_root / 'processed' / self.g3_dataset_id / 'm13_5_renoise' / 'm13_5_summary.json'
        cands = sorted(self.m13_dir.glob('M13_6_evaluation_summary_*.json'))  # timestamped names sort by time
        self.s6_path = cands[-1] if cands else None
        self._json, self._hist, self._sha = {}, {}, {}

        # Availability rule of M13_6_evaluation (cell 1): optional models skipped with a warning.
        self.tag, self.available, self.skipped = {}, [], []
        for key, tag, _aug, req in MODEL_SPECS:
            if (self.m13_dir / tag / VAL_NPZ).exists():
                self.available.append(key)
                self.tag[key] = tag
            else:
                require(not req, f'Required model {key}: missing {self.m13_dir / tag / VAL_NPZ}')
                self.skipped.append(key)
                warnings.warn(f'Optional model {key} ({tag}) skipped: {VAL_NPZ} not found')
        self.tag.update(dict(G1_HIST_SPECS))
        self.checks = []

    def check(self, ok, name):
        require(bool(ok), 'Check failed: ' + name)
        self.checks.append(name)

    def sha(self, path):
        p = str(Path(path).resolve())
        if p not in self._sha:
            self._sha[p] = digest(Path(p))
        return self._sha[p]

    def json(self, path):
        if path is None or not Path(path).exists():
            raise Skip(f'missing input {path}')
        p = str(Path(path).resolve())
        if p not in self._json:
            self._json[p] = json.loads(Path(p).read_text())
        return self._json[p]

    def s3b4a(self):
        return self.json(self.m13_dir / 'M13_3b_4a_evaluation_summary.json')

    def s3a(self):
        return self.json(self.m13_dir / 'M13_3a_evaluation_summary.json')

    def s6(self):
        s = self.json(self.s6_path)
        extra = [m for m in self.available if m not in s['models_used']]
        if extra and not getattr(self, '_warned_s6', False):
            warnings.warn(f'{extra} available on disk but not in {self.s6_path.name}: omitted from summary-based '
                          'figures; re-run M13_6_evaluation to include them')
            self._warned_s6 = True
        return s

    def summary_sha(self, path):
        """sha256 recorded for `path` by any summary already loaded (None if no summary lists it)."""
        p = str(Path(path).resolve())
        for s in self._json.values():
            for v in (s.get('inputs') or {}).values():
                if isinstance(v, dict) and str(Path(v.get('path', '')).resolve()) == p:
                    return v['sha256']
        return None

    def hist_path(self, key):
        return self.m13_dir / self.tag[key] / HIST_NPZ

    def hist(self, key):
        """history.npz with the checks of M13_6_evaluation cell 3, plus SHA256 against the evaluation summaries."""
        if key in self._hist:
            return self._hist[key]
        path = self.hist_path(key)
        if not path.exists():
            raise Skip(f'missing {path}')
        for loader in (self.s3a, self.s3b4a, self.s6):
            try:
                loader()
            except Skip:
                pass
        ref = self.summary_sha(path)
        if ref is not None:
            self.check(self.sha(path) == ref, f'{key}: history SHA256 equals evaluation summary')
        with np.load(path, allow_pickle=False) as a:
            h = {k: a[k] for k in a.files}
        self.check(h['label_names'].tolist() == LABELS, f'{key}: history label order')
        E = len(h['epoch'])
        self.check(h['val_mse'].shape == h['train40k_mse'].shape == h['val_slope'].shape == (E, 3)
                   and h['val_loss'].shape == (E,), f'{key}: history shapes')
        self.check(len(np.unique(h['epoch'])) == E and np.isfinite(h['val_loss']).all(),
                   f'{key}: history epochs unique, val_loss finite')
        self._hist[key] = h
        return h

    def s6_sets(self):
        """Summary sets of M13_6 (cell 13): '3b', '6a' and the augmented architectures it evaluated."""
        s = self.s6()
        return ['3b', '6a'] + [a for a in ('4a_aug', '4b_aug') if a in s['settings']['arch_aug']]

    def s6_arch_models(self, sets=None):
        """Seed models behind the M13_6 architecture sets. `arch_aug` is {set: [seeds]} in current summaries and a
        plain list of single-seed keys in summaries written before the second 4a_aug seed."""
        arch = self.s6()['settings']['arch_aug']
        if not isinstance(arch, dict):
            arch = {a: [a] for a in arch}
        return [m for a, ms in arch.items() if sets is None or a in sets for m in ms]


def save(fig, out_dir, name):
    files = []
    for ext, kw in (('png', dict(dpi=300)), ('pdf', {})):
        p = out_dir / f'{name}.{ext}'
        fig.savefig(p, bbox_inches='tight', **kw)
        files.append(p.name)
    plt.close(fig)
    return files


# ---- Figures ---------------------------------------------------------------------------------------------------
def fig_f1(ctx, out):
    """M13_1_diagnostics cell 8 (RMSE panels only): M11 val RMSE by f220 class x SNR, row-bootstrap CI of M13.1."""
    d = ctx.m13_1_dir
    paths = [d / 'analysis.json', d / 'a_bins.csv', d / 'a_cross.csv']
    for p in paths:
        if not p.exists():
            raise Skip(f'missing {p}')
    analysis = ctx.json(paths[0])
    a_bins, a_cross = pd.read_csv(paths[1]), pd.read_csv(paths[2])
    s3b4a_path = ctx.m13_dir / 'M13_3b_4a_evaluation_summary.json'
    if s3b4a_path.exists():
        ctx.check(analysis['npz_sha256'] == ctx.s3b4a()['inputs']['M11_val']['sha256'],
                  'M13.1 diagnostics computed on the M11 val npz of the M13 evaluations')
    snr_order = analysis['settings']['snr_bins']
    classes = list(a_bins.loc[a_bins.grouping == 'f220_class', 'group'].unique())
    x = np.arange(len(classes))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    for ax, lab in zip(axes, ['total_mass', 'chi_eff']):
        for k, s in enumerate(snr_order):
            t = a_cross[(a_cross.grouping == 'f220_class_x_snr') & (a_cross.label == lab) & (a_cross.snr_bin == s)]
            t = t.set_index('group').reindex(classes)
            require(t['rmse'].notna().all(), f'F1: missing f220 x SNR cells for {lab} {s}')
            xs = x + 0.1 * (k - (len(snr_order) - 1) / 2)
            a, b = s[1:-1].split(',')
            errbars(ax, xs, t.rmse, t.rmse_ci_lo, t.rmse_ci_hi, SEQ3[k % 3], label=f'SNR {a}–{b}')
            sp = t['sparse'].to_numpy(bool)
            if sp.any():  # as M13_1 cell 4: rings mark sparse cells
                ax.scatter(xs[sp], t.rmse.to_numpy()[sp], s=120, facecolors='none', edgecolors=REF, zorder=3)
        ax.set(title=VAR[lab], xticks=x, xlabel=r'$f_{220}$ [Hz]', ylabel='RMSE')
        ax.set_xticklabels([pretty_cls(c) for c in classes])
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='f1_m11_by_f220', files=save(fig, out, 'f1_m11_by_f220'), inputs=paths, models=['M11'])]


def fig_f2(ctx, out):
    """M13_3a_evaluation cell 7, bottom row: val slope (truth on prediction) vs epoch; R2/R3 averaged."""
    epoch_mark = int(ctx.s3a()['settings']['epoch_mark'])
    h = {k: ctx.hist(k) for k in ('R1', 'R2', 'R3')}
    ctx.check(not h['R1']['dropout_off'].astype(bool).any(), 'R1: dropout active in every epoch')
    for k in ('R2', 'R3'):
        off = np.flatnonzero(h[k]['dropout_off'].astype(bool))
        ctx.check(off.size and int(h[k]['epoch'][off[0]]) == epoch_mark, f'{k}: dropout off from epoch {epoch_mark}')
    ctx.check(np.array_equal(h['R2']['epoch'], h['R3']['epoch']), 'R2/R3: same epochs')
    curves = {'R1': (h['R1']['epoch'], h['R1']['val_slope']),
              'G1': (h['R2']['epoch'], (h['R2']['val_slope'] + h['R3']['val_slope']) / 2.0)}
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for j, (ax, lab) in enumerate(zip(axes, LABELS)):
        for s, (ep, v) in curves.items():
            name, c, ls = STYLE[s]
            ax.plot(ep, v[:, j], ls, color=c, label=name)
        ax.axvline(epoch_mark, color=REF, ls=':', lw=1.2, label='Dropout switched off')
        refline(ax, 1.0)
        ax.set(title=VAR[lab], xlabel='Epoch')
    axes[0].set_ylabel('Slope of truth on prediction')
    fig.tight_layout()
    legend_below(fig)
    paths = [ctx.m13_dir / 'M13_3a_evaluation_summary.json'] + [ctx.hist_path(k) for k in h]
    return [dict(figure='f2_slope_vs_epoch', files=save(fig, out, 'f2_slope_vs_epoch'), inputs=paths,
                 models=['R1', 'R2', 'R3'])]


def plot_bias_deciles(rows, sets, out, name):
    """Bias (prediction - truth) vs decile mean of the truth (M13_3a cell 13 / M13_6 cell 13 definition)."""
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        for s, key in sets:
            t = sorted(sel(rows, set=key, label=lab), key=lambda r: r['decile'])
            require(len(t) == 10, f'{name}: expected 10 deciles for {key} {lab}')
            nm, c, ls = STYLE[s]
            ax.plot(col(t, 'center'), col(t, 'bias'), 'o' + ls, color=c, label=nm)
        refline(ax, 0.0)
        ax.set(title=VAR[lab], xlabel='True value (decile mean)')
    axes[0].set_ylabel('Bias (prediction − truth)')
    fig.tight_layout()
    legend_below(fig)
    return save(fig, out, name)


def fig_f3(ctx, out):
    """M13_3a cell 13 deciles, as stored per set (M11, mean R2/R3) in the M13.3b/4a summary (cell 13)."""
    s = ctx.s3b4a()
    files = plot_bias_deciles(s['tails_deciles'], [('M11', 'M11'), ('G1', 'G1')], out, 'f3_bias_vs_truth')
    return [dict(figure='f3_bias_vs_truth', files=files, inputs=[ctx.m13_dir / 'M13_3b_4a_evaluation_summary.json'],
                 models=['M11'] + s['settings']['g1_pair'])]


def plot_band(rows, comparisons, out, name, ylabel, f220_classes=None):
    """2x3 band panels of M13_3b_4a cell 9 / M13_6 cell 11.

    comparisons: [(comparison key, or None when the rows hold a single comparison; colour set key; legend name)].
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.0))
    for i, grouping in enumerate(('f220', 'M_total')):
        for j, lab in enumerate(LABELS):
            ax = axes[i, j]
            for k, (cmp, s, nm) in enumerate(comparisons):
                kw = dict(grouping=grouping, label=lab)
                if cmp is not None:
                    kw['comparison'] = cmp
                t = [r for r in sel(rows, **kw) if r['n'] > 0]
                if grouping == 'f220' and f220_classes is not None:
                    t = [next(r for r in t if r['cls'] == c) for c in f220_classes]
                x = np.arange(len(t))
                errbars(ax, x + 0.15 * (k - (len(comparisons) - 1) / 2), col(t, 'delta_rmse_pct'),
                        col(t, 'ci95_lo_pct'), col(t, 'ci95_hi_pct'), STYLE[s][1], label=nm)
                ax.set_xticks(x)
                ax.set_xticklabels([pretty_cls(r['cls']) for r in t])
            refline(ax, 0.0)
            ax.set(xlabel=r'$f_{220}$ [Hz]' if grouping == 'f220' else r'$M$ [M$_\odot$]')
            if i == 0:
                ax.set_title(VAR[lab])
        axes[i, 0].set_ylabel(ylabel)
    fig.tight_layout()
    if len(comparisons) > 1:
        legend_below(fig)
    return save(fig, out, name)


def fig_f4(ctx, out):
    """M13_3b_4a cell 9: Delta RMSE % (mean 3b vs mean R2/R3) by f220 class and M_total, paired row-bootstrap CI."""
    s = ctx.s3b4a()
    # Non-overlapping f220 classes used for the band reading (same three classes as F1).
    files = plot_band(s['band'], [(None, '3b', 'New data vs old data')], out, 'f4_g3_vs_g1_bands',
                      'ΔRMSE, new vs old data [%]', f220_classes=['<256', '256-512', '>512'])
    return [dict(figure='f4_g3_vs_g1_bands', files=files, inputs=[ctx.m13_dir / 'M13_3b_4a_evaluation_summary.json'],
                 models=s['settings']['g1_pair'] + s['settings']['g3_pair'])]


def fig_f5(ctx, out):
    """Bottom rows of M13_3b_4a cell 7 and M13_6 cell 7: MSE_train40k / MSE_val vs epoch, each seed drawn."""
    order = [k for k, *_ in MODEL_SPECS]
    models = [m for m in order if m in ctx.available]
    h = {m: ctx.hist(m) for m in models}
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for j, (ax, lab) in enumerate(zip(axes, LABELS)):
        for m in models:
            ep, ok = h[m]['epoch'], np.isfinite(h[m]['train40k_mse'][:, j])
            name, c, ls = STYLE[SET_OF[m]]
            ax.plot(ep[ok], h[m]['train40k_mse'][ok, j] / h[m]['val_mse'][ok, j], '.' + ls, color=c, ms=4,
                    lw=1.2, label=name)
        refline(ax, 1.0)
        ax.set(title=VAR[lab], xlabel='Epoch')
    axes[0].set_ylabel(r'MSE$_{\rm train}$ / MSE$_{\rm val}$')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='f5_train_val_ratio', files=save(fig, out, 'f5_train_val_ratio'),
                 inputs=[ctx.hist_path(m) for m in models], models=models)]


def fig_f6(ctx, out):
    """m13_5_summary.json: MSE on training sources (orig / new noise) and on validation sources (val40k)."""
    m = ctx.json(ctx.m13_5_path)
    ctx.check(m['labels'] == LABELS, 'M13.5 summary label order')
    ctx.check(m['dataset_id'] == ctx.g3_dataset_id, 'M13.5 summary dataset is the G3 dataset')
    bars = [('MSE_train_orig', 'training sources, original noise'),
            ('MSE_train_new', 'same sources, new noise realisation'),
            ('MSE_val40k', 'validation sources')]
    entries = []
    for key, tag in RENOISE_TAGS.items():
        if tag not in m['results']:
            warnings.warn(f'F6: {tag} not in {ctx.m13_5_path.name}; skipped')
            continue
        r = m['results'][tag]
        fig, axes = plt.subplots(1, 3, figsize=FIG13)
        for ax, lab in zip(axes, LABELS):
            for k, (fld, nm) in enumerate(bars):
                ax.bar(k, r[lab][fld], width=0.7, color=SEQ3[2 - k], label=nm)
            ax.set(title=VAR[lab], xticks=[])
        axes[0].set_ylabel('MSE (standardised units)')
        fig.tight_layout()
        legend_below(fig)
        name = f'f6_renoise_bars_{key}'
        entries.append(dict(figure=name, files=save(fig, out, name), inputs=[ctx.m13_5_path], models=[key]))
    if not entries:
        raise Skip('no M13.5 checkpoint results')
    return entries


def fig_f7(ctx, out):
    """RMSE vs SNR bin mean (M13_3b_4a cell 13 for M11 / R2-R3, M13_6 cell 13 for the G3 chain)."""
    sa, s6 = ctx.s3b4a(), ctx.s6()
    # Both summaries use G3-HDF5 SNR and the same 3b seeds, so their '3b'/'G3' rows must coincide.
    g3 = sel(sa['snr'], set='G3')
    b3 = sel(s6['snr'], set='3b')
    ctx.check([r['snr_bin'] for r in g3] == [r['snr_bin'] for r in b3]
              and all(np.allclose(col(g3, k), col(b3, k), rtol=SUMMARY_RTOL) for k in ['snr_center'] + LABELS),
              'SNR table: 3b rows of M13.6 summary equal G3 rows of M13.3b/4a summary')
    series = [('M11', sel(sa['snr'], set='M11')), ('G1', sel(sa['snr'], set='G1'))]
    series += [(s, sel(s6['snr'], set=s)) for s in ctx.s6_sets()]
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        for s, t in series:
            require(len(t) == len(sa['settings']['snr_edges']) - 1, f'F7: SNR bins for {s}')
            nm, c, ls = STYLE[s]
            ax.plot(col(t, 'snr_center'), col(t, lab), 'o' + ls, color=c, label=nm)
        ax.set(title=VAR[lab], xlabel='Network SNR (bin mean)')
    axes[0].set_ylabel('RMSE')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='f7_rmse_vs_snr', files=save(fig, out, 'f7_rmse_vs_snr'),
                 inputs=[ctx.m13_dir / 'M13_3b_4a_evaluation_summary.json', ctx.s6_path],
                 models=['M11'] + sa['settings']['g1_pair'] + s6['settings']['b3'] + s6['settings']['a6']
                 + ctx.s6_arch_models())]


def fig_f8(ctx, out):
    """As F3 for the G3 chain, from M13_6 cell 13 (tails_deciles)."""
    s6 = ctx.s6()
    sets = [(s, s) for s in ctx.s6_sets()]
    files = plot_bias_deciles(s6['tails_deciles'], sets, out, 'f8_bias_vs_truth_chain')
    return [dict(figure='f8_bias_vs_truth_chain', files=files, inputs=[ctx.s6_path],
                 models=s6['settings']['b3'] + s6['settings']['a6'] + ctx.s6_arch_models())]


def fig_f9(ctx, out):
    """M13_6 cell 9: Delta RMSE % of each augmented architecture vs mean 6a by tau30 decile, paired row CI."""
    s6 = ctx.s6()
    arch = [a for a in ('4a_aug', '4b_aug') if a in s6['settings']['arch_aug']]
    if not arch:
        raise Skip('no augmented architecture in M13.6 summary')
    x = np.arange(1, 11)
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        for k, a in enumerate(arch):
            t = sorted(sel(s6['tau_deciles'], model=a, label=lab), key=lambda r: r['tau_decile'])
            require(len(t) == 10, f'F9: expected 10 tau30 deciles for {a} {lab}')
            errbars(ax, x + 0.15 * (k - (len(arch) - 1) / 2), col(t, 'delta_rmse_pct'), col(t, 'ci95_lo_pct'),
                    col(t, 'ci95_hi_pct'), STYLE[a][1], label=STYLE[a][0])
        refline(ax, 0.0)
        ax.set(title=VAR[lab], xlabel=r'$\tau_{30}$ decile (1 = shortest)', xticks=x)
    axes[0].set_ylabel('ΔRMSE vs CNN + augmentation [%]')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='f9_context_tau_deciles', files=save(fig, out, 'f9_context_tau_deciles'),
                 inputs=[ctx.s6_path], models=s6['settings']['a6'] + ctx.s6_arch_models(arch))]


def fig_f10(ctx, out):
    """M13_6 cell 11: Delta RMSE % with paired row CI by f220 class and M_total."""
    s6 = ctx.s6()
    cmps = [('6a_vs_3b', '6a', 'Noise augmentation vs none')]
    if '4a_aug' in s6['settings']['arch_aug']:
        cmps.append(('4a_aug_vs_6a', '4a_aug', 'Transformer vs CNN (both augmented)'))
    files = plot_band(s6['band'], cmps, out, 'f10_context_bands', 'ΔRMSE [%]')
    return [dict(figure='f10_context_bands', files=files, inputs=[ctx.s6_path],
                 models=s6['settings']['b3'] + s6['settings']['a6'] + ctx.s6_arch_models([c[1] for c in cmps[1:]]))]


def fig_f11(ctx, out):
    """M13_6 cell 17 (early_vs_final_spearman): Spearman of model order at epoch k vs final order."""
    s6 = ctx.s6()
    names = {'sin_aug': 'Without augmentation', 'con_aug': 'With augmentation', 'G3': 'All new-data models'}
    fig, ax = plt.subplots(figsize=FIG1)
    drawn = []
    for k, (reg, nm) in enumerate(names.items()):
        t = sorted(sel(s6['early_vs_final_spearman'], regime=reg), key=lambda r: r['epoch'])
        rho = col(t, 'spearman_vs_final')
        ok = np.isfinite(rho)
        if not ok.any():
            warnings.warn(f'F11: regime {reg} has no finite Spearman (n < 3 models); omitted')
            continue
        n = max(int(r['n']) for r in t)
        ax.plot(col(t, 'epoch')[ok], rho[ok], 'o-', color=SEQ3[k], label=f'{nm} (n = {n})')
        drawn.append(reg)
    if not drawn:
        plt.close(fig)
        raise Skip('no regime with finite Spearman correlation')
    refline(ax, 0.0)
    ax.set(xlabel='Epoch k', ylabel='Spearman ρ (order at k vs final)', ylim=(-1.05, 1.05))
    ax.legend(loc='lower right')
    fig.tight_layout()
    models = sorted({m for r in s6['early_vs_final_spearman'] if r['regime'] in drawn for m in r['models'].split(',')})
    return [dict(figure='f11_early_ranking', files=save(fig, out, 'f11_early_ranking'), inputs=[ctx.s6_path],
                 models=models)]


def table_t1(ctx, out):
    """Median per-epoch wall time from history.npz (train_m13_3.py records train/val/train40k times per epoch)."""
    keys = [k for k, _ in G1_HIST_SPECS] + ctx.available
    rows, used, missing = [], [], []
    for m in keys:
        try:
            h = ctx.hist(m)
        except Skip:
            continue
        if 'train_time' not in h:
            missing.append(m)
            continue
        tt, tv = h['train_time'].astype(float), h['val_time'].astype(float)
        t40 = h['train40k_time'].astype(float) if 'train40k_time' in h else np.full_like(tt, np.nan)
        # Epoch time = train + val + train40k (train40k only at the epochs where it was evaluated).
        rows.append(dict(model=STYLE[SET_OF[m]][0], run_tag=ctx.tag[m], n_epochs=len(tt),
                         median_epoch_time_s=float(np.median(tt + tv + np.nan_to_num(t40))),
                         median_train_time_s=float(np.median(tt)), median_val_time_s=float(np.median(tv)),
                         median_train40k_time_s=float(np.nanmedian(t40)) if np.isfinite(t40).any() else np.nan,
                         median_data_wait_s=float(np.median(h['data_wait'])) if 'data_wait' in h else np.nan))
        used.append(m)
    if missing:
        warnings.warn(f'T1: history.npz without per-epoch time for {missing}; not computed for them')
    if not rows:
        raise Skip('history.npz does not record per-epoch time')
    p = out / 't1_epoch_time.csv'
    pd.DataFrame(rows).to_csv(p, index=False, float_format='%.4g')
    return [dict(figure='t1_epoch_time', files=[p.name], inputs=[ctx.hist_path(m) for m in used], models=used,
                 note=f'no per-epoch time recorded for {missing}' if missing else None)]


FIGURES = [('F1', fig_f1), ('F2', fig_f2), ('F3', fig_f3), ('F4', fig_f4), ('F5', fig_f5), ('F6', fig_f6),
           ('F7', fig_f7), ('F8', fig_f8), ('F9', fig_f9), ('F10', fig_f10), ('F11', fig_f11), ('T1', table_t1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--overwrite', action='store_true', help='replace an existing figures directory content')
    args = ap.parse_args()

    cfg = json.loads((ROOT / 'configs/experiments/train_m11_final_G1_500k.json').read_text())
    data_root = resolve_data_root(config_data_root=cfg.get('data_root'))
    out = data_root / 'results' / 'M13_presentation' / 'figures'
    manifest_path = out / 'figures_manifest.json'
    require(args.overwrite or not manifest_path.exists(), f'{manifest_path} exists; pass --overwrite to replace')
    out.mkdir(parents=True, exist_ok=True)

    set_style()
    ctx = Ctx(data_root)
    done, skipped = [], []
    for fid, fn in FIGURES:
        try:
            done += fn(ctx, out)
        except Skip as e:
            skipped.append(dict(figure=fid, reason=str(e)))

    manifest = dict(
        created=datetime.now().isoformat(timespec='seconds'), script=str(Path(__file__).resolve()),
        script_sha256=digest(Path(__file__).resolve()), data_root=str(data_root),
        m13_6_summary=str(ctx.s6_path) if ctx.s6_path else None,
        models_available=ctx.available, models_skipped=ctx.skipped, checks=ctx.checks,
        figures=[dict(figure=d['figure'], files=d['files'], models=d['models'],
                      inputs=[dict(path=str(p), sha256=ctx.sha(p)) for p in d['inputs']],
                      **({'note': d['note']} if d.get('note') else {})) for d in done],
        omitted=skipped)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f'Output: {out}')
    print('Generated:')
    for d in done:
        print(f"  {d['figure']}: {', '.join(d['files'])}  [{', '.join(d['models'])}]")
    print('Omitted:' if skipped else 'Omitted: none')
    for s in skipped:
        print(f"  {s['figure']}: {s['reason']}")
    if ctx.skipped:
        print('Optional models not available:', ', '.join(ctx.skipped))


if __name__ == '__main__':
    main()
