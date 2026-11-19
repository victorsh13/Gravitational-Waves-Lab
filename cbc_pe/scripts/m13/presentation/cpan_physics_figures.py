"""CPAN 2026 physics, data and architecture figures F02-F05 and appendix A1-A5, manifest.json and contact_sheet.png.

Called from scripts/m13/make_presentation_figures.py --set cpan_physics. Read-only and descriptive: no waveform is
generated, no network is run (no forward pass), nothing is fitted. Inputs (fixed paths; nothing is searched):
  - clean processed signals L(s) (whitened, before the per-sample z-score). They are stored for TRAIN rows only
    (M13.6 / M13.13 docs; no clean signal exists for val / cal / test), so F02-F04 show train rows:
      G3  (waveform from 30 Hz, no taper: start step)   <G3>/m13_6_aug/memmap/X_clean_train.npy   (M13.6)
      G3T (waveform from 15 Hz, Kaiser taper to 25 Hz)  <G3T>/memmap/X_clean_train.npy            (M13.13; the
          memmap_dir of the 5b config), both in the row order of <G3>/fp16_memmap/row_ids_train.npy;
  - G3T network input X of train rows (<G3T>/fp16_memmap/X_train.npy; only the train40k monitor rows are written,
    the others are zero) and the G3 train labels (<G3>/fp16_memmap/y_train.npy, unstandardized);
  - the G3 split file named by the 5b config (sizes only);
  - M13.13 evaluation summary 20261110T124625Z and the three val prediction npz it records (F04 bars);
  - the M14.5b training config (tensor shapes) and the 5b checkpoint (template_bank tensors only; CPU), located as
    in cpan_results_figures.py (val npz of the M14.5b summary, sha256 checked against M14.5 and M14.7).
F03v3 (slide: H1 input and clean signal, compact legend inside), F03v4 (F03v3 with the legend labels "whitened data"
/ "GW signal") and F03asd (appendix: the H1 ASD of F03v2 alone).
F14a-d (noise construction, H1, F03v4 row: O3b strain segment of the PSD, ASD, Gaussian noise coloured with the PSD
and the same noise whitened by the pipeline; regenerated with the row's seed and the generator calls, from the local
strain cache only) and F15a-c (signal, noise and data of the F03v4 row, same axes).
F14v2a-d (real strain and simulated noise through the pipeline band filter, same axes; ASD of both; simulated noise
whitened) and F15v2a-c (F15 in landscape sizes, wide data panel).
A figure with a missing input is MISSING, one whose check fails is FAILED_CHECK; neither is drawn.
Output: DATA_ROOT/results/presentation/cpan_2026/physics/run_<UTC>/ (manifest.json, contact_sheet.png, figures).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.m12.m12_5_3_posthoc_check import LABELS, digest, npz_to_physical, require  # noqa: E402
from scripts.m13.presentation import architecture_diagrams as AD  # noqa: E402
from scripts.m13.presentation.cpan_results_figures import (  # noqa: E402
    BASE_CFG, COL, CPAN_CFG, CFG, FIGOPT, FULL, SLIDE_SIZE, ST, TAG, TALL, TAU_F_LOW, FailedCheck, Inputs, Missing,
    SLIDE, place_legend, save, save_slide, set_style, slide_rc, inside_legend, tau30, to_py, var_label)

# ---- Fixed artifacts and constants -----------------------------------------------------------------------------
CFG_5B = ROOT / 'configs/experiments/m14/train_m14_5b_4aG3T_tbank_fullcov_s123.json'
M13_13_STAMP = '20261110T124625Z'
FS = 4096.0                     # 16384 samples over the 4 s window of G1/G3 (G3T gates provenance: duration_s 4.0)
G3_F_START = 30.0               # G3 waveform: SEOBNRv4_opt from 30 Hz without taper (M13.12 / M13.13 docs)
PCTS = (10, 50, 90)             # F02: train rows whose chirp mass is nearest to these percentiles of the train labels
MAX_SCAN = 5000                 # F03: candidates (nearest to the train P50) scanned for a written G3T X_train row
F04_PRE_S, F04_LEN_S = 0.1, 0.3  # F04 left: 0.3 s window starting 0.1 s before the located start step
SEED_BOOT, N_BOOT = 131302, 2000  # F04: same B and seed as the M13.13 evaluation (paired rows)
RMSE_RTOL = 1e-6                # RMSE / Delta RMSE re-derived from the same npz vs the M13.13 summary
N_FILTERS_SHOWN = 8             # A5
SPLIT_SIZES = dict(train=400000, val=40000, cal=30000, test=30000)
# Latency axis of F02: order-of-magnitude values given for the talk (not derived from any artifact here).
LATENCY = dict(inference_s=1e-3, lookahead_s=8.0, pe_s=(3600.0, 3 * 86400.0))
DET = ('H1', 'L1', 'V1')
DET_COLORS = ('#1B9E77', '#7570B3', '#D95F02')
ROW_COLORS = ('#1B9E77', '#7570B3', '#D95F02')
C_G3, C_G3T = '#D95F02', '#1B9E77'
C_INFER = '#D95F02'              # F02 inference marker (was style.accent, no longer in cpan_2026.json)
GREY, INK, ERR = '#9A9A9A', '#333333', '#B22222'


class PInputs(Inputs):
    """Inputs of cpan_results_figures (registration, sha256, checks, cache) plus the data of this set."""

    def cfg5b(self):
        c = self.json(CFG_5B, 'M14.5b training config')
        self.check(c['outputs']['checkpoint_tag'] == TAG, f'5b config: checkpoint_tag {TAG}')
        return c

    def big(self, p, what, note):
        p = self.exists(p, what)
        self.nohash[str(p.resolve())] = note
        return np.load(p, mmap_mode='r')

    def train(self):
        """Clean G3 / G3T signals, G3T inputs and labels of the train rows (memmaps; rows are read on demand)."""
        if self.cached('train'):
            return self._cache['train']
        start = len(self.used)
        c = self.cfg5b()
        g3t = self.data_root / 'processed' / c['dataset']['dataset_id']
        g3 = g3t.parent
        self.check(g3.name == 'M13_G3_band1024_w16_500k' and g3t.name == 'm13_13_g3t', '5b config: G3T dataset dir')
        man = self.json(g3 / 'fp16_memmap' / 'manifest.json', 'G3 fp16_memmap manifest')
        attrs = man['source_hdf5']['attrs']
        self.check(attrs['labels'] == ','.join(LABELS), f'G3 memmap: label order {LABELS}')
        self.check(attrs['detector_order'] == ','.join(DET), 'G3 memmap: detector order H1, L1, V1')
        rid = np.load(self.exists(g3 / 'fp16_memmap' / 'row_ids_train.npy', 'G3 train row ids'))
        rid_t = np.load(self.exists(g3t / 'fp16_memmap' / 'row_ids_train.npy', 'G3T train row ids'))
        self.check(np.array_equal(rid, rid_t), 'G3 and G3T fp16_memmap: identical train row ids')
        y = np.load(self.exists(g3 / 'fp16_memmap' / 'y_train.npy', 'G3 train labels')).astype(np.float64)
        self.check(y.shape == (len(rid), 3) and np.all(np.isfinite(y)) and np.all(y[:, 0] <= y[:, 1]),
                   'G3 train labels: shape, finite, chirp mass <= total mass')
        m6 = self.json(g3 / 'm13_6_aug' / 'memmap' / 'manifest.json', 'M13.6 clean-signal manifest')
        mt = self.json(g3t / c['training']['noise_augmentation']['memmap_dir'] / 'manifest.json',
                       'M13.13 clean-signal manifest')
        n = c['model']['template_bank']['n_samples']
        for m, w in ((m6, 'M13.6'), (mt, 'M13.13')):
            self.check(m['X_clean_train']['shape'] == [len(rid), 3, n]
                       and m['X_clean_train']['row_order'].startswith('G3 fp16_memmap row_ids_train'),
                       f'{w} clean signals: shape and G3 train row order')
        note = 'sha256 not computed (memmap of ~39 GB); only the rows listed in the figure entry are read'
        cg3 = self.big(g3 / 'm13_6_aug' / 'memmap' / 'X_clean_train.npy', 'G3 clean train signals', note)
        cg3t = self.big(g3t / c['training']['noise_augmentation']['memmap_dir'] / 'X_clean_train.npy',
                        'G3T clean train signals', note)
        xg3t = self.big(g3t / 'fp16_memmap' / 'X_train.npy', 'G3T train inputs X', note)
        for a, w in ((cg3, 'G3 clean'), (cg3t, 'G3T clean'), (xg3t, 'G3T X_train')):
            self.check(a.shape == (len(rid), 3, n), f'{w}: shape (n_train, 3, {n})')
        yt = np.load(self.exists(g3t / 'fp16_memmap' / 'y_train.npy', 'G3T train labels'))
        d = dict(g3=g3, g3t=g3t, rid=rid, y=y, yt=yt, cg3=cg3, cg3t=cg3t, xg3t=xg3t, n=n,
                 s15=dict(f_lower=mt['s15']['f_lower'], taper_end_hz=mt['s15']['taper_end_hz']))
        return self.store('train', d, start)

    def rows(self):
        """Deterministic train rows: F02 = nearest to the train P10 / P50 / P90 of the chirp mass; F03 / F04 = the
        nearest to P50 whose G3T X_train row is written (train40k monitor rows only; the others are zero)."""
        if self.cached('rows'):
            return self._cache['rows']
        start = len(self.used)
        tr = self.train()
        mc = tr['y'][:, 0]
        q = np.percentile(mc, PCTS)
        f02 = [int(np.argsort(np.abs(mc - qq), kind='stable')[0]) for qq in q]
        order = np.argsort(np.abs(mc - q[1]), kind='stable')[:MAX_SCAN]
        f03 = next((int(p) for p in order if np.any(np.asarray(tr['xg3t'][p]) != 0)), None)
        if f03 is None:
            raise Missing(f'no written G3T X_train row among the {MAX_SCAN} train rows nearest to the P50 chirp mass')
        self.check(np.allclose(tr['yt'][f03], tr['y'][f03], rtol=1e-6, atol=1e-6),
                   'F03 row: G3T and G3 train labels agree')
        d = dict(pcts=list(PCTS), mc_percentiles=q, f02=f02, f03=f03)
        return self.store('rows', d, start)

    def m13_13(self):
        """Per-row squared errors of the three M13.13 val cases and the two Delta RMSE comparisons of F04."""
        if self.cached('m13_13'):
            return self._cache['m13_13']
        start = len(self.used)
        p = self.data_root / 'results' / 'M13_3' / 'M13_13' / f'M13_13_G3T_evaluation_summary_{M13_13_STAMP}.json'
        s = self.json(p, 'M13.13 evaluation summary')
        self.check(s['run_utc'] == M13_13_STAMP and s['labels'] == LABELS and s['split'] == 'val',
                   'M13.13 summary: stamp, label order, val')
        cases = ('G3_on_G3', 'G3_on_G3T', 'G3T_on_G3T')
        P = {}
        for c in cases:
            rec = s['inputs'][f'pred/{c}']
            f = self.exists(rec['path'], f'M13.13 val predictions {c}')
            self.check(self.sha(f) == rec['sha256'], f'M13.13 {c} npz: sha256 equals the summary')
            with np.load(f, allow_pickle=False) as z:  # object members (model_config, ...) are never read
                names = [str(x) for x in z['label_names']]
                y, mu, units = npz_to_physical(z['y_val'].astype(np.float64), z['pred_val'].astype(np.float64),
                                               z['y_mean'].astype(np.float64), z['y_std'].astype(np.float64))
                idx = z['idx_val'].astype(np.int64)
            self.check(names == LABELS, f'M13.13 {c} npz: label order')
            self.check(units == s['prediction_sources'][c]['units'], f'M13.13 {c} npz: units as in the summary')
            P[c] = dict(y=y, mu=mu, idx=idx)
        self.check(all(np.array_equal(P[c]['idx'], P[cases[0]]['idx']) for c in cases)
                   and len(np.unique(P[cases[0]]['idx'])) == len(P[cases[0]]['idx']) == s['n_val'],
                   'M13.13 npz: identical unique val row ids in the three cases')
        self.check(all(np.allclose(P[c]['y'], P[cases[0]]['y'], rtol=1e-6, atol=1e-6) for c in cases),
                   'M13.13 npz: identical labels in the three cases')
        stored = {(m['quantity'], m['label']): m for m in s['measures'] if m['group'] == 'global'}
        E = np.concatenate([(P[c]['mu'] - P[c]['y']) ** 2 for c in cases], axis=1)   # (n, 3 cases x 3 labels)
        rmse = np.sqrt(E.mean(0)).reshape(3, 3)
        for i, c in enumerate(cases):
            self.check(np.allclose(rmse[i], [stored[(f'rmse_{c}', lab)]['value'] for lab in LABELS], rtol=RMSE_RTOL),
                       f'M13.13 {c}: global RMSE equals the summary')
        # (i) model trained with the step (4a-G3), evaluated without it (G3T val) vs with it (G3 val).
        # (ii) model retrained without the step (4a-G3T) vs 4a-G3, both evaluated on G3T val (= drmse_pct_i).
        comp = dict(i=(1, 0), ii=(2, 1))
        point = {k: 100.0 * (rmse[a] / rmse[b] - 1.0) for k, (a, b) in comp.items()}
        self.check(np.allclose(point['ii'], [stored[('drmse_pct_i', lab)]['value'] for lab in LABELS],
                               rtol=RMSE_RTOL, atol=1e-6), 'F04 (ii) equals drmse_pct_i of the M13.13 summary')
        rng = np.random.default_rng(SEED_BOOT)
        n = E.shape[0]
        boot = {k: np.empty((N_BOOT, 3)) for k in comp}
        for b in range(N_BOOT):
            r = np.sqrt(E[rng.integers(0, n, n)].mean(0)).reshape(3, 3)
            for k, (a, c) in comp.items():
                boot[k][b] = 100.0 * (r[a] / r[c] - 1.0)
        ci = {k: np.percentile(v, [2.5, 97.5], axis=0) for k, v in boot.items()}
        stored_ii = np.array([[stored[('drmse_pct_i', lab)][q] for lab in LABELS] for q in ('ci95_lo', 'ci95_hi')])
        d = dict(n=n, rmse=dict(zip(cases, rmse)), point=point, ci=ci, stored_ci_ii=stored_ii,
                 max_abs_ci_diff_ii_vs_summary=float(np.max(np.abs(ci['ii'] - stored_ii))))
        return self.store('m13_13', d, start)

    def templates(self):
        """template_i / template_q of the 5b checkpoint (CPU load; no model is built, no forward)."""
        if self.cached('templates'):
            return self._cache['templates']
        start = len(self.used)
        import torch  # local import: only A5 needs torch
        tb = self.cfg5b()['model']['template_bank']
        ck = self.exists(self.pred('val')['ckpt'], '5b checkpoint')
        # Trusted source (our own training run): full unpickling needed for the non-tensor metadata, as in
        # cpan_results_figures.count_parameters. The VM torch 1.9.1 never runs this script.
        obj = torch.load(ck, map_location='cpu', weights_only=False)
        sd = next((obj[k] for k in ('model_state_dict', 'state_dict', 'model') if isinstance(obj, dict)
                   and isinstance(obj.get(k), dict)), obj if isinstance(obj, dict) else None)
        self.check(sd is not None and 'template_bank.template_i' in sd and 'template_bank.template_q' in sd,
                   '5b checkpoint: template_bank tensors in the state_dict')
        hi = sd['template_bank.template_i'].detach().double().numpy()
        hq = sd['template_bank.template_q'].detach().double().numpy()
        self.check(hi.shape == hq.shape == (tb['n_templates'], tb['template_length']),
                   '5b templates: shape (n_templates, template_length) of the config')
        d = dict(hi=hi, hq=hq, n=tb['n_samples'], ckpt=str(ck), sha=self.sha(ck))
        return self.store('templates', d, start)


# ---- helpers ---------------------------------------------------------------------------------------------------
def contact_sheet(out, entries):
    """Thumbnails of this set in FIGURES order (cpan_results_figures.contact_sheet lists only the results IDs)."""
    ok = [(e['id'], out / f) for e in entries if e['status'] == 'OK' for f in e['files'] if f.endswith('.png')]
    bad = [f"{e['id']} {e['status']}" for e in entries if e['status'] != 'OK']
    ncol = 3
    nrow = max(1, -(-len(ok) // ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.9 * nrow + (0.5 if bad else 0)), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (fid, p) in zip(axes.flat, ok):
        ax.imshow(plt.imread(p))
        ax.set_title(fid, fontsize=16, fontweight='bold', loc='left')
    if bad:
        fig.text(0.01, 0.005, 'Not produced: ' + ', '.join(bad), fontsize=13, color=ERR)
    fig.tight_layout()
    fig.savefig(out / 'contact_sheet.png', dpi=110, facecolor='white')
    plt.close(fig)
    return dict(file='contact_sheet.png', figures=[i for i, _ in ok], not_produced=bad)


def save_diagram(fig, out, name, crop=None):
    """Diagram save: same middle-dot rule, formats and dpi as save(); optional crop in figure inches."""
    bad = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if '·' in t.get_text()]
    require(not bad, f'{name}: middle dot in figure text {bad}')
    bb = Bbox([[crop[0], crop[1]], [crop[2], crop[3]]]) if crop else 'tight'
    files = []
    for ext in ST['formats']:
        fig.savefig(out / f'{name}.{ext}', dpi=ST['dpi'], bbox_inches=bb, facecolor='white')
        files.append(f'{name}.{ext}')
    plt.close(fig)
    return files


def mc_text(mc):
    v = CFG['variables']['chirp_mass']
    return f"{v['label']} = {mc:.1f} {v['unit']}"


def row_info(tr, p):
    return dict(train_position=int(p), hdf5_row_id=int(tr['rid'][p]), labels=dict(zip(LABELS, tr['y'][p].tolist())))


def merger_index(s):
    return int(np.argmax(np.abs(s)))


# ---- F02 -------------------------------------------------------------------------------------------------------
def f02(I, out, name):
    tr, R = I.train(), I.rows()
    fig = plt.figure(figsize=TALL)
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.42)
    inner = outer[0].subgridspec(3, 1, hspace=0.08)
    axes = []
    sig, rows = [], []
    for k, p in enumerate(R['f02']):
        s = np.asarray(tr['cg3t'][p, 0], dtype=np.float64)   # H1
        im = merger_index(s)
        I.check(0 < im < tr['n'] - 1, f'F02 row {p}: merger inside the window')
        sig.append((s, im, tr['y'][p, 0], float(tau30(tr['y'][p, 0]))))
        rows.append(dict(row_info(tr, p), percentile=R['pcts'][k], merger_index=im, tau30_newtonian_s=sig[-1][3]))
    lo = -min(1.25 * max(x[3] for x in sig), max(x[1] for x in sig) / FS)
    for k, (s, im, mc, tau) in enumerate(sig):
        ax = fig.add_subplot(inner[k, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        t = (np.arange(len(s)) - im) / FS
        ax.plot(t, s / np.max(np.abs(s)), color=ROW_COLORS[k], lw=0.9)
        ax.axvline(-tau, color=INK, ls='--', lw=1.0)
        ax.text(-tau, 1.05, f' ≈ {tau:.2f} s above {TAU_F_LOW:.0f} Hz', color=INK, fontsize=ST['annot_size'] - 2,
                va='bottom', ha='left')
        ax.text(0.01, 0.92, f"{mc_text(mc)}  (train P{R['pcts'][k]})", transform=ax.transAxes, va='top',
                fontsize=ST['annot_size'] - 1, color=ROW_COLORS[k])
        ax.set_ylim(-1.15, 1.45)
        ax.set_yticks([])
        ax.grid(False)
        if k < 2:
            ax.tick_params(labelbottom=False)
    axes[0].set_xlim(lo, 0.06)
    axes[1].set_ylabel('whitened H1 strain\n(normalised to peak)')
    axes[-1].set_xlabel('time relative to merger [s]')
    lab = CFG['variables']['chirp_mass']['label']
    axes[0].set_title(f'{lab} controls the chirp: time in band ' + r'$\tau \propto$' + f' {lab}' + r'$^{-5/3}$',
                      loc='left', fontsize=ST['axes_size'])
    # latency axis
    ax = fig.add_subplot(outer[1])
    ax.set_xscale('log')
    ax.set_xlim(2e-4, 2e6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.grid(True, axis='x')
    ax.grid(False, axis='y')
    ticks = [1e-3, 1.0, 60.0, 3600.0, 86400.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['1 ms', '1 s', '1 min', '1 h', '1 day'])
    ax.minorticks_off()
    yy = 0.35
    ax.plot([LATENCY['inference_s']], [yy], 'o', ms=10, color=C_INFER, zorder=3)
    ax.text(LATENCY['inference_s'], yy + 0.22, 'network inference\n(~ms)', ha='center', va='bottom',
            fontsize=ST['annot_size'] - 2)
    ax.plot([LATENCY['lookahead_s']], [yy], 'o', ms=10, color=INK, zorder=3)
    ax.text(LATENCY['lookahead_s'], yy + 0.22, 'data look-ahead of the\nwhitening filter (~8 s)', ha='center',
            va='bottom', fontsize=ST['annot_size'] - 2)
    a, b = LATENCY['pe_s']
    ax.plot([a, b], [yy, yy], color='#7570B3', lw=8, solid_capstyle='butt', zorder=3)
    ax.text(np.sqrt(a * b), yy + 0.22, 'full Bayesian PE\n(hours to days)', ha='center', va='bottom',
            fontsize=ST['annot_size'] - 2)
    ax.set_xlabel('latency (log scale)')
    files = save(fig, out, name)
    return dict(files=files, rows=rows, split='train', detector='H1', signal='G3T clean L(s) (before z-score)',
                note='Clean signals exist for train rows only; rows nearest to the train P10 / P50 / P90 of the chirp '
                     'mass (stable argsort, first row). Each signal normalised to its peak |L(s)|; merger = argmax '
                     '|L(s)| in H1. Dashed line: Newtonian time from 30 Hz to merger with the true chirp mass '
                     '(tau30 of cpan_results_figures). Latency values: order of magnitude given for the talk, not '
                     'derived from artifacts.', latency=LATENCY, mc_percentiles=R['mc_percentiles'])



# ---- F03 -------------------------------------------------------------------------------------------------------
def f03(I, out, name):
    tr, R = I.train(), I.rows()
    p = R['f03']
    x = np.asarray(tr['xg3t'][p], dtype=np.float64)
    s = np.asarray(tr['cg3t'][p], dtype=np.float64)
    # X = (L(s) + L(n) - m) / (sd + eps) per detector; m, sd are not stored, so the clean signal is drawn on the
    # scale of X with the least-squares fit X_d ~ a_d L(s)_d + b_d (a_d estimates 1 / (sd + eps)).
    fit = []
    for d_ in range(3):
        a, b = np.linalg.lstsq(np.column_stack([s[d_], np.ones(tr['n'])]), x[d_], rcond=None)[0]
        I.check(a > 0, f'F03 row {p} {DET[d_]}: positive scale of the clean signal on X')
        fit.append((float(a), float(b)))
    c = I.cfg5b()
    # split_file of the 5b config is '../<name>' relative to the G3T dataset dir, i.e. <G3>/<name>.
    split = I.exists(tr['g3'] / Path(c['dataset']['split_file']).name, 'G3 split file')
    with np.load(split, allow_pickle=False) as z:
        sizes = {k: int(z[f'{k}_idx'].shape[0]) for k in SPLIT_SIZES}
    I.check(sizes == SPLIT_SIZES, f'split sizes {SPLIT_SIZES}')
    t = np.arange(tr['n']) / FS
    fig = plt.figure(figsize=TALL)
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 0.42], hspace=0.38)
    inner = outer[0].subgridspec(3, 1, hspace=0.08)
    axes = []
    for d_ in range(3):
        ax = fig.add_subplot(inner[d_, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        ax.plot(t, x[d_], color=GREY, lw=0.4)
        ax.plot(t, fit[d_][0] * s[d_] + fit[d_][1], color=DET_COLORS[d_], lw=1.3)
        ax.set_ylabel(DET[d_])
        ax.grid(False)
        if d_ < 2:
            ax.tick_params(labelbottom=False)
    axes[0].set_xlim(0, tr['n'] / FS)
    axes[-1].set_xlabel('time [s]')
    axes[0].legend([Line2D([], [], color=GREY, lw=1.2), Line2D([], [], color=INK, lw=1.6)],
                   ['network input X (whitened, z-scored)', 'clean signal (same scale)'], loc='lower left',
                   bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax = fig.add_subplot(outer[1])
    tot = sum(sizes.values())
    left = 0
    cols = dict(train='#7570B3', val='#1B9E77', cal='#E6AB02', test='#D95F02')
    for k, v in sizes.items():
        ax.barh(0, v, left=left, height=1.0, color=cols[k], edgecolor='white', lw=1.5)
        ax.text(left + v / 2, -0.75, f'{k}\n{v // 1000}k', ha='center', va='top', fontsize=ST['annot_size'] - 2)
        left += v
    ax.set_xlim(0, tot)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    files = save(fig, out, name)
    return dict(files=files, row=row_info(tr, p), split_of_row='train', z_scale_fit=dict(zip(DET, fit)),
                split_sizes=sizes, split_file=str(split),
                note='Clean signals exist for train rows only. Row: nearest to the train P50 chirp mass with a '
                     'written G3T X_train row (train40k monitor rows). Clean signal drawn as a_d L(s)_d + b_d, '
                     '(a_d, b_d) least-squares fit of X_d on L(s)_d over the window (z-score statistics not stored).')


# ---- F03 v2 (slide): H1 only, input and clean signal; ASD of the row's PSD ------------------------------------
ASD_BAND = (30.0, 1024.0)     # processing band of the dataset (M13_0d: 30-1024 Hz), shaded in F03v2
ASD_OMITTED = 'ASD omitted: raw O3b strain not available locally'


def block_psd_h1(I, tr, p):
    """Raw off-source PSD of the row's block for H1, recomputed as the generator does (no waveform): 128 s of cached
    GWOSC strain, generate_m11_final_G1.estimate_raw_offsource_psd (Welch, 8 s segments), native 1/8 Hz grid (the
    generator then only interpolates it onto the processing grid; same use as make_presentation_figures d0_psd).
    Returns (f, psd, info) or None when the strain of that interval is not cached locally (case b)."""
    import h5py
    import pandas as pd
    import scripts.generate_m11_final_G1 as gen
    man = I.json(tr['g3'] / 'fp16_memmap' / 'manifest.json', 'G3 fp16_memmap manifest')
    h5 = I.exists(man['source_hdf5']['path'], 'G3 HDF5 (block_id, psd_start, psd_end)')
    I.nohash[str(h5.resolve())] = 'sha256 not computed (dataset HDF5); only block_id, psd_start, psd_end of one row'
    row_id = int(tr['rid'][p])
    with h5py.File(h5, 'r') as h:
        bid = h['block_id'][row_id]
        bid = bid.decode() if isinstance(bid, bytes) else str(bid)
        ps, pe = float(h['psd_start'][row_id]), float(h['psd_end'][row_id])
    keys = np.load(I.exists(tr['g3t'] / 'memmap' / 'psd_key_train.npy', 'train psd_key (block id)'), mmap_mode='r')
    I.check(str(keys[p]) == bid, f'F03v2 row {p}: HDF5 block_id equals the clean-signal psd_key')
    I.check(np.isclose(pe - ps, 128.0), f'F03v2 block {bid}: 128 s off-source reference')
    info = dict(block_id=bid, psd_start=ps, psd_end=pe, hdf5_row_id=row_id)
    manifest = pd.DataFrame([dict(block_id=bid, psd_start=ps, psd_end=pe)])
    try:   # only the file lookup decides case (b); any later failure is a real error
        selected, _ = gen.resolve_cached_psd_files(I.data_root, manifest)
    except Exception as e:  # noqa: BLE001 - generator's require(): no / ambiguous cached file
        info['strain_lookup_error'] = f'{type(e).__name__}: {e}'
        return None, info
    for ifo in gen.DETECTORS:
        f_ = I.exists(selected[bid, ifo][0], f'cached GWOSC strain {ifo}')
        I.nohash[str(f_.resolve())] = 'sha256 not computed (4096 s GWOSC file); 128 s read'
    strains = gen.read_psd_intervals(next(manifest.itertuples()), selected)
    from pycbc.psd import interpolate as pycbc_interpolate
    gen.interpolate = pycbc_interpolate   # the generator binds it lazily in initialize_physics(), not called here
    rate = 4096
    psd = gen.estimate_raw_offsource_psd(strains['H1'], psd_start=ps, psd_end=pe,
                                         delta_f=1.0 / gen.PSD_SEGMENT_DURATION_S,
                                         target_flength=int(round(rate / 2 * gen.PSD_SEGMENT_DURATION_S)) + 1)
    info.update(strain_file_H1=str(selected[bid, 'H1'][0]), welch_segment_s=gen.PSD_SEGMENT_DURATION_S)
    return (psd.sample_frequencies.numpy(), psd.numpy()), info


def f03v2(I, out, name):
    tr, R = I.train(), I.rows()
    p = R['f03']
    x = np.asarray(tr['xg3t'][p, 0], dtype=np.float64)
    s = np.asarray(tr['cg3t'][p, 0], dtype=np.float64)
    a, b = np.linalg.lstsq(np.column_stack([s, np.ones(tr['n'])]), x, rcond=None)[0]   # as F03
    I.check(a > 0, f'F03v2 row {p} H1: positive scale of the clean signal on X')
    asd, info = block_psd_h1(I, tr, p)
    t = np.arange(tr['n']) / FS
    with slide_rc():
        fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
        if asd is not None:
            ax, ax2 = fig.subplots(2, 1, height_ratios=[1.0, 1.0])
        else:
            ax, ax2 = fig.subplots(1, 1), None
        ax.plot(t, x, color=GREY, lw=0.4, label='network input X (whitened, z-scored)')
        ax.plot(t, a * s + b, color=DET_COLORS[0], lw=1.6, label='clean signal (same scale)')
        ax.set(xlim=(0, tr['n'] / FS), xlabel='time [s]', ylabel='H1')
        ax.grid(False)
        inside_legend(ax, ax.get_legend_handles_labels()[0], loc='upper left')
        if ax2 is not None:
            f, psd = asd
            m = (f >= 10) & (f <= FS / 2)
            band = ax2.axvspan(*ASD_BAND, color=COL['validity_band'], lw=0, zorder=0,
                               label=f'{ASD_BAND[0]:.0f}–{ASD_BAND[1]:.0f} Hz band')
            ln, = ax2.loglog(f[m], np.sqrt(psd[m]), color=INK, lw=1.0, label='H1 ASD, O3b (off-source, 128 s)')
            ax2.set(xlim=(10, FS / 2), xlabel='frequency [Hz]', ylabel=r'ASD [1/$\sqrt{\rm Hz}$]')
            ax2.grid(True, which='major')
            inside_legend(ax2, [ln, band], loc='upper right')
        files = save_slide(fig, out, name)
    note = ('Same row and clean-signal scaling as F03 (train row; least-squares fit of X on L(s)). '
            + ('ASD = sqrt of the raw off-source PSD of the row\'s block recomputed from the cached GWOSC strain with '
               'the generator functions (Welch, 8 s segments, 1/8 Hz grid); not the processing-grid PSD itself.'
               if asd is not None else ASD_OMITTED + '.'))
    return dict(files=files, row=row_info(tr, p), split_of_row='train', detector='H1', z_scale_fit=(float(a), float(b)),
                asd='recomputed' if asd is not None else 'omitted', psd_block=info, note=note)


# ---- F03 v3 (slide): H1 input and clean signal only; F03asd (appendix): the ASD alone ----------------------------
F03V3_LABELS = ('network input X (whitened, z-scored)', 'clean signal (same scale)')
F03V4_LABELS = ('whitened data', 'GW signal')


def f03v3(I, out, name, fid='F03v3', labels=F03V3_LABELS):
    """H1 input and clean signal (slide). F03v4 = the same figure with the legend labels F03V4_LABELS."""
    tr, R = I.train(), I.rows()
    p = R['f03']
    x = np.asarray(tr['xg3t'][p, 0], dtype=np.float64)
    s = np.asarray(tr['cg3t'][p, 0], dtype=np.float64)
    a, b = np.linalg.lstsq(np.column_stack([s, np.ones(tr['n'])]), x, rcond=None)[0]   # as F03 / F03v2
    I.check(a > 0, f'{fid} row {p} H1: positive scale of the clean signal on X')
    t = np.arange(tr['n']) / FS
    with slide_rc():
        fig, ax = plt.subplots(figsize=tuple(FIGOPT['F03v3']['size']), layout='constrained')
        ax.plot(t, x, color=COL['neutral'], lw=0.4)
        ax.plot(t, a * s + b, color=COL['signal'], lw=1.6)
        ax.set(xlim=(0, tr['n'] / FS), xlabel='time [s]', ylabel='H1')
        ax.grid(False)
        leg = place_legend(ax, [Line2D([], [], color=COL['neutral'], lw=1.6),
                                Line2D([], [], color=COL['signal'], lw=1.6)],
                           list(labels), prefer=('upper right',), grow=8)
        files = save_slide(fig, out, name, strict=True)
    return dict(files=files, row=row_info(tr, p), split_of_row='train', detector='H1', z_scale_fit=(float(a), float(b)),
                role='main', legend=leg,
                note='Same row and clean-signal scaling as F03 / F03v2 (train row; least-squares fit of X on L(s)); '
                     'H1 only. The ASD is the separate figure F03asd.'
                     + ('' if fid == 'F03v3' else f' As F03v3 except the legend labels {list(labels)}.'))


def f03v4(I, out, name):
    return f03v3(I, out, name, fid='F03v4', labels=F03V4_LABELS)


def f03asd(I, out, name):
    tr, R = I.train(), I.rows()
    p = R['f03']
    asd, info = block_psd_h1(I, tr, p)
    if asd is None:
        raise Missing(f"{ASD_OMITTED} ({info.get('strain_lookup_error', '')})")
    f, psd = asd
    with slide_rc():
        fig, ax = plt.subplots(figsize=tuple(FIGOPT['F03asd']['size']), layout='constrained')
        m = (f >= 10) & (f <= FS / 2)
        band = ax.axvspan(*ASD_BAND, color=COL['validity_band'], lw=0, zorder=0,
                          label=f'{ASD_BAND[0]:.0f}–{ASD_BAND[1]:.0f} Hz band')
        ln, = ax.loglog(f[m], np.sqrt(psd[m]), color=INK, lw=1.4, label='H1 ASD, O3b (off-source, 128 s)')
        ax.set(xlim=(10, FS / 2), xlabel='frequency [Hz]', ylabel=r'ASD [1/$\sqrt{\rm Hz}$]')
        ax.grid(True, which='major')
        leg = place_legend(ax, [ln, band], prefer=('upper right',), grow=4)
        files = save_slide(fig, out, name, strict=True)
    return dict(files=files, row=row_info(tr, p), split_of_row='train', detector='H1', psd_block=info,
                role='appendix', legend=leg,
                note="ASD of the F03v3 row's block: sqrt of the raw off-source PSD recomputed from the cached GWOSC "
                     'strain with the generator functions (Welch, 8 s segments, 1/8 Hz grid), as F03v2; not the '
                     'processing-grid PSD itself.')


# ---- F14 (noise construction, H1, 4 independent PNG) and F15 (signal + noise = data, 3 PNG) -------------------
SEED_F14_ASD_NPERSEG = 4096   # F14d inset only: scipy Welch of the 4 s whitened noise (1 s Hann segments, 50 %)


def welch_method_from_code():
    """Welch parameters of the generator's PSD estimate, read from the code that runs (not assumed):
    estimate_raw_offsource_psd calls TimeSeries.psd(segment_duration) without keywords, TimeSeries.psd sets
    seg_stride = int(seg_len / 2) and passes the rest to pycbc.psd.welch, whose defaults give window / average."""
    import inspect
    import re as re_
    import pycbc
    import pycbc.psd
    import pycbc.types.timeseries as pts
    import scripts.generate_m11_final_G1 as gen
    nr = 'no registrado'
    src_gen = inspect.getsource(gen.estimate_raw_offsource_psd)
    call_ok = 'psd_data.psd(float(psd_segment_duration))' in src_gen
    m = re_.search(r'seg_stride\s*=\s*int\(\s*seg_len\s*/\s*(\d+)\s*\)', inspect.getsource(pts.TimeSeries.psd))
    par = inspect.signature(pycbc.psd.welch).parameters
    return dict(
        function='scripts/generate_m11_final_G1.estimate_raw_offsource_psd -> pycbc TimeSeries.psd -> pycbc.psd.welch'
                 ', then pycbc.psd.interpolate to the target delta_f',
        segment_duration_s=float(gen.PSD_SEGMENT_DURATION_S),
        overlap_fraction=(1.0 - 1.0 / int(m.group(1))) if (m and call_ok) else nr,
        window=par['window'].default if call_ok else nr,
        average=par['avg_method'].default if call_ok else nr,
        pycbc_version=pycbc.__version__,
        read_from='inspect of the installed code: call without keywords in estimate_raw_offsource_psd '
                  f'({"found" if call_ok else "NOT found"}), seg_stride in TimeSeries.psd '
                  f'({"found" if m else "NOT found"}), defaults of pycbc.psd.welch')


def noise_demo(I):
    """H1 of the F03 / F03v4 row: the 128 s O3b strain of its block, the PSD of F03asd, and the row's Gaussian
    processing noise regenerated with the generator calls (noise_from_psd with the row's gaussian_seed_H1, then
    SignalProcessor.process with the G3 processing configuration stored in the G3 HDF5 config_json)."""
    if I.cached('noise_demo'):
        return I._cache['noise_demo']
    start = len(I.used)
    import h5py
    import pandas as pd
    import scripts.generate_m11_final_G1 as gen
    from pycbc.noise import gaussian
    from pycbc.psd import interpolate as pycbc_interpolate
    from src.config import SimulationConfig
    from src.processing import SignalProcessor
    tr, R = I.train(), I.rows()
    p = R['f03']
    asd, info = block_psd_h1(I, tr, p)          # same PSD as F03asd (raw off-source, Welch 8 s, 1/8 Hz)
    if asd is None:
        raise Missing(f"raw O3b strain of block {info['block_id']} not cached locally "
                      f"({info.get('strain_lookup_error', '')}); nothing downloaded")
    man = I.json(tr['g3'] / 'fp16_memmap' / 'manifest.json', 'G3 fp16_memmap manifest')
    h5 = I.exists(man['source_hdf5']['path'], 'G3 HDF5 (row metadata, generator config_json)')
    row_id = int(tr['rid'][p])
    with h5py.File(h5, 'r') as h:
        spec = json.loads(h.attrs['config_json'])
        seed = int(h['gaussian_seed_H1'][row_id])
        p0, p1 = float(h['processing_start'][row_id]), float(h['processing_end'][row_id])
        mods = json.loads(h.attrs['source_module_sha256'])
    gen.interpolate = pycbc_interpolate          # bound lazily by initialize_physics() in the generator
    C = SimulationConfig(**spec['simulation'])
    proc = SignalProcessor(config=C, **spec['signal_processor'])
    I.check(np.isclose(p1 - p0, C.processing_duration), 'F14: processing window of the row = config duration')
    sel_, _ = gen.resolve_cached_psd_files(I.data_root, pd.DataFrame([dict(block_id=info['block_id'],
                                                                             psd_start=info['psd_start'],
                                                                             psd_end=info['psd_end'])]))
    strains = gen.read_psd_intervals(next(pd.DataFrame([dict(block_id=info['block_id'], psd_start=info['psd_start'],
                                                             psd_end=info['psd_end'])]).itertuples()), sel_)
    seg = strains['H1']
    psd_proc = gen.estimate_raw_offsource_psd(seg, psd_start=info['psd_start'], psd_end=info['psd_end'],
                                              delta_f=C.processing_delta_f, target_flength=C.processing_flength)
    noise = gaussian.noise_from_psd(psd=psd_proc, length=C.processing_length, delta_t=C.delta_t, seed=seed)
    noise.start_time = p0
    white = proc.process(strain=noise, psd=psd_proc, detector_name='H1')
    i0 = int(round(C.processing_context_start_seconds / C.delta_t))
    colored = np.asarray(noise.numpy()[i0:i0 + C.length], float)
    white = np.asarray(white.numpy(), float)
    I.check(len(colored) == len(white) == C.length == tr['n'], 'F14: 4 s window of the colored and whitened noise')
    from scipy.signal import welch as sp_welch
    fw, pw = sp_welch(white, fs=1.0 / C.delta_t, window='hann', nperseg=SEED_F14_ASD_NPERSEG,
                      noverlap=SEED_F14_ASD_NPERSEG // 2, average='median')
    here_proc = digest(ROOT / 'src/processing.py')
    d = dict(seg_t=np.arange(len(seg)) * float(seg.delta_t), seg=np.asarray(seg.numpy(), float), asd=asd,
             colored=colored, white=white, white_asd=(fw, np.sqrt(pw)), t=np.arange(C.length) * C.delta_t,
             info=dict(psd_block=info, psd_source_strain_file=info['strain_file_H1'],
                       psd_estimation_segment_s=float(info['psd_end'] - info['psd_start']),
                       welch=welch_method_from_code(), hdf5_row_id=row_id, gaussian_seed_H1=seed,
                       processing_window=(p0, p1), processing_config='G3 HDF5 attrs config_json (simulation, '
                       'signal_processor): ' + json.dumps(spec['signal_processor'], sort_keys=True),
                       psd_for_noise=f'same estimate on the processing grid (delta_f {C.processing_delta_f:.6g} Hz)',
                       src_processing_sha256_matches_dataset=(mods.get('src/processing.py') == here_proc)))
    return I.store('noise_demo', d, start)


def _f14_axes():
    return plt.subplots(figsize=tuple(FIGOPT['F14']['size']), layout='constrained')


def _f14_entry(files, D, extra, note):
    return dict(files=files, role='main', detector='H1', **D['info'], **extra, note=note)


def f14a(I, out, name):
    D = noise_demo(I)
    with slide_rc():
        fig, ax = _f14_axes()
        ax.plot(D['seg_t'], D['seg'], color=COL['f14_line'], lw=0.3, rasterized=True)
        ax.set(xlim=(0, D['seg_t'][-1]), xlabel='time [s]', ylabel='strain')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(False)
        files = save_slide(fig, out, name, strict=True)
    return _f14_entry(files, D, {}, 'Raw GWOSC H1 strain of the PSD reference interval of the F03v4 row (read from '
                                     'the local cache; nothing downloaded).')


def f14b(I, out, name):
    D = noise_demo(I)
    f, psd = D['asd']
    with slide_rc():
        fig, ax = _f14_axes()
        m = (f >= 10) & (f <= FS / 2)
        ax.axvspan(*ASD_BAND, color=COL['validity_band'], lw=0, zorder=0)
        ax.loglog(f[m], np.sqrt(psd[m]), color=INK, lw=1.4)
        ax.set(xlim=(10, FS / 2), xlabel='frequency [Hz]', ylabel=r'ASD [1/$\sqrt{\rm Hz}$]')
        ax.grid(True, which='major')
        files = save_slide(fig, out, name, strict=True)
    return _f14_entry(files, D, dict(band_hz=ASD_BAND), 'Same PSD as F03asd (sqrt of the raw off-source PSD, '
                                                        f'1/8 Hz grid). Shaded: {ASD_BAND[0]:.0f}-{ASD_BAND[1]:.0f} Hz.')


def f14c(I, out, name):
    D = noise_demo(I)
    with slide_rc():
        fig, ax = _f14_axes()
        ax.plot(D['t'], D['colored'], color=COL['f14_line'], lw=0.5)
        ax.set(xlim=(0, D['t'][-1]), xlabel='time [s]', ylabel='strain')
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(False)
        files = save_slide(fig, out, name, strict=True)
    return _f14_entry(files, D, {}, 'Gaussian noise coloured with the PSD (pycbc noise_from_psd, gaussian_seed_H1 of '
                                     'the row, processing length), final 4 s window; strain units, not whitened.')


def f14d(I, out, name):
    D = noise_demo(I)
    fw, aw = D['white_asd']
    with slide_rc():
        fig, ax = _f14_axes()
        ax.plot(D['t'], D['white'], color=COL['f14_line'], lw=0.5)
        ax.set(xlim=(0, D['t'][-1]), xlabel='time [s]', ylabel='whitened strain')
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(False)
        lim = 1.15 * float(np.max(np.abs(D['white'])))
        ax.set_ylim(-lim, 1.9 * lim)                 # headroom for the inset above the trace
        ins = ax.inset_axes(list(FIGOPT['F14']['inset_bounds']))
        m = (fw >= 10) & (fw <= FS / 2)
        ins.axvspan(*ASD_BAND, color=COL['validity_band'], lw=0, zorder=0)
        ins.loglog(fw[m], aw[m], color=INK, lw=1.2)
        ins.set_xlim(10, FS / 2)
        ins.set_xticks([])
        ins.set_yticks([])
        ins.minorticks_off()
        ins.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
        ins.grid(False)
        ins.text(0.04, 0.9, 'ASD', transform=ins.transAxes, ha='left', va='top', fontsize=SLIDE['tick_size'])
        files = save_slide(fig, out, name, strict=True)
    return _f14_entry(files, D, dict(inset_asd='scipy.signal.welch of the 4 s whitened noise: Hann, '
                                                f'{SEED_F14_ASD_NPERSEG} samples, 50 % overlap, median; log-log '
                                                '10-2048 Hz, shaded band as F14b, no tick labels'),
                      'The F14c noise after SignalProcessor.process (whitening with the processing PSD, high- and '
                      'low-pass, crop to the 4 s window), before the per-sample z-score.')


def f15_parts(I):
    """F03v4 row, H1: signal a L(s) + b (least-squares scale on X, as F03), noise = X - (a L(s) + b), data = X."""
    if I.cached('f15'):
        return I._cache['f15']
    start = len(I.used)
    tr, R = I.train(), I.rows()
    p = R['f03']
    x = np.asarray(tr['xg3t'][p, 0], dtype=np.float64)
    s = np.asarray(tr['cg3t'][p, 0], dtype=np.float64)
    a, b = np.linalg.lstsq(np.column_stack([s, np.ones(tr['n'])]), x, rcond=None)[0]   # as F03 / F03v4
    I.check(a > 0, f'F15 row {p} H1: positive scale of the clean signal on X')
    sig = a * s + b
    lim = 1.05 * float(max(np.abs(x).max(), np.abs(sig).max(), np.abs(x - sig).max()))
    d = dict(t=np.arange(tr['n']) / FS, x=x, sig=sig, noise=x - sig, lim=lim, row=row_info(tr, p),
             fit=(float(a), float(b)))
    return I.store('f15', d, start)


def _f15(I, out, name, which):
    D = f15_parts(I)
    with slide_rc():
        fig, ax = plt.subplots(figsize=tuple(FIGOPT['F15']['size']), layout='constrained')
        if which == 'a':
            ax.plot(D['t'], D['sig'], color=COL['signal'], lw=1.4)
        elif which == 'b':
            ax.plot(D['t'], D['noise'], color=COL['f14_line'], lw=0.4)
        else:
            ax.plot(D['t'], D['x'], color=COL['f14_line'], lw=0.4)
            ax.plot(D['t'], D['sig'], color=COL['signal'], lw=0.9, alpha=0.8)
        ax.set(xlim=(0.0, 4.0), ylim=(-D['lim'], D['lim']), xlabel='time [s]', ylabel='H1')
        ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))
        ax.grid(False)
        files = save_slide(fig, out, name, strict=True)
    extra = {}
    if which == 'b':
        try:   # informative: is the residual the regenerated noise of F14d (same seed and processing)?
            w = noise_demo(I)['white']
            extra['corr_with_F14d_whitened_noise'] = float(np.corrcoef(w, D['noise'])[0, 1])
        except Exception as e:  # noqa: BLE001 - informative only; F15b does not depend on it
            extra['corr_with_F14d_whitened_noise'] = f'not computed: {type(e).__name__}: {e}'
    return dict(files=files, role='main', row=D['row'], split_of_row='train', detector='H1', z_scale_fit=D['fit'],
                axes=dict(xlim=(0.0, 4.0), ylim=(-D['lim'], D['lim']), shared_by='F15a, F15b, F15c'), **extra,
                note='Same row and scaling as F03v4: signal = a L(s) + b (whitened clean signal on the scale of X), '
                     'noise = X - (a L(s) + b), data = X (network input: whitened, z-scored). Same axes in F15a-c.')


def f15a(I, out, name):
    return _f15(I, out, name, 'a')


def f15b(I, out, name):
    return _f15(I, out, name, 'b')


def f15c(I, out, name):
    return _f15(I, out, name, 'c')


# ---- F14v2 (noise: real vs simulated, same band filter; 4 landscape PNG) and F15v2 (wide data panel) -----------
def noise_demo_v2(I):
    """F14v2: same PSD block, GWOSC file, row, seed and processing configuration as F14 (noise_demo). The pipeline's
    band filter is SignalProcessor.process with the G3 signal_processor config and whitening_method 'none' (the same
    high-pass and low-pass FIR and crop to the 4 s window; no z-score in that config). It is applied to (a) one
    processing window of the real strain, centred in the 128 s PSD segment, and (c) simulated Gaussian noise
    (noise_from_psd with the same PSD) whose seed is the one of the 50 noise_check.json realisations with the 4 s
    in-band std closest to their median (noise_check_seed); (d) is that noise whitened by the pipeline."""
    if I.cached('noise_demo_v2'):
        return I._cache['noise_demo_v2']
    start = len(I.used)
    import h5py
    import pandas as pd
    import scripts.generate_m11_final_G1 as gen
    from pycbc.noise import gaussian
    from pycbc.psd import interpolate as pycbc_interpolate
    from pycbc.types import TimeSeries
    from src.config import SimulationConfig
    from src.processing import SignalProcessor
    D = noise_demo(I)                            # F14: PSD (F03asd), seed, row, whitened noise
    blk = D['info']['psd_block']
    tr, R = I.train(), I.rows()
    man = I.json(tr['g3'] / 'fp16_memmap' / 'manifest.json', 'G3 fp16_memmap manifest')
    h5 = I.exists(man['source_hdf5']['path'], 'G3 HDF5 (row metadata, generator config_json)')
    row_id = int(tr['rid'][R['f03']])
    with h5py.File(h5, 'r') as h:
        spec = json.loads(h.attrs['config_json'])
        seed = int(h['gaussian_seed_H1'][row_id])
        p0 = float(h['processing_start'][row_id])
    I.check(seed == D['info']['gaussian_seed_H1'] and row_id == D['info']['hdf5_row_id'], 'F14v2: same row and seed as F14')
    NC = noise_check_seed(I, blk, row_id, D['info']['psd_source_strain_file'])
    gen.interpolate = pycbc_interpolate          # bound lazily by initialize_physics() in the generator
    C = SimulationConfig(**spec['simulation'])
    sp = dict(spec['signal_processor'])
    I.check(sp['apply_highpass'] and sp['apply_lowpass'] and not sp.get('apply_standardization', False)
            and sp['output_mode'] == 'crop_to_config',
            'F14v2: pipeline filter = high-pass and low-pass FIR then crop to the window, no z-score')
    band_hz = (float(sp['highpass_frequency']), float(sp['lowpass_frequency']))
    I.check(band_hz == ASD_BAND, f'F14v2: pipeline band {band_hz} equals the shaded band {ASD_BAND}')
    proc = SignalProcessor(config=C, **sp)
    band = SignalProcessor(config=C, **dict(sp, whitening_method='none'))
    df = pd.DataFrame([dict(block_id=blk['block_id'], psd_start=blk['psd_start'], psd_end=blk['psd_end'])])
    sel_, _ = gen.resolve_cached_psd_files(I.data_root, df)
    seg = gen.read_psd_intervals(next(df.itertuples()), sel_)['H1']
    I.check(np.isclose(float(seg.delta_t), C.delta_t, rtol=0, atol=1e-12), 'F14v2: real strain at the config rate')
    L = int(C.processing_length)
    i0 = (len(seg) - L) // 2                     # processing window centred in the 128 s PSD segment
    real = TimeSeries(np.asarray(seg.numpy()[i0:i0 + L], float), delta_t=C.delta_t,
                      epoch=float(seg.start_time) + i0 * C.delta_t)
    real_bp = band.process(strain=real, psd=None, detector_name='H1')
    psd_proc = gen.estimate_raw_offsource_psd(seg, psd_start=blk['psd_start'], psd_end=blk['psd_end'],
                                              delta_f=C.processing_delta_f, target_flength=C.processing_flength)
    noise = gaussian.noise_from_psd(psd=psd_proc, length=L, delta_t=C.delta_t, seed=NC['seed'])
    noise.start_time = p0
    sim_bp = band.process(strain=noise, psd=None, detector_name='H1')
    I.check(np.isclose(float(np.std(sim_bp.numpy())), NC['std_noise_check'], rtol=1e-6),
            f"F14v2c: in-band 4 s std of seed {NC['seed']} reproduces noise_check.json")
    white = np.asarray(proc.process(strain=noise, psd=psd_proc, detector_name='H1').numpy(), float)
    from scipy.signal import welch as sp_welch
    fw, pw = sp_welch(white, fs=1.0 / C.delta_t, window='hann', nperseg=SEED_F14_ASD_NPERSEG,
                      noverlap=SEED_F14_ASD_NPERSEG // 2, average='median')   # inset as F14d
    sim_psd = noise.psd(float(gen.PSD_SEGMENT_DURATION_S))   # same TimeSeries.psd call as the generator estimate
    a, c = np.asarray(real_bp.numpy(), float), np.asarray(sim_bp.numpy(), float)
    I.check(len(a) == len(c) == C.length == tr['n'], 'F14v2: 4 s windows')
    g0 = float(real_bp.start_time)
    I.check(np.isclose(g0, NC['f14v2a_window_start_gps']), 'F14v2a: real window = the window of noise_check.json')
    sa, sc = float(np.std(a)), float(np.std(c))
    lim = 1.05 * float(max(np.abs(a).max(), np.abs(c).max()))
    d = dict(t=np.arange(C.length) * C.delta_t, real=a, sim=c, lim=lim, white=white, white_asd=(fw, np.sqrt(pw)),
             asd_real=D['asd'], asd_sim=(sim_psd.sample_frequencies.numpy(), np.sqrt(sim_psd.numpy())),
             info=dict(psd_block=blk, real_strain_file=D['info']['psd_source_strain_file'],
                       real_window_gps=(g0, g0 + C.length * C.delta_t),
                       real_processing_window_gps=(float(real.start_time), float(real.start_time) + L * C.delta_t),
                       real_window_rule='processing window (context + 4 s + context) centred in the 128 s PSD '
                                        'segment; the 4 s are its crop after the band filter',
                       band_filter=dict(code='src/processing.py SignalProcessor.process, G3 signal_processor config '
                                             "with whitening_method='none'", band_hz=band_hz,
                                        fir_order=sp.get('fir_order'), fir_beta=sp.get('fir_beta'),
                                        remove_corrupted=sp.get('remove_corrupted')),
                       std_in_band=dict(real_a=sa, simulated_c=sc, ratio_c_over_a=sc / sa, unit='strain'),
                       simulated_seed=dict({k: v for k, v in NC.items() if k != 'f14v2a_window_start_gps'},
                                           ratio_sim_over_real=sc / sa),
                       hdf5_row_id=row_id, row_gaussian_seed_H1_not_used=seed, welch=D['info']['welch'],
                       psd_estimation_segment_s=D['info']['psd_estimation_segment_s']))
    return I.store('noise_demo_v2', d, start)


NOISE_CHECK_REL = 'results/presentation/cpan_2026/physics/noise_check.json'


def noise_check_seed(I, blk, row_id, real_file):
    """Seed of the simulated noise of F14v2: among the 50 H1 realisations of noise_check.json (seeds base + i, base
    read from its method text), the one whose 4 s in-band std is closest to their median (first on ties); its
    empirical percentile = 100 * fraction of the 50 stds <= its std."""
    import re as re_
    nc = I.json(I.data_root / NOISE_CHECK_REL, 'F14v2 noise diagnostic (noise_check.json)')
    I.check(nc['psd_block']['block_id'] == blk['block_id'] and int(nc['psd_block']['hdf5_row_id']) == row_id
            and nc['psd_block']['files']['H1'] == real_file, 'noise_check.json: same PSD block, row and H1 file')
    m = re_.search(r'SIM_SEED_BASE\+i \(base (\d+)\)', nc['method']['simulated'])
    I.check(m is not None, 'noise_check.json: simulated seeds = SIM_SEED_BASE + i with the base in method.simulated')
    sim = nc['detectors']['H1']['simulated']
    stds = np.asarray(sim['std_per_realisation'], float)
    I.check(len(stds) == int(sim['std']['n']) == 50 and np.isclose(np.median(stds), sim['std']['median'], rtol=1e-12),
            'noise_check.json H1: 50 realisations, median as stored')
    med = float(np.median(stds))
    dist = np.abs(stds - med)
    i = int(np.argmin(dist))
    tied = [int(k) for k in np.flatnonzero(np.isclose(dist, dist[i], rtol=1e-9, atol=0))]
    return dict(seed=int(m.group(1)) + i, realisation_index=i, std_noise_check=float(stds[i]),
                tied_candidates=[dict(index=k, seed=int(m.group(1)) + k, std=float(stds[k]),
                                      percentile_in_50=float(100.0 * np.mean(stds <= stds[k]))) for k in tied],
                tie_rule='first index (with 50 values the median is the midpoint of the two central stds, which are '
                         'equally close)',
                median_std_50=med, percentile_in_50=float(100.0 * np.mean(stds <= stds[i])),
                rule='seed of the 50 noise_check.json H1 realisations whose in-band 4 s std is closest to the median; '
                     'percentile = 100 x fraction of the 50 stds <= it',
                source=str(I.data_root / NOISE_CHECK_REL),
                f14v2a_window_start_gps=float(nc['detectors']['H1']['f14v2a_window']['start_gps']))


def _f14v2_axes():
    return plt.subplots(figsize=tuple(FIGOPT['F14v2']['size']), layout='constrained')


def _f14v2_trace(I, out, name, key, note):
    D = noise_demo_v2(I)
    with slide_rc():
        fig, ax = _f14v2_axes()
        ax.plot(D['t'], D[key], color=COL['f14_line'], lw=0.5)
        ax.set(xlim=(0.0, 4.0), ylim=(-D['lim'], D['lim']), xlabel='time [s]', ylabel='strain')
        ax.yaxis.set_major_locator(MaxNLocator(4, symmetric=True))
        ax.grid(False)
        files = save_slide(fig, out, name, strict=True)
    return dict(files=files, role='main', detector='H1', **D['info'],
                axes=dict(xlim=(0.0, 4.0), ylim=(-D['lim'], D['lim']), shared_by='F14v2a, F14v2c'), note=note)


def f14v2a(I, out, name):
    return _f14v2_trace(I, out, name, 'real', 'Real O3b H1 strain (local GWOSC cache, nothing downloaded), 4 s at '
                                             'real_window_gps, after the pipeline band filter; strain units.')


def f14v2c(I, out, name):
    return _f14v2_trace(I, out, name, 'sim', 'Simulated Gaussian noise (noise_from_psd with the PSD of F14v2b, seed '
                                            'simulated_seed: the median-std realisation of noise_check.json), same '
                                            'band filter and axes as F14v2a.')


def f14v2b(I, out, name):
    D = noise_demo_v2(I)
    (f, psd), (fs_, asd_s) = D['asd_real'], D['asd_sim']
    with slide_rc():
        fig, ax = _f14v2_axes()
        m, ms = (f >= 10) & (f <= FS / 2), (fs_ >= 10) & (fs_ <= FS / 2)
        ax.axvspan(*ASD_BAND, color=COL['validity_band'], lw=0, zorder=0)
        l1, = ax.loglog(f[m], np.sqrt(psd[m]), color=INK, lw=1.8)
        l2, = ax.loglog(fs_[ms], asd_s[ms], color=COL['signal'], lw=0.8)
        ax.set(xlim=(10, FS / 2), xlabel='frequency [Hz]', ylabel=r'ASD [1/$\sqrt{\rm Hz}$]')
        ax.grid(True, which='major')
        leg = place_legend(ax, [Line2D([], [], color=INK, lw=1.8), Line2D([], [], color=COL['signal'], lw=1.8)],
                           ['O3b data', 'simulated noise'], prefer=('upper right',), grow=4)
        files = save_slide(fig, out, name, strict=True)
    return dict(files=files, role='main', detector='H1', **D['info'], legend=leg, band_hz=ASD_BAND,
                note='Dark: ASD of the real 128 s segment (the F14b / F03asd PSD). Thin: ASD of the simulated noise of '
                     'F14v2c, estimated with the same TimeSeries.psd call (8 s Welch) over its full processing window '
                     '(before the band filter, so the comparison is not shaped by the filter). Shaded: 30-1024 Hz.')


def f14v2d(I, out, name):
    D = noise_demo_v2(I)
    fw, aw = D['white_asd']
    with slide_rc():
        fig, ax = _f14v2_axes()
        ax.plot(D['t'], D['white'], color=COL['f14_line'], lw=0.5)
        ax.set(xlim=(0.0, 4.0), xlabel='time [s]', ylabel='whitened')
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.grid(False)
        lim = 1.15 * float(np.max(np.abs(D['white'])))
        ax.set_ylim(-lim, 1.9 * lim)                 # headroom for the inset above the trace (as F14d)
        ins = ax.inset_axes(list(FIGOPT['F14v2']['inset_bounds']))
        m = (fw >= 10) & (fw <= FS / 2)
        ins.axvspan(*ASD_BAND, color=COL['validity_band'], lw=0, zorder=0)
        ins.loglog(fw[m], aw[m], color=INK, lw=1.2)
        ins.set_xlim(10, FS / 2)
        ins.set_xticks([])
        ins.set_yticks([])
        ins.minorticks_off()
        ins.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
        ins.grid(False)
        ins.text(0.04, 0.9, 'ASD', transform=ins.transAxes, ha='left', va='top', fontsize=SLIDE['tick_size'])
        files = save_slide(fig, out, name, strict=True)
    return dict(files=files, role='main', detector='H1', **D['info'],
                inset_asd=f'as F14d (scipy Welch, Hann, {SEED_F14_ASD_NPERSEG} samples, 50 %, median; no tick labels)',
                note='The F14v2c noise (simulated_seed) whitened as in the pipeline (SignalProcessor.process with the '
                     'G3 config), before the per-sample z-score.')


def _f15v2(I, out, name, which):
    """F15 with landscape sizes: a / b = F15v2.size_ab, c = F15v2.size_c; same row, scaling and axes as F15."""
    D = f15_parts(I)
    size = FIGOPT['F15v2']['size_c' if which == 'c' else 'size_ab']
    with slide_rc():
        fig, ax = plt.subplots(figsize=tuple(size), layout='constrained')
        if which == 'a':
            ax.plot(D['t'], D['sig'], color=COL['signal'], lw=1.4)
        elif which == 'b':
            ax.plot(D['t'], D['noise'], color=COL['f14_line'], lw=0.4)
        else:
            ax.plot(D['t'], D['x'], color=COL['f14_line'], lw=0.4)
            ax.plot(D['t'], D['sig'], color=COL['signal'], lw=0.9, alpha=0.8)
        ax.set(xlim=(0.0, 4.0), ylim=(-D['lim'], D['lim']), xlabel='time [s]', ylabel='H1')
        ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))
        ax.grid(False)
        files = save_slide(fig, out, name, strict=True)
    return dict(files=files, role='main', row=D['row'], split_of_row='train', detector='H1', z_scale_fit=D['fit'],
                size_in=list(size), axes=dict(xlim=(0.0, 4.0), ylim=(-D['lim'], D['lim']),
                                              shared_by='F15v2a, F15v2b, F15v2c (and F15a-c)'),
                note='As F15 (same row, scaling and limits): signal = a L(s) + b, noise = X - (a L(s) + b), '
                     'data = X (network input), with the signal overlaid in F15v2c.')


def f15v2a(I, out, name):
    return _f15v2(I, out, name, 'a')


def f15v2b(I, out, name):
    return _f15v2(I, out, name, 'b')


def f15v2c(I, out, name):
    return _f15v2(I, out, name, 'c')


# ---- F04 -------------------------------------------------------------------------------------------------------
def f04(I, out, name):
    tr, R = I.train(), I.rows()
    M = I.m13_13()
    p = R['f03']
    g3 = np.asarray(tr['cg3'][p, 0], dtype=np.float64)
    g3t = np.asarray(tr['cg3t'][p, 0], dtype=np.float64)
    im = merger_index(g3t)
    tau = float(tau30(tr['y'][p, 0]))
    # The G3 step sits where the 30 Hz start of the G3 waveform falls in the window. It is located as the largest
    # |G3 - G3T| within +-max(0.1 s, tau30 / 2) of the Newtonian start (merger - tau30); both are aligned (G-ALIGN).
    t0 = im - int(round(tau * FS))
    hw = int(round(max(0.1, 0.5 * tau) * FS))
    a, b = max(0, t0 - hw), min(im, t0 + hw)
    I.check(b - a > 10, f'F04 row {p}: Newtonian 30 Hz start inside the window')
    i_step = a + int(np.argmax(np.abs(g3[a:b] - g3t[a:b])))
    w0 = max(0, i_step - int(round(F04_PRE_S * FS)))
    w1 = min(tr['n'], w0 + int(round(F04_LEN_S * FS)))
    fig = plt.figure(figsize=FULL)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    t = np.arange(w0, w1) / FS
    s15 = tr['s15']
    ax.plot(t, g3[w0:w1], color=C_G3, lw=1.4, label=f'abrupt start ({G3_F_START:.0f} Hz)')
    ax.plot(t, g3t[w0:w1], color=C_G3T, lw=1.4,
            label=f"tapered start ({s15['f_lower']:.0f} Hz,\ntaper to {s15['taper_end_hz']:.0f} Hz)")
    ax.axvline(i_step / FS, color=INK, ls=':', lw=1.0)
    ax.set_xlabel('time in window [s]')
    ax.set_ylabel('whitened H1 strain')
    ax.grid(False)
    ax.legend(loc='upper left', fontsize=ST['legend_size'] - 2)
    ax.set_title('same train row, clean signal', loc='left', fontsize=ST['axes_size'] - 2)
    ax = fig.add_subplot(gs[0, 1])
    xs = np.arange(len(LABELS))
    spec = (('i', -0.19, C_G3, '(i) network trained on abrupt-start data:\n     tapered vs abrupt-start inputs'),
            ('ii', 0.19, C_G3T, '(ii) network retrained on tapered data vs\n     network trained on abrupt-start data,\n'
                                '     both on tapered inputs'))
    for k, dx, col, lab in spec:
        v, ci = M['point'][k], M['ci'][k]
        ax.bar(xs + dx, v, width=0.36, color=col, label=lab, yerr=[v - ci[0], ci[1] - v],
               error_kw=dict(ecolor=INK, lw=1.2))
        for j in range(len(LABELS)):
            off = ci[1][j] if v[j] >= 0 else ci[0][j]
            ax.text(xs[j] + dx, off + (1.0 if v[j] >= 0 else -1.0), f'{v[j]:+.1f}', ha='center',
                    va='bottom' if v[j] >= 0 else 'top', fontsize=ST['annot_size'] - 3)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([var_label(lab, unit=False) for lab in LABELS])
    ax.set_ylabel(r'$\Delta$RMSE [%]')
    ax.grid(False, axis='x')
    ymin = min(M['ci'][k][0].min() for k in ('i', 'ii'))
    ymax = max(M['ci'][k][1].max() for k in ('i', 'ii'))
    ax.set_ylim(ymin - 6, ymax + 6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=ST['legend_size'] - 2, ncol=1)
    ax.set_title(f"validation ({M['n'] // 1000}k rows), 95 % paired bootstrap CI", loc='left',
                 fontsize=ST['axes_size'] - 2)
    files = save_slide(fig, out, name)   # same drawing as before; also checks that no internal name is left
    return dict(files=files, row=row_info(tr, p), split_of_row='train', detector='H1',
                step=dict(index=i_step, time_s=i_step / FS, newtonian_start_index=t0, search=[a, b], window=[w0, w1]),
                drmse_pct=dict(i=M['point']['i'], ii=M['point']['ii']),
                ci95=dict(i=M['ci']['i'], ii=M['ci']['ii']), rmse_by_case=M['rmse'],
                ci95_ii_m13_13_summary=M['stored_ci_ii'],
                max_abs_ci_diff_ii_vs_summary=M['max_abs_ci_diff_ii_vs_summary'],
                comparisons={'i': '100 (RMSE[4a-G3 on G3T val] / RMSE[4a-G3 on G3 val] - 1)',
                             'ii': '100 (RMSE[4a-G3T on G3T val] / RMSE[4a-G3 on G3T val] - 1) (= drmse_pct_i)'},
                bootstrap=dict(B=N_BOOT, seed=SEED_BOOT, rows='paired, same resampled rows for every case',
                               interval='percentile 2.5 / 97.5'),
                note='Left: clean signals exist for train rows only; same row as F03. Window of 0.3 s from 0.1 s '
                     'before the located G3 start step (largest |G3 - G3T| around the Newtonian 30 Hz start). '
                     'Right: per-row residuals of the M13.13 val npz (40k rows, 4a s123 models); point values '
                     'checked against the summary, CIs re-derived here.')


# ---- F05 and appendix diagrams ---------------------------------------------------------------------------------
def _diagram(I, out, name, make, crop=None, extra=None):
    c = I.cfg5b()
    with plt.rc_context(AD.RC):
        fig = make(c)
        files = save_diagram(fig, out, name, crop)
    sh = AD.tbank_shapes(c['model'], fs=FS)
    return dict(files=files, shapes=dict(conv=sh['conv'], tokens=sh['tok'], concat_channels=sh['cat'],
                                         head_out=sh['head_out']), **(extra or {}))


def f05(I, out, name):
    return _diagram(I, out, name, lambda c: AD.fig_tbank_overall(c['model'], source=CFG_5B.name))


def a1(I, out, name):
    return _diagram(I, out, name, lambda c: AD.fig_tbank_filterbank(c['model']))


def _reference_matches(I, c):
    """The reused M13.4a diagrams hard-code dilations 1, 2, 4, k = 7, 2 layers, 4 heads and dropout off for the last
    20 % of epochs: they are reused only if the 5b config has the same values."""
    kw = c['model']['kwargs']
    I.check(list(kw['dilations']) == [1, 2, 4] and kw['residual_kernel_size'] == 7 and kw['residual_channels'] == 64
            and kw['agg_layers'] == 2 and kw['agg_heads'] == 4 and kw['agg_pool'] == 8
            and c['training']['dropout_off_fraction'] == 0.2, 'reused M13.4a diagram matches the 5b config')


def a2(I, out, name):
    _reference_matches(I, I.cfg5b())
    return _diagram(I, out, name, lambda c: AD.fig_reference(),
                    crop=AD.CROPS['reference']['arch_cnn_tr_resblock'],
                    extra=dict(note='reused: architecture_diagrams.fig_reference panel (b)'))


def a3(I, out, name):
    _reference_matches(I, I.cfg5b())
    return _diagram(I, out, name, lambda c: AD.fig_reference(),
                    crop=AD.CROPS['reference']['arch_cnn_tr_transformer'],
                    extra=dict(note='reused: architecture_diagrams.fig_reference panel (c)'))


def a4(I, out, name):
    def make(c):
        t = c['training']
        return AD.fig_cholesky_head(c['model'], loss_note=f"Training loss: {t['loss']} (beta = {t['beta_nll']}).")
    return _diagram(I, out, name, make)


def a5(I, out, name):
    T = I.templates()
    hi, hq, n = T['hi'], T['hq'], T['n']
    K = hi.shape[0]
    # As TemplateBank._layout: unit L2 norm; the circular roll does not change |FFT|.
    hi = hi / (np.linalg.norm(hi, axis=1, keepdims=True) + 1e-12)
    hq = hq / (np.linalg.norm(hq, axis=1, keepdims=True) + 1e-12)
    Hi, Hq = np.abs(np.fft.rfft(hi, n=n)), np.abs(np.fft.rfft(hq, n=n))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    sel = [int(k) for k in np.unique(np.round(np.linspace(0, K - 1, N_FILTERS_SHOWN)).astype(int))]
    fig, axes = plt.subplots(2, 4, figsize=TALL, sharex=True, sharey=True)
    peaks = {}
    m = f > 0
    for ax, k in zip(axes.flat, sel):
        pk = max(Hi[k].max(), Hq[k].max())
        ax.plot(f[m], Hi[k][m] / pk, color='#1B9E77', lw=1.0)
        ax.plot(f[m], Hq[k][m] / pk, color='#7570B3', lw=1.0, ls='--')
        peaks[k] = float(f[np.argmax(Hi[k] + Hq[k])])
        ax.set_xscale('log')
        ax.set_title(f'filter {k}: peak {peaks[k]:.0f} Hz', fontsize=ST['tick_size'] - 1, loc='left')
        ax.grid(True, which='major')
    for ax in axes.flat[len(sel):]:
        ax.axis('off')
    axes[0, 0].set_xlim(10, FS / 2)
    axes[0, 0].set_ylim(0, 1.08)
    for ax in axes[1]:
        ax.set_xlabel('frequency [Hz]')
    for ax in axes[:, 0]:
        ax.set_ylabel('|H(f)| / max')
    fig.legend([Line2D([], [], color='#1B9E77', lw=1.4), Line2D([], [], color='#7570B3', lw=1.4, ls='--')],
               [r'$h^{I}_{k}$', r'$h^{Q}_{k}$'], loc='upper right', ncol=2, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle('Learned filters of the final model: amplitude response (whitened domain)', x=0.01, ha='left',
                 fontsize=ST['axes_size'])
    files = save(fig, out, name)
    return dict(files=files, filters_shown=sel, peak_frequency_hz=peaks, n_templates=K, checkpoint=T['ckpt'],
                checkpoint_sha256=T['sha'],
                note='|rfft| of the unit-norm template_i / template_q of the 5b state_dict (CPU, no forward); filters '
                     f'at round(linspace(0, {K - 1}, {N_FILTERS_SHOWN})); each panel normalised to its maximum.')


FIGURES = [('F02', 'F02_chirp_mass_and_latency', f02), ('F03', 'F03_input_and_split', f03),
           ('F03v2', 'F03v2_input_and_psd_H1', f03v2),
           ('F03v3', 'F03v3_input_H1', f03v3), ('F03asd', 'F03asd_psd_H1', f03asd),
           ('F03v4', 'F03v4_input_H1', f03v4),
           ('F14a', 'F14a_noise_strain_segment_H1', f14a), ('F14b', 'F14b_noise_asd_H1', f14b),
           ('F14c', 'F14c_noise_colored_H1', f14c), ('F14d', 'F14d_noise_whitened_H1', f14d),
           ('F15a', 'F15a_signal_H1', f15a), ('F15b', 'F15b_noise_H1', f15b), ('F15c', 'F15c_data_H1', f15c),
           ('F14v2a', 'F14v2a_noise_real_bandpassed_H1', f14v2a), ('F14v2b', 'F14v2b_noise_asd_real_vs_sim_H1', f14v2b),
           ('F14v2c', 'F14v2c_noise_simulated_bandpassed_H1', f14v2c),
           ('F14v2d', 'F14v2d_noise_simulated_whitened_H1', f14v2d),
           ('F15v2a', 'F15v2a_signal_H1', f15v2a), ('F15v2b', 'F15v2b_noise_H1', f15v2b),
           ('F15v2c', 'F15v2c_data_H1', f15v2c),
           ('F04', 'F04_step_vs_taper', f04), ('F05', 'F05_architecture_overall', f05),
           ('A1', 'A1_arch_filter_bank', a1), ('A2', 'A2_arch_residual_block', a2),
           ('A3', 'A3_arch_transformer', a3), ('A4', 'A4_arch_cholesky_head', a4),
           ('A5', 'A5_filter_responses', a5)]
SET_NAME = 'cpan_physics'


def run(data_root, out_base=None, only=None, resume_run=None):
    """Produce the set (same run / manifest / contact-sheet conventions as cpan_results_figures.run).

    only: None = all figures; else a list of IDs (F02-F05, A1-A5). resume_run: an existing run_<UTC> directory of
    this set; the selected figures are regenerated there and manifest.json / contact_sheet.png are merged.
    """
    data_root = Path(data_root)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    valid = [f for f, _, _ in FIGURES]
    sel = valid if not only else [x.strip() for x in only if x.strip()]
    bad = [x for x in sel if x not in valid]
    require(not bad, f'--only: unknown IDs {bad}; valid: {valid}')
    old = None
    if resume_run is not None:
        out = Path(resume_run)
        require(out.is_dir() and out.name.startswith('run_') and (out / 'manifest.json').is_file(),
                f'--resume-run {out}: not a run_<UTC> directory with manifest.json')
        old = json.loads((out / 'manifest.json').read_text())
        require(old.get('set') == SET_NAME, f'--resume-run {out}: manifest is not of the {SET_NAME} set')
    else:
        base = (Path(out_base) if out_base is not None
                else data_root / 'results' / 'presentation' / 'cpan_2026' / 'physics')
        out = base / f'run_{stamp}'
        require(not out.exists(), f'{out} exists; refusing to overwrite')
        out.mkdir(parents=True)
    set_style()
    I = PInputs(data_root)
    here = Path(__file__).resolve()
    script_sha = digest(here)
    helpers = [ROOT / 'scripts/m13/presentation/architecture_diagrams.py',
               ROOT / 'scripts/m13/presentation/cpan_results_figures.py']

    def safe_entry(path):
        try:
            return I.entry(path)
        except Exception as e:  # noqa: BLE001 - an unreadable input must not prevent writing the manifest
            return dict(path=str(path), sha256=None, status='ERROR', error=f'{type(e).__name__}: {e}')

    new = {}
    try:
        for fid, name, fn in FIGURES:
            if fid not in sel:
                continue
            I.begin()
            try:
                res, status = fn(I, out, name), 'OK'
            except Missing as e:
                res, status = dict(reason=str(e)), 'MISSING'
            except FailedCheck as e:
                res, status = dict(reason=f'check failed: {e}'), 'FAILED_CHECK'
            except Exception as e:  # noqa: BLE001 - recorded in the manifest, reported in the exit status
                res, status = dict(reason=f'{type(e).__name__}: {e}'), 'ERROR'
            plt.close('all')
            used = list(dict.fromkeys(str(p) for p in I.used))
            new[fid] = dict(id=fid, name=name, status=status, files=res.pop('files', []), generated_utc=stamp,
                            script_sha256=script_sha, inputs=[safe_entry(p) for p in used], **res)
            print(f'{fid}: {status}' + (f"  ({res['reason']})" if 'reason' in res else ''))
    finally:
        prev = {e['id']: e for e in (old or {}).get('figures', [])}
        entries = [new.get(fid, prev.get(fid)) for fid, _, _ in FIGURES if fid in new or fid in prev]
        try:
            sheet = contact_sheet(out, entries)
        except Exception as e:  # noqa: BLE001
            plt.close('all')
            sheet = dict(file=None, status='ERROR', error=f'{type(e).__name__}: {e}')
        problems = [f"{e['id']}: {e['status']} ({e.get('reason', '')})" for e in new.values() if e['status'] != 'OK']
        if sheet.get('status') == 'ERROR':
            problems.append(f"contact_sheet.png: ERROR ({sheet['error']})")
        cfgs = [dict(path=str(p), sha256=digest(p)) for p in (BASE_CFG, CPAN_CFG, CFG_5B) if p.exists()]
        update = dict(utc=stamp, items=sel, script_sha256=script_sha, problems=problems, configs=cfgs,
                      helpers=[dict(path=str(p), sha256=digest(p)) for p in helpers])
        manifest = dict(
            set=SET_NAME, created_utc=(old or {}).get('created_utc', stamp), updated_utc=stamp, run_dir=str(out),
            data_root=str(data_root), script=dict(path=str(here), sha256=script_sha), configs=cfgs,
            helpers=update['helpers'],
            artifacts=dict(model=TAG, training_config=str(CFG_5B), m13_13_evaluation_stamp=M13_13_STAMP,
                           clean_signals='train rows only (no clean val / cal / test signal is stored)'),
            checks=list(dict.fromkeys(list((old or {}).get('checks', [])) + I.checks)),
            info={**(old or {}).get('info', {}), **to_py(I.info)}, figures=entries,
            contact_sheet=sheet, problems_last_update=problems,
            updates=list((old or {}).get('updates', [])) + [update],
            note='script / config sha256 at top level are those of the last update; each figure entry records the '
                 'script sha256 and UTC stamp that produced it.')
        (out / 'manifest.json').write_text(json.dumps(to_py(manifest), indent=2, ensure_ascii=False))
        print(f'Output: {out}')
    if problems:
        print('Problems (exit code 1):')
        for x in problems:
            print('  ' + x)
        sys.exit(1)
    return out
