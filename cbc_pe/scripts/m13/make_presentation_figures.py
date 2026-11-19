"""Presentation figures for M13 (read-only; descriptive).

Every plotted value is read from an existing artifact; nothing is trained, predicted or re-fitted:
  - M13.1 CSVs (M11 diagnostics by f220 class x SNR)                          -> F1
  - history.npz of the M13.3a / M13.3b / M13.4a / M13.6 runs                   -> F2, F5, T1, v1
  - M13_3b_4a_evaluation_summary.json (deciles, bands, SNR for M11 and G1)     -> F3, F4, F7
  - m13_5_summary.json (re-noise MSEs)                                         -> F6
  - latest M13_6_evaluation_summary_*.json (deciles, tau30, bands, SNR, rho)   -> F7-F11, d2, d3, d5, a1, a2
  - final val npz of M11 and of the M13 runs                                   -> r2, d1, d4, a1 (seed band)
  - latest completed M13_7 run directory (CSVs, summary, cal-only bundle)      -> m1, m1a, m1b, m2-m4, l1, l2
  - M11.14 outputs (lvk_comparison.csv) and its frozen LVK reference CSV       -> r1, l1
  - GWTC-3 PE samples (DATA_ROOT/external/gwtc3_pe)                            -> l1
  - GWOSC-cache strain of one train PSD block; M13.0c pilots / first G3 val rows -> d0
Bins (SNR, tau30 deciles, f220 / M_total classes) and the paired row bootstrap are therefore exactly those of the
notebooks that wrote the summaries. Optional models follow the M13_6_evaluation rule (included if their final val npz
exists); summary-based figures can only show models present in that summary. m1b/m3 re-apply the saved M13_7 Mondrian
bundle (cal-only quantiles) to the saved test predictions, as M13_7 cell 8, and check the medians they reproduce.

Names, colours, line styles and sizes: configs/presentation/labels.json (no display name is written here). 'short'
model names in f9, f10, d1-d5, a1, a2, v1, m*, l1, l2; full 'label' elsewhere. Only neutral greys are fixed here.
Output: --out, default <DATA_ROOT>/results/M13_presentation/figures/ (PNG + PDF) and figures_manifest.json.
"""
import argparse
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_data_root, resolve_processed_artifact  # noqa: E402
from scripts.m12.m12_5_3_posthoc_check import LABELS, digest, npz_to_physical, require  # noqa: E402

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
    ('4c_aug_s123', 'M13_4c_G3aug_s123', True, False),
]
G1_HIST_SPECS = [('R1', 'M13_3a_R1_s123'), ('R2', 'M13_3a_R2_s123'), ('R3', 'M13_3a_R3_s124')]
VAL_NPZ, HIST_NPZ, CAL_TEST_NPZ = 'val_predictions_embeddings.npz', 'history.npz', 'cal_test_predictions_embeddings.npz'
RENOISE_TAGS = {'3b_s123': 'M13_3b_G3_s123', '3b_s124': 'M13_3b_G3_s124', '4a_s123': 'M13_4a_G3_s123'}
SUMMARY_RTOL = 1e-9  # identical quantities written by two notebooks must agree to float round-off
RECOMPUTE_RTOL = 1e-6  # quantity recomputed here from the same npz as a notebook (summation order may differ)
# Set key (= labels.json 'models' key) of each run. 'G1' = mean R2/R3, '3b' = mean of the 3b seeds ('G3' in the
# M13.3b/4a summary), '6a' / '4a_aug' / '4b_aug' / '4c_aug' = mean of the available seeds of that set.
SET_OF = {'R1': 'R1', 'R2': 'G1', 'R3': 'G1', '3b_s123': '3b', '3b_s124': '3b', '4a_s123': '4a_s123',
          '6a_aug_s123': '6a', '6a_aug_s124': '6a', '4a_aug_s123': '4a_aug', '4a_aug_s124': '4a_aug',
          '4b_aug_s123': '4b_aug', '4c_aug_s123': '4c_aug'}
ARCH_SETS = ('4a_aug', '4b_aug', '4c_aug')
D1_RUNS = ('6a_aug_s123', '4b_aug_s123')  # same seed: reference CNN vs DenseNet + transformer (d1, d4)
SEED_BAND_PAIR = ('4a_aug_s123', '4a_aug_s124')  # a1 seed-noise band, definition of M13_6 cell 11
M13_7_RUN = '4b_aug_s123'  # model of the M13_7 provisional pipeline (its results live under this run tag)
PRIMARY_POLICY = 'conservative'  # M11.14 / M13_7 primary Mondrian policy
# GWTC-3 PE release read as M13_7 cell 27 (combined-waveform default analysis, 90 % interval).
PE_VARIANT, PE_GROUP, PE_LEVEL = 'mixed_cosmo', 'C01:Mixed', 0.90
PE_Q = [(1 - PE_LEVEL) / 2, 1 - (1 - PE_LEVEL) / 2]
LVK_REF_NAME = 'GWTC-3-confident_detector_frame_reference_error_propagated.csv'  # frozen M11.14 reference (cell 8)
V1_RUNS = ('6a_aug_s123', '6a_aug_s124', '4a_aug_s123', '4a_aug_s124', '4b_aug_s123', '4c_aug_s123')
DROPOUT_OFF_EPOCH = 64  # training schedule of the M13 runs (dropout switched off from this epoch)
V1_ZOOM = (60, 79)

# ---- Presentation style (configs/presentation/labels.json) -----------------------------------------------------
LAB = json.loads((ROOT / 'configs/presentation/labels.json').read_text())
ST = LAB['style']
REF_SET = LAB['reference_model']
GRID, REF = ST['grid'], ST['refline']
FIG13, FIG1 = tuple(ST['fig13']), tuple(ST['fig1'])
SEQ3 = ST['seq3']  # ordered categories (SNR bins, re-noise bars, rho regimes, detectors, filter lengths)
EVENTS = LAB['event_palette']  # one colour per real event (cycled)
INK, GREY, GREY_LIGHT = '#333333', '#8C8C8C', '#D9D9D9'  # neutral greys (the only colours fixed in this file)
NAUG = LAB['datasets']['naug'].split(' (')[0]
WB = LAB['datasets']['wb'].split(' (')[0]


def sty(key, short=False):
    """(legend name, colour, line style) of a labels.json model key; `short` selects the short name."""
    m = LAB['models'][key]
    return m['short' if short else 'label'], m['color'], m['ls']


def run_label(m, short=True):
    """Legend name of a single-seed run: set name + ' (seed N)'."""
    return f"{sty(SET_OF[m], short)[0]} (seed {m.rsplit('_s', 1)[1]})"


def var_label(lab, unit=True):
    v = LAB['variables'][lab]
    return f"{v['label']} [{v['unit']}]" if unit and v['unit'] else v['label']


def width_label(lab, what='Interval width'):
    u = LAB['variables'][lab]['unit']
    return f'{what} [{u}]' if u else what


class Skip(Exception):
    """A figure cannot be produced from the available inputs (reported, not fatal)."""


def set_style():
    plt.rcParams.update({
        'font.family': ST['font'], 'mathtext.fontset': 'dejavusans',
        'axes.labelsize': ST['axes_size'], 'axes.titlesize': ST['axes_size'],
        'xtick.labelsize': ST['tick_size'], 'ytick.labelsize': ST['tick_size'],
        'legend.fontsize': ST['legend_size'], 'legend.frameon': False,
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


def legend_below(fig, ncol=3, extra=()):
    seen = {h.get_label(): h for h in extra}
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
        self.m11_dir = data_root / 'results' / 'M11_final_G1_500k'
        self.m13_1_dir = self.m11_dir / 'M13_1_diagnostics'
        self.pe_dir = data_root / 'external' / 'gwtc3_pe'
        self.g3_cfg = json.loads((ROOT / 'configs/experiments/m13/train_m13_3b_G3_s123.json').read_text())
        self.g3_dataset_id = self.g3_cfg['dataset']['dataset_id']
        self.m13_5_path = data_root / 'processed' / self.g3_dataset_id / 'm13_5_renoise' / 'm13_5_summary.json'
        cands = sorted(self.m13_dir.glob('M13_6_evaluation_summary_*.json'))  # timestamped names sort by time
        self.s6_path = cands[-1] if cands else None
        self._json, self._hist, self._sha, self._val, self._pe = {}, {}, {}, {}, {}
        self._pe_manifest, self._m13_7, self._m13_7_test = None, None, None
        self.nohash = {}  # resolved path -> reason sha256 is not computed (dataset HDF5 of ~100 GB)
        self.spec_tag = {k: t for k, t, *_ in MODEL_SPECS}

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

    def input_entry(self, path):
        p = str(Path(path).resolve())
        if p in self.nohash:
            return dict(path=str(path), sha256=None, size=Path(p).stat().st_size, note=self.nohash[p])
        return dict(path=str(path), sha256=self.sha(path))

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

    def val_path(self, key):
        return self.m13_dir / self.spec_tag[key] / VAL_NPZ

    def val(self, key, path=None):
        """Final val npz in physical units, sorted by HDF5 row id (as load_pred of M13_6 cell 3)."""
        if key in self._val:
            return self._val[key]
        path = Path(path) if path is not None else self.val_path(key)
        if not path.exists():
            raise Skip(f'missing {path}')
        ref = self.summary_sha(path)
        if ref is not None:
            self.check(self.sha(path) == ref, f'{key}: val npz SHA256 equals evaluation summary')
        with np.load(path, allow_pickle=False) as a:
            need = ['pred_val', 'y_val', 'idx_val', 'y_mean', 'y_std', 'label_names']
            missing = [k for k in need if k not in a.files]
            if missing:
                raise Skip(f'{path.name} lacks {missing}')
            f, y, mean, sd = (a[k].astype(np.float64) for k in ('pred_val', 'y_val', 'y_mean', 'y_std'))
            idx, names = a['idx_val'].astype(np.int64), a['label_names'].tolist()
        self.check(names == LABELS, f'{key}: val npz label order')
        self.check(f.shape == y.shape == (len(idx), 3) and len(np.unique(idx)) == len(idx), f'{key}: val npz shapes')
        y, f, _ = npz_to_physical(y, f, mean, sd)
        for k0, v0 in self._val.items():  # one train-only scaler for every run (M13_6 checks it equals M11's)
            self.check(np.allclose(mean, v0['mean'], rtol=1e-6) and np.allclose(sd, v0['sd'], rtol=1e-6),
                       f'{key}: same train-only label scaler as {k0}')
            break
        o = np.argsort(idx)
        self._val[key] = dict(f=f[o], y=y[o], idx=idx[o], mean=mean, sd=sd, path=path)
        return self._val[key]

    def scaler(self):
        """Train-only label scaler (y_mean, y_std), read from the reference run's val npz."""
        v = self.val(f'{REF_SET}_aug_s123')
        return v['mean'], v['sd']

    def g3_paths(self):
        ds = self.g3_cfg['dataset']
        try:
            h5 = Path(resolve_processed_artifact(data_root=self.data_root, dataset_id=ds['dataset_id'],
                                                 file_name=ds['dataset_file']))
            split = Path(resolve_processed_artifact(data_root=self.data_root, dataset_id=ds['dataset_id'],
                                                    file_name=ds['split_file']))
        except (FileNotFoundError, OSError, ValueError) as e:
            raise Skip(f'G3 dataset not resolvable: {e}')
        for p in (h5, split):
            if not p.exists():
                raise Skip(f'missing {p}')
        self.nohash[str(h5.resolve())] = 'sha256 not computed (dataset HDF5); only the listed rows/columns are read'
        return h5, split

    def s6_sets(self):
        """Summary sets of M13_6 (cell 15): '3b', the reference '6a' and the augmented architectures it evaluated."""
        s = self.s6()
        return ['3b', REF_SET] + [a for a in ARCH_SETS if a in s['settings']['arch_aug']]

    def s6_seeds(self, sets):
        """{set: seed models} of the M13_6 summary, for the requested sets present in it (missing ones warned)."""
        st = self.s6()['settings']
        arch = st['arch_aug'] if isinstance(st['arch_aug'], dict) else {a: [a] for a in st['arch_aug']}
        known = {'3b': st['b3'], REF_SET: st['a6'], **arch}
        miss = [s for s in sets if s not in known]
        if miss:
            warnings.warn(f'{miss} not in {self.s6_path.name}; omitted')
        return {s: known[s] for s in sets if s in known}

    def s6_arch_models(self, sets=None):
        """Seed models behind the M13_6 architecture sets. `arch_aug` is {set: [seeds]} in current summaries and a
        plain list of single-seed keys in summaries written before the second 4a_aug seed."""
        arch = self.s6()['settings']['arch_aug']
        if not isinstance(arch, dict):
            arch = {a: [a] for a in arch}
        return [m for a, ms in arch.items() if sets is None or a in sets for m in ms]

    def m13_7(self):
        """(run dir, summary path, summary) of the latest M13_7 run that wrote its summary (one run_<stamp>/run)."""
        if self._m13_7 is None:
            base = self.m13_dir / self.spec_tag[M13_7_RUN] / 'M13_7_pipeline_provisional'
            runs = [d for d in sorted(base.glob('run_*')) if any(d.glob('M13_7_summary_*.json'))]
            if not runs:
                raise Skip(f'no completed M13_7 run in {base}')
            spath = sorted(runs[-1].glob('M13_7_summary_*.json'))[-1]
            s = self.json(spath)
            self.check(s['model'] == self.spec_tag[M13_7_RUN], 'M13_7 summary model tag')
            self.check(s['real_events']['primary_policy'] == PRIMARY_POLICY, 'M13_7 primary policy')
            self._m13_7 = (runs[-1], spath, s)
        return self._m13_7

    def m13_7_file(self, name):
        p = self.m13_7()[0] / name
        if not p.exists():
            raise Skip(f'missing {p}')
        return p

    def m13_7_test(self):
        """Per-row test widths (physical) and Mondrian bins of the primary policy, re-applying the saved cal-only
        bundle to the saved test predictions exactly as M13_7 cell 8; medians checked against the M13_7 summary."""
        if self._m13_7_test is not None:
            return self._m13_7_test
        import pickle
        from copy import copy
        from dataclasses import replace
        try:
            from src.conformal.pipeline import apply_mondrian
        except ImportError as e:
            raise Skip(f'src.conformal not importable: {e}')
        run, spath, s = self.m13_7()
        prov = s['provenance']
        bundles = sorted(run.glob('mondrian_*_calibrators_cal_only.pkl'))
        if len(bundles) != 1:
            raise Skip(f'expected one cal-only Mondrian bundle in {run}; found {len(bundles)}')
        bundle = bundles[0]
        self.check(self.sha(bundle) == prov['bundle_sha256'], 'M13_7 Mondrian bundle SHA256 equals M13_7 summary')
        pred_path = self.m13_dir / self.spec_tag[M13_7_RUN] / CAL_TEST_NPZ
        if not pred_path.exists():
            raise Skip(f'missing {pred_path}')
        self.check(self.sha(pred_path) == prov['predictions_sha256'],
                   'M13_7 cal/test predictions SHA256 equals M13_7 summary')
        with np.load(pred_path, allow_pickle=False) as z:
            self.check(z['label_names'].tolist() == LABELS, 'M13_7 cal/test npz label order')
            sd = z['y_std'].astype(np.float64)
            idx = z['idx_test'].astype(np.int64)
            pred, emb = z['pred_test'].astype(np.float64), z['emb_test'].astype(np.float64)
        with bundle.open('rb') as fh:
            calibrators = pickle.load(fh)['calibrators']  # own artifact, identity checked above
        widths, bins = {}, {}
        for (policy, lab), system in calibrators.items():
            if policy != PRIMARY_POLICY:
                continue
            j = system.label_index
            fitted = replace(system.fitted, difficulty_model=copy(system.fitted.difficulty_model))
            pr = apply_mondrian(fitted, pred_target=pred,
                                target_embedding=emb if system.taxonomy_mode == 'difficulty' else None)
            widths[lab] = (pr.upper[:, j] - pr.lower[:, j]) * sd[j]
            bins[lab] = np.asarray(pr.bin_indices[:, j], dtype=np.int64)
        ref = {r['label']: r['median_width_phys'] for r in s['test']['summary'] if r['final_policy'] == PRIMARY_POLICY}
        self.check(set(widths) == set(LABELS) and all(np.isclose(np.median(widths[k]), ref[k], rtol=SUMMARY_RTOL)
                                                      for k in LABELS),
                   're-applied Mondrian widths reproduce the M13_7 test median widths')
        self._m13_7_test = dict(widths=widths, bins=bins, idx=idx, inputs=[spath, bundle, pred_path])
        return self._m13_7_test

    def m11_real(self):
        """(path, rows) of the M11.14 real-event outputs for M11, primary policy (point + Mondrian interval)."""
        path = self.m11_dir / 'real_events' / 'multi_event' / 'lvk_comparison.csv'
        if not path.exists():
            raise Skip(f'missing {path}')
        t = pd.read_csv(path)
        need = {'event', 'label', 'model', 'policy', 'prediction', 'lower', 'upper', 'lvk_central'}
        if need - set(t.columns):
            raise Skip(f'{path.name} lacks {sorted(need - set(t.columns))}')
        t = t[(t.model == 'M11') & (t.policy == PRIMARY_POLICY)]
        if t.empty:
            raise Skip(f'no M11 {PRIMARY_POLICY} rows in {path.name}')
        return path, t

    def pe(self, event):
        """(samples [Mc, M, chi_eff] detector frame, path) as M13_7 cells 27/29, or None if no PE file."""
        if event in self._pe:
            return self._pe[event]
        base = re.sub(r'-v\d+$', '', str(event))
        hits = sorted(self.pe_dir.glob(f'*{base}_PEDataRelease_{PE_VARIANT}.h5')) if self.pe_dir.is_dir() else []
        require(len(hits) <= 1, f'Ambiguous PE files for {event}: {hits}')
        if not hits:
            self._pe[event] = None
            return None
        path = hits[0]
        if self._pe_manifest is None:
            self._pe_manifest = {f['key']: f for m in sorted(self.pe_dir.glob('manifest_*.json'))
                                 for f in json.loads(m.read_text())['files']}
        entry = self._pe_manifest.get(path.name)
        self.check(entry is not None and entry['size'] == path.stat().st_size,
                   f'{path.name}: in the download manifest with the same size')
        with h5py.File(path, 'r') as h:
            require(PE_GROUP in h and 'posterior_samples' in h[PE_GROUP], f'{path.name}: no {PE_GROUP}/posterior_samples')
            ds = h[PE_GROUP]['posterior_samples']
            names = ds.dtype.names
            mass_fields = ['total_mass'] if 'total_mass' in names else ['mass_1', 'mass_2']
            need = ['chirp_mass', *mass_fields, 'chi_eff', 'chirp_mass_source', 'redshift']
            require(not [n for n in need if n not in names], f'{path.name}: missing PE fields')
            c = {n: np.asarray(ds[n], dtype=float) for n in need}
        total = c['total_mass'] if 'total_mass' in c else c['mass_1'] + c['mass_2']
        samples = np.column_stack([c['chirp_mass'], total, c['chi_eff']])
        require(np.isfinite(samples).all(), f'{path.name}: non-finite samples')
        # No `_source` suffix = detector frame in pesummary (same check as M13_7 cell 27).
        dev = np.max(np.abs(c['chirp_mass'] / ((1 + c['redshift']) * c['chirp_mass_source']) - 1))
        self.check(dev < 1e-6, f'{path.name}: chirp_mass is detector frame')
        self._pe[event] = (samples, path)
        return self._pe[event]


def save(fig, out_dir, name):
    files = []
    for ext, kw in (('png', dict(dpi=ST['dpi'])), ('pdf', {})):
        p = out_dir / f'{name}.{ext}'
        fig.savefig(p, bbox_inches='tight', **kw)
        files.append(p.name)
    plt.close(fig)
    return files


def joint_figure(rows, bins=80, gridsize=75):
    """Hexbin prediction vs truth with true/predicted marginal line histograms (M08 evaluation, cells 56-58).

    rows: [(header, prediction colour, legend name, y (n, 3), f (n, 3))], physical units. One explicit GridSpec:
    per variable [main | right marginal | gap] with the top marginal above the main panel; one colorbar per row
    (shared log norm) in the last column; y labels only on the main panels.
    """
    nr = len(rows)
    hr = []
    for i in range(nr):
        hr += [0.45 if i == 0 else 1.35, 1.1, 5.0]  # header gap, top marginal, main
    wr = []
    for j in range(len(LABELS)):
        wr += [5.0, 1.1, 1.35 if j < len(LABELS) - 1 else 0.3]
    wr.append(0.2)
    fig = plt.figure(figsize=(21, 6.3 * nr + 0.4))
    gs = fig.add_gridspec(len(hr), len(wr), height_ratios=hr, width_ratios=wr, hspace=0.06, wspace=0.06)
    for i, (header, pc, name, y, f) in enumerate(rows):
        r0, hbs = 3 * i + 1, []
        for j, lab in enumerate(LABELS):
            t, p = y[:, j], f[:, j]
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            pad = 0.04 * (hi - lo if hi > lo else 1.0)
            lims = (lo - pad, hi + pad)
            ax = fig.add_subplot(gs[r0 + 1, 3 * j])
            top = fig.add_subplot(gs[r0, 3 * j], sharex=ax)
            right = fig.add_subplot(gs[r0 + 1, 3 * j + 1], sharey=ax)
            hbs.append(ax.hexbin(t, p, gridsize=gridsize, mincnt=1, norm=LogNorm(), cmap='viridis', linewidths=0.0))
            ax.plot(lims, lims, '--', lw=1.5, color=INK, label='Ideal prediction')
            ax.set(xlim=lims, ylim=lims, xlabel='True ' + var_label(lab), ylabel='Predicted ' + var_label(lab))
            ax.grid(True, axis='both')
            edges = np.linspace(lims[0], lims[1], bins + 1)
            c = 0.5 * (edges[1:] + edges[:-1])
            dt, dp = (np.histogram(v, bins=edges, density=True)[0] for v in (t, p))
            top.plot(c, dt, color=GREY, lw=2.0, label='True')
            top.plot(c, dp, '--', color=pc, lw=2.0, label=f'Predicted: {name}')
            right.plot(dt, c, color=GREY, lw=2.0)
            right.plot(dp, c, '--', color=pc, lw=2.0)
            top.set_ylim(bottom=0.0)
            right.set_xlim(left=0.0)
            for a in (top, right):  # density scale carries no information here: no ticks, no labels
                a.grid(False)
                a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        norm = LogNorm(vmin=1.0, vmax=max(float(np.max(h.get_array())) for h in hbs))
        for h in hbs:
            h.set_norm(norm)
        fig.colorbar(hbs[0], cax=fig.add_subplot(gs[r0 + 1, -1])).set_label('Count (log scale)')
        head = fig.add_subplot(gs[r0 - 1, :-1])
        head.axis('off')
        head.text(0.5, 0.0, header, ha='center', va='bottom', fontsize=ST['axes_size'], fontweight='bold',
                  transform=head.transAxes)
    return fig


def half_violins(ax, samples, x, width=0.7):
    """Left half violins of the LVK samples, one colour per event, with the 5 % / 95 % quantiles drawn as
    segments over the half-violin width only (M13_7 cell 31, redesigned)."""
    kw = dict(positions=x, widths=width, showextrema=False)
    try:
        parts = ax.violinplot(samples, side='low', **kw)
    except TypeError:  # matplotlib < 3.9 has no `side`: clip each full violin to x <= its centre
        parts = ax.violinplot(samples, **kw)
        for b, x0 in zip(parts['bodies'], x):
            for path in b.get_paths():
                path.vertices[:, 0] = np.minimum(path.vertices[:, 0], x0)
    for k, b in enumerate(parts['bodies']):
        c = EVENTS[k % len(EVENTS)]
        b.set_facecolor(c)
        b.set_edgecolor(c)
        b.set_alpha(0.6)
    for x0, s in zip(x, samples):
        ax.hlines(np.quantile(s, PE_Q), x0 - width / 2, x0, color=INK, lw=1.0)
    return [Patch(facecolor=GREY, edgecolor=GREY, alpha=0.6, label='LVK'),
            Line2D([], [], color=INK, lw=1.0, label='5 % and 95 % quantiles')]


def event_axis(ax, events, lab):
    ax.set_xticks(np.arange(len(events)))
    ax.set_xticklabels(events, rotation=90, fontsize=ST['tick_size'] - 3)
    ax.set_title(var_label(lab, unit=False))
    u = LAB['variables'][lab]['unit']
    if u:
        ax.set_ylabel(f'{u} (detector frame)')


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
        ax.set(title=var_label(lab), xticks=x, xlabel=r'$f_{220}$ [Hz]', ylabel='RMSE')
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
            name, c, ls = sty(s)
            ax.plot(ep, v[:, j], ls, color=c, label=name)
        ax.axvline(epoch_mark, color=REF, ls=':', lw=1.2, label='Dropout switched off')
        refline(ax, 1.0)
        ax.set(title=var_label(lab), xlabel='Epoch')
    axes[0].set_ylabel('Slope of truth on prediction')
    fig.tight_layout()
    legend_below(fig)
    paths = [ctx.m13_dir / 'M13_3a_evaluation_summary.json'] + [ctx.hist_path(k) for k in h]
    return [dict(figure='f2_slope_vs_epoch', files=save(fig, out, 'f2_slope_vs_epoch'), inputs=paths,
                 models=['R1', 'R2', 'R3'])]


def plot_bias_deciles(rows, sets, out, name, scaler=None, short=False):
    """Bias (prediction - truth) vs decile mean of the truth (M13_3a cell 13 / M13_6 cell 15 definition).
    With `scaler` = (y_mean, y_std) both axes are shown in standardised units (train-only scaler)."""
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for j, (ax, lab) in enumerate(zip(axes, LABELS)):
        mu, sd = (0.0, 1.0) if scaler is None else (scaler[0][j], scaler[1][j])
        for s, key in sets:
            t = sorted(sel(rows, set=key, label=lab), key=lambda r: r['decile'])
            require(len(t) == 10, f'{name}: expected 10 deciles for {key} {lab}')
            nm, c, ls = sty(s, short)
            ax.plot((col(t, 'center') - mu) / sd, col(t, 'bias') / sd, 'o' + ls, color=c, label=nm)
        refline(ax, 0.0)
        if scaler is None:
            ax.set(title=var_label(lab), xlabel='True value (decile mean)')
        else:
            ax.set(title=var_label(lab, unit=False), xlabel='True value, standardised (decile mean)')
    axes[0].set_ylabel('Bias (prediction − truth)' + ('' if scaler is None else ', standardised'))
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
    """2x3 band panels of M13_3b_4a cell 9 / M13_6 cell 13.

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
                        col(t, 'ci95_lo_pct'), col(t, 'ci95_hi_pct'), sty(s)[1], label=nm)
                ax.set_xticks(x)
                ax.set_xticklabels([pretty_cls(r['cls']) for r in t])
            refline(ax, 0.0)
            ax.set(xlabel=r'$f_{220}$ [Hz]' if grouping == 'f220' else var_label('total_mass'))
            if i == 0:
                ax.set_title(var_label(lab))
        axes[i, 0].set_ylabel(ylabel)
    fig.tight_layout()
    legend_below(fig)
    return save(fig, out, name)


def fig_f4(ctx, out):
    """M13_3b_4a cell 9: Delta RMSE % (mean 3b vs mean R2/R3) by f220 class and M_total, paired row-bootstrap CI."""
    s = ctx.s3b4a()
    # Non-overlapping f220 classes used for the band reading (same three classes as F1).
    files = plot_band(s['band'], [(None, '3b', f"{sty('3b')[0]} vs {sty('G1')[0]}")], out, 'f4_g3_vs_g1_bands',
                      'ΔRMSE vs reference [%]', f220_classes=['<256', '256-512', '>512'])
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
            name, c, ls = sty(SET_OF[m])
            ax.plot(ep[ok], h[m]['train40k_mse'][ok, j] / h[m]['val_mse'][ok, j], '.' + ls, color=c, ms=4,
                    lw=1.2, label=name)
        refline(ax, 1.0)
        ax.set(title=var_label(lab), xlabel='Epoch')
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
            ax.set(title=var_label(lab), xticks=[])
        axes[0].set_ylabel('MSE (standardised units)')
        fig.tight_layout()
        legend_below(fig)
        name = f'f6_renoise_bars_{key}'
        entries.append(dict(figure=name, files=save(fig, out, name), inputs=[ctx.m13_5_path], models=[key]))
    if not entries:
        raise Skip('no M13.5 checkpoint results')
    return entries


def fig_f7(ctx, out):
    """RMSE vs SNR bin mean (M13_3b_4a cell 13 for M11 / R2-R3, M13_6 cell 15 for the G3 chain)."""
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
            nm, c, ls = sty(s)
            ax.plot(col(t, 'snr_center'), col(t, lab), 'o' + ls, color=c, label=nm)
        ax.set(title=var_label(lab), xlabel='Network SNR (bin mean)')
    axes[0].set_ylabel('RMSE')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='f7_rmse_vs_snr', files=save(fig, out, 'f7_rmse_vs_snr'),
                 inputs=[ctx.m13_dir / 'M13_3b_4a_evaluation_summary.json', ctx.s6_path],
                 models=['M11'] + sa['settings']['g1_pair'] + s6['settings']['b3'] + s6['settings']['a6']
                 + ctx.s6_arch_models())]


def fig_f8(ctx, out):
    """As F3 for the G3 chain, from M13_6 cell 15 (tails_deciles)."""
    s6 = ctx.s6()
    sets = [(s, s) for s in ctx.s6_sets()]
    files = plot_bias_deciles(s6['tails_deciles'], sets, out, 'f8_bias_vs_truth_chain')
    return [dict(figure='f8_bias_vs_truth_chain', files=files, inputs=[ctx.s6_path],
                 models=s6['settings']['b3'] + s6['settings']['a6'] + ctx.s6_arch_models())]


def plot_tau_deltas(ax, s6, arch, lab, name):
    x = np.arange(1, 11)
    for k, a in enumerate(arch):
        t = sorted(sel(s6['tau_deciles'], model=a, label=lab), key=lambda r: r['tau_decile'])
        require(len(t) == 10, f'{name}: expected 10 tau30 deciles for {a} {lab}')
        errbars(ax, x + 0.15 * (k - (len(arch) - 1) / 2), col(t, 'delta_rmse_pct'), col(t, 'ci95_lo_pct'),
                col(t, 'ci95_hi_pct'), sty(a)[1], label=sty(a, short=True)[0])
    refline(ax, 0.0)
    ax.set(title=var_label(lab, unit=False), xlabel=r'$\tau_{30}$ decile (1 = shortest)', xticks=x)


def fig_f9(ctx, out):
    """M13_6 cell 9: Delta RMSE % of each augmented architecture vs mean 6a by tau30 decile, paired row CI."""
    s6 = ctx.s6()
    arch = [a for a in ARCH_SETS if a in s6['settings']['arch_aug']]
    if not arch:
        raise Skip('no augmented architecture in M13.6 summary')
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        plot_tau_deltas(ax, s6, arch, lab, 'F9')
    axes[0].set_ylabel('ΔRMSE vs reference [%]')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='f9_context_tau_deciles', files=save(fig, out, 'f9_context_tau_deciles'),
                 inputs=[ctx.s6_path], models=s6['settings']['a6'] + ctx.s6_arch_models(arch))]


def fig_f10(ctx, out):
    """M13_6 cell 13: Delta RMSE % of each augmented architecture vs mean 6a by f220 class and M_total, row CI."""
    s6 = ctx.s6()
    arch = [a for a in ARCH_SETS if a in s6['settings']['arch_aug']]
    if not arch:
        raise Skip('no augmented architecture in M13.6 summary')
    cmps = [(f'{a}_vs_{REF_SET}', a, sty(a, short=True)[0]) for a in arch]
    files = plot_band(s6['band'], cmps, out, 'f10_context_bands', 'ΔRMSE vs reference [%]')
    return [dict(figure='f10_context_bands', files=files, inputs=[ctx.s6_path],
                 models=s6['settings']['a6'] + ctx.s6_arch_models(arch))]


def fig_f11(ctx, out):
    """M13_6 cell 19 (early_vs_final_spearman): Spearman of model order at epoch k vs final order."""
    s6 = ctx.s6()
    names = {'sin_aug': f'Without {NAUG}', 'con_aug': f'With {NAUG}', 'G3': f'All {WB} models'}
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
        rows.append(dict(model=sty(SET_OF[m])[0], run_tag=ctx.tag[m], n_epochs=len(tt),
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


# ---- Real events: M11 baseline and M13_7 -----------------------------------------------------------------------
def v1_panel(ax, curves, xkey, xlabel):
    """Validation loss curves; second seed dashed; vertical line at dropout off; inset on epochs V1_ZOOM."""
    ins = ax.inset_axes([0.47, 0.42, 0.5, 0.5])
    for m, h, x in curves:
        _, c, _ = sty(SET_OF[m])
        ls = '--' if not m.endswith('_s123') else '-'
        ax.plot(x, h['val_loss'], ls, color=c, lw=1.4, label=run_label(m))
        z = (h['epoch'] >= V1_ZOOM[0]) & (h['epoch'] <= V1_ZOOM[1])
        ins.plot(x[z], h['val_loss'][z], ls, color=c, lw=1.2)
    for a in (ax, ins):
        xs = [x[np.flatnonzero(h['epoch'] == DROPOUT_OFF_EPOCH)] for _, h, x in curves]
        if xkey == 'epoch':
            a.axvline(DROPOUT_OFF_EPOCH, color=GREY, ls=':', lw=1.2, label='Dropout switched off' if a is ax else None)
        else:  # per-run time of the dropout-off epoch differs; mark each run with a tick on the curve
            for (m, h, x), xe in zip(curves, xs):
                if xe.size:
                    a.plot(xe, h['val_loss'][h['epoch'] == DROPOUT_OFF_EPOCH], '|', color=GREY, ms=12, mew=1.5,
                           label='Dropout switched off' if a is ax else None)
    ax.set_yscale('log')
    ax.set(xlabel=xlabel, ylabel='Validation loss (standardised MSE)')
    ax.grid(True, axis='both', which='major')
    ins.set_title(f'epochs {V1_ZOOM[0]}–{V1_ZOOM[1]}', fontsize=ST['tick_size'] - 2)
    ins.tick_params(labelsize=ST['tick_size'] - 3)
    ins.grid(True, axis='both')


def fig_v1(ctx, out):
    """val_loss (standardised MSE, history.npz) vs epoch and vs cumulative optimisation time ('train_time', the field
    of T1's median_train_time_s), augmented runs; M13_6 section 7 histories."""
    runs = [m for m in V1_RUNS if m in ctx.available]
    if not runs:
        raise Skip('no augmented run with a history')
    h = {m: ctx.hist(m) for m in runs}
    off = {m: int(h[m]['epoch'][np.flatnonzero(h[m]['dropout_off'].astype(bool))[0]])
           for m in runs if 'dropout_off' in h[m] and h[m]['dropout_off'].astype(bool).any()}
    if any(e != DROPOUT_OFF_EPOCH for e in off.values()):
        warnings.warn(f'v1: dropout-off epoch in history differs from {DROPOUT_OFF_EPOCH}: {off}')
    timed = [m for m in runs if 'train_time' in h[m]]
    if timed and len(timed) < len(runs):
        warnings.warn(f'v1: no per-epoch train_time for {sorted(set(runs) - set(timed))}; omitted from the time panel')
    if not timed:
        warnings.warn('v1: no history records per-epoch train_time; time panel omitted')
    fig, axes = plt.subplots(1, 2 if timed else 1, figsize=(FIG13[0], 5.0) if timed else FIG1, squeeze=False)
    v1_panel(axes[0, 0], [(m, h[m], h[m]['epoch'].astype(float)) for m in runs], 'epoch', 'Epoch')
    if timed:
        v1_panel(axes[0, 1], [(m, h[m], np.cumsum(h[m]['train_time'].astype(float)) / 3600.0) for m in timed],
                 'time', 'Cumulative training time [h]')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='v1_val_loss', files=save(fig, out, 'v1_val_loss'), inputs=[ctx.hist_path(m) for m in runs],
                 models=runs, note=None if timed else 'time panel omitted: no per-epoch train_time')]


def fig_r1(ctx, out):
    """M11.14: frozen LVK reference (best ± first-order propagated error, cell 8) and M11 conservative point +
    Mondrian interval (M11.14 outputs) per event. No violins."""
    path, t = ctx.m11_real()
    ref_path = ctx.data_root / 'lvk_references' / LVK_REF_NAME
    if not ref_path.exists():
        raise Skip(f'missing {ref_path}')
    ref = pd.read_csv(ref_path)
    require(not ref.event.duplicated().any(), 'r1: duplicate LVK event reference')
    ref = ref.set_index('event')
    events = list(t.event.drop_duplicates())
    require(all(e in ref.index for e in events), 'r1: M11 event without LVK reference')
    fig, axes = plt.subplots(1, 3, figsize=(max(FIG13[0], 1.05 * len(events)), 5.6))
    for j, (ax, lab) in enumerate(zip(axes, LABELS)):
        tab = t[t.label == lab].set_index('event').reindex(events)
        require(tab.prediction.notna().all(), f'r1: missing M11 rows for {lab}')
        best, lo, hi = (ref.loc[events, f'{lab}_lvk{s}'].to_numpy(float) for s in ('', '_lower', '_upper'))
        ctx.check(np.allclose(best, tab.lvk_central.to_numpy(float), rtol=SUMMARY_RTOL),
                  f'r1: M11.14 lvk_central equals the frozen LVK reference ({lab})')
        x = np.arange(len(events))
        nm, c, _ = sty('LVK_prop')
        ax.errorbar(x - 0.12, best, yerr=[best - lo, hi - best], fmt='_', ms=12, mew=2, color=c, ecolor=c,
                    capsize=3, label=nm)
        yerr = np.clip([tab.prediction - tab.lower, tab.upper - tab.prediction], 0, None)
        nm, c, _ = sty('M11')
        ax.errorbar(x + 0.12, tab.prediction, yerr=yerr, fmt='s', ms=5, color=c, ecolor=c, capsize=3, label=nm)
        event_axis(ax, events, lab)
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='r1_m11_vs_lvk', files=save(fig, out, 'r1_m11_vs_lvk'), inputs=[path, ref_path],
                 models=['M11'], note=f'{len(events)} M11 real events; M11 interval = Mondrian {PRIMARY_POLICY}')]


def fig_r2(ctx, out):
    """M11 val: hexbin prediction vs truth with marginals (M08 evaluation notebook cells 34, 56-58), physical."""
    path = Path(ctx.s3b4a()['inputs']['M11_val']['path'])
    v = ctx.val('M11', path)
    nm, c, _ = sty('M11')
    fig = joint_figure([(f'{nm}: validation', c, nm, v['y'], v['f'])])
    legend_below(fig)
    return [dict(figure='r2_m11_pred_vs_true', files=save(fig, out, 'r2_m11_pred_vs_true'), inputs=[path],
                 models=['M11'])]


# ---- New data and architectures (G3 chain, val) ----------------------------------------------------------------
def fig_d0_psd(ctx, out):
    """ASD of H1/L1/V1 for the first train row's PSD block: the raw 128 s off-source Welch (8 s) PSD of the
    generator (generate_m11_final_G1.estimate_raw_offsource_psd, reused by the G3 generator for whitening).
    Plotted on its native 1/8 Hz grid; the generator only interpolates it onto the processing grid."""
    try:
        from scripts.generate_m11_final_G1 import (DETECTORS, PSD_SEGMENT_DURATION_S, estimate_raw_offsource_psd,
                                                   read_psd_intervals, resolve_cached_psd_files)
    except ImportError as e:
        raise Skip(f'generator PSD functions not importable: {e}')
    h5, split = ctx.g3_paths()
    with np.load(split) as a:
        row_id = int(np.min(a['train_idx']))
    with h5py.File(h5, 'r') as h:
        ctx.check(h.attrs.get('experiment_id') == ctx.g3_dataset_id, 'd0: HDF5 is the G3 dataset')
        bid = h['block_id'][row_id]
        bid = bid.decode() if isinstance(bid, bytes) else str(bid)
        manifest = pd.DataFrame([dict(block_id=bid, psd_start=float(h['psd_start'][row_id]),
                                      psd_end=float(h['psd_end'][row_id]))])
    selected, _ = resolve_cached_psd_files(ctx.data_root, manifest)
    row = next(manifest.itertuples())
    strains = read_psd_intervals(row, selected)
    rate = 4096
    fig, ax = plt.subplots(figsize=FIG1)
    ax.axvspan(512, 1024, color=GRID, alpha=0.8, lw=0, label='512–1024 Hz')
    for k, ifo in enumerate(DETECTORS):
        import scripts.generate_m11_final_G1 as _gen          # el generador importa interpolate de forma perezosa
        from pycbc.psd import interpolate as _pycbc_interpolate
        _gen.interpolate = _pycbc_interpolate                  # en initialize_physics(); aquí no se llama
        psd = estimate_raw_offsource_psd(strains[ifo], psd_start=row.psd_start, psd_end=row.psd_end,
                                         delta_f=1.0 / PSD_SEGMENT_DURATION_S,
                                         target_flength=int(round(rate / 2 * PSD_SEGMENT_DURATION_S)) + 1)
        f, p = psd.sample_frequencies.numpy(), psd.numpy()
        m = (f >= 20) & (f <= 1100)
        ax.loglog(f[m], np.sqrt(p[m]), color=SEQ3[k], lw=0.9, label=ifo)
    ax.set_xlim(20, 1100)
    ax.set_xlabel('Frequency [Hz]', fontsize=ST['axes_size'])
    ax.set_ylabel(r'ASD [strain / $\sqrt{\rm Hz}$]', fontsize=ST['axes_size'])
    ax.tick_params(axis='both', which='both', labelsize=ST['tick_size'])
    ax.grid(True, axis='both', which='major')
    ax.legend(loc='upper right', fontsize=ST['legend_size'])
    fig.tight_layout()
    files = sorted({p for p, _ in selected.values()})
    return [dict(figure='d0_psd_lines', files=save(fig, out, 'd0_psd_lines'), inputs=[split, h5] + files, models=[],
                 note=f'train HDF5 row {row_id}, PSD block {bid}')]


def fig_d0_whitening(ctx, out):
    """M13.0c (plot_median_psds): median Welch PSD of the processed X, 0.5 s vs 16 s whitening filter."""
    try:
        from scipy.signal import welch
        from scripts.m13.m13_0c_whitening_sweep import (DETECTORS, PILOT_ROWS, SAMPLE_RATE, WELCH_NPERSEG,
                                                        analyse_pilot, dataset_id)
    except ImportError as e:
        raise Skip(f'M13.0c functions not importable: {e}')
    proc = ctx.data_root / 'processed'
    p05 = proc / dataset_id(0.5) / f'pilot_{PILOT_ROWS}.h5'
    if not p05.exists():
        raise Skip(f'missing {p05}')
    f, med = analyse_pilot(p05)[:2]
    curves = {0.5: (med, f'{PILOT_ROWS} M13.0c pilot rows')}
    inputs = [p05]
    p16 = proc / dataset_id(16.0) / f'pilot_{PILOT_ROWS}.h5'
    if p16.exists():
        f16, med16 = analyse_pilot(p16)[:2]
        curves[16.0] = (med16, f'{PILOT_ROWS} M13.0c pilot rows')
        inputs.append(p16)
    else:
        # No 16 s pilot: first val rows of the G3 dataset (16 s filter), same Welch settings as analyse_pilot.
        h5, split = ctx.g3_paths()
        with np.load(split) as a:
            rows = np.sort(a['val_idx'])[:PILOT_ROWS]
        with h5py.File(h5, 'r') as h:
            ctx.check(h.attrs.get('experiment_id') == ctx.g3_dataset_id, 'd0: HDF5 is the G3 dataset')
            ctx.check(float(h.attrs['whitening_max_filter_duration_s']) == 16.0, 'd0: G3 whitening filter is 16 s')
            ctx.check((h['status'][rows] == 1).all(), 'd0: G3 val rows generated')
            X = h['X'][rows].astype(np.float64)
        f16, p = welch(X, fs=SAMPLE_RATE, nperseg=WELCH_NPERSEG, axis=-1)
        curves[16.0] = (np.median(p, axis=0), f'first {PILOT_ROWS} val rows')
        inputs += [split, h5]
    ctx.check(np.array_equal(f, f16), 'd0: same Welch frequency grid for both filters')
    fig, axes = plt.subplots(1, 3, figsize=FIG13, sharey=True)
    for k, (ax, ifo) in enumerate(zip(axes, DETECTORS)):
        ax.axvspan(512, 1024, color=GRID, alpha=0.8, lw=0)
        for i, (dur, (m, src)) in enumerate(curves.items()):
            ax.semilogy(f, m[k], lw=0.9, color=SEQ3[2 * i], label=f'{dur:g} s whitening filter')
        ax.set_xlim(20, 1100)
        ax.set(title=ifo, xlabel='Frequency [Hz]')
    axes[0].set_ylabel('Median PSD of processed X')
    fig.tight_layout()
    legend_below(fig, ncol=2)
    note = '; '.join(f'{d:g} s: {s}' for d, (_, s) in curves.items())
    if not p16.exists():
        note += ' (pilot rows are the first manifest rows, not the same sources as val rows)'
    return [dict(figure='d0_whitening_filters', files=save(fig, out, 'd0_whitening_filters'), inputs=inputs,
                 models=[], note=note)]


def fig_d1(ctx, out):
    """Val hexbin prediction vs truth with marginals (as r2), reference CNN vs DenseNet + transformer, same seed."""
    v = {m: ctx.val(m) for m in D1_RUNS}
    ctx.check(np.array_equal(v[D1_RUNS[0]]['idx'], v[D1_RUNS[1]]['idx']), 'd1: identical val row ids')
    fig = joint_figure([(run_label(m), sty(SET_OF[m])[1], sty(SET_OF[m], short=True)[0], v[m]['y'], v[m]['f'])
                        for m in D1_RUNS])
    legend_below(fig)
    return [dict(figure='d1_pred_vs_true', files=save(fig, out, 'd1_pred_vs_true'),
                 inputs=[v[m]['path'] for m in D1_RUNS], models=list(D1_RUNS))]


def fig_d2(ctx, out):
    """RMSE in standardised units per set (mean of seeds, error bar = seed range), M13_6 cell 5 metrics."""
    s6 = ctx.s6()
    _, sd = ctx.scaler()
    seeds = ctx.s6_seeds(['3b', REF_SET, *ARCH_SETS])
    rm = {(r['model'], r['label']): r['rmse'] for r in s6['metrics']}
    R = {s: np.array([[rm[m, lab] / sd[j] for j, lab in enumerate(LABELS)] for m in ms]) for s, ms in seeds.items()}
    ref = R[REF_SET].mean(0)
    delta = {s: 100.0 * (r.mean(0) / ref - 1.0) for s, r in R.items()}  # delta_pct of M13_6 cell 5
    for a in ARCH_SETS:
        t = sel(s6['arch_comparisons'], comparison=f'{a}_vs_{REF_SET}')
        if a in R and t:
            ctx.check(np.allclose([delta[a][LABELS.index(r['label'])] for r in t], col(t, 'delta_rmse_pct'),
                                  rtol=SUMMARY_RTOL, atol=1e-9), f'd2: {a} ΔRMSE equals M13_6 arch_comparisons')
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for j, (ax, lab) in enumerate(zip(axes, LABELS)):
        for k, (s, r) in enumerate(R.items()):
            nm, c, _ = sty(s, short=True)
            m = r[:, j].mean()
            err = [[m - r[:, j].min()], [r[:, j].max() - m]] if len(r) == 2 else None
            ax.bar(k, m, width=0.7, color=c, label=nm, yerr=err, capsize=3, ecolor=INK)
            txt = 'ref.' if s == REF_SET else f'{delta[s][j]:+.1f} %'
            ax.text(k, r[:, j].max() * 1.01, txt, ha='center', va='bottom', fontsize=ST['legend_size'] - 1)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        ax.set(title=var_label(lab, unit=False), xticks=[])
    axes[0].set_ylabel('RMSE (standardised units)')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='d2_rmse_std', files=save(fig, out, 'd2_rmse_std'),
                 inputs=[ctx.s6_path, ctx.val_path(f'{REF_SET}_aug_s123')],
                 models=[m for ms in seeds.values() for m in ms],
                 note='error bar = range of the two seeds (none for single-seed sets); not a confidence interval')]


def fig_d3(ctx, out):
    """As F8 in standardised units, reference and augmented architectures only."""
    s6 = ctx.s6()
    sets = [(s, s) for s in ctx.s6_sets() if s != '3b']
    files = plot_bias_deciles(s6['tails_deciles'], sets, out, 'd3_bias_deciles_std', scaler=ctx.scaler(),
                              short=True)
    return [dict(figure='d3_bias_deciles_std', files=files, inputs=[ctx.s6_path, ctx.val_path(f'{REF_SET}_aug_s123')],
                 models=s6['settings']['a6'] + ctx.s6_arch_models())]


def fig_d4(ctx, out):
    """Histogram of standardised val residuals (prediction - truth), same seed, identical rows."""
    v = {m: ctx.val(m) for m in D1_RUNS}
    ctx.check(np.array_equal(v[D1_RUNS[0]]['idx'], v[D1_RUNS[1]]['idx']), 'd4: identical val row ids')
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for j, (ax, lab) in enumerate(zip(axes, LABELS)):
        r = {m: (v[m]['f'][:, j] - v[m]['y'][:, j]) / v[m]['sd'][j] for m in D1_RUNS}
        edges = np.linspace(min(x.min() for x in r.values()), max(x.max() for x in r.values()), 121)
        for m in D1_RUNS:
            _, c, ls = sty(SET_OF[m])
            ax.hist(r[m], bins=edges, histtype='step', lw=1.6, color=c, ls=ls, label=run_label(m))
        ax.axvline(0.0, color=REF, lw=0.9, zorder=1)
        ax.set_yscale('log')
        ax.set(title=var_label(lab, unit=False), xlabel='Residual (prediction − truth), standardised')
    axes[0].set_ylabel('Count')
    fig.tight_layout()
    legend_below(fig, ncol=2)
    return [dict(figure='d4_residuals', files=save(fig, out, 'd4_residuals'),
                 inputs=[v[m]['path'] for m in D1_RUNS], models=list(D1_RUNS))]


def fig_d5(ctx, out):
    """RMSE (standardised) vs SNR bin mean, M13_6 cell 15 (snr table), reference and augmented architectures."""
    s6 = ctx.s6()
    _, sd = ctx.scaler()
    sets = [s for s in ctx.s6_sets() if s != '3b']
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for j, (ax, lab) in enumerate(zip(axes, LABELS)):
        for s in sets:
            t = sel(s6['snr'], set=s)
            require(len(t) == len(s6['settings']['snr_edges']) - 1, f'd5: SNR bins for {s}')
            nm, c, ls = sty(s, short=True)
            ax.plot(col(t, 'snr_center'), col(t, lab) / sd[j], 'o' + ls, color=c, label=nm)
        ax.set(title=var_label(lab, unit=False), xlabel='Network SNR (bin mean)')
    axes[0].set_ylabel('RMSE (standardised units)')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='d5_rmse_vs_snr', files=save(fig, out, 'd5_rmse_vs_snr'),
                 inputs=[ctx.s6_path, ctx.val_path(f'{REF_SET}_aug_s123')],
                 models=s6['settings']['a6'] + ctx.s6_arch_models())]


def fig_a1(ctx, out):
    """F9 plus a band of the 4a_aug seed noise per tau30 decile (M13_6 cell 11 definition, all three labels)."""
    from scripts.m13.m13_1_diagnostics import tau_newtonian
    s6 = ctx.s6()
    arch = [a for a in ARCH_SETS if a in s6['settings']['arch_aug']]
    if not arch:
        raise Skip('no augmented architecture in M13.6 summary')
    noise, inputs = None, [ctx.s6_path]
    if all(m in ctx.available for m in SEED_BAND_PAIR):
        v = {m: ctx.val(m) for m in SEED_BAND_PAIR}
        a, b = SEED_BAND_PAIR
        ctx.check(np.array_equal(v[a]['idx'], v[b]['idx']), 'a1: identical val row ids for the seed pair')
        Y = v[a]['y']
        tau30 = tau_newtonian(Y[:, 0])
        edges = np.asarray(s6['tau_edges_s'], float)
        ctx.check(np.allclose(np.quantile(tau30, np.linspace(0, 1, 11)), edges, rtol=RECOMPUTE_RTOL),
                  'a1: tau30 decile edges equal M13_6 summary')
        tdec = np.clip(np.searchsorted(edges[1:-1], tau30, side='right'), 0, 9)
        noise = np.empty((10, 3))
        for d in range(10):
            g = tdec == d
            r = [np.sqrt(np.mean((v[m]['f'][g] - Y[g])**2, axis=0)) for m in SEED_BAND_PAIR]
            noise[d] = 100.0 * np.abs(r[0] - r[1]) / np.mean(r, axis=0)
        ref = sel(s6.get('seed_noise_by_decile', {}).get('noise', []), grouping='tau30')
        if ref:
            ctx.check(np.allclose(noise[:, LABELS.index('chi_eff')],
                                  [next(r for r in ref if r['cls'] == f'D{d + 1}')['noise_4a_aug_pct'] for d in range(10)],
                                  rtol=RECOMPUTE_RTOL), 'a1: chi_eff seed noise equals M13_6 section 3b')
        inputs += [v[m]['path'] for m in SEED_BAND_PAIR]
    else:
        warnings.warn(f'a1: {SEED_BAND_PAIR} not both available; seed-noise band omitted')
    x = np.arange(1, 11)
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for j, (ax, lab) in enumerate(zip(axes, LABELS)):
        if noise is not None:
            ax.bar(x, 2 * noise[:, j], bottom=-noise[:, j], width=0.8, color=sty('4a_aug')[1], alpha=0.15, lw=0,
                   label=f"± seed-to-seed spread, {sty('4a_aug', short=True)[0]}")
        plot_tau_deltas(ax, s6, arch, lab, 'a1')
    axes[0].set_ylabel('ΔRMSE vs reference [%]')
    fig.tight_layout()
    legend_below(fig, ncol=2)
    return [dict(figure='a1_tau_deciles_all', files=save(fig, out, 'a1_tau_deciles_all'), inputs=inputs,
                 models=s6['settings']['a6'] + ctx.s6_arch_models(arch) + (list(SEED_BAND_PAIR) if noise is not None
                                                                            else []),
                 note='CI = paired row bootstrap of val (row sampling, not seed variation); band = '
                      '100|RMSE_s123 - RMSE_s124| / mean, two seeds only')]


def fig_a2(ctx, out):
    """M13_6 cell 11: chi_eff RMSE by tau30 decile for every run of section 3b."""
    s6 = ctx.s6()
    sn = s6.get('seed_noise_by_decile')
    if not sn:
        raise Skip(f'{ctx.s6_path.name} has no seed_noise_by_decile (section 3b)')
    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for m in sn['models']:
        t = [next(r for r in sn['per_model'] if r['grouping'] == 'tau30' and r['cls'] == f'D{d}' and r['model'] == m)
             for d in x]
        _, c, _ = sty(SET_OF[m])
        second = not m.endswith('_s123')
        ax.plot(x, col(t, 'rmse'), 's--' if second else 'o-', color=c, ms=4, lw=1.3, label=run_label(m))
    ax.set(xlabel=r'$\tau_{30}$ decile (1 = shortest)', ylabel='RMSE ' + var_label('chi_eff'), xticks=x)
    fig.tight_layout()
    legend_below(fig, ncol=2)
    return [dict(figure='a2_chi_rmse_tau', files=save(fig, out, 'a2_chi_rmse_tau'), inputs=[ctx.s6_path],
                 models=list(sn['models']))]


# ---- M13_7: Mondrian on test and real events -------------------------------------------------------------------
def fig_m1(ctx, out):
    """M13_7 cells 8/12 (test, conservative): median width per Mondrian bin and coverage per bin."""
    _, spath, s = ctx.m13_7()
    path = ctx.m13_7_file('test_coverage_per_bin.csv')
    t = pd.read_csv(path)
    nominal, tol = float(s['test']['criterion']['nominal']), float(s['test']['criterion']['abs_coverage_gap_max'])
    _, c, _ = sty('M13_7')
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        b = t[(t.final_policy == PRIMARY_POLICY) & (t.label == lab) & (t.n > 0)].sort_values('mondrian_bin')
        require(len(b) > 0, f'm1: no bins for {lab}')
        x, n, cov = b.mondrian_bin.to_numpy(), b.n.to_numpy(float), b.coverage.to_numpy(float)
        ax.bar(x, b.median_width_phys, width=0.7, color=c, alpha=0.55, label='Median interval width')
        ax.set(title=var_label(lab, unit=False), xlabel='Mondrian bin', ylabel='Median ' + width_label(lab).lower(),
               xticks=x)
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.spines['right'].set_visible(True)
        h95 = 1.96 * np.sqrt(nominal * (1 - nominal) / n)  # binomial 95 % band of the coverage at this bin's n
        ax2.bar(x, 2 * h95, bottom=nominal - h95, width=0.7, color=REF, alpha=0.3, lw=0,
                label='Binomial 95 % band (bin n)')
        ax2.axhspan(nominal - tol, nominal + tol, color=GRID, alpha=0.6, lw=0, label=f'±{100 * tol:g} points')
        ax2.axhline(nominal, color=INK, lw=1.0, ls='--', label=f'Nominal {nominal:.2f}')
        ax2.plot(x, cov, 'o', color=INK, label='Coverage per bin')
        lo = min(cov.min(), (nominal - h95).min(), nominal - tol)
        hi = max(cov.max(), (nominal + h95).max(), nominal + tol)
        ax2.set_ylim(lo - 0.01, hi + 0.01)
        ax2.set_ylabel('Coverage (test)')
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='m1_mondrian_width_coverage', files=save(fig, out, 'm1_mondrian_width_coverage'),
                 inputs=[path, spath], models=[M13_7_RUN])]


def m13_7_bins(ctx, lab):
    _, spath, s = ctx.m13_7()
    path = ctx.m13_7_file('test_coverage_per_bin.csv')
    t = pd.read_csv(path)
    b = t[(t.final_policy == PRIMARY_POLICY) & (t.label == lab) & (t.n > 0)].sort_values('mondrian_bin')
    require(len(b) > 0, f'M13_7: no Mondrian bins for {lab}')
    return b, [path, spath], s


def fig_m1a(ctx, out):
    """M13_7 cell 8 (test, conservative): coverage per Mondrian bin, nominal line, ±tolerance band of the fixed
    criterion and binomial 95 % band at each bin's n (sampling spread expected at nominal coverage)."""
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        b, inputs, s = m13_7_bins(ctx, lab)
        nominal = float(s['test']['criterion']['nominal'])
        tol = float(s['test']['criterion']['abs_coverage_gap_max'])
        x, n, cov = b.mondrian_bin.to_numpy(), b.n.to_numpy(float), b.coverage.to_numpy(float)
        h95 = 1.96 * np.sqrt(nominal * (1 - nominal) / n)
        ax.bar(x, 2 * h95, bottom=nominal - h95, width=0.7, color=GREY, alpha=0.35, lw=0,
               label='Binomial 95 % band (bin n)')
        ax.axhspan(nominal - tol, nominal + tol, color=GREY_LIGHT, alpha=0.6, lw=0, label=f'±{100 * tol:g} points')
        ax.axhline(nominal, color=INK, lw=1.0, ls='--', label=f'Nominal {nominal:.2f}')
        ax.plot(x, cov, 'o', color=sty('M13_7')[1], label='Coverage per bin')
        lo = min(cov.min(), (nominal - h95).min(), nominal - tol)
        hi = max(cov.max(), (nominal + h95).max(), nominal + tol)
        ax.set_ylim(lo - 0.01, hi + 0.01)
        ax.set(title=var_label(lab, unit=False), xlabel='Mondrian bin', xticks=x)
    axes[0].set_ylabel('Coverage (test)')
    fig.tight_layout()
    legend_below(fig, ncol=4)
    return [dict(figure='m1a_mondrian_coverage', files=save(fig, out, 'm1a_mondrian_coverage'), inputs=inputs,
                 models=[M13_7_RUN])]


def fig_m1b(ctx, out):
    """Median test interval width per Mondrian bin with IQR whiskers (conservative). Per-row widths from the
    re-applied bundle; per-bin n and median checked against test_coverage_per_bin.csv of M13_7."""
    r = ctx.m13_7_test()
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        b, inputs, _ = m13_7_bins(ctx, lab)
        x = b.mondrian_bin.to_numpy()
        w, k = r['widths'][lab], r['bins'][lab]
        q = np.array([np.quantile(w[k == i], [0.25, 0.5, 0.75]) for i in x])
        ctx.check(np.array_equal([int((k == i).sum()) for i in x], b.n.to_numpy(int))
                  and np.allclose(q[:, 1], b.median_width_phys.to_numpy(float), rtol=SUMMARY_RTOL),
                  f'm1b: per-bin n and median width equal M13_7 test_coverage_per_bin ({lab})')
        ax.bar(x, q[:, 1], width=0.7, color=sty('M13_7')[1], alpha=0.75, yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
               capsize=3, ecolor=INK, label='Median width (whiskers: interquartile range)')
        ax.set(title=var_label(lab, unit=False), xlabel='Mondrian bin', xticks=x,
               ylabel=width_label(lab, 'Median interval width'))
    fig.tight_layout()
    legend_below(fig)
    return [dict(figure='m1b_mondrian_width', files=save(fig, out, 'm1b_mondrian_width'),
                 inputs=inputs + r['inputs'][1:], models=[M13_7_RUN])]


def fig_m4(ctx, out):
    """M13_7 cell 9 (descriptive, test, conservative): mean interval width of M11 (its frozen test metrics) vs
    M13_7, with the % change. Different predictors and quantiles: descriptive only."""
    _, spath, s = ctx.m13_7()
    rows = s['test'].get('width_vs_M11_descriptive')
    if not rows:
        raise Skip('M13_7 summary has no width_vs_M11_descriptive (M11 test metrics were not found)')
    rows = {r['label']: r for r in rows if r['final_policy'] == PRIMARY_POLICY}
    require(set(rows) == set(LABELS), 'm4: width comparison does not cover the three labels')
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        r = rows[lab]
        ctx.check(np.isclose(r['mean_width_ratio_vs_M11'], r['mean_width_phys'] / r['mean_width_phys_M11'],
                             rtol=SUMMARY_RTOL), f'm4: width ratio consistent ({lab})')
        for k, (key, v) in enumerate((('M11', r['mean_width_phys_M11']), ('M13_7', r['mean_width_phys']))):
            nm, c, _ = sty(key, short=True)
            ax.bar(k, v, width=0.6, color=c, label=nm)
        ax.text(1, r['mean_width_phys'], f"{100 * (r['mean_width_ratio_vs_M11'] - 1):+.1f} %", ha='center',
                va='bottom', fontsize=ST['legend_size'])
        ax.set_ylim(0, 1.12 * max(r['mean_width_phys'], r['mean_width_phys_M11']))
        ax.set(title=var_label(lab, unit=False), xticks=[], ylabel=width_label(lab, 'Mean interval width'))
    fig.tight_layout()
    legend_below(fig, ncol=2)
    return [dict(figure='m4_width_vs_m11', files=save(fig, out, 'm4_width_vs_m11'), inputs=[spath],
                 models=['M11', M13_7_RUN], note='descriptive: different predictors and Mondrian quantiles')]


def fig_m2(ctx, out):
    """M13_7 cell 14: fixed test examples (seed fixed before results) with prediction, interval and truth."""
    _, spath, s = ctx.m13_7()
    path = ctx.m13_7_file('test_fixed_examples.csv')
    t = pd.read_csv(path)
    nominal = float(s['test']['criterion']['nominal'])
    nm, c, _ = sty('M13_7', short=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 13))
    for ax, lab in zip(axes, LABELS):
        e = t[t.label == lab].reset_index(drop=True)
        x = np.arange(len(e))
        ax.errorbar(x, e.prediction, yerr=[e.prediction - e.lower, e.upper - e.prediction], fmt='o', color=c,
                    capsize=3, label=f'{nm}: prediction and {int(round(100 * nominal))} % interval')
        ax.plot(x, e.true, 'x', color=INK, ms=8, label='True value')
        ax.set_xticks(x)
        ax.set_xticklabels(e.hdf5_row, rotation=90, fontsize=ST['tick_size'] - 4)
        ax.set_ylabel(var_label(lab))
    axes[-1].set_xlabel('Test HDF5 row id')
    fig.tight_layout()
    legend_below(fig, ncol=2)
    return [dict(figure='m2_examples', files=save(fig, out, 'm2_examples'), inputs=[path, spath], models=[M13_7_RUN],
                 note=f"seed {s['test']['fixed_examples']['seed']}, n = {s['test']['fixed_examples']['n']}")]


def fig_m3(ctx, out):
    """M13_7 cell 13 (first figure): test interval width vs network SNR, conservative. Widths come from re-applying
    the saved cal-only bundle to the saved test predictions exactly as cell 8 (Ctx.m13_7_test)."""
    r = ctx.m13_7_test()
    widths, idx = r['widths'], r['idx']
    h5, _ = ctx.g3_paths()
    with h5py.File(h5, 'r') as h:
        ctx.check(h.attrs.get('experiment_id') == ctx.g3_dataset_id, 'm3: HDF5 is the G3 dataset')
        snr = np.asarray(h['final_network_snr'][:], dtype=float)[idx]
    require(np.isfinite(snr).all(), 'm3: non-finite SNR')
    q_edges = np.unique(np.quantile(snr, np.linspace(0, 1, 13)))  # 12 quantile bins, as cell 13
    which = np.clip(np.digitize(snr, q_edges[1:-1]), 0, len(q_edges) - 2)
    centers = np.array([np.median(snr[which == k]) for k in range(len(q_edges) - 1)])
    fig, axes = plt.subplots(1, 3, figsize=FIG13)
    for ax, lab in zip(axes, LABELS):
        w = widths[lab]
        ax.hexbin(snr, w, gridsize=60, bins='log', mincnt=1, cmap='viridis')
        q = np.array([np.quantile(w[which == k], [0.25, 0.5, 0.75]) for k in range(len(q_edges) - 1)])
        ax.plot(centers, q[:, 1], 'o-', color=INK, label='Median (SNR quantile bins)')
        ax.fill_between(centers, q[:, 0], q[:, 2], color=INK, alpha=0.2, label='Interquartile range')
        ax.set(title=var_label(lab, unit=False), xlabel='Network SNR', ylabel=width_label(lab))
    fig.tight_layout()
    legend_below(fig, ncol=2)
    return [dict(figure='m3_width_vs_snr', files=save(fig, out, 'm3_width_vs_snr'),
                 inputs=r['inputs'] + [h5], models=[M13_7_RUN])]


def fig_l1(ctx, out):
    """M13_7 cell 31, redesigned: per label, left half violins of the LVK PE samples (one colour per event) with
    GWOSC best and 5 % / 95 % segments over the half-violin width; to the right of the centre, M11 (M11.14 point +
    Mondrian conservative interval) at +0.15 and M13_7 (point + interval) at +0.32."""
    _, spath, s = ctx.m13_7()
    pe_path, ev_path = ctx.m13_7_file('real_lvk_pe_comparison.csv'), ctx.m13_7_file('real_per_event_comparison.csv')
    meta = s['real_events'].get('lvk_posterior_samples') or {}
    ctx.check(meta.get('variant') == PE_VARIANT and meta.get('group') == PE_GROUP
              and float(meta.get('level', np.nan)) == PE_LEVEL, 'l1: LVK PE settings equal those of M13_7')
    pe_table, per_event = pd.read_csv(pe_path), pd.read_csv(ev_path)
    main_pe = pe_table[pe_table.policy == PRIMARY_POLICY]
    try:
        m11_path, m11 = ctx.m11_real()
    except Skip as e:
        warnings.warn(f'l1: M11 omitted ({e})')
        m11_path, m11 = None, None
    entries, pe_files = [], set()
    for j, lab in enumerate(LABELS):
        table = main_pe[main_pe.label == lab].reset_index(drop=True)
        if table.empty:
            continue
        pe = [ctx.pe(e) for e in table.event]
        require(all(p is not None for p in pe), f'l1: PE file of an M13_7 event not found in {ctx.pe_dir}')
        samples = [p[0][:, j] for p in pe]
        ctx.check(np.allclose([np.quantile(x, PE_Q[0]) for x in samples], table.lvk_pe_lower90, rtol=SUMMARY_RTOL)
                  and np.allclose([np.quantile(x, PE_Q[1]) for x in samples], table.lvk_pe_upper90, rtol=SUMMARY_RTOL),
                  f'l1: PE samples reproduce the M13_7 90 % bounds ({lab})')
        pe_files |= {p[1] for p in pe}
        x, width = np.arange(len(table)), 0.7
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(table) + 3), 6))
        handles = half_violins(ax, samples, x, width)
        nm, c, _ = sty('LVK_best', short=True)
        ax.hlines(table.lvk_best, x - width / 2, x, color=c, lw=2.2, label=nm)
        has_m11 = False
        if m11 is not None:
            t11 = m11[m11.label == lab].set_index('event').reindex(table.event)
            if t11.prediction.notna().all():
                if 'prediction_M11' in per_event:
                    p11 = per_event[per_event.label == lab].set_index('event').prediction_M11.reindex(table.event)
                    ctx.check(np.allclose(p11.to_numpy(float), t11.prediction.to_numpy(float), rtol=SUMMARY_RTOL),
                              f'l1: M11 points of M13_7 equal M11.14 outputs ({lab})')
                yerr = np.clip([t11.prediction - t11.lower, t11.upper - t11.prediction], 0, None)
                nm, c, _ = sty('M11', short=True)
                ax.errorbar(x + 0.15, t11.prediction, yerr=yerr, fmt='s', ms=5, color=c, ecolor=c, capsize=2,
                            label=nm)
                has_m11 = True
            else:
                warnings.warn(f'l1: M11.14 rows missing for some M13_7 events ({lab}); M11 omitted')
        yerr = np.clip([table.prediction - table.lower, table.upper - table.prediction], 0, None)
        nm, c, _ = sty('M13_7', short=True)
        ax.errorbar(x + 0.32, table.prediction, yerr=yerr, fmt='o', ms=5, color=c, ecolor=c, capsize=2, label=nm)
        event_axis(ax, list(table.event), lab)
        fig.tight_layout()
        legend_below(fig, ncol=3, extra=handles)
        name = f'l1_violin_{lab}'
        entries.append(dict(figure=name, files=save(fig, out, name),
                            inputs=[spath, pe_path, ev_path] + ([m11_path] if has_m11 else []) + sorted(pe_files),
                            models=[M13_7_RUN] + (['M11'] if has_m11 else [])))
    if not entries:
        raise Skip(f'no {PRIMARY_POLICY} rows in {pe_path.name}')
    return entries


def fig_l2(ctx, out):
    """M13_7 cell 25 (absolute_discrepancy): |prediction - GWOSC best| per event, M11 and M13_7."""
    _, spath, _ = ctx.m13_7()
    path = ctx.m13_7_file('real_per_event_comparison.csv')
    per_event = pd.read_csv(path)
    models = [m for m in ('M11', 'M13_7') if f'absolute_discrepancy_{m}' in per_event]
    if not models:
        raise Skip(f'{path.name} has no absolute_discrepancy columns')
    shifts = {'M11': -0.16, 'M13_7': 0.16} if len(models) == 2 else {models[0]: 0.0}
    n_ev = per_event.event.nunique()
    fig, axes = plt.subplots(1, 3, figsize=(max(FIG13[0], 1.05 * n_ev), 5.6))
    for ax, lab in zip(axes, LABELS):
        table = per_event[per_event.label == lab].reset_index(drop=True)
        x = np.arange(len(table))
        for m in models:
            nm, c, _ = sty(m, short=True)
            ax.bar(x + shifts[m], table[f'absolute_discrepancy_{m}'], width=0.3, color=c, label=nm)
        event_axis(ax, list(table.event), lab)
        ax.set_ylabel('|prediction − GWOSC best|' + (f" [{LAB['variables'][lab]['unit']}]"
                                                    if LAB['variables'][lab]['unit'] else ''))
    fig.tight_layout()
    legend_below(fig, ncol=2)
    return [dict(figure='l2_discrepancy', files=save(fig, out, 'l2_discrepancy'), inputs=[spath, path],
                 models=[M13_7_RUN] + (['M11'] if 'M11' in models else []))]


FIGURES = [('F1', fig_f1), ('F2', fig_f2), ('F3', fig_f3), ('F4', fig_f4), ('F5', fig_f5), ('F6', fig_f6),
           ('F7', fig_f7), ('F8', fig_f8), ('F9', fig_f9), ('F10', fig_f10), ('F11', fig_f11), ('T1', table_t1),
           ('v1', fig_v1), ('r1', fig_r1), ('r2', fig_r2), ('d0_psd', fig_d0_psd), ('d0_whitening', fig_d0_whitening),
           ('d1', fig_d1), ('d2', fig_d2), ('d3', fig_d3), ('d4', fig_d4), ('d5', fig_d5), ('a1', fig_a1),
           ('a2', fig_a2), ('m1', fig_m1), ('m1a', fig_m1a), ('m1b', fig_m1b), ('m2', fig_m2), ('m3', fig_m3),
           ('m4', fig_m4), ('l1', fig_l1), ('l2', fig_l2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', type=Path, default=None,
                    help='output directory (default <DATA_ROOT>/results/M13_presentation/figures)')
    ap.add_argument('--overwrite', action='store_true', help='replace an existing figures directory content')
    ap.add_argument('--set', dest='fig_set', choices=('m13', 'cpan_results', 'cpan_physics'), default='m13',
                    help="'m13' (default): the figures of this file. 'cpan_results': F06-F12 + numbers.json of "
                         'scripts/m13/presentation/cpan_results_figures.py, written to a new run_<UTC> directory under '
                         '--out (default <DATA_ROOT>/results/presentation/cpan_2026); --overwrite is ignored. '
                         "'cpan_physics': F02-F05 + appendix A1-A5 of scripts/m13/presentation/cpan_physics_figures.py "
                         '(default --out <DATA_ROOT>/results/presentation/cpan_2026/physics); --only takes those IDs '
                         'and --resume-run a run_<UTC> of that set')
    ap.add_argument('--only', default=None,
                    help="cpan_results only: comma-separated IDs to produce: groups F06-F12, output IDs (F09a, "
                         "F07_chi_eff, F12_main; their whole group is regenerated) and/or 'numbers', e.g. F11,F12")
    ap.add_argument('--resume-run', type=Path, default=None,
                    help='cpan_results only: existing run_<UTC> directory; the --only items are regenerated there and '
                         'manifest.json / numbers.json / contact_sheet.png are merged instead of creating a new run')
    args = ap.parse_args()
    require(args.fig_set in ('cpan_results', 'cpan_physics') or (args.only is None and args.resume_run is None),
            '--only and --resume-run require --set cpan_results or cpan_physics')

    cfg = json.loads((ROOT / 'configs/experiments/train_m11_final_G1_500k.json').read_text())
    data_root = resolve_data_root(config_data_root=cfg.get('data_root'))
    if args.fig_set == 'cpan_results':
        from scripts.m13.presentation.cpan_results_figures import run as run_cpan
        run_cpan(data_root, args.out, only=args.only.split(',') if args.only else None, resume_run=args.resume_run)
        return
    if args.fig_set == 'cpan_physics':
        from scripts.m13.presentation.cpan_physics_figures import run as run_phys
        run_phys(data_root, args.out, only=args.only.split(',') if args.only else None, resume_run=args.resume_run)
        return
    out = args.out if args.out is not None else data_root / 'results' / 'M13_presentation' / 'figures'
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
            warnings.warn(f'{fid} omitted: {e}')
            skipped.append(dict(figure=fid, reason=str(e)))

    manifest = dict(
        created=datetime.now().isoformat(timespec='seconds'), script=str(Path(__file__).resolve()),
        script_sha256=digest(Path(__file__).resolve()), data_root=str(data_root),
        labels_config=dict(path=str(ROOT / 'configs/presentation/labels.json'),
                           sha256=digest(ROOT / 'configs/presentation/labels.json')),
        reference_model=REF_SET, reference_note=LAB.get('reference_note'),
        m13_6_summary=str(ctx.s6_path) if ctx.s6_path else None,
        m13_7_run=str(ctx._m13_7[0]) if ctx._m13_7 else None,
        models_available=ctx.available, models_skipped=ctx.skipped, checks=ctx.checks,
        figures=[dict(figure=d['figure'], files=d['files'], models=d['models'],
                      inputs=[ctx.input_entry(p) for p in d['inputs']],
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
