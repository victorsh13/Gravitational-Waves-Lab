"""CPAN 2026 result figures F06-F12, numbers.json, manifest.json and contact_sheet.png (read-only; descriptive).
Slide versions (one variable per figure, legend inside, test except F08s on val): F06m, F07b, F08s, F09c, F10m, F11v2,
F12v2. F07b also re-applies the M14.7 cal-only Mondrian bundle (inherited geometry, score r) to the test npz; F09c
reads numbers.json of the run it is written to.
v2 / v3 versions (same values and checks as the version they replace; compact legend inside the axes at the location
overlapping the fewest drawn artists; style checked before saving): F10mv2, F06mv2, F07bv2, F08sv2 (manifest lists
the 68 configurations per label), F11v3, F12v3, F09av2 (appendix, in the F09 group); F07q (backup) = per-SNR-decile
90 % quantiles on cal. _chirp_mass = main slide, _total_mass / _chi_eff = appendix (manifest field 'role').
v3 / v4 versions: F10mv3 (finer hexbin, soft grid), F06mv3 (linear axes with the same range, both distributions in each
marginal), F11v4 (F11v2 event colours, solid median and 5 / 95 % lines), F11h (test: F11v4 chirp mass with swapped
axes), F12v4 (F12v2 LVK colours, zero-overlap legend search), F13 (optional: F09c subset as normalised errors).
F06mv4 (F06mv3 without the tercile text, legend upper left, marginals on the shared range, chirp mass 0-16) and F13v2
(F13 with larger legend, coverages in the labels, blue ellipse-only points).
F06cv2 (F06c with thicker outlined lines), F16 / F16b (test RMSE and RMS(sigma) in 10 equal-size groups of network SNR /
true value, bootstrap band), table_test (manifest numbers for the talk table, no figure), F17 (training history of the
final model from its saved history.npz, run_info.json and log; nothing re-run).
F17v2 (training curves in the style of the M13_6_evaluation notebook: val and train-monitor MSE, val RMSE per
parameter, log y), F18a / F18b (back-up: F12v4 in the (chirp mass, chi_eff) and (total mass, chi_eff) planes, radius
of each plane on cal), loss_definition (manifest only: the loss as implemented, lines cited; code read, not run).
F17v3 (history keys described from the training code; val_loss and M14.5 val RMSE checks before drawing; val MSE
per parameter), F18c / F18d (group F18grid: the nine events, F12 style, in the two chi_eff planes).
F06c (cone, test): predicted sigma vs signed error with y = +-sigma and y = +-q sigma (cal q), fractions inside in
the legend and the manifest.

Called from scripts/m13/make_presentation_figures.py --set cpan_results. Nothing is trained, predicted or re-fitted
with a network; every value comes from saved artifacts of the final model (M14.5b) and its interval selection (M14.7):
  - M14.5b summary 20261113T150740Z and the cal / val / test prediction npz it records (mu, Sigma, y, idx);
  - M14.7 run 20261113T152305Z: summary (68-config grid on val, test confirmation) and selection JSON;
  - M14.5b real-events JSON (run 20261113T154408Z; written next to the run_<stamp> folder by M14_5_real_events)
    and the events prediction npz it records;
  - GWTC-3 PE samples (DATA_ROOT/external/gwtc3_pe), files listed in that JSON;
  - target_network_snr (and chirp_mass, only to check row alignment) of the G3 HDF5 recorded by both summaries;
  - the 5b checkpoint (CPU, weights only) to count parameters; no forward pass.
Re-derived quantities (single-bin conformal quantiles on cal, 3D / 2D ellipsoid radii on cal, coverages, tau-decile
correlations, real-event intervals) are checked against the values the notebooks stored. A figure with a missing
input is reported as MISSING and a figure whose check fails as FAILED_CHECK; neither is drawn.
Names, colours and sizes: configs/presentation/labels.json deep-merged with configs/presentation/cpan_2026.json.
Output: DATA_ROOT/results/presentation/cpan_2026/run_<UTC>/.
"""
import json
import math
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text  # noqa: E402
import matplotlib.transforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.m12.m12_5_3_posthoc_check import LABELS, digest, npz_to_physical, require  # noqa: E402

# ---- Artifacts of the final model (fixed stamps; nothing is searched) ------------------------------------------
TAG = 'M14_5b_4a_G3T_tbank_fullcov_s123'
M14_5_STAMP = '20261113T150740Z'
M14_7_STAMP = '20261113T152305Z'
REAL_STAMP = '20261113T154408Z'
LEVEL = 0.90
COV_ATOL = 2e-4    # coverage re-derived here vs stored: the M14.7 bundle adds a 1e-10 jitter to scores (ties)
Q_RTOL = 1e-6      # conformal quantile / radius / width re-derived from the same float32 npz
VOL_RTOL = 1e-4    # geometric-mean volume (float32 Sigma, summation order)
TAU_F_LOW, T_SUN = 30.0, 4.925490947e-6  # Newtonian tau30 of the predicted chirp mass (checked vs stored edges)
SEED_BOOT, N_BOOT = 20261113, 1000        # F09 bootstrap of the empirical correlation (rows within a decile)
SEED_PE_SUBSAMPLE, N_PE_SCATTER = 20261114, 3000
PAIRS = (('chirp_mass', 'total_mass'), ('total_mass', 'chi_eff'))

# ---- Presentation style ----------------------------------------------------------------------------------------
BASE_CFG = ROOT / 'configs/presentation/labels.json'
CPAN_CFG = ROOT / 'configs/presentation/cpan_2026.json'


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def resolve_color_aliases(cols):
    """A colour value '@key' is an alias of colors.key (e.g. method_norm = '@network'); resolved once at load."""
    for k in list(cols):
        v, seen = cols[k], set()
        while isinstance(v, str) and v.startswith('@'):
            require(v[1:] in cols and v not in seen, f'cpan_2026.json colors.{k}: bad alias {cols[k]}')
            seen.add(v)
            v = cols[v[1:]]
        cols[k] = v
    return cols


CFG = deep_merge(json.loads(BASE_CFG.read_text()), json.loads(CPAN_CFG.read_text()))
resolve_color_aliases(CFG['colors'])
ST, METH, COL, CMAP = CFG['style'], CFG['methods'], CFG['colors'], CFG['cmap']
ALPHA, FIGOPT = CFG.get('alpha', {}), CFG['figure_options']
EVC = COL['events']
FULL, TALL, WIDE = tuple(ST['fig_full']), tuple(ST['fig_tall']), tuple(ST['fig_wide'])
SINGLE = ST['single']
SINGLE_SIZE = tuple(SINGLE['size'])
SNR_NAMES = ('Low', 'Mid', 'High')
LOG_LABELS = ('chirp_mass', 'total_mass')  # F06: log axes for the masses, linear for chi_eff
# Slide versions (F06m, F07b, F08s, F09c, F10m, F11v2, F12v2): one variable per figure, legend inside the axes.
SLIDE = ST['slide']
SLIDE_SIZE, SLIDE_FULL = tuple(SLIDE['size']), tuple(ST['slide_full'])
# Internal experiment / dataset / architecture names that must not appear in slide text (M11, M14.5, G3, G3T, 4a).
INTERNAL_NAMES = re.compile(r'\b(M1\d|G\dT?|4[abc])\b')
EXPECTED_JOINT_3D = dict(box=0.814, ellipsoid=0.901)  # F09c text: global test joint coverage, checked vs numbers.json
MONDRIAN_WIDTH_RTOL = 1e-5  # F07b: Mondrian test mean width re-derived from the bundle vs M14.7 (float32 offsets)
# v2 / v3 figures: compact legend inside the axes, placed where it overlaps the fewest drawn artists.
LEG = ST['legend_compact']
MIN_AXES_PT = float(ST['min_axes_pt'])
MARGINAL_FRAC = float(FIGOPT['marginal_frac'])
LEGEND_LOCS = ('upper right', 'upper left', 'lower right', 'lower left', 'upper center', 'lower center',
               'center right', 'center left')


def set_style():
    plt.rcParams.update({
        'font.family': ST['font'], 'mathtext.fontset': 'dejavusans', 'font.size': ST['tick_size'],
        'axes.labelsize': ST['axes_size'], 'axes.titlesize': ST['axes_size'],
        'xtick.labelsize': ST['tick_size'], 'ytick.labelsize': ST['tick_size'],
        'legend.fontsize': ST['legend_size'], 'legend.frameon': False,
        'figure.facecolor': COL['background'], 'axes.facecolor': COL['background'],
        'savefig.facecolor': COL['background'], 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': COL['grid'], 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'lines.linewidth': 1.8, 'errorbar.capsize': 3,
    })


def single_rc():
    """Font sizes of the one-figure-per-slide versions (>= 16 pt)."""
    return plt.rc_context({'font.size': SINGLE['tick_size'], 'axes.labelsize': SINGLE['axes_size'],
                           'axes.titlesize': SINGLE['axes_size'], 'xtick.labelsize': SINGLE['tick_size'],
                           'ytick.labelsize': SINGLE['tick_size'], 'legend.fontsize': SINGLE['legend_size']})


def slide_rc():
    """Font sizes of the slide versions (axes >= 20 pt, legend >= 18 pt)."""
    return plt.rc_context({'font.size': SLIDE['annot_size'], 'axes.labelsize': SLIDE['axes_size'],
                           'axes.titlesize': SLIDE['axes_size'], 'xtick.labelsize': SLIDE['tick_size'],
                           'ytick.labelsize': SLIDE['tick_size'], 'legend.fontsize': SLIDE['legend_size'],
                           'legend.title_fontsize': SLIDE['legend_size']})


def inside_legend(ax, handles, labels=None, loc='best', **kw):
    """Legend inside the axes on a semi-transparent white background."""
    labels = labels if labels is not None else [h.get_label() for h in handles]
    return ax.legend(handles, labels, loc=loc, frameon=True, framealpha=SLIDE['legend_alpha'],
                     facecolor=COL['background'], edgecolor=COL['grid'], **kw)


def save_slide(fig, out, name, strict=False):
    """save() plus the slide rule: no internal names (M11, G3, G3T, 4a, ...) in any figure text.
    strict (v2 / v3 figures): also check_style() and save with style.slide.pad_inches (bbox tight, no extra margin)."""
    bad = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if INTERNAL_NAMES.search(t.get_text())]
    require(not bad, f'{name}: internal names in figure text {bad}')
    if strict:
        check_style(fig, name)
        return save(fig, out, name, pad=SLIDE['pad_inches'])
    return save(fig, out, name)


def legend_kw():
    """Compact legend: style.legend_compact font size and frame alpha, tight paddings."""
    return dict(frameon=True, fontsize=LEG['fontsize'], title_fontsize=LEG['fontsize'], framealpha=LEG['framealpha'],
                facecolor=COL['background'], edgecolor=COL['grid'], borderpad=0.35, labelspacing=0.3,
                handlelength=1.6, handletextpad=0.5, borderaxespad=0.4, columnspacing=1.0)


def _densify(xy, step=3.0):
    """Points every `step` display pixels along a polyline (so a line crossing a legend counts)."""
    xy = xy[np.isfinite(xy).all(1)]
    if len(xy) < 2:
        return xy
    d = np.diff(xy, axis=0)
    n = np.clip(np.ceil(np.hypot(d[:, 0], d[:, 1]) / step).astype(int), 1, 2000)
    i = np.repeat(np.arange(len(d)), n)
    frac = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n, n)
    return np.vstack([xy[i] + frac[:, None] * d[i], xy[-1:]])


def _obstacles(ax, renderer):
    """Drawn artists of `ax` in display coordinates: points (markers, hexbin cells, densified lines and outlines),
    filled shapes (violins, filled histograms, bars) and text boxes. Background bands (zorder < 0.5) are ignored."""
    from matplotlib.collections import LineCollection, PathCollection
    pts, fills, texts = [], [], []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        pts.append(_densify(xy) if ln.get_linestyle() not in ('None', 'none', '', ' ') else xy)
    for c in ax.collections:
        if not c.get_visible() or c.get_zorder() < 0.5:
            continue
        offs = np.asarray(c.get_offsets(), float)
        if isinstance(c, PathCollection) or len(offs) > 1:   # scatter markers, hexbin cells
            # drawn centre = transform(path centre) + offset_transform(offset): exact for scatter (identity path
            # transform, data offsets) and hexbin (data path transform, AffineDeltaTransform offsets, also log axes)
            paths = c.get_paths()
            ctr = np.asarray(paths[0].vertices, float).mean(0) if paths else np.zeros(2)
            pts.append(c.get_transform().transform(ctr[None]) + c.get_offset_transform().transform(offs))
            continue
        T = c.get_transform()
        fc = np.asarray(c.get_facecolor())
        filled = (not isinstance(c, LineCollection) and getattr(c, 'filled', True) and fc.size > 0
                  and np.any(fc.reshape(-1, 4)[:, 3] > 0))
        for p in c.get_paths():
            if filled:
                fills.append((p, T))
            else:
                pts += [_densify(np.asarray(v)) for v in p.to_polygons(T, closed_only=False)]
    for p in ax.patches:
        if not p.get_visible() or p.get_zorder() < 0.5:
            continue
        T = p.get_transform()
        if p.get_fill():
            fills.append((p.get_path(), T))
        else:
            pts += [_densify(np.asarray(v)) for v in p.get_path().to_polygons(T, closed_only=False)]
    for t in ax.texts:
        if t.get_visible() and t.get_text():
            texts.append(t.get_window_extent(renderer))
    pts = [q[np.isfinite(q).all(1)] for q in pts if len(q)]
    return (np.vstack(pts) if pts else np.zeros((0, 2))), fills, texts


def _overlap(bb, obs):
    """Overlap score of a legend box: points inside + 5 x grid samples inside filled shapes + 1000 x text overlap."""
    from matplotlib.transforms import Bbox
    pts, fills, texts = obs
    n_pts = int(np.sum((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0)
                       & (pts[:, 1] <= bb.y1))) if len(pts) else 0
    gx, gy = np.meshgrid(np.linspace(bb.x0, bb.x1, 12), np.linspace(bb.y0, bb.y1, 6))
    g = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(g), bool)
    for path, T in fills:
        inside |= path.contains_points(g, transform=T)
    txt = 0.0
    for t in texts:
        x = Bbox.intersection(bb, t)
        if x is not None and t.width * t.height > 0:
            txt += x.width * x.height / (t.width * t.height)
    return n_pts + 5 * int(inside.sum()) + 1000.0 * txt


def _grow_top(ax, frac=0.1):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * (hi / lo) ** frac if ax.get_yscale() == 'log' else hi + frac * (hi - lo))


def place_legend(axes, handles, labels=None, prefer=(), grow=0, locs=None, also=(), **kw):
    """Compact legend inside one of `axes`, at the location (of `prefer` first, then LEGEND_LOCS) whose box overlaps
    the fewest drawn artists (_overlap; ties keep the earlier location). With grow > 0 and every location
    overlapping, the y range is extended upwards by 10 % per step (at most `grow` steps) and the search repeated.
    locs: only these locations (e.g. F13 'upper left'); the overlap is still measured and returned.
    also: axes whose drawn artists also count as obstacles (e.g. the twin axes under a twinx legend, F17).
    Returns the choice (stored in the manifest)."""
    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    labels = labels if labels is not None else [h.get_label() for h in handles]
    fig = axes[0].figure
    locs = list(locs) if locs else list(dict.fromkeys(list(prefer) + list(LEGEND_LOCS)))
    kws = dict(legend_kw(), **kw)
    best, step = None, 0
    for step in range(grow + 1):
        fig.draw_without_rendering()
        r = fig.canvas.get_renderer()
        best = None
        for ia, ax in enumerate(axes):
            obs, abb = _obstacles(ax, r), ax.get_window_extent(r)
            for ex in also:
                o2 = _obstacles(ex, r)
                obs = (np.vstack([obs[0], o2[0]]), obs[1] + o2[1], obs[2] + o2[2])
            for loc in locs:
                leg = ax.legend(handles, labels, loc=loc, **kws)
                bb = leg.get_window_extent(r)
                leg.remove()
                fits = bb.x0 >= abb.x0 - 1 and bb.x1 <= abb.x1 + 1 and bb.y0 >= abb.y0 - 1 and bb.y1 <= abb.y1 + 1
                s = _overlap(bb, obs) + (0.0 if fits else 1e6)
                if best is None or s < best[0]:
                    best = (s, ia, loc)
        if best[0] == 0 or step == grow:
            break
        for ax in axes:
            _grow_top(ax)
    s, ia, loc = best
    axes[ia].legend(handles, labels, loc=loc, **kws)
    return dict(loc=loc, axes_index=ia, overlap_score=float(s), headroom_steps=step,
                rule='compact legend inside the axes at the location overlapping the fewest drawn artists')


TEXT_CORNERS = {'lower right': (0.98, 0.03, 'right', 'bottom'), 'upper left': (0.02, 0.97, 'left', 'top'),
                'upper right': (0.98, 0.97, 'right', 'top'), 'lower left': (0.02, 0.03, 'left', 'bottom')}


def place_text(ax, text, prefer=('lower right',), **kw):
    """Annotation text (no box) at the axes corner overlapping the fewest drawn artists (same score as
    place_legend; ties keep the earlier corner). Call before place_legend so the legend avoids it."""
    fig = ax.figure
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    obs = _obstacles(ax, r)
    best = None
    for c in dict.fromkeys(list(prefer) + list(TEXT_CORNERS)):
        x, y, ha, va = TEXT_CORNERS[c]
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **kw)
        s = _overlap(t.get_window_extent(r), obs)
        t.remove()
        if best is None or s < best[0]:
            best = (s, c)
    x, y, ha, va = TEXT_CORNERS[best[1]]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **kw)
    return dict(loc=best[1], overlap_score=float(best[0]))


def soft_grid(*axes):
    """Light 'whitegrid': lines at the major ticks in colors.grid_soft, below the data (axes.axisbelow)."""
    for a in axes:
        a.set_axisbelow(True)
        a.grid(True, which='major', color=COL['grid_soft'], lw=float(ST['grid_soft_lw']))


def check_style(fig, name):
    """v2 / v3 rules: no figure-level (outside) legend; every legend inside its axes with font size
    style.legend_compact.fontsize; axis labels and tick labels >= style.min_axes_pt."""
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    require(not fig.legends, f'{name}: figure-level legend (outside the axes)')
    small, outside = [], []
    for ax in fig.axes:
        txt = [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
        small += [f'{t.get_text()} ({t.get_fontsize():.0f} pt)' for t in txt
                  if t.get_visible() and t.get_text() and t.get_fontsize() < MIN_AXES_PT - 1e-6]
        leg = ax.get_legend()
        if leg is not None:
            bb, abb = leg.get_window_extent(r), ax.get_window_extent(r)
            if not (bb.x0 >= abb.x0 - 1 and bb.x1 <= abb.x1 + 1 and bb.y0 >= abb.y0 - 1 and bb.y1 <= abb.y1 + 1):
                outside.append('legend box outside its axes')
            outside += [f'legend text {t.get_fontsize():.0f} pt' for t in leg.get_texts()
                        if abs(t.get_fontsize() - LEG['fontsize']) > 1e-6]
    require(not small, f'{name}: axis text below {MIN_AXES_PT:.0f} pt: {small}')
    require(not outside, f'{name}: legend rule: {outside}')


def role(lab):
    """_chirp_mass = main slide; _total_mass / _chi_eff = appendix."""
    return 'main' if lab == 'chirp_mass' else 'appendix'


def fs():
    """Annotation font size of the current style context."""
    return plt.rcParams['font.size']


def log_ticks(ax, lo, hi):
    """<= 5 labelled major ticks on both log axes (1-2-5, else 1-3, else decades); minor ticks unlabelled."""
    k0, k1 = math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1
    t = []
    for subs in ((1, 2, 5), (1, 3), (1,), tuple(range(1, 10))):
        t = [m * 10.0 ** k for k in range(k0, k1 + 1) for m in subs if lo <= m * 10.0 ** k <= hi]
        if 2 <= len(t) <= 5:
            break
    if len(t) > 5:
        t = t[::math.ceil(len(t) / 5)]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(t))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_formatter(NullFormatter())


def lin_ticks(ax):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(5))
        axis.set_minor_formatter(NullFormatter())


def var_label(lab, unit=True):
    v = CFG['variables'][lab]
    return f"{v['label']} [{v['unit']}]" if unit and v['unit'] else v['label']


def unit_sfx(lab):
    u = CFG['variables'][lab]['unit']
    return f' [{u}]' if u else ''


def short_event(e, events):
    s = e.split('_')[0]
    return s if sum(x.split('_')[0] == s for x in events) == 1 else e


def save(fig, out, name, pad=None):
    bad = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if '\u00b7' in t.get_text()]
    require(not bad, f'{name}: middle dot in figure text {bad}')
    files = []
    extra = {} if pad is None else dict(pad_inches=pad)
    for ext in ST['formats']:
        p = out / f'{name}.{ext}'
        fig.savefig(p, dpi=ST['dpi'], bbox_inches='tight', **extra)
        files.append(p.name)
    plt.close(fig)
    return files


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# ---- Small numerics --------------------------------------------------------------------------------------------
def kth(scores, level=LEVEL):
    """Split-conformal quantile: k-th smallest score, k = ceil((n + 1) level) (same rank as M14.5 / M14.7)."""
    n = len(scores)
    k = int(math.ceil((n + 1) * level))
    return float(np.partition(scores, k - 1)[k - 1]), k


def quantile_groups(x, n):
    edges = np.quantile(x, np.linspace(0, 1, n + 1))
    return np.searchsorted(edges[1:-1], x, side='right')


def maha(d, dim):
    """s = r^T Sigma^-1 r in standardized units, first `dim` labels (2D = chirp_mass, total_mass)."""
    r = d['r_std'][:, :dim]
    sol = np.linalg.solve(d['cov'][:, :dim, :dim], r[..., None])[..., 0]
    return np.einsum('ni,ni->n', r, sol)


def tau30(mc):
    return (5.0 / 256.0) * (np.pi * TAU_F_LOW) ** (-8.0 / 3.0) * (mc * T_SUN) ** (-5.0 / 3.0)


def hits(d, j, q, key):
    """Inside the single-bin symmetric 90 % interval of label j: score r (physical) or r/sigma (standardized)."""
    r = d['r_std'][:, j]
    if key == 'r':
        return np.abs(r * d['sd'][j]) <= q['r']
    return np.abs(r) <= q['r_sigma'] * d['sig_std'][:, j]


def widths(d, j, q, key):
    return np.full(d['n'], 2.0 * q['r']) if key == 'r' else 2.0 * q['r_sigma'] * d['sigma'][:, j]


def ellipse_xy(mu2, cov2_phys, radius, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    L = np.linalg.cholesky(cov2_phys)
    return mu2[:, None] + math.sqrt(radius) * (L @ np.vstack([np.cos(t), np.sin(t)]))


def hpd_contour(ax, xy, level, color):
    """Contour of the smallest binned region holding `level` of the samples (80 x 80 histogram, 3 x 3 box blur)."""
    lo, hi = np.percentile(xy, [0.1, 99.9], axis=0)
    pad = 0.2 * (hi - lo)
    H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=80, range=[[lo[0] - pad[0], hi[0] + pad[0]],
                                                                    [lo[1] - pad[1], hi[1] + pad[1]]])
    P = np.pad(H, 1)
    H = sum(P[i:i + H.shape[0], j:j + H.shape[1]] for i in range(3) for j in range(3)) / 9.0
    v = np.sort(H.ravel())[::-1]
    thr = v[np.searchsorted(np.cumsum(v) / len(xy), level)]
    ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, levels=[thr], colors=[color],
               linewidths=1.8)


class Missing(Exception):
    """An input artifact is not where the summaries / docs record it."""


class FailedCheck(Exception):
    """A re-derived quantity does not reproduce the value stored by the notebook."""


# ---- Inputs, checks and cached loads ---------------------------------------------------------------------------
class Inputs:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        m13 = self.data_root / 'results' / 'M13_3'
        self.p5 = m13 / 'M14_5' / TAG / f'M14_5_summary_{TAG}_{M14_5_STAMP}.json'
        self.d7 = m13 / 'M14_7' / TAG / f'run_{M14_7_STAMP}'
        self.p7 = self.d7 / f'M14_7_summary_{M14_7_STAMP}.json'
        self.p7sel = self.d7 / f'M14_7_selection_val_{M14_7_STAMP}.json'
        self.dreal = m13 / 'M14_5' / 'real_events'
        self.preal = self.dreal / f'M14_5_real_events_{TAG}_{REAL_STAMP}.json'
        self.pe_dir = self.data_root / 'external' / 'gwtc3_pe'
        self._sha, self._json, self._pred, self._pe, self._cache, self._cache_used = {}, {}, {}, {}, {}, {}
        self._h5 = None
        self.nohash, self.checks, self.info, self.used = {}, [], {}, []

    def begin(self):
        self.used = []

    def cached(self, key):
        """True (and the inputs it was derived from re-registered for the current figure) if `key` is cached."""
        if key in self._cache:
            self.used += self._cache_used[key]
            return True
        return False

    def store(self, key, value, start):
        self._cache[key], self._cache_used[key] = value, list(self.used[start:])
        return value

    def exists(self, p, what):
        p = Path(p)
        self.used.append(p)
        if not p.exists():
            raise Missing(f'{what}: {p}')
        return p

    def check(self, ok, name):
        if not bool(ok):
            raise FailedCheck(name)
        if name not in self.checks:
            self.checks.append(name)

    def sha(self, p):
        k = str(Path(p).resolve())
        if k not in self._sha:
            self._sha[k] = digest(Path(k))
        return self._sha[k]

    def entry(self, p):
        p = Path(p)
        if not p.exists():
            return dict(path=str(p), sha256=None, status='MISSING')
        if p.is_dir():  # safety net: a directory is never hashed (its content is not walked)
            warnings.warn(f'input registered as a directory, not hashed: {p}')
            return dict(path=str(p), kind='directory', sha256=None)
        k = str(p.resolve())
        if k in self.nohash:
            return dict(path=str(p), sha256=None, size=p.stat().st_size, note=self.nohash[k])
        return dict(path=str(p), sha256=self.sha(p))

    def json(self, p, what):
        p = self.exists(p, what)
        k = str(p.resolve())
        if k not in self._json:
            self._json[k] = json.loads(p.read_text())
        return self._json[k]

    def s5(self):
        s = self.json(self.p5, 'M14.5b summary')
        self.check(s['model_tag'] == TAG and s['summary_stamp_utc'] == M14_5_STAMP, 'M14.5 summary: model and stamp')
        return s

    def s7(self):
        s = self.json(self.p7, 'M14.7 summary')
        self.check(s['model'] == TAG and s['run_stamp'] == M14_7_STAMP, 'M14.7 summary: model and stamp')
        return s

    def sel7(self):
        s, s7 = self.json(self.p7sel, 'M14.7 selection'), self.s7()
        self.check(s['selected'] == s7['selected'] and s7['selection_json']['sha256'] == self.sha(self.p7sel),
                   'M14.7 selection JSON: same configuration and sha256 as the M14.7 summary')
        return s

    def real(self):
        # The run directory only has to exist; it is not an input (nothing in it is read by F11 / F12). The files
        # read are registered by json() / exists(): this JSON, the events npz and each GWTC-3 PE file.
        run_dir = self.dreal / f'run_{REAL_STAMP}'
        if not run_dir.is_dir():
            raise Missing(f'M14.5 real-events run directory: {run_dir}')
        s = self.json(self.preal, 'M14.5 real-events JSON')
        self.check(s['model_tag'] == TAG and s['run_stamp_utc'] == REAL_STAMP
                   and Path(s['run_dir']).name == f'run_{REAL_STAMP}', 'real-events JSON: model, stamp, run dir')
        return s

    # -- predictions -------------------------------------------------------------------------------------------
    def pred(self, split):
        s5, s7 = self.s5(), self.s7()
        rec = s5['inputs']['pred'][split] if split != 'test' else s5['amendment_2']['test']['inputs']['model']
        p = self.exists(rec['path'], f'{split} predictions')
        if split in self._pred:
            return self._pred[split]
        self.check(self.sha(p) == rec['sha256'] == s7['inputs']['predictions'][split]['sha256'],
                   f'{split} npz: sha256 equals M14.5 and M14.7 summaries')
        with np.load(p, allow_pickle=False) as z:  # object members (model_config) are never read
            names = [str(x) for x in z['label_names']]
            mean, sd = z['y_mean'].astype(np.float64), z['y_std'].astype(np.float64)
            y_s, f_s = z[f'y_{split}'].astype(np.float64), z[f'pred_{split}'].astype(np.float64)
            cov = z[f'cov_{split}'].astype(np.float64)
            sig_npz = z[f'sigma_{split}'].astype(np.float64)
            idx = z[f'idx_{split}'].astype(np.int64)
            ckpt = str(z['checkpoint_file'])
        self.check(names == LABELS, f'{split} npz: label order {LABELS}')
        y, mu, units = npz_to_physical(y_s, f_s, mean, sd)
        # Sigma is stored in standardized units (M14.5 design): the npz must be standardized for D Sigma D to hold.
        self.check(units == 'standardized', f'{split} npz: y / pred standardized (train-only scaler)')
        sig_std = np.sqrt(np.einsum('nii->ni', cov))
        self.check(np.max(np.abs(sig_std - sig_npz)) < 1e-5, f'{split} npz: sigma = sqrt(diag Sigma)')
        self.check(len(np.unique(idx)) == len(idx), f'{split} npz: unique HDF5 row ids')
        self._pred[split] = dict(y=y, mu=mu, r_std=y_s - f_s, cov=cov, sig_std=sig_std, sigma=sig_std * sd,
                                 cov_phys=cov * np.outer(sd, sd)[None], sd=sd, mean=mean, idx=idx, n=len(idx),
                                 ckpt=ckpt, path=p)
        return self._pred[split]

    def snr(self, split):
        """target_network_snr of the HDF5 rows of `split` (nominal SNR, the source of the M14.3 / M14.7 terciles)."""
        s5, s7 = self.s5(), self.s7()
        self.check(str(s5['inputs']['g3_h5']) == str(s7['inputs']['g3_h5']), 'G3 HDF5: same file in M14.5 and M14.7')
        h5 = self.exists(s5['inputs']['g3_h5'], 'G3 HDF5 (SNR)')
        self.nohash[str(h5.resolve())] = ('sha256 not computed (dataset HDF5); only target_network_snr and '
                                          'chirp_mass are read, rows by the npz idx')
        if self._h5 is None:
            with h5py.File(h5, 'r') as h:
                self._h5 = (np.asarray(h['target_network_snr'][:], float), np.asarray(h['chirp_mass'][:], float))
        d = self.pred(split)
        snr, cm = self._h5
        self.check(np.allclose(cm[d['idx']], d['y'][:, 0], rtol=1e-4, atol=1e-3),
                   f'{split}: HDF5 chirp_mass at idx equals npz truth (row alignment)')
        return snr[d['idx']]

    def snr_cuts(self):
        s5, s7 = self.s5(), self.s7()
        c7 = np.asarray(s7['criteria']['snr_tercile_cuts_val_m14_3'], float)
        self.check(np.array_equal(c7, np.asarray(s5['groups']['snr_cuts_val'], float)),
                   'SNR tercile cuts: same in M14.5 and M14.7')
        if 'snr_cuts_recomputed_val' not in self.info:  # informative only (quantile convention of M14.3 not re-read)
            self.info['snr_cuts_recomputed_val'] = dict(stored=c7, np_quantile=np.quantile(self.snr('val'),
                                                                                           [1 / 3, 2 / 3]))
        return c7

    # -- single-bin intervals and ellipsoid radii, fitted on cal only -----------------------------------------
    def intervals(self):
        if self.cached('q'):
            return self._cache['q']
        start = len(self.used)
        cal, val, s7, sel = self.pred('cal'), self.pred('val'), self.s7(), self.sel7()
        q = {}
        for j, lab in enumerate(LABELS):
            cfg = sel['selected'][lab]
            self.check((cfg['taxonomy_mode'], cfg['interval_mode'], cfg['n_bins'], cfg['score'])
                       == ('none', 'symmetric', 1, 'r/sigma'), f'{lab}: M14.7 selection is single bin, symmetric, r/sigma')
            q_rs, k = kth(np.abs(cal['r_std'][:, j]) / cal['sig_std'][:, j])
            q_r, _ = kth(np.abs(cal['r_std'][:, j]) * cal['sd'][j])
            self.check(k == s7['bins_val_selected'][lab][0]['k_rank'], f'{lab}: conformal rank k equals M14.7')
            q[lab] = dict(r=q_r, r_sigma=q_rs, k=k)
            for key, score in (('r', 'r'), ('r_sigma', 'r/sigma')):
                g = grid_entry(s7['grid_val'], lab, dict(taxonomy_mode='none', interval_mode='symmetric', n_bins=1,
                                                         n_neighbors=None, score=score))
                self.check(g is not None, f'{lab}: single-bin symmetric {score} in M14.7 grid')
                h, w = hits(val, j, q[lab], key), widths(val, j, q[lab], key)
                self.check(abs(h.mean() - g['coverage']) <= COV_ATOL
                           and np.isclose(w.mean(), g['mean_width_phys'], rtol=Q_RTOL),
                           f'{lab}: single-bin {score} val coverage and mean width reproduce M14.7')
        return self.store('q', q, start)

    def radii(self):
        if self.cached('R'):
            return self._cache['R']
        start = len(self.used)
        cal, s5 = self.pred('cal'), self.s5()
        R = {}
        for dim, key in ((3, '3D'), (2, '2D')):
            R[key], k = kth(maha(cal, dim))
            self.check(k == s5['c3']['k_rank'] and np.isclose(R[key], s5['c3']['radius_S'][key], rtol=Q_RTOL),
                       f'S ellipsoid {key}: cal radius reproduces M14.5')
        return self.store('R', R, start)

    # -- real events ------------------------------------------------------------------------------------------
    def events(self):
        if self.cached('events'):
            return self._cache['events']
        start = len(self.used)
        R, q, test = self.real(), self.intervals(), self.pred('test')
        rec = R['inputs']['events_pred']
        p = self.exists(rec['path'], 'events prediction npz')
        self.check(self.sha(p) == rec['sha256'], 'events npz: sha256 equals real-events JSON')
        with np.load(p, allow_pickle=False) as z:
            names = [str(x) for x in z['event_names']]
            pred, cov = z['pred'].astype(np.float64), z['cov'].astype(np.float64)
            mean, sd = z['y_mean'].astype(np.float64), z['y_std'].astype(np.float64)
            self.check([str(x) for x in z['label_names']] == LABELS, 'events npz: label order')
        self.check(np.array_equal(mean, test['mean']) and np.array_equal(sd, test['sd']),
                   'events npz: same train-only scaler as the test npz')
        order = list(R['event_order_by_lvk_chirp'])
        self.check(sorted(order) == sorted(names), 'events: order list equals npz events')
        per = {(r['event'], r['label']): r for r in R['per_event_label']}
        out = dict(order=order, mu={}, sigma={}, cov_phys={}, lo={}, hi={}, samples={},
                   oos=dict(R.get('out_of_support') or {}))
        for j, lab in enumerate(LABELS):
            self.check(np.isclose(q[lab]['r_sigma'], R['main_intervals']['offsets_standardized_r_over_sigma'][lab][1],
                                  rtol=Q_RTOL), f'{lab}: cal quantile equals the real-events offsets')
        for e in order:
            k = names.index(e)
            mu, sig = pred[k] * sd + mean, np.sqrt(np.diag(cov[k])) * sd
            qv = np.array([q[lab]['r_sigma'] for lab in LABELS])
            lo, hi = mu - qv * sig, mu + qv * sig
            for j, lab in enumerate(LABELS):
                r = per[(e, lab)]
                self.check(np.allclose([mu[j], lo[j], hi[j]], [r['mu'], r['main_lower'], r['main_upper']], rtol=Q_RTOL),
                           f'{e} {lab}: mu and 90 % interval reproduce the real-events JSON')
            out['mu'][e], out['sigma'][e], out['lo'][e], out['hi'][e] = mu, sig, lo, hi
            out['cov_phys'][e] = cov[k] * np.outer(sd, sd)
            out['samples'][e] = self.pe(e)
            for j, lab in enumerate(LABELS):
                r, s = per[(e, lab)], out['samples'][e][:, j]
                self.check(np.allclose(np.quantile(s, [0.5, 0.05, 0.95]),
                                       [r['lvk_pe_median'], r['lvk_pe_lower90'], r['lvk_pe_upper90']], rtol=1e-9),
                           f'{e} {lab}: PE samples reproduce the stored LVK median and 5-95 % bounds')
        return self.store('events', out, start)

    def pe(self, e):
        R = self.real()
        lvk = R['lvk_posterior']
        info = lvk['files'][e]
        path = self.exists(info['file'], f'GWTC-3 PE file of {e}')
        if e in self._pe:
            return self._pe[e]
        self.check(path.parent.resolve() == self.pe_dir.resolve(), f'{e}: PE file in DATA_ROOT/external/gwtc3_pe')
        self.check(path.stat().st_size == info['size'], f'{e}: PE file size equals real-events JSON')
        with h5py.File(path, 'r') as h:
            ds = h[lvk['group']]['posterior_samples']
            names = ds.dtype.names
            c = {n: np.asarray(ds[n], float) for n in ('chirp_mass', 'chi_eff', 'chirp_mass_source', 'redshift',
                                                       'total_mass', 'mass_1', 'mass_2') if n in names}
        total = c['total_mass'] if info['total_mass_from'] == 'total_mass' else c['mass_1'] + c['mass_2']
        s = np.column_stack([c['chirp_mass'], total, c['chi_eff']])
        self.check(len(s) == info['n_samples'] and np.isfinite(s).all(), f'{e}: PE sample count and finiteness')
        if 'chirp_mass_source' in c and 'redshift' in c:  # no `_source` suffix = detector frame (as M13_7 cell 27)
            dev = np.max(np.abs(c['chirp_mass'] / ((1 + c['redshift']) * c['chirp_mass_source']) - 1))
            self.check(dev < 1e-6, f'{e}: PE chirp_mass is detector frame')
        self._pe[e] = s
        return s


def grid_entry(grid, lab, cfg):
    keys = [k for k in ('taxonomy_mode', 'interval_mode', 'n_bins', 'n_neighbors', 'score') if k in cfg]
    hit = [g for g in grid if g['label'] == lab and all(g[k] == cfg[k] for k in keys)]
    return hit[0] if len(hit) == 1 else None


# ---- Figures ---------------------------------------------------------------------------------------------------
# Each group function computes once (same calculations and checks as before) and draws a combined figure (appendix)
# and, where listed in FIGURES, one figure per label (suffix _<label>) with the same panel function.
def emit_per_label(out, gid, draw, numbers, rows, note=None, layout=(1, 3), figsize=None, ncol=3, ncol_single=2,
                   sharex=False, sharey=False, finish=None):
    """draw(ax, lab, combined) -> legend handles; finish(fig, axes, combined) for figure-level items (colorbar)."""
    extra = {'note': note} if note else {}
    fig, axes = plt.subplots(*layout, figsize=figsize or FULL, layout='constrained', sharex=sharex, sharey=sharey)
    axes = list(np.ravel(axes))
    handles = [draw(ax, lab, True) for ax, lab in zip(axes, LABELS)][-1]
    if finish:
        finish(fig, axes, True)
    fig.legend(handles=handles, loc='outside lower center', ncol=ncol)
    res = {gid: dict(files=save(fig, out, NAME[gid]), numbers=numbers, rows=rows, **extra)}
    for lab in LABELS:
        oid = f'{gid}_{lab}'
        with single_rc():
            fig, ax = plt.subplots(figsize=SINGLE_SIZE, layout='constrained')
            handles = draw(ax, lab, False)
            if finish:
                finish(fig, [ax], False)
            fig.legend(handles=handles, loc='outside lower center', ncol=ncol_single)
            res[oid] = dict(files=save(fig, out, NAME[oid]), numbers=numbers.get(lab, numbers), rows=rows, **extra)
    return res


def f06(I, out):
    """RMS residual vs RMS predicted sigma in 20 quantile groups of sigma (val); RMS(sigma)/RMSE by SNR tercile."""
    v, cuts = I.pred('val'), I.snr_cuts()
    t = np.digitize(I.snr('val'), cuts)
    D, nums = {}, {}
    for j, lab in enumerate(LABELS):
        s, r = v['sigma'][:, j], v['y'][:, j] - v['mu'][:, j]
        g = quantile_groups(s, 20)
        cnt = np.bincount(g, minlength=20)
        rs, rr = np.sqrt(np.bincount(g, s ** 2, 20) / cnt), np.sqrt(np.bincount(g, r ** 2, 20) / cnt)
        ratio = [float(np.sqrt(np.mean(s[t == k] ** 2)) / np.sqrt(np.mean(r[t == k] ** 2))) for k in range(3)]
        D[lab] = dict(rs=rs, rr=rr, ratio=ratio, lo=0.8 * min(rs.min(), rr.min()), hi=1.25 * max(rs.max(), rr.max()))
        nums[lab] = dict(rms_sigma_by_group=rs, rms_residual_by_group=rr, rows_per_group=cnt,
                         rms_sigma_over_rmse_by_snr_tercile=dict(zip(SNR_NAMES, ratio)),
                         rows_per_snr_tercile=np.bincount(t, minlength=3))

    def draw(ax, lab, combined):
        d = D[lab]
        lo, hi = d['lo'], d['hi']
        ax.plot([lo, hi], [lo, hi], '--', color=COL['refline'], lw=1.3, label='$y = x$')
        ax.plot(d['rs'], d['rr'], 'o', color=COL['final_model'], ms=7, label='Group of predicted $\\sigma$ (5 % of rows each)')
        if lab in LOG_LABELS:
            ax.set(xscale='log', yscale='log')
            log_ticks(ax, lo, hi)
        else:
            lin_ticks(ax)
        ax.set(xlim=(lo, hi), ylim=(lo, hi))
        ax.grid(True, which='major', axis='both')
        ax.set_xlabel('RMS predicted $\\sigma$' + unit_sfx(lab))
        ax.set_ylabel('RMS residual' + unit_sfx(lab))
        ax.set_title(var_label(lab, unit=False))
        ax.text(0.04, 0.97, 'RMS($\\sigma$)/RMSE by SNR:\n' + '\n'.join(f'{n} {x:.2f}' for n, x in zip(SNR_NAMES, d['ratio'])),
                transform=ax.transAxes, va='top', ha='left', fontsize=fs())
        return ax.get_legend_handles_labels()[0]

    rows = dict(split='val', n=v['n'], sigma_groups='20 quantile groups of predicted sigma per label',
                snr_tercile_cuts=cuts, snr='target_network_snr (nominal)')
    return emit_per_label(out, 'F06', draw, nums, rows, ncol=2, ncol_single=1)


def f07(I, out):
    """Coverage of the single-bin symmetric 90 % interval by SNR tercile: score r vs r/sigma, val and test."""
    q, s7, cuts = I.intervals(), I.s7(), I.snr_cuts()
    gap = float(s7['criteria']['cond_max_gap'])
    nums, rows, pts = {lab: {} for lab in LABELS}, {}, {lab: [] for lab in LABELS}
    for split, marker, dx in (('val', 'o', -0.08), ('test', 's', 0.08)):
        d = I.pred(split)
        t = np.digitize(I.snr(split), cuts)
        rows[split] = dict(n=d['n'], per_tercile=np.bincount(t, minlength=3))
        for j, lab in enumerate(LABELS):
            for key, score, dd in (('r', 'r', -0.16), ('r_sigma', 'r/sigma', 0.16)):
                h = hits(d, j, q[lab], key)
                cov = [float(h[t == k].mean()) for k in range(3)]
                ref = None
                if split == 'val':
                    ref = grid_entry(s7['grid_val'], lab, dict(taxonomy_mode='none', interval_mode='symmetric',
                                                               n_bins=1, n_neighbors=None, score=score))
                elif key == 'r_sigma':
                    ref = next(g for g in s7['test'] if g['role'] == 'selected' and g['label'] == lab)
                if ref is not None:
                    self_ok = all(abs(cov[k] - ref[f'cov_SNR-T{k + 1}']) <= COV_ATOL for k in range(3))
                    I.check(self_ok, f'F07 {lab} {split} {score}: SNR-tercile coverage reproduces M14.7')
                pts[lab].append((np.arange(3) + dd + dx, cov, marker, COL[f'score_{key}']))
                nums[lab][f'{split}/{score}'] = dict(zip(SNR_NAMES, cov))

    def draw(ax, lab, combined):
        for x, cov, marker, c in pts[lab]:
            ax.plot(x, cov, marker, color=c, ms=9, mec=COL['marker_edge'], mew=0.8)
        ax.axhspan(LEVEL - gap, LEVEL + gap, color=COL['validity_band'], zorder=0)
        ax.axhline(LEVEL, color=COL['nominal_line'], lw=1.2, zorder=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(SNR_NAMES)
        ax.set_xlim(-0.6, 2.6)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('SNR tercile')
        ax.set_title(var_label(lab, unit=False))
        if not combined or lab == LABELS[0]:
            ax.set_ylabel('Empirical coverage')
        return [Patch(color=COL['score_r'], label=METH['r']['label']),
                Patch(color=COL['score_r_sigma'], label=METH['r_sigma']['label']),
                Line2D([], [], ls='', marker='o', color=COL['neutral'], ms=9, label='Validation'),
                Line2D([], [], ls='', marker='s', color=COL['neutral'], ms=9, label='Test'),
                Line2D([], [], color=COL['nominal_line'], lw=1.2, label='Nominal 0.90'),
                Patch(color=COL['validity_band'], label=f'Validity band ($\\pm${gap:.3f})')]

    rows = dict(splits=rows, snr_tercile_cuts=cuts, quantiles_cal=q,
                configuration='single bin, symmetric; score r vs r/sigma; quantiles fitted on cal')
    note = (f'Band = M14.7 criterion 3 (|coverage - 0.90| <= {gap}), read from M14.7 summary criteria.cond_max_gap; '
            'it is not a binomial band. Val values reproduce M14.7 grid_val; test r/sigma reproduces M14.7 test '
            '(selected); test r is re-derived here (not stored by M14.7).')
    return emit_per_label(out, 'F07', draw, nums, rows, note=note, sharey=True, ncol=3, ncol_single=2)


def f08(I, out):
    """M14.7 grid on val: mean width vs maximum conditional deviation, per label."""
    s7, sel = I.s7(), I.sel7()
    gap, m11 = float(s7['criteria']['cond_max_gap']), s7['inherited_m11']['geometry']

    def size(nb):
        return 25.0 + 6.0 * np.asarray(nb, float)

    nums, D = {}, {}
    for lab in LABELS:
        rows = [g for g in s7['grid_val'] if g['label'] == lab]
        I.check(len(rows) == s7['search_space']['n_per_label'][lab], f'F08 {lab}: grid size equals n_per_label')
        nums[lab] = dict(n_configs=len(rows), n_valid=sum(g['valid'] for g in rows))
        D[lab] = dict(scores=[], rings=[])
        for key, score in (('r', 'r'), ('r_sigma', 'r/sigma')):
            rr = [g for g in rows if g['score'] == score]
            D[lab]['scores'].append(dict(
                key=key, x=np.array([g['mean_width_phys'] for g in rr]), y=np.array([g['max_cond_gap'] for g in rr]),
                nb=np.array([g['n_bins'] for g in rr]), ok=np.array([bool(g['valid']) for g in rr])))
            g11 = grid_entry(rows, lab, dict(m11[lab], score=score))
            I.check(g11 is not None, f'F08 {lab}: M11 geometry with score {score} in the grid')
            D[lab]['rings'].append(g11)
            nums[lab][f'm11_geometry/{score}'] = dict(mean_width=g11['mean_width_phys'],
                                                      max_cond_gap=g11['max_cond_gap'], valid=g11['valid'])
        gs = grid_entry(rows, lab, sel['selected'][lab])
        sv = sel['selected_val'][lab]
        I.check(gs is not None and gs['key_id'] == sv['key_id'] and gs['mean_width_phys'] == sv['mean_width_phys'],
                f'F08 {lab}: selected configuration in the grid equals the selection JSON')
        D[lab]['selected'] = gs
        nums[lab]['selected'] = dict(mean_width=gs['mean_width_phys'], max_cond_gap=gs['max_cond_gap'])

    def draw(ax, lab, combined):
        ax.axhspan(0, gap, color=COL['validity_band'], zorder=0)
        for d in D[lab]['scores']:
            col, ok = COL[f"score_{d['key']}"], d['ok']
            ax.scatter(d['x'][ok], d['y'][ok], s=size(d['nb'][ok]), color=col, alpha=0.85,
                       edgecolors=COL['marker_edge'], linewidths=0.6, zorder=3)
            ax.scatter(d['x'][~ok], d['y'][~ok], s=size(d['nb'][~ok]), facecolors='none', edgecolors=col,
                       linewidths=1.3, zorder=3)
        for g11 in D[lab]['rings']:
            ax.scatter([g11['mean_width_phys']], [g11['max_cond_gap']], s=size(g11['n_bins']) + 260,
                       facecolors='none', edgecolors=COL['m11_ring'], linewidths=2.0, zorder=4)
        gs = D[lab]['selected']  # drawn last, above everything
        ax.scatter([gs['mean_width_phys']], [gs['max_cond_gap']], marker='*', s=1040, color=COL['selected'],
                   edgecolors=COL['selected_edge'], linewidths=1.5, zorder=10)
        ax.set_xlabel('Mean width' + unit_sfx(lab))
        ax.set_title(var_label(lab, unit=False))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if not combined or lab == LABELS[0]:
            ax.set_ylabel('Max. conditional deviation')
        h = [Line2D([], [], ls='', marker='o', ms=10, color=COL['score_r'], label=METH['r']['label']),
             Line2D([], [], ls='', marker='o', ms=10, color=COL['score_r_sigma'], label=METH['r_sigma']['label']),
             Line2D([], [], ls='', marker='o', ms=10, mfc='none', mec=COL['neutral'], mew=1.3, label='Not valid'),
             Line2D([], [], ls='', marker='*', ms=24, color=COL['selected'], mec=COL['selected_edge'], label='Selected'),
             Line2D([], [], ls='', marker='o', ms=16, mfc='none', mec=COL['m11_ring'], mew=2.0,
                    label='M11 Mondrian geometry'),
             Patch(color=COL['validity_band'], label=f'Conditional validity ($\\leq${gap:.3f})')]
        return h + [Line2D([], [], ls='', marker='o', color=COL['neutral'], ms=math.sqrt(size(nb)), label=f'{nb} bins')
                    for nb in (1, 8, 48)]

    rows = dict(split='val', n_eval=s7['splits']['n']['val'], source='M14.7 summary grid_val')
    return emit_per_label(out, 'F08', draw, nums, rows, ncol=5, ncol_single=3,
                          note='Filled = valid under M14.7 criteria 1-3; marker area grows with the number of bins; '
                               'star = selected configuration (drawn on top, 2x area).')


def f09(I, out):
    """F09a: empirical vs Sigma-implied error correlation by decile of predicted tau30 (val).
    F09b: three val rows (median tau30 of each tau tercile) in the (chirp mass, total mass) plane."""
    s5, s7, cal, v = I.s5(), I.s7(), I.pred('cal'), I.pred('val')
    q, R = I.intervals(), I.radii()
    I.check((cal['mu'][:, 0] > 0).all() and (v['mu'][:, 0] > 0).all(), 'F09: predicted chirp mass > 0 (tau30)')
    edges = np.asarray(s5['groups']['tau_pred_edges_cal_s'], float)
    tau_cal, tau_val = tau30(cal['mu'][:, 0]), tau30(v['mu'][:, 0])
    I.check(np.allclose(np.quantile(tau_cal, np.linspace(0, 1, 11)), edges, rtol=1e-5),
            'F09: Newtonian tau30 of predicted chirp mass reproduces the stored cal decile edges')
    dec = np.searchsorted(edges[1:-1], tau_val, side='right')
    stored = {(r['group'], r['pair']): r for r in s5['descriptive']['correlations']}
    z = v['r_std'] / v['sig_std']
    rng = np.random.default_rng(SEED_BOOT)
    corr = {}
    for (a, b) in PAIRS:
        i, j = LABELS.index(a), LABELS.index(b)
        rho = v['cov'][:, i, j] / (v['sig_std'][:, i] * v['sig_std'][:, j])
        emp, imp, lo, hi, ns = [], [], [], [], []
        for k in range(10):
            m = np.flatnonzero(dec == k)
            e_k, i_k = float(np.corrcoef(z[m, i], z[m, j])[0, 1]), float(rho[m].mean())
            st = stored.get((f'D{k + 1}', f'{a}/{b}'))
            I.check(st is not None and st['n'] == len(m) and abs(st['empirical_z'] - e_k) < 1e-6
                    and abs(st['implied_mean'] - i_k) < 1e-6, f'F09 D{k + 1} {a}/{b}: n and correlations reproduce M14.5')
            ii = m[rng.integers(0, len(m), size=(N_BOOT, len(m)))]
            xa, xb = z[ii, i], z[ii, j]
            xa, xb = xa - xa.mean(1, keepdims=True), xb - xb.mean(1, keepdims=True)
            boot = (xa * xb).sum(1) / np.sqrt((xa ** 2).sum(1) * (xb ** 2).sum(1))
            l_k, h_k = np.quantile(boot, [0.025, 0.975])
            for lst, val in ((emp, e_k), (imp, i_k), (lo, l_k), (hi, h_k), (ns, len(m))):
                lst.append(val)
        corr[f'{a}/{b}'] = dict(empirical=np.array(emp), ci95_lo=np.array(lo), ci95_hi=np.array(hi),
                                implied_mean=imp, n=ns)

    cuts = np.asarray(s7['criteria']['tau_tercile_cuts_cal'], float)
    I.info['tau_tercile_cuts_recomputed_cal'] = dict(stored=cuts, np_quantile=np.quantile(tau_cal, [1 / 3, 2 / 3]))
    ter = np.digitize(tau_val, cuts)
    examples, geo = [], []
    for k in range(3):
        m = np.flatnonzero(ter == k)
        med = float(np.median(tau_val[m]))
        row = int(m[np.argmin(np.abs(tau_val[m] - med))])  # first row closest to the tercile median (deterministic)
        mu2, y2 = v['mu'][row, :2], v['y'][row, :2]
        half = np.array([q[lab]['r_sigma'] for lab in LABELS[:2]]) * v['sigma'][row, :2]
        s2 = float(maha(dict(r_std=v['r_std'][row:row + 1], cov=v['cov'][row:row + 1]), 2)[0])
        geo.append(dict(E=ellipse_xy(mu2, v['cov_phys'][row, :2, :2], R['2D']), mu2=mu2, y2=y2, half=half))
        examples.append(dict(tercile=k + 1, row=row, idx=int(v['idx'][row]), tau30_pred_s=float(tau_val[row]),
                             tercile_median_tau_s=med, mu=mu2, truth=y2, box_half_width=half,
                             truth_in_ellipse=s2 <= R['2D'], truth_in_box=bool(np.all(np.abs(y2 - mu2) <= half))))
    rows = dict(split='val', n=v['n'], decile_edges='M14.5 tau_pred_edges_cal_s (cal)',
                tercile_cuts='M14.7 tau_tercile_cuts_cal',
                bootstrap=dict(seed=SEED_BOOT, B=N_BOOT, unit='val rows within decile'),
                selection='per tau tercile: first val row with tau30 closest to the tercile median')

    # F09a: correlation by decile, alone
    pair_key = {'chirp_mass/total_mass': 'pair_mc_mtot', 'total_mass/chi_eff': 'pair_mtot_chi'}
    with single_rc():
        fig, ax = plt.subplots(figsize=SINGLE_SIZE, layout='constrained')
        x = np.arange(1, 11)
        handles = []
        for (a, b) in PAIRS:
            c, d = COL[pair_key[f'{a}/{b}']], corr[f'{a}/{b}']
            ax.plot(x, d['implied_mean'], '-', color=c, lw=2.2)
            ax.errorbar(x, d['empirical'], yerr=[d['empirical'] - d['ci95_lo'], d['ci95_hi'] - d['empirical']],
                        fmt='o', color=c, ms=7, capsize=3)
            handles.append(Patch(color=c, label=f'{var_label(a, False)} and {var_label(b, False)}'))
        ax.set(xlabel='Predicted signal duration (decile, short → long)', ylabel='Error correlation',
               xticks=[1, 4, 7, 10])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        handles += [Line2D([], [], ls='', marker='o', color=COL['neutral'], ms=7, label='Empirical (95 % bootstrap CI)'),
                    Line2D([], [], color=COL['neutral'], lw=2.2, label='Implied by learned $\\Sigma$')]
        fig.legend(handles=handles, loc='outside lower center', ncol=2)
        files_a = save(fig, out, NAME['F09a'])

    # F09av2 (appendix): same values as F09a, slide fonts, compact legend inside the axes
    with slide_rc():
        fig, ax = plt.subplots(figsize=SLIDE_SIZE, layout='constrained')
        x = np.arange(1, 11)
        handles = []
        for (a, b) in PAIRS:
            c, d = COL[pair_key[f'{a}/{b}']], corr[f'{a}/{b}']
            ax.plot(x, d['implied_mean'], '-', color=c, lw=2.6)
            ax.errorbar(x, d['empirical'], yerr=[d['empirical'] - d['ci95_lo'], d['ci95_hi'] - d['empirical']],
                        fmt='o', color=c, ms=9, capsize=4, lw=2.0)
            handles.append(Patch(color=c, label=f'{var_label(a, False)} and {var_label(b, False)}'))
        ax.set(xlabel='Predicted signal duration (decile, short → long)', ylabel='Error correlation',
               xticks=[1, 4, 7, 10])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        handles += [Line2D([], [], ls='', marker='o', color=COL['neutral'], ms=9, label='Empirical (95 % bootstrap CI)'),
                    Line2D([], [], color=COL['neutral'], lw=2.6, label='Implied by learned $\\Sigma$')]
        leg_a2 = place_legend(ax, handles, grow=4)
        files_a2 = save_slide(fig, out, NAME['F09av2'], strict=True)

    # F09b: the three ellipse examples
    long_ex = examples[2]
    annotate = (not long_ex['truth_in_box']) and long_ex['truth_in_ellipse']
    with single_rc():
        fig, axes = plt.subplots(1, 3, figsize=WIDE, layout='constrained')
        for k, (a2, g, title) in enumerate(zip(axes, geo, ('Short signal', 'Medium signal', 'Long signal'))):
            a2.plot(g['E'][0], g['E'][1], color=COL['ellipse'], lw=2.2)
            a2.add_patch(Rectangle(g['mu2'] - g['half'], 2 * g['half'][0], 2 * g['half'][1], fill=False,
                                   ec=COL['box'], lw=2.0, ls='--'))
            a2.plot(*g['mu2'], 'o', color=COL['prediction'], ms=6)
            a2.plot(*g['y2'], 'x', color=COL['truth'], ms=11, mew=2.5)
            if k == 2 and annotate:
                a2.annotate('truth outside box,\ninside ellipse', xy=g['y2'], xytext=(0.04, 0.97),
                            textcoords='axes fraction', va='top', ha='left', fontsize=fs(), color=COL['annotation'],
                            arrowprops=dict(arrowstyle='->', color=COL['annotation'], lw=1.2))
            a2.set_title(title)
            a2.set_xlabel(var_label('chirp_mass'))
            if k == 0:
                a2.set_ylabel(var_label('total_mass'))
            a2.xaxis.set_major_locator(MaxNLocator(3))
            a2.yaxis.set_major_locator(MaxNLocator(4))
            a2.grid(True, axis='both')
        handles = [Line2D([], [], color=COL['ellipse'], lw=2.2, label='Conformal 2D ellipse'),
                   Line2D([], [], color=COL['box'], lw=2.0, ls='--', label='Intervals'),
                   Line2D([], [], ls='', marker='o', color=COL['prediction'], ms=6, label='Prediction $\\mu$'),
                   Line2D([], [], ls='', marker='x', color=COL['truth'], ms=11, mew=2.5, label='True value')]
        fig.legend(handles=handles, loc='outside lower center', ncol=4)
        files_b = save(fig, out, NAME['F09b'])
    note_b = ('Annotation on the long-signal example drawn only because the computed flags say truth_in_box = False '
              'and truth_in_ellipse = True.' if annotate else
              'Annotation "truth outside box, inside ellipse" NOT drawn: the computed flags of the long-signal '
              f"example are truth_in_box = {long_ex['truth_in_box']}, truth_in_ellipse = {long_ex['truth_in_ellipse']}.")
    note_a = ('Empirical = Pearson correlation of r/sigma within the decile (equals M14.5 empirical_z); '
              'implied = mean of Sigma_ij / sqrt(Sigma_ii Sigma_jj). Deciles of predicted tau30.')
    return {'F09a': dict(files=files_a, numbers=dict(correlation=corr), rows=rows, note=note_a),
            'F09av2': dict(files=files_a2, numbers=dict(correlation=corr), rows=rows, role='appendix', legend=leg_a2,
                           note=note_a + ' Same values as F09a; legend inside the axes.'),
            'F09b': dict(files=files_b, numbers=dict(examples=examples, radius_2D_cal=R['2D'],
                                                     long_signal_annotation=annotate), rows=rows, note=note_b)}


def f10(I, out):
    """Prediction vs truth on test (hexbin, log counts), y = x and median prediction in 25 quantile bins of truth."""
    d = I.pred('test')
    D, nums = {}, {}
    for j, lab in enumerate(LABELS):
        t, p = d['y'][:, j], d['mu'][:, j]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = 0.03 * (hi - lo)
        g = quantile_groups(t, 25)
        xm = np.array([np.median(t[g == k]) for k in range(25)])
        ym = np.array([np.median(p[g == k]) for k in range(25)])
        D[lab] = dict(t=t, p=p, lims=(lo - pad, hi + pad), xm=xm, ym=ym)
        nums[lab] = dict(median_truth_by_bin=xm, median_prediction_by_bin=ym)

    def draw(ax, lab, combined):
        e = D[lab]
        ax.hexbin(e['t'], e['p'], gridsize=55, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'], linewidths=0.0)
        ax.plot(e['lims'], e['lims'], '--', color=COL['identity_line'], lw=1.4)
        ax.plot(e['xm'], e['ym'], '-', color=COL['median_line'], lw=2.6)
        ax.set(xlim=e['lims'], ylim=e['lims'], xlabel='True ' + var_label(lab), ylabel='Predicted ' + var_label(lab))
        ax.set_aspect('equal')
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        return [Line2D([], [], ls='--', color=COL['identity_line'], lw=1.4, label='$y = x$'),
                Line2D([], [], color=COL['median_line'], lw=2.6, label='Median prediction (25 bins of truth)')]

    def finish(fig, axes, combined):
        hbs = [ax.collections[0] for ax in axes]  # the hexbin is the first collection of each axes
        norm = LogNorm(vmin=1.0, vmax=max(float(np.max(h.get_array())) for h in hbs))
        for h in hbs:
            h.set_norm(norm)
        fig.colorbar(hbs[0], ax=axes, shrink=0.8 if combined else 0.9).set_label('Count')

    return emit_per_label(out, 'F10', draw, nums, dict(split='test', n=d['n']), ncol=2, ncol_single=1, finish=finish)


def half_violins(ax, samples, x, colors, width=0.7, alpha=0.65, edge=True, orientation='vertical'):
    """Half violins: vertical = left (low) half at each x; horizontal (F11h) = upper (high) half at each y = x."""
    kw = dict(positions=x, widths=width, showextrema=False)
    if orientation == 'horizontal':
        try:
            parts = ax.violinplot(samples, side='high', orientation='horizontal', **kw)
        except TypeError:  # matplotlib < 3.10: full horizontal violin clipped to y >= its centre
            parts = ax.violinplot(samples, vert=False, **kw)
            for b, y0 in zip(parts['bodies'], x):
                for path in b.get_paths():
                    path.vertices[:, 1] = np.maximum(path.vertices[:, 1], y0)
    else:
        try:
            parts = ax.violinplot(samples, side='low', **kw)
        except TypeError:  # matplotlib < 3.9: clip each full violin to x <= its centre
            parts = ax.violinplot(samples, **kw)
            for b, x0 in zip(parts['bodies'], x):
                for path in b.get_paths():
                    path.vertices[:, 0] = np.minimum(path.vertices[:, 0], x0)
    for b, c in zip(parts['bodies'], colors):
        b.set_facecolor(c)
        b.set_edgecolor(c if edge else 'none')
        b.set_alpha(alpha)
    return parts


def violin_left_edge(body, x0, y):
    """x of the drawn left (low-side) outline of a half violin at height y (interpolated on the body path)."""
    v = np.vstack([p.vertices for p in body.get_paths()])
    v = v[np.isfinite(v).all(1) & (v[:, 0] < x0 - 1e-9)]
    if not len(v):
        return float(x0)
    o = np.argsort(v[:, 1], kind='stable')
    return float(np.interp(y, v[o, 1], v[o, 0]))


def violin_upper_edge(body, y0, x):
    """y of the drawn upper (high-side) outline of a horizontal half violin at abscissa x (F11h)."""
    v = np.vstack([p.vertices for p in body.get_paths()])
    v = v[np.isfinite(v).all(1) & (v[:, 1] > y0 + 1e-9)]
    if not len(v):
        return float(y0)
    o = np.argsort(v[:, 0], kind='stable')
    return float(np.interp(x, v[o, 0], v[o, 1]))


def f11(I, out):
    """Real events (ordered by LVK median chirp mass): LVK half violins, LVK median and 5-95 %, mu and 90 % interval."""
    ev = I.events()
    order = ev['order']
    x = np.arange(len(order))
    colors = [EVC[e] for e in order]
    D, nums = {}, {}
    for j, lab in enumerate(LABELS):
        smp = [ev['samples'][e][:, j] for e in order]
        qs = np.array([np.quantile(s, [0.05, 0.5, 0.95]) for s in smp])
        mu = np.array([ev['mu'][e][j] for e in order])
        lo, hi = np.array([ev['lo'][e][j] for e in order]), np.array([ev['hi'][e][j] for e in order])
        D[lab] = dict(smp=smp, qs=qs, mu=mu, lo=lo, hi=hi)
        nums[lab] = {e: dict(lvk_q05=qs[k, 0], lvk_median=qs[k, 1], lvk_q95=qs[k, 2], mu=mu[k], lower=lo[k],
                             upper=hi[k]) for k, e in enumerate(order)}

    def draw(ax, lab, combined):
        d = D[lab]
        half_violins(ax, d['smp'], x, colors)
        qs = d['qs']
        ax.errorbar(x - 0.1, qs[:, 1], yerr=[qs[:, 1] - qs[:, 0], qs[:, 2] - qs[:, 1]], fmt='_',
                    color=COL['lvk_summary'], ms=10, mew=2, capsize=0, lw=1.6)
        ax.errorbar(x + 0.18, d['mu'], yerr=[d['mu'] - d['lo'], d['hi'] - d['mu']], fmt='o', color=COL['final_model'],
                    ms=6, capsize=3, lw=2.0)
        ax.set_ylabel(var_label(lab))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xticks(x)
        ax.set_xticklabels([short_event(e, order) for e in order], rotation=30, ha='right')
        for tl, e in zip(ax.get_xticklabels(), order):
            if e in ev['oos']:
                tl.set_color(COL['neutral'])
        return [Patch(color=COL['lvk_neutral'], alpha=0.65, label='LVK posterior (GWTC-3)'),
                Line2D([], [], color=COL['lvk_summary'], marker='_', ms=10, mew=2, lw=1.6, label='LVK median and 5-95 %'),
                Line2D([], [], color=COL['final_model'], marker='o', ms=6, lw=2.0,
                       label=METH['final_model']['label'] + ': $\\mu$ and 90 % interval')]

    rows = dict(events=order, order='LVK median chirp mass (real-events JSON event_order_by_lvk_chirp)')
    note = ('Grey tick label = out of training support per the real-events JSON: '
            + json.dumps(ev['oos'], ensure_ascii=False) + '. Event names shortened to GWyymmdd when unique.')
    return emit_per_label(out, 'F11', draw, nums, rows, note=note, layout=(3, 1), figsize=TALL, sharex=True,
                          ncol=3, ncol_single=1)


def f12(I, out):
    """3 x 3 grid and F12_main (configured events in one row) in (chirp mass, total mass): LVK samples and 90 %
    contour, conformal 2D ellipse, intervals, mu."""
    ev, R, real = I.events(), I.radii(), I.real()
    stored = {r['event']: r for r in real['ellipsoid_per_event']}
    order = ev['order']
    main = list(CFG['real_events']['main_events'])
    require(len(main) == 3 and all(e in order for e in main),
            f'cpan_2026.json real_events.main_events must be 3 of {order}: {main}')
    D, nums = {}, {}
    for k, e in enumerate(order):
        S = ev['samples'][e][:, :2]
        dm = np.median(S, axis=0) - ev['mu'][e][:2]  # r^T Sigma^-1 r is invariant to the D rescaling
        s2 = float(dm @ np.linalg.solve(ev['cov_phys'][e][:2, :2], dm))
        I.check(np.isclose(s2, stored[e]['s_lvk_median_2D'], rtol=1e-5),
                f'F12 {e}: Mahalanobis score of the LVK median reproduces the real-events JSON')
        sub = np.random.default_rng([SEED_PE_SUBSAMPLE, k]).choice(len(S), min(len(S), N_PE_SCATTER), replace=False)
        mu2 = ev['mu'][e][:2]
        E = ellipse_xy(mu2, ev['cov_phys'][e][:2, :2], R['2D'])
        lo, hi = ev['lo'][e][:2], ev['hi'][e][:2]
        pts = np.vstack([np.percentile(S, [0.5, 99.5], axis=0), E.T, lo, hi])
        a, b = pts.min(0), pts.max(0)
        pad = 0.08 * (b - a)
        D[e] = dict(S=S, sub=sub, mu2=mu2, E=E, lo=lo, hi=hi, xlim=(a[0] - pad[0], b[0] + pad[0]),
                    ylim=(a[1] - pad[1], b[1] + pad[1]))
        nums[e] = dict(mu=mu2, lower=lo, upper=hi, s_lvk_median_2D=s2, lvk_median_in_ellipse=s2 <= R['2D'])

    def draw(ax, e, xlabel, ylabel):
        d, c = D[e], EVC[e]
        ax.scatter(d['S'][d['sub'], 0], d['S'][d['sub'], 1], s=3, color=c, alpha=0.25, linewidths=0, rasterized=True)
        hpd_contour(ax, d['S'], LEVEL, c)
        ax.plot(d['E'][0], d['E'][1], color=COL['ellipse'], lw=2.0)
        ax.add_patch(Rectangle(d['lo'], d['hi'][0] - d['lo'][0], d['hi'][1] - d['lo'][1], fill=False, ec=COL['box'],
                               lw=1.8, ls='--'))
        ax.plot(*d['mu2'], 'o', color=COL['prediction'], ms=5)
        ax.set(xlim=d['xlim'], ylim=d['ylim'])
        ax.set_title(short_event(e, order) + (' (out of support)' if e in ev['oos'] else ''), fontsize=fs())
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.grid(True, axis='both')
        if xlabel:
            ax.set_xlabel(var_label('chirp_mass'))
        if ylabel:
            ax.set_ylabel(var_label('total_mass'))

    def handles():
        return [Patch(color=COL['lvk_neutral'], alpha=0.5, label='LVK samples and 90 % region'),
                Line2D([], [], color=COL['ellipse'], lw=2.0, label='Conformal 2D ellipse'),
                Line2D([], [], color=COL['box'], lw=1.8, ls='--', label='Intervals'),
                Line2D([], [], ls='', marker='o', color=COL['prediction'], ms=5, label='Prediction $\\mu$')]

    fig, axes = plt.subplots(3, 3, figsize=TALL, layout='constrained')
    for k, (ax, e) in enumerate(zip(axes.flat, order)):
        draw(ax, e, k // 3 == 2, k % 3 == 0)
    for ax in list(axes.flat)[len(order):]:
        ax.axis('off')
    fig.legend(handles=handles(), loc='outside lower center', ncol=4)
    files_grid = save(fig, out, NAME['F12'])
    with single_rc():
        fig, axes = plt.subplots(1, 3, figsize=WIDE, layout='constrained')
        for k, (ax, e) in enumerate(zip(axes, main)):
            draw(ax, e, True, k == 0)
        fig.legend(handles=handles(), loc='outside lower center', ncol=2)
        files_main = save(fig, out, NAME['F12_main'])
    rows = dict(events=order, radius_2D_cal=R['2D'],
                lvk_scatter=f'{N_PE_SCATTER} samples per event, default_rng([{SEED_PE_SUBSAMPLE}, k]), '
                            'k = position in the LVK chirp-mass order (same subsample in F12 and F12_main)',
                lvk_contour='90 % highest-density region of an 80 x 80 histogram with 3 x 3 box blur')
    return {'F12': dict(files=files_grid, numbers=nums, rows=rows),
            'F12_main': dict(files=files_main, numbers={e: nums[e] for e in main},
                             rows=dict(rows, events=main, selection='cpan_2026.json real_events.main_events'))}


# ---- Slide versions (test; F08s on validation) -----------------------------------------------------------------
def marginal_axes(fig, width_ratios=(4, 1, 0.15), height_ratios=(1, 4)):
    """Main axes with a marginal on top and on the right, plus a colour-bar axes."""
    gs = fig.add_gridspec(2, 3, width_ratios=list(width_ratios), height_ratios=list(height_ratios))
    ax = fig.add_subplot(gs[1, 0])
    top, right = fig.add_subplot(gs[0, 0], sharex=ax), fig.add_subplot(gs[1, 1], sharey=ax)
    for a in (top, right):
        a.grid(False)
    top.tick_params(labelbottom=False)
    top.set_yticks([])
    top.spines['left'].set_visible(False)
    right.tick_params(labelleft=False)
    right.set_xticks([])
    right.spines['bottom'].set_visible(False)
    return ax, top, right, fig.add_subplot(gs[1, 2])


def f10m(I, out):
    """Prediction vs truth on test (hexbin, log counts), y = x, marginals of truth and prediction."""
    d = I.pred('test')
    res = {}
    for j, lab in enumerate(LABELS):
        t, p = d['y'][:, j], d['mu'][:, j]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = 0.03 * (hi - lo)
        lims = (lo - pad, hi + pad)
        bins = np.linspace(*lims, 61)
        with slide_rc():
            fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
            ax, top, right, cax = marginal_axes(fig)
            hb = ax.hexbin(t, p, gridsize=55, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'], linewidths=0.0,
                           extent=(*lims, *lims))
            ax.plot(lims, lims, '--', color=COL['identity_line'], lw=1.8)
            ax.set(xlim=lims, ylim=lims, xlabel='True ' + var_label(lab), ylabel='Predicted ' + var_label(lab))
            ax.grid(False)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            for a, o in ((top, 'vertical'), (right, 'horizontal')):
                a.hist(t, bins=bins, histtype='step', color=COL['truth'], lw=2.0, orientation=o)
                a.hist(p, bins=bins, histtype='step', color=COL['network'], lw=2.0, orientation=o)
            fig.colorbar(hb, cax=cax).set_label('Count')
            inside_legend(ax, [Line2D([], [], ls='--', color=COL['identity_line'], lw=1.8, label='$y = x$'),
                               Line2D([], [], color=COL['truth'], lw=2.0, label='True values'),
                               Line2D([], [], color=COL['network'], lw=2.0, label='Network prediction')],
                          loc='upper left')
            files = save_slide(fig, out, NAME[f'F10m_{lab}'])
        res[f'F10m_{lab}'] = dict(files=files, rows=dict(split='test', n=d['n']),
                                  note='Marginals: 60 equal-width bins over the axis range; top = along the x axis, '
                                       'right = along the y axis; both show truth and prediction.')
    return res


def f06m(I, out):
    """Predicted sigma vs |y - mu| on test (hexbin), RMS |y - mu| in 20 quantile groups of sigma, y = x, marginals;
    RMS(sigma)/RMSE by SNR tercile (val cuts, as F06 / F07)."""
    d, cuts = I.pred('test'), I.snr_cuts()
    t = np.digitize(I.snr('test'), cuts)
    res = {}
    for j, lab in enumerate(LABELS):
        s, a = d['sigma'][:, j], np.abs(d['y'][:, j] - d['mu'][:, j])
        g = quantile_groups(s, 20)
        cnt = np.bincount(g, minlength=20)
        rs, rr = np.sqrt(np.bincount(g, s ** 2, 20) / cnt), np.sqrt(np.bincount(g, a ** 2, 20) / cnt)
        ratio = [float(np.sqrt(np.mean(s[t == k] ** 2)) / np.sqrt(np.mean(a[t == k] ** 2))) for k in range(3)]
        log = lab in LOG_LABELS
        keep = a > 0 if log else np.ones(len(a), bool)   # log axes: |y - mu| = 0 cannot be drawn (counted below)
        if log:
            xl, yl = (0.9 * s.min(), 1.1 * s.max()), (float(np.quantile(a[keep], 1e-3)), 1.1 * a.max())
            bx, by = np.geomspace(*xl, 50), np.geomspace(*yl, 50)
        else:
            xl, yl = (0.0, 1.05 * s.max()), (0.0, 1.05 * a.max())
            bx, by = np.linspace(*xl, 50), np.linspace(*yl, 50)
        scale = 'log' if log else 'linear'
        with slide_rc():
            fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
            ax, top, right, cax = marginal_axes(fig)
            hb = ax.hexbin(s[keep], a[keep], gridsize=50, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'],
                           linewidths=0.0, xscale=scale, yscale=scale)
            lo_, hi_ = max(xl[0], yl[0]), min(xl[1], yl[1])
            ax.plot([lo_, hi_], [lo_, hi_], '--', color=COL['refline'], lw=1.8, label='$y = x$')
            ax.plot(rs, rr, 'o-', color=COL['network'], ms=8, lw=2.2, mec=COL['marker_edge'], mew=0.8,
                    label='RMS $|y-\\mu|$ in 20 groups of $\\sigma$')
            ax.set(xscale=scale, yscale=scale, xlim=xl, ylim=yl)
            if log:
                log_ticks(ax, min(xl[0], yl[0]), max(xl[1], yl[1]))
            else:
                lin_ticks(ax)
            ax.grid(False)
            ax.set_xlabel('Predicted $\\sigma$' + unit_sfx(lab))
            ax.set_ylabel('$|y-\\mu|$' + unit_sfx(lab))
            top.hist(s, bins=bx, histtype='step', color=COL['network'], lw=2.0)
            right.hist(a[keep], bins=by, histtype='step', color=COL['truth'], lw=2.0, orientation='horizontal')
            fig.colorbar(hb, cax=cax).set_label('Count')
            title = 'RMS($\\sigma$)/RMSE by SNR tercile\n' + ',  '.join(f'{n} {x:.2f}' for n, x in zip(SNR_NAMES, ratio))
            inside_legend(ax, ax.get_legend_handles_labels()[0], loc='upper left', title=title)
            files = save_slide(fig, out, NAME[f'F06m_{lab}'])
        res[f'F06m_{lab}'] = dict(
            files=files, rows=dict(split='test', n=d['n'], snr_tercile_cuts=cuts, snr='target_network_snr (nominal)'),
            numbers=dict(rms_sigma_by_group=rs, rms_abs_error_by_group=rr, rows_per_group=cnt,
                         rms_sigma_over_rmse_by_snr_tercile=dict(zip(SNR_NAMES, ratio)),
                         rows_per_snr_tercile=np.bincount(t, minlength=3), n_zero_error_not_drawn=int((~keep).sum())),
            note='20 quantile groups of predicted sigma on test; SNR tercile cuts of val (M14.5 / M14.7), as F06.')
    return res


def inherited_test(I):
    """Per-row test intervals of the inherited Mondrian geometry with score r, re-applying the cal-only M14.7 bundle
    as its metadata 'application' says; test coverage and mean width checked against M14.7 (role inherited_m11_r)."""
    if I.cached('inherited_test'):
        return I._cache['inherited_test']
    start = len(I.used)
    import pickle
    from copy import copy
    from dataclasses import replace
    from src.conformal.pipeline import apply_mondrian
    s7, d = I.s7(), I.pred('test')
    p = I.exists(s7['bundle']['path'], 'M14.7 cal-only calibrator bundle')
    I.check(I.sha(p) == s7['bundle']['sha256'], 'M14.7 bundle: sha256 equals the M14.7 summary')
    with open(p, 'rb') as fh:
        b = pickle.load(fh)  # own artifact, identity checked above
    meta = b['metadata']
    I.check(meta['calibration_space'] == 'standardized' and list(meta['label_names']) == LABELS
            and np.allclose(meta['y_std'], d['sd']) and np.allclose(meta['y_mean'], d['mean']),
            'M14.7 bundle: standardized space, label order and train-only scaler of the test npz')
    with np.load(d['path'], allow_pickle=False) as z:
        f_std, emb = z['pred_test'].astype(np.float64), z['emb_test'].astype(np.float64)
    out = {}
    for j, lab in enumerate(LABELS):
        e = b['inherited_m11'][lab]['r']
        sysm, geo = e['system'], s7['inherited_m11']['geometry'][lab]
        I.check(sysm.label_index == j and e['config']['score'] == 'r'
                and all(e['config'][k] == geo[k] for k in ('taxonomy_mode', 'interval_mode', 'n_bins')),
                f'{lab}: bundle inherited geometry (score r) equals the M14.7 summary')
        if sysm.n_bins == 1:
            bins = np.zeros(d['n'], dtype=np.int64)
        else:
            fitted = replace(sysm.fitted, difficulty_model=copy(sysm.fitted.difficulty_model))
            pr = apply_mondrian(fitted, pred_target=f_std,
                                target_embedding=emb if sysm.taxonomy_mode == 'difficulty' else None)
            bins = np.asarray(pr.bin_indices[:, j], dtype=np.int64)
        off = np.asarray(sysm.fitted.calibrator.intervals_[j], dtype=np.float64)[bins]
        lo = (f_std[:, j] + off[:, 0]) * d['sd'][j] + d['mean'][j]
        hi = (f_std[:, j] + off[:, 1]) * d['sd'][j] + d['mean'][j]
        y = d['y'][:, j]
        h, w = (y >= lo) & (y <= hi), hi - lo
        ref = next(r for r in s7['test'] if r['role'] == 'inherited_m11_r' and r['label'] == lab)
        I.check(abs(h.mean() - ref['coverage']) <= COV_ATOL
                and np.isclose(w.mean(), ref['mean_width_phys'], rtol=MONDRIAN_WIDTH_RTOL),
                f'{lab}: inherited Mondrian (r) test coverage and mean width reproduce M14.7')
        out[lab] = dict(hit=h, width=w, bins=bins)
    return I.store('inherited_test', out, start)


def f07b(I, out):
    """Mean width and coverage of the 90 % interval by network SNR (10 equal-size test groups), three methods."""
    q, s7, d = I.intervals(), I.s7(), I.pred('test')
    M = inherited_test(I)
    gap = float(s7['criteria']['cond_max_gap'])
    snr = I.snr('test')
    g = quantile_groups(snr, 10)
    xs = np.array([np.median(snr[g == k]) for k in range(10)])
    res = {}
    for j, lab in enumerate(LABELS):
        ref = next(r for r in s7['test'] if r['role'] == 'selected' and r['label'] == lab)
        hs, ws = hits(d, j, q[lab], 'r_sigma'), widths(d, j, q[lab], 'r_sigma')
        I.check(abs(hs.mean() - ref['coverage']) <= COV_ATOL and np.isclose(ws.mean(), ref['mean_width_phys'],
                                                                             rtol=Q_RTOL),
                f'F07b {lab}: selected test coverage and mean width reproduce M14.7')
        meth = [('raw_single', COL['score_r'], hits(d, j, q[lab], 'r'), widths(d, j, q[lab], 'r')),
                ('mondrian_raw', COL['mondrian_raw'], M[lab]['hit'], M[lab]['width']),
                ('normalised_selected', COL['network'], hs, ws)]
        per = {k: dict(mean_width=np.array([w[g == i].mean() for i in range(10)]),
                       coverage=np.array([h[g == i].mean() for i in range(10)]), overall_coverage=float(h.mean()),
                       overall_mean_width=float(w.mean())) for k, _, h, w in meth}
        with slide_rc():
            fig, (a1, a2) = plt.subplots(2, 1, figsize=SLIDE_SIZE, sharex=True, layout='constrained')
            for k, c, _, _ in meth:
                kw = dict(color=c, ms=8, lw=2.4, mec=COL['marker_edge'], mew=0.8, label=METH[k]['label'])
                a1.plot(xs, per[k]['mean_width'], 'o-', **kw)
                a2.plot(xs, per[k]['coverage'], 'o-', **kw)
            band = a2.axhspan(LEVEL - gap, LEVEL + gap, color=COL['validity_band'], zorder=0,
                              label=f'validity band ($\\pm${gap:.3f})')
            nom = a2.axhline(LEVEL, color=COL['nominal_line'], lw=1.4, zorder=1, label='nominal 0.90')
            a1.set_ylabel('Mean width' + unit_sfx(lab))
            a2.set_ylabel('Coverage')
            a2.set_xlabel('Network SNR (median of 10 equal-size groups)')
            a1.set_title(f'{var_label(lab, unit=False)}: 90 % intervals on test', loc='left')
            for a in (a1, a2):
                a.yaxis.set_major_locator(MaxNLocator(4))
            inside_legend(a1, a1.get_legend_handles_labels()[0], loc='best')
            inside_legend(a2, [nom, band], loc='best')
            files = save_slide(fig, out, NAME[f'F07b_{lab}'])
        res[f'F07b_{lab}'] = dict(
            files=files, numbers=dict(snr_group_median=xs, rows_per_group=np.bincount(g, minlength=10), **per),
            rows=dict(split='test', n=d['n'], quantiles='fitted on cal only', snr='target_network_snr (nominal)'),
            note=f'Band = M14.7 cond_max_gap ({gap}); not a binomial band. raw single bin: cal quantile of |y - mu|; '
                 'Mondrian raw: inherited geometry of the previous pipeline, score r, M14.7 cal-only bundle '
                 '(coverage and width checked vs M14.7 test); normalised selected: M14.7 selection (checked).')
    return res


def f08s(I, out):
    """M14.7 configuration search on validation: mean width vs max conditional deviation (per label)."""
    s7, sel = I.s7(), I.sel7()
    gap = float(s7['criteria']['cond_max_gap'])
    res = {}

    def size(nb):
        return 40.0 + 8.0 * np.asarray(nb, float)

    for lab in LABELS:
        rows = [g for g in s7['grid_val'] if g['label'] == lab]
        I.check(len(rows) == s7['search_space']['n_per_label'][lab], f'F08s {lab}: grid size equals n_per_label')
        gs = grid_entry(rows, lab, sel['selected'][lab])
        sv = sel['selected_val'][lab]
        I.check(gs is not None and gs['key_id'] == sv['key_id'] and gs['mean_width_phys'] == sv['mean_width_phys'],
                f'F08s {lab}: selected configuration in the grid equals the selection JSON')
        modes = sorted({g['interval_mode'] for g in rows})
        mk = {'symmetric': 'o', 'asymmetric': '^'} if len(modes) > 1 else {m: 'o' for m in modes}
        with slide_rc():
            fig, ax = plt.subplots(figsize=SLIDE_SIZE, layout='constrained')
            for score, key in (('r', 'score_r'), ('r/sigma', 'network')):
                for m in modes:
                    rr = [g for g in rows if g['score'] == score and g['interval_mode'] == m]
                    if not rr:
                        continue
                    x = np.array([g['mean_width_phys'] for g in rr])
                    y = np.array([g['max_cond_gap'] for g in rr])
                    nb = np.array([g['n_bins'] for g in rr])
                    ok = np.array([bool(g['valid']) for g in rr])
                    ax.scatter(x[ok], y[ok], s=size(nb[ok]), marker=mk[m], color=COL[key], alpha=0.85,
                               edgecolors=COL['marker_edge'], linewidths=0.6, zorder=3)
                    ax.scatter(x[~ok], y[~ok], s=size(nb[~ok]), marker=mk[m], facecolors='none', edgecolors=COL[key],
                               linewidths=1.4, zorder=3)
            ax.scatter([gs['mean_width_phys']], [gs['max_cond_gap']], marker='*', s=1100, color=COL['selected'],
                       edgecolors=COL['selected_edge'], linewidths=1.5, zorder=10)
            thr = ax.axhline(gap, ls=':', color=COL['nominal_line'], lw=2.0, label=f'validity threshold ({gap:.3f})')
            ax.set_xlabel('Mean width' + unit_sfx(lab))
            ax.set_ylabel('Max. conditional deviation')
            ax.set_title(f'{var_label(lab, unit=False)}: validation (selection set)', loc='left')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            h = [Line2D([], [], ls='', marker='o', ms=12, color=COL['score_r'], label=METH['raw_single']['label']),
                 Line2D([], [], ls='', marker='o', ms=12, color=COL['network'], label=METH['normalised']['label'])]
            if len(modes) > 1:
                h += [Line2D([], [], ls='', marker=mk[m], ms=12, color=COL['neutral'], label=m) for m in modes]
            h += [Line2D([], [], ls='', marker='o', ms=math.sqrt(size(nb)), color=COL['neutral'], label=f'{nb} bins')
                  for nb in (1, 8, 48)]
            h += [Line2D([], [], ls='', marker='o', ms=12, mfc='none', mec=COL['neutral'], mew=1.4,
                         label='fails another criterion'),
                  Line2D([], [], ls='', marker='*', ms=26, color=COL['selected'], mec=COL['selected_edge'],
                         label='selected'), thr]
            inside_legend(ax, h, loc='best', ncol=2)
            files = save_slide(fig, out, NAME[f'F08s_{lab}'])
        res[f'F08s_{lab}'] = dict(
            files=files, numbers=dict(n_configs=len(rows), n_valid=sum(g['valid'] for g in rows), interval_modes=modes,
                                      selected=dict(mean_width=gs['mean_width_phys'], max_cond_gap=gs['max_cond_gap'])),
            rows=dict(split='val', n_eval=s7['splits']['n']['val'], source='M14.7 summary grid_val'),
            note='Filled = valid under M14.7 criteria 1-3; hollow = fails criterion 1 or 2 or 3; area grows with the '
                 'number of bins. The inherited geometry is not marked.')
    return res


def f09c(I, out):
    """Short signals (predicted tau30 tercile 1, test): normalised errors of chirp mass and total mass, box of the two
    90 % intervals and conformal 2D ellipse; joint coverages in the group and global 3D (checked vs numbers.json)."""
    s7, d, q, R = I.s7(), I.pred('test'), I.intervals(), I.radii()
    I.check((d['mu'][:, 0] > 0).all(), 'F09c: predicted chirp mass > 0 (tau30)')
    cuts = np.asarray(s7['criteria']['tau_tercile_cuts_cal'], float)
    m = np.digitize(tau30(d['mu'][:, 0]), cuts) == 0
    z = d['r_std'][:, :2] / d['sig_std'][:, :2]
    qb = np.array([q[lab]['r_sigma'] for lab in LABELS[:2]])
    in_box = np.all([hits(d, j, q[lab], 'r_sigma') for j, lab in enumerate(LABELS[:2])], axis=0)
    in_ell = maha(d, 2) <= R['2D']
    b3 = float(np.all([hits(d, j, q[lab], 'r_sigma') for j, lab in enumerate(LABELS)], axis=0).mean())
    e3 = float((maha(d, 3) <= R['3D']).mean())
    nj = I.json(Path(out) / 'numbers.json', 'numbers.json of this run (joint test coverage)')
    jt = nj.get('joint_test', {})
    I.check(isinstance(jt, dict) and 'ellipsoid_3d' in jt
            and abs(jt['box_of_three_intervals']['coverage'] - b3) <= COV_ATOL
            and abs(jt['ellipsoid_3d']['coverage'] - e3) <= COV_ATOL,
            'F09c: global 3D box / ellipsoid test coverage equals numbers.json')
    I.check(f"{b3:.3f}" == f"{EXPECTED_JOINT_3D['box']:.3f}" and f"{e3:.3f}" == f"{EXPECTED_JOINT_3D['ellipsoid']:.3f}",
            f'F09c: global 3D coverages round to {EXPECTED_JOINT_3D}')
    rho = d['cov'][:, 0, 1] / (d['sig_std'][:, 0] * d['sig_std'][:, 1])
    rbar = float(rho[m].mean())
    E = ellipse_xy(np.zeros(2), np.array([[1.0, rbar], [rbar, 1.0]]), R['2D'])
    cats = [('region_both', in_box & in_ell, 'inside both'), ('region_ellipse_only', in_ell & ~in_box, 'ellipse only'),
            ('region_box_only', in_box & ~in_ell, 'box only'), ('region_outside', ~(in_box | in_ell), 'outside both')]
    lim = 1.25 * max(np.abs(E).max(), qb.max())
    zz = z[m]
    n_beyond = int(np.sum(np.any(np.abs(zz) > lim, axis=1)))
    cb, ce = float(in_box[m].mean()), float(in_ell[m].mean())
    with slide_rc():
        fig, ax = plt.subplots(figsize=SLIDE_SIZE, layout='constrained')
        for key, sel, lab_ in cats:
            ss = sel[m]
            ax.scatter(zz[ss, 0], zz[ss, 1], s=10 if key == 'region_both' else 22, color=COL[key], alpha=0.6,
                       linewidths=0, rasterized=True, label=f'{lab_} ({100 * ss.mean():.1f} %)',
                       zorder=2 if key == 'region_both' else 3)
        ax.add_patch(Rectangle(-qb, 2 * qb[0], 2 * qb[1], fill=False, ec=COL['box'], lw=2.4, ls='--', zorder=4,
                               label='box of the two 90 % intervals'))
        ax.plot(E[0], E[1], color=COL['network'], lw=2.8, zorder=5, label='conformal 2D ellipse')
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
        ax.set_aspect('equal')
        ax.set_xlabel('$(y-\\mu)/\\sigma$ of ' + var_label('chirp_mass', unit=False))
        ax.set_ylabel('$(y-\\mu)/\\sigma$ of ' + var_label('total_mass', unit=False))
        ax.set_title('Short signals (shortest third of predicted durations), test', loc='left')
        inside_legend(ax, ax.get_legend_handles_labels()[0], loc='upper left', markerscale=2)
        ax.text(0.98, 0.03, f'joint coverage, short signals (2D):\nbox {cb:.3f},  ellipse {ce:.3f}\n'
                            f'all test rows (3D):\nbox {b3:.3f},  ellipsoid {e3:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=SLIDE['legend_size'],
                bbox=dict(facecolor=COL['background'], alpha=SLIDE['legend_alpha'], edgecolor=COL['grid']))
        files = save_slide(fig, out, NAME['F09c'])
    return {'F09c': dict(
        files=files, numbers=dict(n_rows_group=int(m.sum()), coverage_2d_group=dict(box=cb, ellipse=ce),
                                  coverage_3d_test=dict(box=b3, ellipsoid=e3), mean_implied_correlation_group=rbar,
                                  box_half_widths_r_over_sigma=qb, radius_2D_cal=R['2D'], points_beyond_axes=n_beyond,
                                  fractions_group={k: float(s_[m].mean()) for k, s_, _ in cats}),
        rows=dict(split='test', n=d['n'], group='predicted tau30 below the first M14.7 tercile cut (cal)',
                  tau_tercile_cuts_cal=cuts),
        note='Point colours use the per-row criteria of the method (row Sigma for the ellipse); the drawn ellipse uses '
             'the mean implied correlation of the group, so points near its edge may be coloured by their own '
             'ellipse. Points beyond the axes are not drawn (counted).')}


def f11v2(I, out):
    """Real events per variable (slide): LVK half violins with median and 5-95 %, network mu and 90 % interval."""
    ev = I.events()
    order = ev['order']
    x = np.arange(len(order))
    colors = [EVC[e] for e in order]
    ylims = CFG['figure_options']['F11v2']['ylim']
    res = {}
    for j, lab in enumerate(LABELS):
        smp = [ev['samples'][e][:, j] for e in order]
        qs = np.array([np.quantile(s_, [0.05, 0.5, 0.95]) for s_ in smp])
        mu = np.array([ev['mu'][e][j] for e in order])
        lo, hi = np.array([ev['lo'][e][j] for e in order]), np.array([ev['hi'][e][j] for e in order])
        yl = [None if v is None else float(v) for v in (ylims.get(lab) or [None, None])]
        with slide_rc():
            fig, ax = plt.subplots(figsize=SLIDE_FULL, layout='constrained')
            half_violins(ax, smp, x, colors)
            ax.errorbar(x - 0.1, qs[:, 1], yerr=[qs[:, 1] - qs[:, 0], qs[:, 2] - qs[:, 1]], fmt='_',
                        color=COL['lvk_summary'], ms=14, mew=2.5, capsize=0, lw=2.0)
            ax.errorbar(x + 0.18, mu, yerr=[mu - lo, hi - mu], fmt='o', color=COL['network'], ms=9, capsize=4, lw=2.6)
            ax.set_ylabel(var_label(lab))
            ax.set_ylim(*yl)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xticks(x)
            ax.set_xticklabels([short_event(e, order) for e in order], rotation=30, ha='right')
            for tl, e in zip(ax.get_xticklabels(), order):
                if e in ev['oos']:
                    tl.set_color(COL['neutral'])
            lvk = (Patch(facecolor=COL['lvk_neutral'], alpha=0.65),
                   Line2D([], [], color=COL['lvk_summary'], marker='_', ms=14, mew=2.5, lw=2.0))
            net = Line2D([], [], color=COL['network'], marker='o', ms=9, lw=2.6)
            inside_legend(ax, [lvk, net], ['LVK posterior (GWTC-3), median and 5–95 %',
                                           'Network prediction, 90 % conformal interval'],
                          loc='upper left', handler_map={tuple: HandlerTuple(ndivide=None)})
            files = save_slide(fig, out, NAME[f'F11v2_{lab}'])
        top = ax.get_ylim()[1]
        res[f'F11v2_{lab}'] = dict(
            files=files, numbers={e: dict(lvk_q05=qs[k, 0], lvk_median=qs[k, 1], lvk_q95=qs[k, 2], mu=mu[k],
                                          lower=lo[k], upper=hi[k]) for k, e in enumerate(order)},
            rows=dict(events=order, order='LVK median chirp mass', ylim_config=yl),
            note='Grey tick label = out of training support: ' + json.dumps(ev['oos'], ensure_ascii=False)
                 + '. Above the upper y limit (not drawn): '
                 + json.dumps([e for k, e in enumerate(order) if max(qs[k, 2], hi[k], smp[k].max()) > top]))
    return res


def f12v2(I, out):
    """Configured main events in (chirp mass, total mass): LVK samples and 90 % contour, conformal 2D ellipse, mu."""
    ev, R, real = I.events(), I.radii(), I.real()
    stored = {r['event']: r for r in real['ellipsoid_per_event']}
    order = ev['order']
    main = list(CFG['real_events']['main_events'])
    require(len(main) == 3 and all(e in order for e in main),
            f'cpan_2026.json real_events.main_events must be 3 of {order}: {main}')
    nums = {}
    with slide_rc():
        fig, axes = plt.subplots(1, 3, figsize=SLIDE_SIZE, layout='constrained')
        for i, (ax, e) in enumerate(zip(axes, main)):
            k = order.index(e)   # same LVK subsample as F12 / F12_main
            S, mu2, C2 = ev['samples'][e][:, :2], ev['mu'][e][:2], ev['cov_phys'][e][:2, :2]
            dm = np.median(S, axis=0) - mu2
            s2 = float(dm @ np.linalg.solve(C2, dm))
            I.check(np.isclose(s2, stored[e]['s_lvk_median_2D'], rtol=1e-5),
                    f'F12v2 {e}: Mahalanobis score of the LVK median reproduces the real-events JSON')
            sub = np.random.default_rng([SEED_PE_SUBSAMPLE, k]).choice(len(S), min(len(S), N_PE_SCATTER), replace=False)
            E = ellipse_xy(mu2, C2, R['2D'])
            pts = np.vstack([np.percentile(S, [0.5, 99.5], axis=0), E.T])
            a, b = pts.min(0), pts.max(0)
            pad = 0.08 * (b - a)
            ax.scatter(S[sub, 0], S[sub, 1], s=4, color=EVC[e], alpha=0.25, linewidths=0, rasterized=True)
            hpd_contour(ax, S, LEVEL, EVC[e])
            ax.plot(E[0], E[1], color=COL['network'], lw=2.8)
            ax.plot(*mu2, 'o', color=COL['network'], ms=10, mec=COL['marker_edge'], mew=1.0)
            ax.set(xlim=(a[0] - pad[0], b[0] + pad[0]), ylim=(a[1] - pad[1], b[1] + pad[1]))
            ax.set_title(short_event(e, order) + (' (out of support)' if e in ev['oos'] else ''))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(True, axis='both')
            ax.set_xlabel(var_label('chirp_mass'))
            if i == 0:
                ax.set_ylabel(var_label('total_mass'))
            nums[e] = dict(mu=mu2, s_lvk_median_2D=s2, lvk_median_in_ellipse=s2 <= R['2D'])
        inside_legend(axes[0], [Patch(color=COL['lvk_neutral'], alpha=0.5, label='LVK 90 % region'),
                                Line2D([], [], color=COL['network'], lw=2.8, label='Network ellipse (90 %)'),
                                Line2D([], [], ls='', marker='o', color=COL['network'], ms=10, label='Network $\\mu$')],
                      loc='best')
        files = save_slide(fig, out, NAME['F12v2'])
    return {'F12v2': dict(files=files, numbers=nums,
                          rows=dict(events=main, selection='cpan_2026.json real_events.main_events',
                                    radius_2D_cal=R['2D'], lvk_scatter=f'{N_PE_SCATTER} samples, as F12'))}


# ---- v2 / v3 slide versions: compact legend inside the axes (place_legend), check_style before saving ---------
MARGINAL_RATIOS = ((1.0, MARGINAL_FRAC, 0.04), (MARGINAL_FRAC, 1.0))   # top height / right width = frac of the panel


def f10m_v2(I, out):
    """F10m with smaller marginals: true values as filled histogram, network prediction as line (test)."""
    d = I.pred('test')
    res = {}
    for j, lab in enumerate(LABELS):
        t, p = d['y'][:, j], d['mu'][:, j]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = 0.03 * (hi - lo)
        lims = (lo - pad, hi + pad)
        bins = np.linspace(*lims, 61)
        with slide_rc():
            fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
            ax, top, right, cax = marginal_axes(fig, *MARGINAL_RATIOS)
            hb = ax.hexbin(t, p, gridsize=55, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'], linewidths=0.0,
                           extent=(*lims, *lims))
            ax.plot(lims, lims, '--', color=COL['identity_line'], lw=1.8)
            ax.set(xlim=lims, ylim=lims, xlabel='True ' + var_label(lab), ylabel='Predicted ' + var_label(lab))
            ax.grid(False)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            for a, o in ((top, 'vertical'), (right, 'horizontal')):
                a.hist(t, bins=bins, histtype='stepfilled', color=COL['true_fill'], alpha=ALPHA['true_fill'],
                       edgecolor='none', lw=0, orientation=o)
                a.hist(p, bins=bins, histtype='step', color=COL['pred_line'], lw=2.2, orientation=o)
            fig.colorbar(hb, cax=cax).set_label('Count')
            leg = place_legend(ax, [Line2D([], [], ls='--', color=COL['identity_line'], lw=1.8, label='$y = x$'),
                                    Patch(facecolor=COL['true_fill'], alpha=ALPHA['true_fill'], edgecolor='none',
                                          label='True values'),
                                    Line2D([], [], color=COL['pred_line'], lw=2.2, label='Network prediction')],
                               prefer=('upper left', 'lower right'))
            files = save_slide(fig, out, NAME[f'F10mv2_{lab}'], strict=True)
        res[f'F10mv2_{lab}'] = dict(
            files=files, rows=dict(split='test', n=d['n']), role=role(lab), legend=leg,
            note=f'As F10m; marginals {MARGINAL_FRAC:.2f} of the panel (top height, right width); 60 equal-width bins; '
                 'true values = filled histogram, network prediction = line.')
    return res


def f06m_v2(I, out):
    """F06m with the F10mv2 marginals, RMS(sigma)/RMSE box at the bottom right and compact legend (test)."""
    d, cuts = I.pred('test'), I.snr_cuts()
    t = np.digitize(I.snr('test'), cuts)
    res = {}
    for j, lab in enumerate(LABELS):
        s, a = d['sigma'][:, j], np.abs(d['y'][:, j] - d['mu'][:, j])
        g = quantile_groups(s, 20)
        cnt = np.bincount(g, minlength=20)
        rs, rr = np.sqrt(np.bincount(g, s ** 2, 20) / cnt), np.sqrt(np.bincount(g, a ** 2, 20) / cnt)
        ratio = [float(np.sqrt(np.mean(s[t == k] ** 2)) / np.sqrt(np.mean(a[t == k] ** 2))) for k in range(3)]
        log = lab in LOG_LABELS
        keep = a > 0 if log else np.ones(len(a), bool)   # log axes: |y - mu| = 0 cannot be drawn (counted below)
        if log:
            xl, yl = (0.9 * s.min(), 1.1 * s.max()), (float(np.quantile(a[keep], 1e-3)), 1.1 * a.max())
            bx, by = np.geomspace(*xl, 50), np.geomspace(*yl, 50)
        else:
            xl, yl = (0.0, 1.05 * s.max()), (0.0, 1.05 * a.max())
            bx, by = np.linspace(*xl, 50), np.linspace(*yl, 50)
        scale = 'log' if log else 'linear'
        with slide_rc():
            fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
            ax, top, right, cax = marginal_axes(fig, *MARGINAL_RATIOS)
            hb = ax.hexbin(s[keep], a[keep], gridsize=50, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'],
                           linewidths=0.0, xscale=scale, yscale=scale)
            lo_, hi_ = max(xl[0], yl[0]), min(xl[1], yl[1])
            ref, = ax.plot([lo_, hi_], [lo_, hi_], '--', color=COL['refline'], lw=1.8, label='$y = x$')
            grp, = ax.plot(rs, rr, 'o-', color=COL['network'], ms=8, lw=2.4, mec=COL['marker_edge'], mew=0.8,
                           label='RMS $|y-\\mu|$ in 20 groups of $\\sigma$')
            ax.set(xscale=scale, yscale=scale, xlim=xl, ylim=yl)
            if log:
                log_ticks(ax, min(xl[0], yl[0]), max(xl[1], yl[1]))
            else:
                lin_ticks(ax)
            ax.grid(False)
            ax.set_xlabel('Predicted $\\sigma$' + unit_sfx(lab))
            ax.set_ylabel('$|y-\\mu|$' + unit_sfx(lab))
            top.hist(s, bins=bx, histtype='step', color=COL['pred_line'], lw=2.2)
            right.hist(a[keep], bins=by, histtype='stepfilled', color=COL['true_fill'], alpha=ALPHA['true_fill'],
                       edgecolor='none', lw=0, orientation='horizontal')
            fig.colorbar(hb, cax=cax).set_label('Count')
            ax.text(0.98, 0.03, 'RMS($\\sigma$)/RMSE by SNR tercile\n'
                    + ',  '.join(f'{n} {x:.2f}' for n, x in zip(SNR_NAMES, ratio)),
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=SLIDE['tick_size'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COL['background'], alpha=LEG['framealpha'],
                              edgecolor=COL['grid']))
            leg = place_legend(ax, [ref, grp], prefer=('upper left',))
            files = save_slide(fig, out, NAME[f'F06mv2_{lab}'], strict=True)
        res[f'F06mv2_{lab}'] = dict(
            files=files, role=role(lab), legend=leg,
            rows=dict(split='test', n=d['n'], snr_tercile_cuts=cuts, snr='target_network_snr (nominal)'),
            numbers=dict(rms_sigma_by_group=rs, rms_abs_error_by_group=rr, rows_per_group=cnt,
                         rms_sigma_over_rmse_by_snr_tercile=dict(zip(SNR_NAMES, ratio)),
                         rows_per_snr_tercile=np.bincount(t, minlength=3), n_zero_error_not_drawn=int((~keep).sum())),
            note='As F06m (same values). Marginals: top = predicted sigma (line), right = |y - mu| (filled).')
    return res


def f07b_v2(I, out):
    """F07b with the method colours / names of the config: one width for all, one width per bin (Mondrian), width
    proportional to sigma (normalised, selected). Test, 10 equal-size SNR groups."""
    q, s7, d = I.intervals(), I.s7(), I.pred('test')
    M = inherited_test(I)
    gap = float(s7['criteria']['cond_max_gap'])
    snr = I.snr('test')
    g = quantile_groups(snr, 10)
    xs = np.array([np.median(snr[g == k]) for k in range(10)])
    res = {}
    for j, lab in enumerate(LABELS):
        ref = next(r for r in s7['test'] if r['role'] == 'selected' and r['label'] == lab)
        hs, ws = hits(d, j, q[lab], 'r_sigma'), widths(d, j, q[lab], 'r_sigma')
        I.check(abs(hs.mean() - ref['coverage']) <= COV_ATOL and np.isclose(ws.mean(), ref['mean_width_phys'],
                                                                             rtol=Q_RTOL),
                f'F07bv2 {lab}: selected test coverage and mean width reproduce M14.7')
        meth = [('one_width_all', COL['method_raw'], hits(d, j, q[lab], 'r'), widths(d, j, q[lab], 'r')),
                ('one_width_bin', COL['method_mondrian'], M[lab]['hit'], M[lab]['width']),
                ('width_sigma', COL['method_norm'], hs, ws)]
        per = {k: dict(mean_width=np.array([w[g == i].mean() for i in range(10)]),
                       coverage=np.array([h[g == i].mean() for i in range(10)]), overall_coverage=float(h.mean()),
                       overall_mean_width=float(w.mean())) for k, _, h, w in meth}
        with slide_rc():
            fig, (a1, a2) = plt.subplots(2, 1, figsize=SLIDE_SIZE, sharex=True, layout='constrained')
            for k, c, _, _ in meth:
                kw = dict(color=c, ms=9, lw=2.8, mec=COL['marker_edge'], mew=0.8, label=METH[k]['label'])
                a1.plot(xs, per[k]['mean_width'], 'o-', **kw)
                a2.plot(xs, per[k]['coverage'], 'o-', **kw)
            band = a2.axhspan(LEVEL - gap, LEVEL + gap, color=COL['validity_band'], zorder=0,
                              label=f'validity band ($\\pm${gap:.3f})')
            nom = a2.axhline(LEVEL, color=COL['nominal_line'], lw=1.4, zorder=1, label='nominal 0.90')
            a1.set_ylabel('Mean width' + unit_sfx(lab))
            a2.set_ylabel('Coverage')
            a2.set_xlabel('Network SNR (median of 10 equal-size groups)')
            a1.set_title(f'{var_label(lab, unit=False)}: 90 % intervals on test', loc='left')
            for a in (a1, a2):
                a.yaxis.set_major_locator(MaxNLocator(4))
            leg1 = place_legend(a1, a1.get_legend_handles_labels()[0], grow=4)
            leg2 = place_legend(a2, [nom, band], grow=4)
            files = save_slide(fig, out, NAME[f'F07bv2_{lab}'], strict=True)
        res[f'F07bv2_{lab}'] = dict(
            files=files, role=role(lab), legend=dict(width=leg1, coverage=leg2),
            numbers=dict(snr_group_median=xs, rows_per_group=np.bincount(g, minlength=10), **per),
            rows=dict(split='test', n=d['n'], quantiles='fitted on cal only', snr='target_network_snr (nominal)'),
            note=f'Same values as F07b (one_width_all = raw single bin, one_width_bin = Mondrian raw, width_sigma = '
                 f'normalised selected). Band = M14.7 cond_max_gap ({gap}); not a binomial band.')
    return res


def f07q(I, out):
    """Backup, calibration split: per SNR decile (10 equal-size cal groups), split-conformal 90 % quantile (rank
    k = ceil((n_bin + 1) 0.90), as kth()) of |y - mu| (physical) and of |y - mu|/sigma; reference line = global cal
    quantile of |y - mu|/sigma (equals the selected single-bin quantile, checked)."""
    q, d = I.intervals(), I.pred('cal')
    snr = I.snr('cal')
    g = quantile_groups(snr, 10)
    cnt = np.bincount(g, minlength=10)
    I.check(len(cnt) == 10 and cnt.min() > 0, 'F07q: ten non-empty cal SNR deciles')
    xs = np.array([np.median(snr[g == k]) for k in range(10)])
    res = {}
    for j, lab in enumerate(LABELS):
        ae = np.abs(d['r_std'][:, j]) * d['sd'][j]          # |y - mu| (physical), score r of intervals()
        z = np.abs(d['r_std'][:, j]) / d['sig_std'][:, j]   # |y - mu| / sigma, score r/sigma of intervals()
        qa = [kth(ae[g == k]) for k in range(10)]
        qz = [kth(z[g == k]) for k in range(10)]
        qg, kg = kth(z)
        I.check(np.isclose(qg, q[lab]['r_sigma'], rtol=Q_RTOL) and kg == q[lab]['k'],
                f'F07q {lab}: global cal quantile of |y - mu|/sigma equals the selected single-bin quantile')
        with slide_rc():
            fig, (a1, a2) = plt.subplots(1, 2, figsize=tuple(FIGOPT['F07q']['size']), sharex=True,
                                         layout='constrained')
            a1.plot(xs, [v for v, _ in qa], 'o-', color=COL['method_raw'], ms=9, lw=2.8, mec=COL['marker_edge'],
                    mew=0.8)
            pz, = a2.plot(xs, [v for v, _ in qz], 'o-', color=COL['method_norm'], ms=9, lw=2.8,
                          mec=COL['marker_edge'], mew=0.8, label='per SNR decile')
            hq = a2.axhline(qg, color=COL['nominal_line'], ls='--', lw=2.0, label=f'global $q$ = {qg:.3f}')
            a1.set_ylabel('90 % quantile of $|y-\\mu|$' + unit_sfx(lab))
            a2.set_ylabel('90 % quantile of $|y-\\mu|/\\sigma$')
            a1.set_title(f'{var_label(lab, unit=False)}: calibration', loc='left')
            for a in (a1, a2):
                a.set_xlabel('Network SNR (decile median)')
                a.yaxis.set_major_locator(MaxNLocator(5))
                a.xaxis.set_major_locator(MaxNLocator(5))
            leg = place_legend(a2, [pz, hq], grow=4)
            files = save_slide(fig, out, NAME[f'F07q_{lab}'], strict=True)
        res[f'F07q_{lab}'] = dict(
            files=files, role='backup', legend=leg,
            numbers=dict(snr_decile_median=xs, rows_per_decile=cnt, q90_abs_error=[v for v, _ in qa],
                         q90_abs_error_over_sigma=[v for v, _ in qz], k_rank_per_decile=[k for _, k in qz],
                         global_q_abs_error_over_sigma=qg, global_k_rank=kg),
            rows=dict(split='cal', n=d['n'], snr='target_network_snr (nominal)', groups='10 equal-size SNR deciles of cal'),
            note='Quantile = split-conformal rank k = ceil((n + 1) 0.90) within each decile (kth), the same rule as the '
                 'global quantile; |y - mu|/sigma is unit-free (standardized ratio equals the physical ratio).')
    return res


TAXONOMY_NAMES = {'none': 'single bin', 'prediction': 'Mondrian on the prediction',
                  'difficulty': 'Mondrian on difficulty (kNN)'}


def f08s_v2(I, out):
    """F08s with pastel score colours, selected star in colors.network, y = worst group |coverage - 0.90|; the
    manifest entry lists the 68 configurations of the label (M14.7 summary grid_val)."""
    s7, sel = I.s7(), I.sel7()
    gap = float(s7['criteria']['cond_max_gap'])
    sp = s7['search_space']
    n_expected = ((1 if sp['one_bin'] else 0) + len(sp['n_bins_grid']) * len(sp['taxonomy_modes'])) \
        * len(sp['interval_modes']) * len(sp['scores'])
    res = {}

    def size(nb):
        return 40.0 + 8.0 * np.asarray(nb, float)

    for lab in LABELS:
        rows = [g for g in s7['grid_val'] if g['label'] == lab]
        I.check(len(rows) == s7['search_space']['n_per_label'][lab] == n_expected,
                f'F08sv2 {lab}: grid size equals n_per_label and the search space')
        gs = grid_entry(rows, lab, sel['selected'][lab])
        sv = sel['selected_val'][lab]
        I.check(gs is not None and gs['key_id'] == sv['key_id'] and gs['mean_width_phys'] == sv['mean_width_phys'],
                f'F08sv2 {lab}: selected configuration in the grid equals the selection JSON')
        configs = [dict(key_id=g['key_id'], taxonomy=g['taxonomy_mode'], taxonomy_name=TAXONOMY_NAMES.get(
                            g['taxonomy_mode'], g['taxonomy_mode']), n_neighbors=g['n_neighbors'], n_bins=g['n_bins'],
                        interval_mode=g['interval_mode'], score=g['score'], valid=bool(g['valid']),
                        coverage=g['coverage'], mean_width=g['mean_width_phys'], max_cond_gap=g['max_cond_gap'],
                        selected=g['key_id'] == gs['key_id'])
                   for g in sorted(rows, key=lambda g: g['key_id'])]
        I.check(len({(c['taxonomy'], c['n_bins'], c['n_neighbors'], c['interval_mode'], c['score']) for c in configs})
                == len(configs), f'F08sv2 {lab}: configurations are distinct')
        modes = sorted({g['interval_mode'] for g in rows})
        mk = {'symmetric': 'o', 'asymmetric': '^'} if len(modes) > 1 else {m: 'o' for m in modes}
        with slide_rc():
            fig, ax = plt.subplots(figsize=SLIDE_SIZE, layout='constrained')
            for score, key in (('r', 'config_r'), ('r/sigma', 'config_r_sigma')):
                for m in modes:
                    rr = [g for g in rows if g['score'] == score and g['interval_mode'] == m]
                    if not rr:
                        continue
                    x = np.array([g['mean_width_phys'] for g in rr])
                    y = np.array([g['max_cond_gap'] for g in rr])
                    nb = np.array([g['n_bins'] for g in rr])
                    ok = np.array([bool(g['valid']) for g in rr])
                    ax.scatter(x[ok], y[ok], s=size(nb[ok]), marker=mk[m], color=COL[key], alpha=0.9,
                               edgecolors=COL['marker_edge'], linewidths=0.6, zorder=3)
                    ax.scatter(x[~ok], y[~ok], s=size(nb[~ok]), marker=mk[m], facecolors='none', edgecolors=COL[key],
                               linewidths=1.6, zorder=3)
            ax.scatter([gs['mean_width_phys']], [gs['max_cond_gap']], marker='*', s=1100, color=COL['network'],
                       edgecolors=COL['selected_edge'], linewidths=1.5, zorder=10)
            thr = ax.axhline(gap, ls=':', color=COL['nominal_line'], lw=2.0, label=f'validity threshold ({gap:.3f})')
            ax.set_xlabel('Mean width' + unit_sfx(lab))
            ax.set_ylabel('worst group: |coverage − 0.90|')
            ax.set_title(f'{var_label(lab, unit=False)}: validation (selection set)', loc='left')
            ax.xaxis.set_major_locator(MaxNLocator(5))
            h = [Line2D([], [], ls='', marker='o', ms=12, color=COL['config_r'], label=METH['raw_single']['label']),
                 Line2D([], [], ls='', marker='o', ms=12, color=COL['config_r_sigma'], label=METH['normalised']['label'])]
            if len(modes) > 1:
                h += [Line2D([], [], ls='', marker=mk[m], ms=12, color=COL['neutral'], label=m) for m in modes]
            h += [Line2D([], [], ls='', marker='o', ms=math.sqrt(size(nb)), color=COL['neutral'], label=f'{nb} bins')
                  for nb in (1, 8, 48)]
            h += [Line2D([], [], ls='', marker='o', ms=12, mfc='none', mec=COL['neutral'], mew=1.6,
                         label='fails another criterion'),
                  Line2D([], [], ls='', marker='*', ms=26, color=COL['network'], mec=COL['selected_edge'],
                         label='selected'), thr]
            leg = place_legend(ax, h, ncol=2, prefer=('upper right',))
            files = save_slide(fig, out, NAME[f'F08sv2_{lab}'], strict=True)
        res[f'F08sv2_{lab}'] = dict(
            files=files, role=role(lab), legend=leg,
            numbers=dict(n_configs=len(rows), n_valid=sum(g['valid'] for g in rows), interval_modes=modes,
                         selected=dict(key_id=gs['key_id'], mean_width=gs['mean_width_phys'],
                                       max_cond_gap=gs['max_cond_gap'])),
            configurations=configs,
            search_space=dict(sp, n_expected_per_label=n_expected,
                              composition='(1 single bin + n_bins_grid x taxonomy_modes) x interval_modes x scores'),
            rows=dict(split='val', n_eval=s7['splits']['n']['val'], source='M14.7 summary grid_val'),
            note='y = max_cond_gap of M14.7 (largest |coverage - 0.90| over the tau and SNR tercile groups). Filled = '
                 'valid under M14.7 criteria 1-3; hollow = fails criterion 1 or 2 or 3; area grows with the number '
                 'of bins. difficulty = kNN difficulty taxonomy (n_neighbors).')
    return res


def f11v3(I, out):
    """Real events per variable: one-colour LVK half violins with median (solid) and 5 % / 95 % (dashed) drawn across
    the violin width at their height; network mu and 90 % interval in colors.network."""
    ev = I.events()
    order = ev['order']
    x = np.arange(len(order))
    ylims = FIGOPT['F11v3']['ylim']
    res = {}
    for j, lab in enumerate(LABELS):
        smp = [ev['samples'][e][:, j] for e in order]
        qs = np.array([np.quantile(s_, [0.05, 0.5, 0.95]) for s_ in smp])
        mu = np.array([ev['mu'][e][j] for e in order])
        lo, hi = np.array([ev['lo'][e][j] for e in order]), np.array([ev['hi'][e][j] for e in order])
        yl = [None if v is None else float(v) for v in (ylims.get(lab) or [None, None])]
        with slide_rc():
            fig, ax = plt.subplots(figsize=SLIDE_FULL, layout='constrained')
            parts = half_violins(ax, smp, x, [COL['lvk_fill']] * len(order), alpha=ALPHA['lvk_fill'], edge=False)
            for k, (b, x0) in enumerate(zip(parts['bodies'], x)):
                for yv, ls, lw in ((qs[k, 1], '-', 2.4), (qs[k, 0], '--', 1.8), (qs[k, 2], '--', 1.8)):
                    ax.plot([violin_left_edge(b, x0, yv), x0], [yv, yv], ls=ls, color=COL['lvk_line'], lw=lw,
                            solid_capstyle='butt', zorder=3)
            ax.errorbar(x + 0.18, mu, yerr=[mu - lo, hi - mu], fmt='o', color=COL['network'], ms=9, capsize=4, lw=2.6,
                        zorder=4)
            ax.set_ylabel(var_label(lab))
            ax.set_ylim(*yl)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xticks(x)
            ax.set_xticklabels([short_event(e, order) for e in order], rotation=30, ha='right')
            for tl, e in zip(ax.get_xticklabels(), order):
                if e in ev['oos']:
                    tl.set_color(COL['neutral'])
            handles = [Patch(facecolor=COL['lvk_fill'], alpha=ALPHA['lvk_fill'], edgecolor='none',
                             label='LVK posterior (GWTC-3)'),
                       Line2D([], [], color=COL['lvk_line'], lw=2.4, label='LVK median'),
                       Line2D([], [], color=COL['lvk_line'], lw=1.8, ls='--', label='LVK 5 % and 95 %'),
                       Line2D([], [], color=COL['network'], marker='o', ms=9, lw=2.6,
                              label='Network $\\mu$, 90 % conformal interval')]
            leg = place_legend(ax, handles, ncol=2)
            files = save_slide(fig, out, NAME[f'F11v3_{lab}'], strict=True)
        top = ax.get_ylim()[1]
        res[f'F11v3_{lab}'] = dict(
            files=files, role=role(lab), legend=leg,
            numbers={e: dict(lvk_q05=qs[k, 0], lvk_median=qs[k, 1], lvk_q95=qs[k, 2], mu=mu[k],
                             lower=lo[k], upper=hi[k]) for k, e in enumerate(order)},
            rows=dict(events=order, order='LVK median chirp mass', ylim_config=yl),
            note='Grey tick label = out of training support: ' + json.dumps(ev['oos'], ensure_ascii=False)
                 + '. Median / 5 % / 95 % segments span the drawn half-violin width at that height. Above the upper '
                 'y limit (partly not drawn): '
                 + json.dumps([e for k, e in enumerate(order) if max(qs[k, 2], hi[k], smp[k].max()) > top]))
    return res


def f12v3(I, out):
    """Configured main events in (chirp mass, total mass), three square panels: LVK samples (grey) and 90 % contour
    (dark grey), conformal 2D ellipse (colors.network), mu (black)."""
    ev, R, real = I.events(), I.radii(), I.real()
    stored = {r['event']: r for r in real['ellipsoid_per_event']}
    order = ev['order']
    main = list(CFG['real_events']['main_events'])
    require(len(main) == 3 and all(e in order for e in main),
            f'cpan_2026.json real_events.main_events must be 3 of {order}: {main}')
    nums = {}
    with slide_rc():
        fig, axes = plt.subplots(1, 3, figsize=tuple(FIGOPT['F12v3']['size']), layout='constrained', sharey=False)
        for i, (ax, e) in enumerate(zip(axes, main)):
            k = order.index(e)   # same LVK subsample as F12 / F12_main / F12v2
            S, mu2, C2 = ev['samples'][e][:, :2], ev['mu'][e][:2], ev['cov_phys'][e][:2, :2]
            dm = np.median(S, axis=0) - mu2
            s2 = float(dm @ np.linalg.solve(C2, dm))
            I.check(np.isclose(s2, stored[e]['s_lvk_median_2D'], rtol=1e-5),
                    f'F12v3 {e}: Mahalanobis score of the LVK median reproduces the real-events JSON')
            sub = np.random.default_rng([SEED_PE_SUBSAMPLE, k]).choice(len(S), min(len(S), N_PE_SCATTER), replace=False)
            E = ellipse_xy(mu2, C2, R['2D'])
            pts = np.vstack([np.percentile(S, [0.5, 99.5], axis=0), E.T])
            a, b = pts.min(0), pts.max(0)
            pad = 0.08 * (b - a)
            ax.scatter(S[sub, 0], S[sub, 1], s=5, color=COL['lvk_fill'], alpha=ALPHA['lvk_samples'], linewidths=0,
                       rasterized=True)
            hpd_contour(ax, S, LEVEL, COL['lvk_contour'])
            ax.plot(E[0], E[1], color=COL['network'], lw=2.8)
            ax.plot(*mu2, 'o', color=COL['mu'], ms=9)
            ax.set(xlim=(a[0] - pad[0], b[0] + pad[0]), ylim=(a[1] - pad[1], b[1] + pad[1]))
            ax.set_box_aspect(1)
            ax.set_title(short_event(e, order) + (' (out of support)' if e in ev['oos'] else ''))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(True, axis='both')
            ax.set_xlabel(var_label('chirp_mass'))
            if i == 0:
                ax.set_ylabel(var_label('total_mass'))
            nums[e] = dict(mu=mu2, s_lvk_median_2D=s2, lvk_median_in_ellipse=s2 <= R['2D'])
        leg = place_legend(list(axes), [
            Line2D([], [], ls='', marker='o', ms=8, color=COL['lvk_fill'], label='LVK samples'),
            Line2D([], [], color=COL['lvk_contour'], lw=1.8, label='LVK 90 % region'),
            Line2D([], [], color=COL['network'], lw=2.8, label='Network ellipse (90 %)'),
            Line2D([], [], ls='', marker='o', color=COL['mu'], ms=9, label='Network $\\mu$')])
        files = save_slide(fig, out, NAME['F12v3'], strict=True)
    return {'F12v3': dict(files=files, numbers=nums, role='main', legend=dict(leg, panel=main[leg['axes_index']]),
                          rows=dict(events=main, selection='cpan_2026.json real_events.main_events',
                                    radius_2D_cal=R['2D'], lvk_scatter=f'{N_PE_SCATTER} samples, as F12'))}


# ---- v3 / v4 slide versions: finer hexbin and soft grid (F10mv3, F06mv3), F11v2 palette (F11v4, F11h), F12v2
# palette (F12v4), short-signal normalised errors (F13) ------------------------------------------------------------
HEX_FINE = float(FIGOPT['hexbin_fine_factor'])   # gridsize = round(factor x the v2 gridsize)


def f10m_v3(I, out):
    """F10mv2 with a finer hexbin and a soft grid in the main panel and the marginals (test)."""
    d = I.pred('test')
    gs = int(round(HEX_FINE * 55))
    res = {}
    for j, lab in enumerate(LABELS):
        t, p = d['y'][:, j], d['mu'][:, j]
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = 0.03 * (hi - lo)
        lims = (lo - pad, hi + pad)
        bins = np.linspace(*lims, 61)
        with slide_rc():
            fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
            ax, top, right, cax = marginal_axes(fig, *MARGINAL_RATIOS)
            hb = ax.hexbin(t, p, gridsize=gs, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'], linewidths=0.0,
                           extent=(*lims, *lims))
            ax.plot(lims, lims, '--', color=COL['identity_line'], lw=1.8)
            ax.set(xlim=lims, ylim=lims, xlabel='True ' + var_label(lab), ylabel='Predicted ' + var_label(lab))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            for a, o in ((top, 'vertical'), (right, 'horizontal')):
                a.hist(t, bins=bins, histtype='stepfilled', color=COL['true_fill'], alpha=ALPHA['true_fill'],
                       edgecolor='none', lw=0, orientation=o)
                a.hist(p, bins=bins, histtype='step', color=COL['pred_line'], lw=2.2, orientation=o)
            soft_grid(ax, top, right)
            fig.colorbar(hb, cax=cax).set_label('Count')
            leg = place_legend(ax, [Line2D([], [], ls='--', color=COL['identity_line'], lw=1.8, label='$y = x$'),
                                    Patch(facecolor=COL['true_fill'], alpha=ALPHA['true_fill'], edgecolor='none',
                                          label='True values'),
                                    Line2D([], [], color=COL['pred_line'], lw=2.2, label='Network prediction')],
                               prefer=('upper left', 'lower right'))
            files = save_slide(fig, out, NAME[f'F10mv3_{lab}'], strict=True)
        res[f'F10mv3_{lab}'] = dict(
            files=files, rows=dict(split='test', n=d['n']), role=role(lab), legend=leg, hexbin_gridsize=gs,
            note=f'As F10mv2 (same values); hexbin gridsize {gs} (F10mv2: 55); soft grid at the major ticks.')
    return res


def f06m_v3(I, out):
    """Predicted sigma vs |y - mu| on test, LINEAR axes with the same range on both (y = x is the diagonal): 0 to
    max(percentile of |y - mu|, max sigma); finer hexbin, soft grid; both distributions in each marginal; RMS
    |y - mu| in 20 quantile groups of sigma; RMS(sigma)/RMSE by SNR tercile (val cuts) as text without a box."""
    d, cuts = I.pred('test'), I.snr_cuts()
    t = np.digitize(I.snr('test'), cuts)
    pct = float(FIGOPT['F06mv3']['abs_error_percentile'])
    gs = int(round(HEX_FINE * 50))
    res = {}
    for j, lab in enumerate(LABELS):
        s, a = d['sigma'][:, j], np.abs(d['y'][:, j] - d['mu'][:, j])
        g = quantile_groups(s, 20)
        cnt = np.bincount(g, minlength=20)
        rs, rr = np.sqrt(np.bincount(g, s ** 2, 20) / cnt), np.sqrt(np.bincount(g, a ** 2, 20) / cnt)
        ratio = [float(np.sqrt(np.mean(s[t == k] ** 2)) / np.sqrt(np.mean(a[t == k] ** 2))) for k in range(3)]
        L = float(max(np.percentile(a, pct), s.max()))
        inside = (a <= L) & (s <= L)
        bins = np.linspace(0.0, L, 61)
        with slide_rc():
            fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
            ax, top, right, cax = marginal_axes(fig, *MARGINAL_RATIOS)
            hb = ax.hexbin(s[inside], a[inside], gridsize=gs, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'],
                           linewidths=0.0, extent=(0.0, L, 0.0, L))
            ref, = ax.plot([0.0, L], [0.0, L], '--', color=COL['refline'], lw=1.8, label='$y = x$')
            grp, = ax.plot(rs, rr, 'o-', color=COL['network'], ms=8, lw=2.4, mec=COL['marker_edge'], mew=0.8,
                           label='RMS $|y-\\mu|$ in 20 groups of $\\sigma$')
            ax.set(xlim=(0.0, L), ylim=(0.0, L))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xlabel('Predicted $\\sigma$' + unit_sfx(lab))
            ax.set_ylabel('$|y-\\mu|$' + unit_sfx(lab))
            for m_, o in ((top, 'vertical'), (right, 'horizontal')):
                m_.hist(s, bins=bins, histtype='stepfilled', color=COL['true_fill'], alpha=ALPHA['true_fill'],
                        edgecolor='none', lw=0, orientation=o)
                m_.hist(a, bins=bins, histtype='step', color=COL['pred_line'], lw=2.2, orientation=o)
            soft_grid(ax, top, right)
            fig.colorbar(hb, cax=cax).set_label('Count')
            txt = place_text(ax, 'RMS($\\sigma$)/RMSE by SNR tercile\n'
                             + ',  '.join(f'{n} {x:.2f}' for n, x in zip(SNR_NAMES, ratio)),
                             prefer=('lower right', 'upper left'), fontsize=SLIDE['tick_size'])
            leg = place_legend(ax, [ref, grp,
                                    Patch(facecolor=COL['true_fill'], alpha=ALPHA['true_fill'], edgecolor='none',
                                          label='predicted $\\sigma$'),
                                    Line2D([], [], color=COL['pred_line'], lw=2.2, label='$|y - \\mu|$')],
                               prefer=('upper left',))
            files = save_slide(fig, out, NAME[f'F06mv3_{lab}'], strict=True)
        res[f'F06mv3_{lab}'] = dict(
            files=files, role=role(lab), legend=leg, text=txt, hexbin_gridsize=gs,
            rows=dict(split='test', n=d['n'], snr_tercile_cuts=cuts, snr='target_network_snr (nominal)'),
            numbers=dict(rms_sigma_by_group=rs, rms_abs_error_by_group=rr, rows_per_group=cnt,
                         rms_sigma_over_rmse_by_snr_tercile=dict(zip(SNR_NAMES, ratio)),
                         rows_per_snr_tercile=np.bincount(t, minlength=3),
                         axis_limit=L, axis_limit_rule=f'max(percentile {pct:g} of |y - mu|, max sigma)',
                         n_outside_axes=int((~inside).sum()), n_abs_error_above_limit=int((a > L).sum()),
                         n_sigma_above_limit=int((s > L).sum()),
                         group_points_outside_axes=int(np.sum((rs > L) | (rr > L)))),
            note='Same values as F06m / F06mv2; linear axes, same range [0, axis_limit] on x and y. Rows outside the '
                 'axes are not drawn in the hexbin nor in the marginals (counted). Both marginals: predicted sigma '
                 '(filled) and |y - mu| (line), 60 equal-width bins.')
    return res


def _f11_values(ev, order, j):
    smp = [ev['samples'][e][:, j] for e in order]
    qs = np.array([np.quantile(s_, [0.05, 0.5, 0.95]) for s_ in smp])
    mu = np.array([ev['mu'][e][j] for e in order])
    lo, hi = np.array([ev['lo'][e][j] for e in order]), np.array([ev['hi'][e][j] for e in order])
    return smp, qs, mu, lo, hi


def _f11_handles(order):
    """Legend of F11v4 / F11h: LVK posterior (three of the event colours), one entry for the median (thicker) and
    5 / 95 % lines, network mu and interval."""
    pick = [order[0], order[len(order) // 2], order[-1]]
    lvk = tuple(Patch(facecolor=EVC[e], edgecolor=EVC[e], alpha=0.65) for e in pick)
    qln = (Line2D([], [], color=COL['lvk_line'], lw=F11_MEDIAN_LW), Line2D([], [], color=COL['lvk_line'], lw=F11_Q_LW))
    net = Line2D([], [], color=COL['network'], marker='o', ms=9, lw=2.6)
    return ([lvk, qln, net], ['LVK posterior (GWTC-3)', 'LVK median and 5 / 95 %',
                              'Network $\\mu$, 90 % conformal interval'],
            dict(handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}))


F11_MEDIAN_LW, F11_Q_LW = 3.0, 1.6


def _f11_numbers(order, qs, mu, lo, hi):
    return {e: dict(lvk_q05=qs[k, 0], lvk_median=qs[k, 1], lvk_q95=qs[k, 2], mu=mu[k], lower=lo[k], upper=hi[k])
            for k, e in enumerate(order)}


def f11v4(I, out):
    """Real events per variable: LVK half violins in the F11v2 event colours, median (thicker) and 5 % / 95 % as
    solid black lines across the violin, network mu and 90 % interval in colors.network."""
    ev = I.events()
    order = ev['order']
    x = np.arange(len(order))
    colors = [EVC[e] for e in order]
    ylims = FIGOPT['F11v4']['ylim']
    res = {}
    for j, lab in enumerate(LABELS):
        smp, qs, mu, lo, hi = _f11_values(ev, order, j)
        yl = [None if v is None else float(v) for v in (ylims.get(lab) or [None, None])]
        with slide_rc():
            fig, ax = plt.subplots(figsize=SLIDE_FULL, layout='constrained')
            parts = half_violins(ax, smp, x, colors)   # alpha 0.65 with edge, as F11v2
            for k, (b, x0) in enumerate(zip(parts['bodies'], x)):
                for yv, lw in ((qs[k, 1], F11_MEDIAN_LW), (qs[k, 0], F11_Q_LW), (qs[k, 2], F11_Q_LW)):
                    ax.plot([violin_left_edge(b, x0, yv), x0], [yv, yv], '-', color=COL['lvk_line'], lw=lw,
                            solid_capstyle='butt', zorder=3)
            ax.errorbar(x + 0.18, mu, yerr=[mu - lo, hi - mu], fmt='o', color=COL['network'], ms=9, capsize=4, lw=2.6,
                        zorder=4)
            ax.set_ylabel(var_label(lab))
            ax.set_ylim(*yl)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xticks(x)
            ax.set_xticklabels([short_event(e, order) for e in order], rotation=30, ha='right')
            for tl, e in zip(ax.get_xticklabels(), order):
                if e in ev['oos']:
                    tl.set_color(COL['neutral'])
            h, l_, kw = _f11_handles(order)
            leg = place_legend(ax, h, l_, **kw)
            files = save_slide(fig, out, NAME[f'F11v4_{lab}'], strict=True)
        top = ax.get_ylim()[1]
        res[f'F11v4_{lab}'] = dict(
            files=files, role=role(lab), legend=leg, numbers=_f11_numbers(order, qs, mu, lo, hi),
            rows=dict(events=order, order='LVK median chirp mass', ylim_config=yl),
            note='Grey tick label = out of training support: ' + json.dumps(ev['oos'], ensure_ascii=False)
                 + '. Violin colours = colors.events (as F11v2). Median / 5 % / 95 % lines span the drawn half-violin '
                 'width at that height. Above the upper y limit (partly not drawn): '
                 + json.dumps([e for k, e in enumerate(order) if max(qs[k, 2], hi[k], smp[k].max()) > top]))
    return res


def f11h(I, out):
    """F11v4 (chirp mass only) with the axes swapped: events on y (lightest LVK median on top), parameter on x;
    upward half violins, network mu and 90 % interval below each violin."""
    ev = I.events()
    order = ev['order']
    lab = 'chirp_mass'
    j = LABELS.index(lab)
    smp, qs, mu, lo, hi = _f11_values(ev, order, j)
    n = len(order)
    rank = np.empty(n, int)
    rank[np.argsort(qs[:, 1], kind='stable')] = np.arange(n)
    y = (n - 1 - rank).astype(float)            # lightest LVK median at the top
    xl = [None if v is None else float(v) for v in (FIGOPT['F11h']['xlim'].get(lab) or [None, None])]
    with slide_rc():
        fig, ax = plt.subplots(figsize=tuple(FIGOPT['F11h']['size']), layout='constrained')
        parts = half_violins(ax, smp, y, [EVC[e] for e in order], orientation='horizontal')
        for k, (b, y0) in enumerate(zip(parts['bodies'], y)):
            for xv, lw in ((qs[k, 1], F11_MEDIAN_LW), (qs[k, 0], F11_Q_LW), (qs[k, 2], F11_Q_LW)):
                ax.plot([xv, xv], [y0, violin_upper_edge(b, y0, xv)], '-', color=COL['lvk_line'], lw=lw,
                        solid_capstyle='butt', zorder=3)
        ax.errorbar(mu, y - 0.18, xerr=[mu - lo, hi - mu], fmt='o', color=COL['network'], ms=9, capsize=4, lw=2.6,
                    zorder=4)
        ax.set_xlabel(var_label(lab))
        ax.set_xlim(*xl)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.set_yticks(y)
        ax.set_yticklabels([short_event(e, order) for e in order])
        for tl, e in zip(ax.get_yticklabels(), order):
            if e in ev['oos']:
                tl.set_color(COL['neutral'])
        ax.set_ylim(-0.6, n - 1 + 0.6)
        h, l_, kw = _f11_handles(order)
        leg = place_legend(ax, h, l_, prefer=('lower right',), **kw)
        files = save_slide(fig, out, NAME['F11h'], strict=True)
    right_lim = ax.get_xlim()[1]
    return {'F11h': dict(
        files=files, role='test', legend=leg, numbers=_f11_numbers(order, qs, mu, lo, hi),
        rows=dict(events=order, order='LVK median chirp mass, lightest at the top', xlim_config=xl),
        note='Same values as F11v4_chirp_mass, axes swapped. Grey tick label = out of training support: '
             + json.dumps(ev['oos'], ensure_ascii=False) + '. Beyond the right x limit (partly not drawn): '
             + json.dumps([e for k, e in enumerate(order) if max(qs[k, 2], hi[k], smp[k].max()) > right_lim]))}


def f12v4(I, out):
    """F12v3 (three square panels) with the F12v2 LVK colours (colors.events: samples and 90 % contour), conformal
    2D ellipse in colors.network, mu in black; legend at a zero-overlap location of any panel (headroom if none)."""
    ev, R, real = I.events(), I.radii(), I.real()
    stored = {r['event']: r for r in real['ellipsoid_per_event']}
    order = ev['order']
    main = list(CFG['real_events']['main_events'])
    require(len(main) == 3 and all(e in order for e in main),
            f'cpan_2026.json real_events.main_events must be 3 of {order}: {main}')
    nums = {}
    with slide_rc():
        fig, axes = plt.subplots(1, 3, figsize=tuple(FIGOPT['F12v4']['size']), layout='constrained', sharey=False)
        for i, (ax, e) in enumerate(zip(axes, main)):
            k = order.index(e)   # same LVK subsample as F12 / F12_main / F12v2 / F12v3
            S, mu2, C2 = ev['samples'][e][:, :2], ev['mu'][e][:2], ev['cov_phys'][e][:2, :2]
            dm = np.median(S, axis=0) - mu2
            s2 = float(dm @ np.linalg.solve(C2, dm))
            I.check(np.isclose(s2, stored[e]['s_lvk_median_2D'], rtol=1e-5),
                    f'F12v4 {e}: Mahalanobis score of the LVK median reproduces the real-events JSON')
            sub = np.random.default_rng([SEED_PE_SUBSAMPLE, k]).choice(len(S), min(len(S), N_PE_SCATTER), replace=False)
            E = ellipse_xy(mu2, C2, R['2D'])
            pts = np.vstack([np.percentile(S, [0.5, 99.5], axis=0), E.T])
            a, b = pts.min(0), pts.max(0)
            pad = 0.08 * (b - a)
            ax.scatter(S[sub, 0], S[sub, 1], s=4, color=EVC[e], alpha=0.25, linewidths=0, rasterized=True)  # as F12v2
            hpd_contour(ax, S, LEVEL, EVC[e])
            ax.plot(E[0], E[1], color=COL['network'], lw=2.8)
            ax.plot(*mu2, 'o', color=COL['mu'], ms=9)
            ax.set(xlim=(a[0] - pad[0], b[0] + pad[0]), ylim=(a[1] - pad[1], b[1] + pad[1]))
            ax.set_box_aspect(1)
            ax.set_title(short_event(e, order) + (' (out of support)' if e in ev['oos'] else ''))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(True, axis='both')
            ax.set_xlabel(var_label('chirp_mass'))
            if i == 0:
                ax.set_ylabel(var_label('total_mass'))
            nums[e] = dict(mu=mu2, s_lvk_median_2D=s2, lvk_median_in_ellipse=s2 <= R['2D'])
        smp_h = tuple(Line2D([], [], ls='', marker='o', ms=7, color=EVC[e], alpha=0.6) for e in main)
        cnt_h = tuple(Line2D([], [], color=EVC[e], lw=1.8) for e in main)
        leg = place_legend(list(axes), [smp_h, cnt_h, Line2D([], [], color=COL['network'], lw=2.8),
                                        Line2D([], [], ls='', marker='o', color=COL['mu'], ms=9)],
                           ['LVK samples', 'LVK 90 % region', 'Network ellipse (90 %)', 'Network $\\mu$'],
                           grow=int(FIGOPT['F12v4']['legend_max_headroom_steps']),
                           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
        files = save_slide(fig, out, NAME['F12v4'], strict=True)
    return {'F12v4': dict(files=files, numbers=nums, role='main',
                          legend=dict(leg, panel=main[leg['axes_index']], zero_overlap=leg['overlap_score'] == 0),
                          rows=dict(events=main, selection='cpan_2026.json real_events.main_events',
                                    radius_2D_cal=R['2D'], lvk_scatter=f'{N_PE_SCATTER} samples, as F12'),
                          note='LVK samples (alpha 0.25) and 90 % contour in colors.events, as F12v2. Legend: first '
                               'location (all three panels) with overlap 0; if none, y headroom +10 % per step on '
                               'all panels (legend.headroom_steps); legend.overlap_score = remaining overlap.')}


def f13(I, out):
    """Short signals (predicted tau30 tercile 1 of the cal cuts, test; same subset and criteria as F09c): normalised
    errors (y - mu)/sigma of chirp mass vs total mass; grey inside the box of the two 90 % intervals, teal outside the
    box and inside the conformal 2D ellipse, dark grey outside both; bands of the two intervals, dashed box, ellipse
    (mean implied correlation of the group, as F09c)."""
    s7, d, q, R = I.s7(), I.pred('test'), I.intervals(), I.radii()
    I.check((d['mu'][:, 0] > 0).all(), 'F13: predicted chirp mass > 0 (tau30)')
    cuts = np.asarray(s7['criteria']['tau_tercile_cuts_cal'], float)
    m = np.digitize(tau30(d['mu'][:, 0]), cuts) == 0
    z = d['r_std'][:, :2] / d['sig_std'][:, :2]
    qb = np.array([q[lab]['r_sigma'] for lab in LABELS[:2]])
    in_box = np.all([hits(d, j, q[lab], 'r_sigma') for j, lab in enumerate(LABELS[:2])], axis=0)
    in_ell = maha(d, 2) <= R['2D']
    rho = d['cov'][:, 0, 1] / (d['sig_std'][:, 0] * d['sig_std'][:, 1])
    rbar = float(rho[m].mean())
    E = ellipse_xy(np.zeros(2), np.array([[1.0, rbar], [rbar, 1.0]]), R['2D'])
    cb, ce = float(in_box[m].mean()), float(in_ell[m].mean())
    cats = [('f13_in_box', in_box, 'inside the box', 6),
            ('f13_ellipse_only', in_ell & ~in_box, 'outside the box, inside the ellipse', 12),
            ('f13_outside', ~(in_box | in_ell), 'outside both', 12)]
    lim = float(FIGOPT['F13']['lim'])
    zz = z[m]
    n_beyond = int(np.sum(np.any(np.abs(zz) > lim, axis=1)))
    with slide_rc():
        fig, ax = plt.subplots(figsize=tuple(FIGOPT['F13']['size']), layout='constrained')
        band = dict(color=COL['f13_band'], alpha=ALPHA['f13_band'], lw=0, zorder=0.1)
        ax.axvspan(-qb[0], qb[0], **band)
        ax.axhspan(-qb[1], qb[1], **band)
        handles = []
        for key, sel, lab_, size in cats:
            ss = sel[m]
            ax.scatter(zz[ss, 0], zz[ss, 1], s=size, color=COL[key], linewidths=0, rasterized=True,
                       zorder=2 if key == 'f13_in_box' else 3)
            handles.append(Line2D([], [], ls='', marker='o', ms=7, color=COL[key], label=lab_))
        ax.add_patch(Rectangle(-qb, 2 * qb[0], 2 * qb[1], fill=False, ec=COL['f13_box'], lw=2.2, ls='--', zorder=4))
        ax.plot(E[0], E[1], color=COL['network'], lw=2.8, zorder=5)
        handles += [Line2D([], [], color=COL['f13_box'], lw=2.2, ls='--',
                           label=f'box of the two 90 % intervals: coverage {cb:.3f}'),
                    Line2D([], [], color=COL['network'], lw=2.8, label=f'conformal 2D ellipse: coverage {ce:.3f}')]
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(7, integer=True))
        soft_grid(ax)
        ax.set_xlabel('$(y-\\mu)/\\sigma$ of ' + var_label('chirp_mass', unit=False))
        ax.set_ylabel('$(y-\\mu)/\\sigma$ of ' + var_label('total_mass', unit=False))
        leg = place_legend(ax, handles, locs=('upper left',))
        files = save_slide(fig, out, NAME['F13'], strict=True)
    return {'F13': dict(
        files=files, role='optional', legend=leg,
        numbers=dict(n_rows_group=int(m.sum()), coverage_2d_group=dict(box=cb, ellipse=ce),
                     mean_implied_correlation_group=rbar, box_half_widths_r_over_sigma=qb, radius_2D_cal=R['2D'],
                     axis_limit=lim, points_beyond_axes=n_beyond,
                     fractions_group={k: float(s_[m].mean()) for k, s_, _, _ in cats}),
        rows=dict(split='test', n=d['n'], group='predicted tau30 below the first M14.7 tercile cut (cal), as F09c',
                  tau_tercile_cuts_cal=cuts),
        note='Same subset, criteria and drawn ellipse as F09c. Point colours use the per-row criteria (row Sigma for '
             'the ellipse); the drawn ellipse uses the mean implied correlation of the group. Points beyond the axes '
             'are not drawn (counted). No text in the figure besides the legend.')}


def hex_overlap(hb, leg, min_count=1):
    """Hexbin cells with count > min_count whose drawn hexagon touches the legend box (display coordinates;
    hexagon = transform(path vertices) + offset_transform(cell offset), as _obstacles)."""
    fig = leg.figure
    fig.draw_without_rendering()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    cnt = np.asarray(hb.get_array(), float)
    offs = np.asarray(hb.get_offsets(), float)[cnt > min_count]
    if not len(offs):
        return 0
    hexv = hb.get_transform().transform(np.asarray(hb.get_paths()[0].vertices, float))   # (k, 2)
    ctr = hb.get_offset_transform().transform(offs)                                     # (n, 2)
    v = ctr[:, None, :] + hexv[None, :, :]   # transData(hexagon around 0) + AffineDelta(cell centre)
    x0, x1 = v[..., 0].min(1), v[..., 0].max(1)
    y0, y1 = v[..., 1].min(1), v[..., 1].max(1)
    return int(np.sum((x1 >= bb.x0) & (x0 <= bb.x1) & (y1 >= bb.y0) & (y0 <= bb.y1)))


def f06m_v4(I, out):
    """F06mv3 without the RMS(sigma)/RMSE text: legend inside at the upper left (checked against hexbin cells with
    count > 1, stored in the manifest); marginals share the main axis range, histogram scale fitted to the marginal;
    axis range 0 to figure_options.F06mv4.fixed_limit (chirp mass) or max(percentile of |y - mu|, max sigma)."""
    d, cuts = I.pred('test'), I.snr_cuts()
    t = np.digitize(I.snr('test'), cuts)
    opt = FIGOPT['F06mv4']
    pct = float(opt['abs_error_percentile'])
    gs = int(round(HEX_FINE * 50))
    res = {}
    for j, lab in enumerate(LABELS):
        s, a = d['sigma'][:, j], np.abs(d['y'][:, j] - d['mu'][:, j])
        g = quantile_groups(s, 20)
        cnt = np.bincount(g, minlength=20)
        rs, rr = np.sqrt(np.bincount(g, s ** 2, 20) / cnt), np.sqrt(np.bincount(g, a ** 2, 20) / cnt)
        ratio = [float(np.sqrt(np.mean(s[t == k] ** 2)) / np.sqrt(np.mean(a[t == k] ** 2))) for k in range(3)]
        fixed = opt['fixed_limit'].get(lab)
        if fixed is not None:
            L, rule = float(fixed), 'cpan_2026.json figure_options.F06mv4.fixed_limit'
        else:
            L, rule = float(max(np.percentile(a, pct), s.max())), f'max(percentile {pct:g} of |y - mu|, max sigma)'
        inside = (a <= L) & (s <= L)
        bins = np.linspace(0.0, L, 61)
        with slide_rc():
            fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
            ax, top, right, cax = marginal_axes(fig, *MARGINAL_RATIOS)   # top sharex, right sharey with ax
            hb = ax.hexbin(s[inside], a[inside], gridsize=gs, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'],
                           linewidths=0.0, extent=(0.0, L, 0.0, L))
            ref, = ax.plot([0.0, L], [0.0, L], '--', color=COL['refline'], lw=1.8, label='$y = x$')
            grp, = ax.plot(rs, rr, 'o-', color=COL['network'], ms=8, lw=2.4, mec=COL['marker_edge'], mew=0.8,
                           label='RMS $|y-\\mu|$ in 20 groups of $\\sigma$')
            ax.set(xlim=(0.0, L), ylim=(0.0, L))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_xlabel('Predicted $\\sigma$' + unit_sfx(lab))
            ax.set_ylabel('$|y-\\mu|$' + unit_sfx(lab))
            hmax = 0.0
            for m_, o in ((top, 'vertical'), (right, 'horizontal')):
                n1, _, _ = m_.hist(s[s <= L], bins=bins, histtype='stepfilled', color=COL['true_fill'],
                                   alpha=ALPHA['true_fill'], edgecolor='none', lw=0, orientation=o)
                n2, _, _ = m_.hist(a[a <= L], bins=bins, histtype='step', color=COL['pred_line'], lw=2.2,
                                   orientation=o)
                hmax = max(hmax, float(np.max(n1)), float(np.max(n2)))
            top.set_ylim(0.0, 1.05 * hmax)      # histogram scale fitted to the marginal (both marginals)
            right.set_xlim(0.0, 1.05 * hmax)
            ax.set(xlim=(0.0, L), ylim=(0.0, L))   # shared range re-asserted after the marginal histograms
            soft_grid(ax, top, right)
            fig.colorbar(hb, cax=cax).set_label('Count')
            leg = place_legend(ax, [ref, grp,
                                    Patch(facecolor=COL['true_fill'], alpha=ALPHA['true_fill'], edgecolor='none',
                                          label='predicted $\\sigma$'),
                                    Line2D([], [], color=COL['pred_line'], lw=2.2, label='$|y - \\mu|$')],
                               locs=('upper left',))
            leg['hex_cells_count_gt1_under_legend'] = hex_overlap(hb, ax.get_legend(), 1)
            files = save_slide(fig, out, NAME[f'F06mv4_{lab}'], strict=True)
        res[f'F06mv4_{lab}'] = dict(
            files=files, role=role(lab), legend=leg, hexbin_gridsize=gs,
            rows=dict(split='test', n=d['n'], snr_tercile_cuts=cuts, snr='target_network_snr (nominal)'),
            numbers=dict(rms_sigma_by_group=rs, rms_abs_error_by_group=rr, rows_per_group=cnt,
                         rms_sigma_over_rmse_by_snr_tercile=dict(zip(SNR_NAMES, ratio)),
                         rows_per_snr_tercile=np.bincount(t, minlength=3),
                         axis_limit=L, axis_limit_rule=rule,
                         n_outside_axes=int((~inside).sum()), n_abs_error_above_limit=int((a > L).sum()),
                         n_sigma_above_limit=int((s > L).sum()),
                         group_points_outside_axes=int(np.sum((rs > L) | (rr > L))),
                         marginal_count_limit=1.05 * hmax),
            note='Same values as F06mv3 (RMS(sigma)/RMSE by SNR tercile kept here, not drawn). Linear axes, same range '
                 '[0, axis_limit] on x and y; marginals share it (top sharex, right sharey), count axis 0 to 1.05 x '
                 'the largest bin of both marginals. Rows outside the axes are not drawn (counted). '
                 'legend.hex_cells_count_gt1_under_legend = hexbin cells with count > 1 touching the legend box.')
    return res


def f13_v2(I, out):
    """F13 with a larger legend (markers x 2, coverages in the labels), blue 'outside box, inside ellipse' points
    and 'error in X / sigma(X)' axis labels; same subset, criteria and values as F13 / F09c."""
    s7, d, q, R = I.s7(), I.pred('test'), I.intervals(), I.radii()
    I.check((d['mu'][:, 0] > 0).all(), 'F13v2: predicted chirp mass > 0 (tau30)')
    cuts = np.asarray(s7['criteria']['tau_tercile_cuts_cal'], float)
    m = np.digitize(tau30(d['mu'][:, 0]), cuts) == 0
    z = d['r_std'][:, :2] / d['sig_std'][:, :2]
    qb = np.array([q[lab]['r_sigma'] for lab in LABELS[:2]])
    in_box = np.all([hits(d, j, q[lab], 'r_sigma') for j, lab in enumerate(LABELS[:2])], axis=0)
    in_ell = maha(d, 2) <= R['2D']
    rho = d['cov'][:, 0, 1] / (d['sig_std'][:, 0] * d['sig_std'][:, 1])
    rbar = float(rho[m].mean())
    E = ellipse_xy(np.zeros(2), np.array([[1.0, rbar], [rbar, 1.0]]), R['2D'])
    cb, ce = float(in_box[m].mean()), float(in_ell[m].mean())
    cats = [('f13_in_box', in_box, 'inside the box', 6),
            ('f13v2_ellipse_only', in_ell & ~in_box, 'outside box, inside ellipse', 12),
            ('f13_outside', ~(in_box | in_ell), 'outside both', 12)]
    lim = float(FIGOPT['F13']['lim'])
    zz = z[m]
    n_beyond = int(np.sum(np.any(np.abs(zz) > lim, axis=1)))
    ms = 2 * 7   # legend markers x 2 (F13: 7)
    with slide_rc():
        fig, ax = plt.subplots(figsize=tuple(FIGOPT['F13']['size']), layout='constrained')
        band = dict(color=COL['f13_band'], alpha=ALPHA['f13_band'], lw=0, zorder=0.1)
        ax.axvspan(-qb[0], qb[0], **band)
        ax.axhspan(-qb[1], qb[1], **band)
        handles = []
        for key, sel, lab_, size in cats:
            ss = sel[m]
            ax.scatter(zz[ss, 0], zz[ss, 1], s=size, color=COL[key], linewidths=0, rasterized=True,
                       zorder=2 if key == 'f13_in_box' else 3)
            handles.append(Line2D([], [], ls='', marker='o', ms=ms, color=COL[key], label=lab_))
        ax.add_patch(Rectangle(-qb, 2 * qb[0], 2 * qb[1], fill=False, ec=COL['f13_box'], lw=2.2, ls='--', zorder=4))
        ax.plot(E[0], E[1], color=COL['network'], lw=2.8, zorder=5)
        handles += [Line2D([], [], color=COL['f13_box'], lw=2.2, ls='--',
                           label=f'box of the two 90 % intervals ({100 * cb:.1f} %)'),
                    Line2D([], [], color=COL['network'], lw=2.8,
                           label=f'conformal ellipse from $\\Sigma$ ({100 * ce:.1f} %)')]
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(7, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(7, integer=True))
        soft_grid(ax)
        vm, vt = var_label('chirp_mass', unit=False), var_label('total_mass', unit=False)
        ax.set_xlabel(f'error in {vm} / $\\sigma$({vm})')
        ax.set_ylabel(f'error in {vt} / $\\sigma$({vt})')
        leg = place_legend(ax, handles, locs=('upper left',))   # overlap_score = points (and lines) under the box
        files = save_slide(fig, out, NAME['F13v2'], strict=True)
    return {'F13v2': dict(
        files=files, role='optional', legend=leg,
        numbers=dict(n_rows_group=int(m.sum()), coverage_2d_group=dict(box=cb, ellipse=ce),
                     mean_implied_correlation_group=rbar, box_half_widths_r_over_sigma=qb, radius_2D_cal=R['2D'],
                     axis_limit=lim, points_beyond_axes=n_beyond,
                     fractions_group={k: float(s_[m].mean()) for k, s_, _, _ in cats}),
        rows=dict(split='test', n=d['n'], group='predicted tau30 below the first M14.7 tercile cut (cal), as F09c',
                  tau_tercile_cuts_cal=cuts),
        note='Same values as F13 / F09c. Legend coverages = coverage_2d_group (computed here). legend.overlap_score: '
             'drawn points and densified lines inside the legend box (0 = no overlap).')}


def f06c(I, out, fid='F06c', lines=None):
    """Cone (test): predicted sigma (x, 0 to L) vs signed error y - mu (y, -L to L); finer hexbin and soft grid as
    F10mv3; lines y = +-sigma and y = +-q sigma (q = single-bin r/sigma conformal quantile of cal, I.intervals());
    fractions inside +-sigma and +-q sigma in the legend and the manifest; marginals sigma (top) and y - mu (right).
    lines: style of the +-sigma / +-q sigma lines (None = F06c; F06cv2 passes thicker lines with a white outline)."""
    lines = lines or dict(sig=dict(color=COL['refline'], lw=1.8), q=dict(color=COL['network'], lw=2.4))
    d, q = I.pred('test'), I.intervals()
    opt = FIGOPT['F06c']
    pct = float(opt['abs_error_percentile'])
    gs = int(round(HEX_FINE * 55))   # as F10mv3
    res = {}
    for j, lab in enumerate(LABELS):
        s, e = d['sigma'][:, j], d['y'][:, j] - d['mu'][:, j]
        a = np.abs(e)
        qs = float(q[lab]['r_sigma'])
        in1 = a <= s
        inq = a <= qs * s
        inq_method = hits(d, j, q[lab], 'r_sigma')   # the method's own criterion (standardized units)
        f1, fq = float(in1.mean()), float(inq.mean())
        I.check(abs(fq - float(inq_method.mean())) <= COV_ATOL,
                f'{fid} {lab}: fraction inside +-q sigma (physical) equals the r/sigma interval coverage (standardized)')
        fixed = opt['fixed_limit'].get(lab)
        if fixed is not None:
            L, rule = float(fixed), 'cpan_2026.json figure_options.F06c.fixed_limit'
        else:
            L, rule = float(max(np.percentile(a, pct), s.max())), f'max(percentile {pct:g} of |y - mu|, max sigma)'
        inside = (s <= L) & (a <= L)
        with slide_rc():
            fig = plt.figure(figsize=SLIDE_SIZE, layout='constrained')
            ax, top, right, cax = marginal_axes(fig, *MARGINAL_RATIOS)   # top sharex, right sharey with ax
            hb = ax.hexbin(s[inside], e[inside], gridsize=gs, mincnt=1, norm=LogNorm(), cmap=CMAP['hexbin'],
                           linewidths=0.0, extent=(0.0, L, -L, L))
            xs = np.array([0.0, L])
            l1, = ax.plot(xs, xs, '--', **lines['sig'], label=f'inside $\\pm\\sigma$: {100 * f1:.1f} %')
            ax.plot(xs, -xs, '--', **lines['sig'])
            lq, = ax.plot(xs, qs * xs, '-', **lines['q'], label=f'inside $\\pm q\\sigma$: {100 * fq:.1f} %')
            ax.plot(xs, -qs * xs, '-', **lines['q'])
            ax.set(xlim=(0.0, L), ylim=(-L, L))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(7, symmetric=True))
            ax.set_xlabel('Predicted $\\sigma$' + unit_sfx(lab))
            ax.set_ylabel('$y-\\mu$' + unit_sfx(lab))
            nt, _, _ = top.hist(s[s <= L], bins=np.linspace(0.0, L, 61), histtype='stepfilled',
                                color=COL['true_fill'], alpha=ALPHA['true_fill'], edgecolor='none', lw=0)
            nr, _, _ = right.hist(e[a <= L], bins=np.linspace(-L, L, 81), histtype='step', color=COL['pred_line'],
                                  lw=2.2, orientation='horizontal')
            top.set_ylim(0.0, 1.05 * float(nt.max()))
            right.set_xlim(0.0, 1.05 * float(nr.max()))
            ax.set(xlim=(0.0, L), ylim=(-L, L))
            soft_grid(ax, top, right)
            fig.colorbar(hb, cax=cax).set_label('Count')
            leg = place_legend(ax, [l1, lq], locs=('upper left',))
            leg['hex_cells_count_gt1_under_legend'] = hex_overlap(hb, ax.get_legend(), 1)
            files = save_slide(fig, out, NAME[f'{fid}_{lab}'], strict=True)
        res[f'{fid}_{lab}'] = dict(
            files=files, role=role(lab), legend=leg, hexbin_gridsize=gs,
            rows=dict(split='test', n=d['n']),
            numbers=dict(q_r_sigma_cal=qs, k_rank_cal=q[lab]['k'], fraction_inside_1sigma=f1,
                         fraction_inside_q_sigma=fq, n_inside_1sigma=int(in1.sum()), n_inside_q_sigma=int(inq.sum()),
                         axis_limit=L, axis_limit_rule=rule, n_outside_axes=int((~inside).sum()),
                         n_abs_error_above_limit=int((a > L).sum()), n_sigma_above_limit=int((s > L).sum())),
            note='q = single-bin symmetric r/sigma conformal quantile on cal (the selected M14.7 interval). Fractions '
                 'over all test rows (also those outside the axes). Rows outside the axes are not drawn (counted). '
                 'Marginals: top = predicted sigma (60 bins on [0, L]), right = y - mu (80 bins on [-L, L]); count '
                 'axis to 1.05 x the largest bin. legend.hex_cells_count_gt1_under_legend = hexbin cells with count '
                 '> 1 touching the legend box.')
    return res


def f06c_v2(I, out):
    """F06c with more visible lines: +-sigma dark grey dashed, +-q sigma colors.network solid, white outline."""
    from matplotlib import patheffects as pe
    o = FIGOPT['F06cv2']

    def outline(lw):
        return [pe.Stroke(linewidth=lw + float(o['outline_extra_lw']), foreground=COL['background']), pe.Normal()]

    lines = dict(sig=dict(color=COL['sigma_line_dark'], lw=float(o['sigma_lw']), path_effects=outline(o['sigma_lw'])),
                 q=dict(color=COL['network'], lw=float(o['q_lw']), path_effects=outline(o['q_lw'])))
    res = f06c(I, out, fid='F06cv2', lines=lines)
    for r in res.values():
        r['note'] = (f"As F06c (same values and checks); lines: +-sigma {COL['sigma_line_dark']} dashed lw {o['sigma_lw']}, "
                     f"+-q sigma colors.network solid lw {o['q_lw']}, both with a white outline (lw + "
                     f"{o['outline_extra_lw']}). " + r['note'])
    return res


SEED_F16, N_BOOT_F16, N_GROUPS_F16 = 20261117, 1000, 10


def _error_curves(I, out, fid, by):
    """Test: RMSE and RMS(predicted sigma) in 10 equal-size groups of `by` (network SNR or true value), x = group
    median; 95 % bootstrap band of the RMSE (rows resampled within each group, B = N_BOOT_F16, seed SEED_F16;
    one generator, labels then groups in order). Legend in the first panel only."""
    d = I.pred('test')
    snr = I.snr('test') if by == 'snr' else None
    rng = np.random.default_rng(SEED_F16)
    nums = {}
    with slide_rc():
        fig, axes = plt.subplots(1, 3, figsize=tuple(FIGOPT['F16']['size']), layout='constrained')
        for j, (ax, lab) in enumerate(zip(axes, LABELS)):
            x = snr if by == 'snr' else d['y'][:, j]
            e2 = (d['y'][:, j] - d['mu'][:, j]) ** 2
            s2 = d['sigma'][:, j] ** 2
            g = quantile_groups(x, N_GROUPS_F16)
            xm = np.array([np.median(x[g == k]) for k in range(N_GROUPS_F16)])
            cnt = np.bincount(g, minlength=N_GROUPS_F16)
            rmse = np.sqrt(np.bincount(g, e2, N_GROUPS_F16) / cnt)
            rms_s = np.sqrt(np.bincount(g, s2, N_GROUPS_F16) / cnt)
            ci = np.empty((N_GROUPS_F16, 2))
            for k in range(N_GROUPS_F16):
                ek = e2[g == k]
                bs = np.sqrt(ek[rng.integers(0, len(ek), (N_BOOT_F16, len(ek)))].mean(1))
                ci[k] = np.percentile(bs, [2.5, 97.5])
            band = ax.fill_between(xm, ci[:, 0], ci[:, 1], color=COL['network'], alpha=ALPHA['f16_band'], lw=0,
                                   zorder=1)
            ln, = ax.plot(xm, rmse, 'o-', color=COL['network'], ms=8, lw=2.4, zorder=3)
            ls, = ax.plot(xm, rms_s, '--', color=COL['rms_sigma'], lw=2.4, zorder=2)
            ax.set_xlabel('network SNR' if by == 'snr' else 'True ' + var_label(lab))
            ax.set_ylabel('error in ' + var_label(lab))
            ax.set_ylim(0.0, 1.12 * float(max(ci[:, 1].max(), rms_s.max())))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            if j == 0:
                h = [(band, ln), ls]
            nums[lab] = dict(group_median_x=xm, rows_per_group=cnt, rmse=rmse, rmse_ci95=ci, rms_sigma=rms_s,
                             rms_sigma_over_rmse=rms_s / rmse)
        leg = place_legend(axes[0], h, ['RMSE (95 % bootstrap)', 'RMS of predicted $\\sigma$'], grow=4,
                           handler_map={tuple: HandlerTuple(ndivide=1)})
        files = save_slide(fig, out, NAME[fid], strict=True)
    return {fid: dict(
        files=files, role='main' if by == 'snr' else 'appendix', legend=leg, numbers=nums,
        rows=dict(split='test', n=d['n'], x='target_network_snr (nominal network SNR)' if by == 'snr'
                  else 'true value of each parameter', groups=f'{N_GROUPS_F16} equal-size quantile groups, x = median'),
        bootstrap=dict(B=N_BOOT_F16, seed=SEED_F16, level=0.95, scheme='rows resampled with replacement within each '
                       'group; percentile interval; one generator, labels then groups in order'),
        note='RMSE = sqrt(mean (y - mu)^2); RMS(sigma) = sqrt(mean sigma^2) of the predicted marginal sigma.')}


def f16(I, out):
    return _error_curves(I, out, 'F16', 'snr')


def f16b(I, out):
    return _error_curves(I, out, 'F16b', 'truth')


def table_test(I, out):
    """Numbers for the talk table (no figure), test, per parameter: RMSE, RMS(sigma)/RMSE by SNR tercile (val cuts,
    as F06m), fraction inside +-sigma, coverage and mean width of the selected 90 % conformal interval, cal q.
    Coverage and width checked against the M14.7 test record and against the values quoted on the slide."""
    d, q, s7, cuts = I.pred('test'), I.intervals(), I.s7(), I.snr_cuts()
    t = np.digitize(I.snr('test'), cuts)
    tab = {}
    for j, lab in enumerate(LABELS):
        e = d['y'][:, j] - d['mu'][:, j]
        s = d['sigma'][:, j]
        h, w = hits(d, j, q[lab], 'r_sigma'), widths(d, j, q[lab], 'r_sigma')
        ref = next(g for g in s7['test'] if g['role'] == 'selected' and g['label'] == lab)
        I.check(abs(h.mean() - ref['coverage']) <= COV_ATOL and np.isclose(w.mean(), ref['mean_width_phys'], rtol=Q_RTOL),
                f'table_test {lab}: test coverage and mean width reproduce M14.7 test (selected)')
        cov_q, w_q, nd = EXPECTED_TABLE[lab]
        I.check(f'{h.mean():.3f}' == f'{cov_q:.3f}' and f'{w.mean():.{nd}f}' == f'{w_q:.{nd}f}',
                f'table_test {lab}: coverage rounds to {cov_q:.3f} and mean width to {w_q:.{nd}f}')
        rmse = float(np.sqrt(np.mean(e ** 2)))
        ratio = [float(np.sqrt(np.mean(s[t == k] ** 2)) / np.sqrt(np.mean(e[t == k] ** 2))) for k in range(3)]
        tab[lab] = dict(rmse=rmse, unit=CFG['variables'][lab]['unit'] or None,
                        rms_sigma_over_rmse_by_snr_tercile=dict(zip(SNR_NAMES, ratio)),
                        fraction_inside_1sigma=float(np.mean(np.abs(e) <= s)),
                        coverage_90_conformal=float(h.mean()), mean_width_90_conformal=float(w.mean()),
                        q_r_over_sigma_cal=float(q[lab]['r_sigma']), k_rank_cal=q[lab]['k'])
    return {'table_test': dict(
        files=[], role='table', table_test=tab,
        rows=dict(split='test', n=d['n'], snr_tercile_cuts_val=cuts, rows_per_snr_tercile=np.bincount(t, minlength=3)),
        note='Same arrays as F06c / F16 (test npz of the final model). Interval: single bin, symmetric, score '
             'r/sigma, cal-only quantile q (M14.7 selection); width = 2 q sigma.')}


EXPECTED_TABLE = dict(chirp_mass=(0.904, 13.72, 2), total_mass=(0.901, 25.05, 2), chi_eff=(0.904, 0.513, 3))
EPOCH_LINE = re.compile(r'^epoch (\d+)/(\d+) \| .*?\| train (-?[\d.]+) \| val (-?[\d.]+)( \*)? .*?\| elapsed ([\d.]+) min')


def training_logs(I):
    """Saved training artifacts of the final model (nothing is re-run): history.npz and run_info.json next to the
    predictions (results/M13_3/<TAG>/, as the val npz checkpoint and the log record) and logs/m14/train_<TAG>.log."""
    if I.cached('training'):
        return I._cache['training']
    start = len(I.used)
    ckpt = Path(I.pred('val')['ckpt'])
    rdir = I.data_root / 'results' / 'M13_3' / TAG
    info = I.json(rdir / 'run_info.json', 'training run_info.json')
    I.check(info.get('tag') == TAG and info['config']['outputs']['checkpoint_tag'] == TAG,
            f'run_info.json: tag {TAG}')
    hp = I.exists(rdir / 'history.npz', 'training history.npz')
    with np.load(hp, allow_pickle=False) as z:
        H = {k: np.asarray(z[k]) for k in z.files}
    I.check([str(x) for x in H['label_names']] == list(LABELS), 'history.npz: label order')
    logp = I.exists(I.data_root / 'logs' / 'm14' / f'train_{TAG}.log', 'training log')
    lines = logp.read_text(errors='replace').splitlines()
    I.check(any(str(hp) in ln and ln.startswith('saved') for ln in lines)
            and any(str(rdir / 'run_info.json') in ln for ln in lines),
            'training log: wrote this history.npz and run_info.json')
    ep = [m.groups() for m in map(EPOCH_LINE.match, lines) if m]
    n = len(H['epoch'])
    I.check(len(ep) == n and [int(e[0]) for e in ep] == list(H['epoch'].astype(int)),
            'training log: one epoch line per history.npz epoch')
    I.check(np.allclose([float(e[2]) for e in ep], H['train_loss'], atol=6e-6)
            and np.allclose([float(e[3]) for e in ep], H['val_loss'], atol=6e-6),
            'training log: train / val loss equal history.npz (5 decimals)')
    elapsed = np.array([float(e[5]) for e in ep])
    reg = info['procedure'].get('registered_model')
    if ckpt.name == f'{TAG}_final.pt' and reg == 'last epoch':
        chosen, crit = int(H['epoch'][-1]), 'last epoch (run_info procedure.registered_model; checkpoint _final.pt)'
    elif ckpt.name == f'{TAG}_bestval.pt':
        chosen, crit = int(H['epoch'][int(np.argmin(H['val_loss']))]), 'minimum val loss (checkpoint _bestval.pt)'
    else:
        I.check(False, f'chosen epoch: checkpoint {ckpt.name} with registered_model {reg!r} not understood')
    nr = 'no registrado'
    per_epoch_s = H['train_time'] + H['val_time'] + H.get('train40k_time', 0.0) + H.get('data_wait', 0.0)
    d = dict(H=H, chosen=chosen, run_info=info, info=dict(
        n_epochs=n, max_epochs_config=info['config']['training']['max_epochs'], chosen_epoch_0based=chosen,
        chosen_criterion=crit, checkpoint=str(ckpt), best_val_loss_epoch_0based=int(H['epoch'][np.argmin(H['val_loss'])]),
        dropout_off_epoch_0based=info['procedure'].get('dropout_off_epoch_0based', nr),
        time_total_min_log_elapsed=float(elapsed[-1]), time_mean_per_epoch_min_log_elapsed=float(elapsed[-1] / n),
        time_mean_per_epoch_s_history=float(np.mean(per_epoch_s)),
        time_note='log lines carry the cumulative "elapsed" (min) only; no wall-clock time per epoch is logged',
        run_start_timestamp=info.get('timestamp', nr), wall_clock_end=nr,
        gpu=info.get('gpu', nr), cuda_visible_devices=info.get('cuda_visible_devices', nr),
        cuda=(info.get('versions') or {}).get('cuda', nr), host=info.get('host', nr),
        n_train=info['data'].get('n_train', nr), n_val=info['data'].get('n_val', nr),
        n_train_monitor=(info['data'].get('train_monitor') or {}).get('n', nr),
        loss_train=info['procedure'].get('loss', nr),
        loss_val='val_loss = mean over labels of the val MSE of mu (standardized labels); the log line and '
                 'train_m13_3.py (val_loss = val_mse.mean()) — not the beta-NLL',
        files=dict(history=str(hp), run_info=str(rdir / 'run_info.json'), log=str(logp))))
    return I.store('training', d, start)


def f17(I, out):
    """Training of the final model from its saved history: left train beta-NLL and val loss (as logged) vs epoch,
    right val RMSE / final val RMSE per parameter; thin vertical line at the chosen checkpoint epoch."""
    T = training_logs(I)
    H, ch = T['H'], T['chosen']
    ep = H['epoch'].astype(int)
    rel = np.sqrt(H['val_mse']) / np.sqrt(H['val_mse'][-1])
    with slide_rc():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=tuple(FIGOPT['F17']['size']), layout='constrained')
        lt, = a1.plot(ep, H['train_loss'], '-', color=COL['train_curve'], lw=2.4)
        a1b = a1.twinx()
        a1b.spines['right'].set_visible(True)
        lv, = a1b.plot(ep, H['val_loss'], '-', color=COL['network'], lw=2.4)
        a1.set_xlabel('epoch')
        a1.set_ylabel('train $\\beta$-NLL', color=COL['train_curve'])
        a1b.set_ylabel('val MSE (standardized)', color=COL['network'])
        a1b.grid(False)
        for a in (a1, a1b, a2):
            a.axvline(ch, color=COL['chosen_epoch'], lw=1.0, zorder=0.6)
            a.yaxis.set_major_locator(MaxNLocator(5))
        hs = [a2.plot(ep, rel[:, j], '-', color=COL[f'param_{lab}'], lw=2.4)[0] for j, lab in enumerate(LABELS)]
        a2.set(xlabel='epoch', ylabel='RMSE / final RMSE')
        mk = Line2D([], [], color=COL['chosen_epoch'], lw=1.0)
        leg1 = place_legend(a1b, [lt, lv, mk], ['train $\\beta$-NLL', 'val MSE', 'chosen checkpoint'], also=[a1],
                            grow=4)
        leg2 = place_legend(a2, hs + [mk], [var_label(lab, unit=False) for lab in LABELS] + ['chosen checkpoint'],
                            grow=4)
        files = save_slide(fig, out, NAME['F17'], strict=True)
    return {'F17': dict(
        files=files, role='main', legend=dict(left=leg1, right=leg2), training=T['info'],
        numbers=dict(final_val_rmse_standardized=dict(zip(LABELS, np.sqrt(H['val_mse'][-1]))),
                     val_rmse_over_final_at_epoch0=dict(zip(LABELS, rel[0]))),
        note='Read from the saved history.npz (log lines checked against it); nothing re-run. Left: train loss = '
             'beta-NLL of the training batches (dropout and noise augmentation on); val loss = mean val MSE of mu in '
             'standardized units, as logged (the val beta-NLL is not logged), on its own axis. Right: sqrt(val MSE) '
             'per parameter divided by its last-epoch value.')}


# ---- training code of the final model (read as text, never imported or run): loss_definition, F17v2 -----------
TRAIN_SCRIPT_REL = 'scripts/m13/train_m13_3.py'
NETWORK_REL = 'src/models/network.py'


def _find_line(I, lines, needle, what, lo=1, hi=None):
    """1-based number of the first line in [lo, hi] containing `needle` (check: it exists)."""
    hi = len(lines) if hi is None else hi
    hit = next((i for i in range(lo, hi + 1) if needle in lines[i - 1]), None)
    I.check(hit is not None, f'{what}: line with {needle!r} in lines {lo}-{hi}')
    return hit


def training_code(I):
    """Locate, in the training script and network module whose sha256 run_info.json records, the lines that define
    the loss, the covariance parametrization and the logged metrics (ast for function ranges; text for lines)."""
    if I.cached('training_code'):
        return I._cache['training_code']
    import ast
    start = len(I.used)
    info = training_logs(I)['run_info']
    tcfg, mcfg = info['config']['training'], info['config']['model']
    src = {}
    for rel in (TRAIN_SCRIPT_REL, NETWORK_REL):
        p = I.exists(ROOT / rel, f'training source {rel}')
        I.check(I.sha(p) == info['sha256'][rel], f'{rel}: sha256 equals run_info.json (the code that trained)')
        txt = p.read_text()
        fn = {}
        for node in ast.walk(ast.parse(txt)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                fn.setdefault(node.name, (node.lineno, node.end_lineno))
        src[rel] = dict(lines=txt.splitlines(), fn=fn)
    T, N = src[TRAIN_SCRIPT_REL], src[NETWORK_REL]

    def tl(needle, f=None):
        lo, hi = T['fn'][f] if f else (1, None)
        return _find_line(I, T['lines'], needle, TRAIN_SCRIPT_REL + (f' {f}()' if f else ''), lo, hi)

    def nl(needle, f=None):
        lo, hi = N['fn'][f] if f else (1, None)
        return _find_line(I, N['lines'], needle, NETWORK_REL + (f' {f}()' if f else ''), lo, hi)

    I.check(tcfg['loss'] == 'beta_nll_full' and mcfg.get('covariance') == 'full' and mcfg.get('heteroscedastic'),
            'run_info config: loss beta_nll_full, model covariance full, heteroscedastic')
    I.check('beta_nll_full_loss' in T['fn'] and 'full_covariance_from_output' in N['fn'],
            'training code: beta_nll_full_loss and full_covariance_from_output defined')
    L = dict(
        loss_fn=T['fn']['beta_nll_full_loss'],
        loss_residual=tl('r = y - out[:, :3]', 'beta_nll_full_loss'),
        loss_log_d=tl('log_d = out[:, 3:6]', 'beta_nll_full_loss'),
        loss_e1=tl('e1 = r[:, 0] * inv_d[:, 0]', 'beta_nll_full_loss'),
        loss_e3=tl('e3 = (r[:, 2] - l31 * e1 - l32 * e2) * inv_d[:, 2]', 'beta_nll_full_loss'),
        loss_nll=tl('nll = 0.5 * e ** 2 + log_d', 'beta_nll_full_loss'),
        loss_beta=tl('nll = nll * torch.exp(2.0 * beta * log_d).detach()', 'beta_nll_full_loss'),
        loss_mean=tl('return nll.mean()', 'beta_nll_full_loss'),
        criterion=tl('return beta_nll_full_loss(out, y, beta)'),
        beta_from_config=tl('beta = float(tcfg["beta_nll"]) if hetero else None'),
        train_loss_sum=tl('loss_sum += loss.detach() * xb.shape[0]'),
        train_loss=tl('train_loss = float(loss_sum) / n_seen'),
        augment=tl('xb = augment_on_gpu(xc, noise, sign, rev, device)'),
        predict_eval=tl('model.eval()', 'predict'),
        label_metrics=T['fn']['label_metrics'],
        mse=tl('mse = np.mean((f - t) ** 2, axis=0)', 'label_metrics'),
        val_metrics=tl('val_mse, val_slope = label_metrics(val_out[0], val_out[2])'),
        train40k_loader=tl('tr_loader = make_loader(ds["train"]'),
        train40k_rows=tl('np.random.default_rng(TRAIN_MONITOR_SEED).choice(len(ds["train"]), TRAIN_MONITOR_N'),
        train40k_metrics=tl('tr_mse, tr_slope = label_metrics(tr_out[0], tr_out[2])'),
        val_loss=tl('val_loss = float(val_mse.mean())'),
        best_ckpt=tl('improved = val_loss < best_val'),
        net_clamp_bounds=(nl('LOG_SIGMA_MIN = -7.0'), nl('LOG_SIGMA_MAX = 3.0')),
        net_clamp=nl('log_sigma = torch.clamp(out[:, k:2 * k], self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX)'),
        net_full_cov_fn=N['fn']['full_covariance_from_output'],
        net_L_diag=nl('L = torch.diag_embed(torch.exp(out[:, k:2 * k]))', 'full_covariance_from_output'),
        net_sigma=nl('return mu, L, torch.bmm(L, L.transpose(1, 2))', 'full_covariance_from_output'))
    I.check('[-7.0, 3.0]' in info['procedure']['loss'], 'run_info procedure.loss: log L_ii clamp [-7.0, 3.0]')
    d = dict(lines=L, beta=float(tcfg['beta_nll']), files={k: str(ROOT / k) for k in src},
             sha256={k: info['sha256'][k] for k in src}, train_src=T)
    return I.store('training_code', d, start)


def _ref(rel, ln):
    return f'{rel}:{ln[0]}-{ln[1]}' if isinstance(ln, tuple) else f'{rel}:{ln}'


def loss_definition(I, out):
    """No figure: the loss of the final model as implemented (lines cited), its beta, the Sigma parametrization and
    the logged validation metric. The source is read as text (sha256 = run_info.json); nothing is imported or run."""
    C = training_code(I)
    L, b = C['lines'], C['beta']
    tr, nw = TRAIN_SCRIPT_REL, NETWORK_REL
    block = dict(
        function='beta_nll_full_loss (run_info config.training.loss = "beta_nll_full"; selected by criterion() '
                 f'at {_ref(tr, L["criterion"])})',
        defined_at=_ref(tr, L['loss_fn']),
        per_sample=('standardized labels y (train-only y_mean, y_std), r = y - mu, e = L^-1 r by forward substitution '
                    '(e1 = r1 / L11, e2 = (r2 - L21 e1) / L22, e3 = (r3 - L31 e1 - L32 e2) / L33); per label i: '
                    'l_i = (e_i^2 / 2 + log L_ii) * sg(L_ii^2)^beta; per sample: (1/3) sum_i l_i'),
        average='mean over the 3 labels and the B rows of the batch (nll.mean() over the B x 3 array), B = 256',
        beta_0_limit='with beta = 0 and summing over labels: 0.5 r^T Sigma^-1 r + 0.5 log det Sigma (Gaussian NLL '
                     'up to a constant), as the function docstring states',
        beta=b,
        beta_with_full_covariance=('the weight is applied per label to the Cholesky diagonal: sg(L_ii^2)^beta = '
                                   'exp(2 beta log L_ii).detach(), i.e. L_ii^(2 beta) (= L_ii for beta = 0.5); it is '
                                   'not Sigma_ii (= sum_j<=i L_ij^2, equal to L_11^2 only for i = 1) nor det Sigma'),
        stop_gradient='yes: .detach() on the weight (no gradient through it)',
        sigma_parametrization=('head outputs [mu_1..3, log L_11, log L_22, log L_33, L21, L31, L32] (standardized '
                               'units); log L_ii clamped to [-7.0, 3.0] in the model forward (LOG_SIGMA_MIN / '
                               'LOG_SIGMA_MAX), L_ii = exp(log L_ii) > 0, off-diagonal L_ij unconstrained, L lower '
                               'triangular, Sigma = L L^T'),
        training_loss_logged=('train_loss = batch-size-weighted mean of the loss over the epoch\'s training batches '
                              '(model.train(): dropout on until epoch 64, noise augmentation on)'),
        validation_metric_logged=('val_loss = mean over the 3 labels of the val MSE of mu, standardized units, '
                                  'computed in eval mode (no dropout, no augmentation) after each epoch; also per-label '
                                  'val_mse and OLS slope; the val beta-NLL is not computed. val_loss selects '
                                  '_bestval.pt; the registered model is the last epoch (_final.pt)'),
        lines={k: _ref(nw if k.startswith('net_') else tr, v) for k, v in L.items()},
        files=C['files'], sha256_equals_run_info=C['sha256'])
    return {'loss_definition': dict(
        files=[], role='table', loss_definition=block,
        note='Read from the training code that run_info.json records (sha256 checked); every cited line is located '
             'by its text and checked to exist; the code is not imported or executed.')}


def f17_v2(I, out):
    """F17 in the style of the M13_6_evaluation training-curve cell (val '-', train40k '.', same colour; log y;
    'epoch (0-based)'; 'MSE (std. units)'): left mean val and train-monitor MSE of mu, right val RMSE per parameter
    (standardized, log y); thin marks at the checkpoint epoch and at the dropout switch-off epoch."""
    T = training_logs(I)
    H, ch, ti = T['H'], T['chosen'], T['info']
    ep = H['epoch'].astype(int)
    opt = FIGOPT['F17v2']
    I.check(np.allclose(H['val_loss'], H['val_mse'].mean(1), rtol=1e-6, atol=1e-9),
            'history.npz: val_loss = mean over labels of val_mse')
    off = ti['dropout_off_epoch_0based']
    I.check(isinstance(off, int) and int(np.argmax(H['dropout_off'])) == off and bool(H['dropout_off'][off])
            and not bool(H['dropout_off'][off - 1]), 'history.npz dropout_off switches at run_info dropout_off_epoch')
    has_tr = 'train40k_mse' in H
    if has_tr:
        C = training_code(I)
        tr = H['train40k_mse']
        ok = np.isfinite(tr).all(1)
    keys = ['epoch', 'val_mse'] + (['train40k_mse'] if has_tr else []) + ['val_loss (check)', 'dropout_off (check)']
    lw, ms = float(opt['line_lw']), float(opt['marker_ms'])
    with slide_rc():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=tuple(opt['size']), layout='constrained', sharex=True)
        a1.plot(ep, H['val_mse'].mean(1), '-', color=COL['network'], lw=lw)
        if has_tr:
            a1.plot(ep[ok], tr[ok].mean(1), '.', color=COL['network'], ms=ms)
        hr = [a2.plot(ep, np.sqrt(H['val_mse'][:, j]), '-', color=COL[f'param_{lab}'], lw=lw)[0]
              for j, lab in enumerate(LABELS)]
        for a in (a1, a2):
            a.set_yscale('log')
            a.axvline(ch, color=COL['chosen_epoch'], lw=1.0, zorder=0.6)
            a.axvline(off, color=COL['dropout_off_epoch'], lw=1.0, ls=':', zorder=0.6)
            a.set_xlabel('epoch (0-based)')
            a.set_xlim(ep[0] - 1, ep[-1] + 1)
        a1.set_ylabel('MSE (std. units)')
        a2.set_ylabel('val RMSE (std. units)')
        mk = [Line2D([], [], color=COL['chosen_epoch'], lw=1.0),
              Line2D([], [], color=COL['dropout_off_epoch'], lw=1.0, ls=':')]
        lab1 = (['validation'] + (['train (40 000 fixed rows)'] if has_tr else [])
                + [f'checkpoint (epoch {ch})', f'dropout off (epoch {off})'])
        h1 = [Line2D([], [], color=COL['network'], lw=lw)] + (
            [Line2D([], [], ls='', marker='.', color=COL['network'], ms=ms * 1.5)] if has_tr else [])
        leg1 = place_legend(a1, h1 + mk, lab1, grow=4)
        leg2 = place_legend(a2, hr, [var_label(lab, unit=False) for lab in LABELS], grow=4)
        files = save_slide(fig, out, NAME['F17v2'], strict=True)
    nr = 'no registrado'
    if has_tr:
        trdef = dict(present_in_history=True, key='train40k_mse', rows='40000 fixed train rows (default_rng(133400))',
                     mode='eval mode after each epoch (predict(): model.eval(), no_grad)', dropout_active=False,
                     noise_augmentation_active=False,
                     inputs='fixed X of the run fp16 memmap (ds["train"]), not the augmented training batches',
                     same_definition_as_val='yes: label_metrics() MSE of mu per label, standardized; mean over labels',
                     code={k: _ref(TRAIN_SCRIPT_REL, C['lines'][k])
                           for k in ('predict_eval', 'mse', 'train40k_loader', 'train40k_rows', 'train40k_metrics',
                                     'val_metrics')},
                     epochs_evaluated=int(ok.sum()))
    else:
        trdef = dict(present_in_history=False, note='train MSE not in history.npz; not reconstructed',
                     dropout_active=nr, noise_augmentation_active=nr)
    return {'F17v2': dict(
        files=files, role='main', legend=dict(left=leg1, right=leg2), training=ti, history_keys_used=keys,
        train_mse=trdef, marks=dict(checkpoint_epoch_0based=ch, dropout_off_epoch_0based=off),
        style_source='notebooks/m13/M13_6_evaluation.ipynb, training-curve cell (val_mse "-" lw 1.2, train40k_mse "." '
                     'ms 3, one colour per model, yscale log, xlabel "epoch (0-based)", ylabel "MSE (std. units)"); '
                     f'adapted: one model, mean over labels on the left, lw {lw:g} / ms {ms:g} for the slide size',
        numbers=dict(final_val_mse_mean=float(H['val_mse'][-1].mean()),
                     final_train_mse_mean=float(tr[-1].mean()) if has_tr else nr,
                     final_val_rmse_standardized=dict(zip(LABELS, np.sqrt(H['val_mse'][-1])))),
        note='Read from the saved history.npz only (log lines checked against it, nothing re-run). The train curve is '
             'the train-monitor MSE (same definition as val); the logged train loss (beta-NLL with dropout and '
             'augmentation) is not drawn.')}


# ---- F18 (back-up): real-event ellipses in the planes with chi_eff ---------------------------------------------
F18_PLANES = (('F18a', 'chirp_mass', 'chi_eff'), ('F18b', 'total_mass', 'chi_eff'))


def pair_maha(d, i, j):
    """s = r^T Sigma^-1 r in standardized units for the label pair (i, j); equals maha(d, 2) for (0, 1)."""
    ix = [i, j]
    r = d['r_std'][:, ix]
    sol = np.linalg.solve(d['cov'][:, ix][:, :, ix], r[..., None])[..., 0]
    return np.einsum('ni,ni->n', r, sol)


def pair_radii(I):
    """90 % split-conformal radius on cal for each chi_eff plane, same rule (kth, same rank) as the stored 2D radius;
    the code is checked to reproduce the stored (chirp_mass, total_mass) radius. No stored value exists for these
    planes; their test coverage is recorded (informative)."""
    if I.cached('R_pairs'):
        return I._cache['R_pairs']
    start = len(I.used)
    cal, test, R, s5 = I.pred('cal'), I.pred('test'), I.radii(), I.s5()
    r01, k01 = kth(pair_maha(cal, 0, 1))
    I.check(k01 == s5['c3']['k_rank'] and np.isclose(r01, R['2D'], rtol=Q_RTOL),
            'pair radius code reproduces the stored 2D radius (chirp_mass, total_mass)')
    res = {}
    for _, a, b in F18_PLANES:
        i, j = LABELS.index(a), LABELS.index(b)
        r, k = kth(pair_maha(cal, i, j))
        res[(a, b)] = dict(radius=r, k_rank=k, n_cal=cal['n'],
                           test_coverage=float(np.mean(pair_maha(test, i, j) <= r)), n_test=test['n'])
    return I.store('R_pairs', res, start)


def f18(I, out):
    """F12v4 (same events, palette, style and legend search) in the planes (chirp mass, chi_eff) and (total mass,
    chi_eff): LVK samples and 90 % contour, conformal 2D ellipse of that plane, mu."""
    ev, real, RP, R = I.events(), I.real(), pair_radii(I), I.radii()
    stored = {r['event']: r for r in real['ellipsoid_per_event']}
    order = ev['order']
    main = list(CFG['real_events']['main_events'])
    require(len(main) == 3 and all(e in order for e in main),
            f'cpan_2026.json real_events.main_events must be 3 of {order}: {main}')

    def inside(S, mu, C, r):
        dm = S - mu
        return np.einsum('ni,ni->n', dm, np.linalg.solve(C, dm.T).T) <= r

    for e in main:   # the fraction code on the stored plane reproduces the real-events JSON
        S, mu, C = ev['samples'][e][:, :2], ev['mu'][e][:2], ev['cov_phys'][e][:2, :2]
        I.check(abs(float(inside(S, mu, C, R['2D']).mean()) - stored[e]['lvk_frac_in_S_2D']) <= 1e-3,
                f'F18 {e}: LVK fraction inside the 2D ellipse reproduces the real-events JSON (chirp_mass, total_mass)')
    res = {}
    for fid, a, b in F18_PLANES:
        i, j = LABELS.index(a), LABELS.index(b)
        rad = RP[(a, b)]['radius']
        nums = {}
        with slide_rc():
            fig, axes = plt.subplots(1, 3, figsize=tuple(FIGOPT['F18']['size']), layout='constrained', sharey=False)
            for n_, (ax, e) in enumerate(zip(axes, main)):
                k = order.index(e)   # same LVK subsample as F12 / F12v4
                S = ev['samples'][e][:, [i, j]]
                mu2 = ev['mu'][e][[i, j]]
                C2 = ev['cov_phys'][e][np.ix_([i, j], [i, j])]
                sub = np.random.default_rng([SEED_PE_SUBSAMPLE, k]).choice(len(S), min(len(S), N_PE_SCATTER),
                                                                         replace=False)
                E = ellipse_xy(mu2, C2, rad)
                pts = np.vstack([np.percentile(S, [0.5, 99.5], axis=0), E.T])
                lo, hi = pts.min(0), pts.max(0)
                pad = 0.08 * (hi - lo)
                ax.scatter(S[sub, 0], S[sub, 1], s=4, color=EVC[e], alpha=0.25, linewidths=0, rasterized=True)
                hpd_contour(ax, S, LEVEL, EVC[e])
                ax.plot(E[0], E[1], color=COL['network'], lw=2.8)
                ax.plot(*mu2, 'o', color=COL['mu'], ms=9)
                ax.set(xlim=(lo[0] - pad[0], hi[0] + pad[0]), ylim=(lo[1] - pad[1], hi[1] + pad[1]))
                ax.set_box_aspect(1)
                ax.set_title(short_event(e, order) + (' (out of support)' if e in ev['oos'] else ''))
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.grid(True, axis='both')
                ax.set_xlabel(var_label(a))
                if n_ == 0:
                    ax.set_ylabel(var_label(b))
                med = np.median(S, axis=0)
                nums[e] = dict(mu=mu2, lvk_frac_in_ellipse_2D=float(inside(S, mu2, C2, rad).mean()),
                               lvk_median=med, lvk_median_in_ellipse=bool(inside(med[None], mu2, C2, rad)[0]),
                               n_lvk_samples=len(S))
            smp_h = tuple(Line2D([], [], ls='', marker='o', ms=7, color=EVC[e], alpha=0.6) for e in main)
            cnt_h = tuple(Line2D([], [], color=EVC[e], lw=1.8) for e in main)
            leg = place_legend(list(axes), [smp_h, cnt_h, Line2D([], [], color=COL['network'], lw=2.8),
                                            Line2D([], [], ls='', marker='o', color=COL['mu'], ms=9)],
                               ['LVK samples', 'LVK 90 % region', 'Network ellipse (90 %)', 'Network $\\mu$'],
                               grow=int(FIGOPT['F18']['legend_max_headroom_steps']),
                               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
            files = save_slide(fig, out, NAME[fid], strict=True)
        res[fid] = dict(files=files, numbers=nums, role='appendix', plane=[a, b],
                        legend=dict(leg, panel=main[leg['axes_index']], zero_overlap=leg['overlap_score'] == 0),
                        radius=dict(RP[(a, b)], rule='90 % split-conformal on cal of r^T Sigma^-1 r for this pair, '
                                                     'kth rank as the stored 2D radius; computed here (no stored '
                                                     'value for this plane); test_coverage informative'),
                        rows=dict(events=main, selection='cpan_2026.json real_events.main_events',
                                  lvk_scatter=f'{N_PE_SCATTER} samples, as F12'),
                        note=f'As F12v4 in the plane ({a}, {b}): marginal 2x2 block of the predicted Sigma (physical '
                             'units). Fraction = LVK samples with (s - mu)^T Sigma_2^-1 (s - mu) <= radius (code '
                             'checked on the stored chirp_mass / total_mass fractions).')
    return res


# ---- F17v3: history keys described from the training code; checks before drawing ------------------------------
EXPECTED_VAL_RMSE_M14_5 = dict(chirp_mass=5.19, total_mass=8.60, chi_eff=0.167)
VAL_RMSE_RTOL = 0.005
VAL_LOSS_RTOL = 1e-6


def _assigns(line, var):
    """True if `line` assigns `var` (plain, chained, tuple or augmented assignment)."""
    if re.match(rf'^\s*{re.escape(var)}\s*[-+*/]=', line):
        return True
    parts = re.split(r'(?<![=<>!+\-*/])=(?!=)', line.split('#')[0])
    return len(parts) > 1 and var in re.findall(r'\w+', ' '.join(parts[:-1]))


def history_keys(I, H):
    """Every key of history.npz with the training-code lines that produce it: the record dict of the epoch loop
    ("key": variable) and the assignments of that variable inside the loop (label_names: the np.savez line)."""
    C = training_code(I)
    T = C['train_src']['lines']
    rel = TRAIN_SCRIPT_REL
    loop = _find_line(I, T, 'for epoch in range(start_epoch, max_epochs):', rel)
    r0 = _find_line(I, T, 'rec = {"epoch": epoch', rel, loop)
    r1 = next(k for k in range(r0, len(T) + 1) if T[k - 1].rstrip().endswith('}'))
    pairs = dict(re.findall(r'"(\w+)":\s*([\w.]+)', ' '.join(T[r0 - 1:r1])))
    save = _find_line(I, T, 'np.savez(p_hist', rel)
    b0 = _find_line(I, T, 'model.train()', rel, loop)                     # training-batch loop of the epoch
    b1 = _find_line(I, T, 'train_loss = float(loss_sum) / n_seen', rel, b0)
    out = {}
    for k in H:
        a = np.asarray(H[k])
        e = dict(shape=list(a.shape), dtype=str(a.dtype))
        if k == 'label_names':
            e.update(code=[f'{rel}:{save}: {T[save - 1].strip()}'])
        elif k in pairs:
            var = pairs[k]
            ln = [loop] if var == 'epoch' else [n for n in range(loop, r0) if _assigns(T[n - 1], var)]
            e.update(record=f'{rel}:{r0}-{r1}: "{k}": {var}',
                     code=[f'{rel}:{n}: {T[n - 1].strip()}' for n in ln] or ['no encontrado en el código'],
                     in_training_batch_loop=any(b0 <= n <= b1 for n in ln))
        else:
            e.update(code=['no encontrado en el código'])
        out[k] = e
    batch_mse = [k for k, e in out.items() if 'mse' in k.lower() and e.get('in_training_batch_loop')]
    return out, batch_mse, dict(epoch_loop=loop, record=(r0, r1), training_batch_loop=(b0, b1), savez=save)


def f17_v3(I, out):
    """F17v2 with the history keys listed and two checks before drawing: (a) mean of the per-parameter val MSE =
    val_loss at every epoch; (b) last-epoch sqrt(val MSE_i) x train y_std_i = M14.5 val RMSE. Left: train MSE of mu
    on the training batches and val MSE on one log axis if history has one; otherwise train beta-NLL (left, linear)
    and val MSE (right, log). Right: val MSE per parameter (log). Marks at the checkpoint and dropout-off epochs."""
    T = training_logs(I)
    H, ch, ti, ri = T['H'], T['chosen'], T['info'], T['run_info']
    ep = H['epoch'].astype(int)
    opt = FIGOPT['F17v3']
    keys, batch_mse, where = history_keys(I, H)
    off = ti['dropout_off_epoch_0based']
    rel_a = np.abs(H['val_mse'].mean(1) - H['val_loss']) / np.abs(H['val_loss'])
    stats = I.exists(ri['data']['label_stats_file'], 'train-only label stats (y_std)')
    with np.load(stats, allow_pickle=False) as z:
        y_std = z['y_std'].astype(np.float64)
        st_names = [str(x) for x in z['label_names']] if 'label_names' in z.files else list(LABELS)
    rmse = np.sqrt(H['val_mse'][-1]) * y_std
    stored = {r['label']: r['rmse_model'] for r in I.s5()['c1']['table']}
    exp = np.array([EXPECTED_VAL_RMSE_M14_5[lab] for lab in LABELS])
    st = np.array([stored[lab] for lab in LABELS])
    checks = dict(
        a_val_loss_equals_mean_val_mse=dict(rtol=VAL_LOSS_RTOL, max_rel_diff=float(rel_a.max()),
                                            n_epochs=len(ep), ok=bool(np.all(rel_a <= VAL_LOSS_RTOL))),
        b_last_epoch_val_rmse=dict(y_std_train=dict(zip(LABELS, y_std)), y_std_file=str(stats),
                                   rmse_from_history=dict(zip(LABELS, rmse)),
                                   expected_quoted=EXPECTED_VAL_RMSE_M14_5,
                                   m14_5_summary_c1_rmse_model=dict(zip(LABELS, st)),
                                   rel_diff_vs_quoted=dict(zip(LABELS, rmse / exp - 1)),
                                   rel_diff_vs_summary=dict(zip(LABELS, rmse / st - 1)), rtol=VAL_RMSE_RTOL,
                                   ok=bool(st_names == list(LABELS) and np.all(np.abs(rmse / exp - 1) <= VAL_RMSE_RTOL)
                                           and np.all(np.abs(rmse / st - 1) <= VAL_RMSE_RTOL))))
    branch = ('train MSE of mu on the training batches (' + batch_mse[0] + ') and val MSE, one log axis'
              if batch_mse else 'no train MSE of mu on the training batches in history.npz: train beta-NLL '
                                '(train_loss, left axis, linear) and val MSE (right axis, log)')
    # Written first (manifest 'info' keeps it also when a check fails and the figure is not drawn).
    I.info['F17v3'] = dict(history_keys=keys, code_ranges=where, train_batch_mse_keys=batch_mse, left_panel=branch,
                           checks=checks)
    I.check(checks['a_val_loss_equals_mean_val_mse']['ok'],
            f'F17v3 (a): val_loss = mean of the 3 val MSE at every epoch (rtol {VAL_LOSS_RTOL})')
    I.check(checks['b_last_epoch_val_rmse']['ok'],
            f'F17v3 (b): last-epoch sqrt(val MSE) x y_std reproduces the M14.5 val RMSE within {VAL_RMSE_RTOL:.1%}')
    I.check(isinstance(off, int) and int(np.argmax(H['dropout_off'])) == off,
            'history.npz dropout_off switches at run_info dropout_off_epoch')
    lw, ms = float(opt['line_lw']), float(opt['marker_ms'])
    c_tr, c_va = COL['train_curve'], COL['network']
    with slide_rc():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=tuple(opt['size']), layout='constrained', sharex=True)
        if batch_mse:
            a1.plot(ep, H['val_mse'].mean(1), '-', color=c_va, lw=lw)
            a1.plot(ep, np.asarray(H[batch_mse[0]]).reshape(len(ep), -1).mean(1), '.', color=c_tr, ms=ms)
            a1.set_yscale('log')
            a1.set_ylabel('MSE (std. units)')
            hl = [Line2D([], [], color=c_va, lw=lw), Line2D([], [], ls='', marker='.', color=c_tr, ms=ms * 1.5)]
            ll, also, lax = ['validation MSE', 'train MSE (training batches)'], [], a1
        else:
            a1.plot(ep, H['train_loss'], '.', color=c_tr, ms=ms)
            a1b = a1.twinx()
            a1b.spines['right'].set_visible(True)
            a1b.plot(ep, H['val_mse'].mean(1), '-', color=c_va, lw=lw)
            a1b.set_yscale('log')
            a1b.grid(False)
            a1.set_ylabel('train $\\beta$-NLL', color=c_tr)
            a1b.set_ylabel('val MSE (std. units)', color=c_va)
            a1.tick_params(axis='y', colors=c_tr)
            a1b.tick_params(axis='y', which='both', colors=c_va)
            hl = [Line2D([], [], ls='', marker='.', color=c_tr, ms=ms * 1.5), Line2D([], [], color=c_va, lw=lw)]
            ll, also, lax = ['train $\\beta$-NLL (left axis)', 'val MSE of $\\mu$ (right axis)'], [a1], a1b
        hr = [a2.plot(ep, H['val_mse'][:, j], '-', color=COL[f'param_{lab}'], lw=lw)[0] for j, lab in enumerate(LABELS)]
        a2.set_yscale('log')
        a2.set_ylabel('val MSE (std. units)')
        for a in (a1, a2):
            a.axvline(ch, color=COL['chosen_epoch'], lw=1.0, zorder=0.6)
            a.axvline(off, color=COL['dropout_off_epoch'], lw=1.0, ls=':', zorder=0.6)
            a.set_xlabel('epoch (0-based)')
            a.set_xlim(ep[0] - 1, ep[-1] + 1)
        mk = [Line2D([], [], color=COL['chosen_epoch'], lw=1.0),
              Line2D([], [], color=COL['dropout_off_epoch'], lw=1.0, ls=':')]
        leg1 = place_legend(lax, hl + mk, ll + [f'checkpoint (epoch {ch})', f'dropout off (epoch {off})'],
                            also=also, grow=4)
        leg2 = place_legend(a2, hr, [var_label(lab, unit=False) for lab in LABELS], grow=4)
        files = save_slide(fig, out, NAME['F17v3'], strict=True)
    return {'F17v3': dict(
        files=files, role='main', legend=dict(left=leg1, right=leg2), training=ti, history_keys=keys,
        left_panel=branch, checks=checks, marks=dict(checkpoint_epoch_0based=ch, dropout_off_epoch_0based=off),
        style_source='notebooks/m13/M13_6_evaluation.ipynb training-curve cell, as F17v2 (val "-", train ".", log y '
                     'for MSE, "epoch (0-based)", "MSE (std. units)")',
        note='Read from the saved history.npz only. The train beta-NLL (train_loss) is the batch-size-weighted mean '
             'over the epoch with dropout (until the dropout-off epoch) and noise augmentation on; the val MSE is in '
             'eval mode. history keys and their code lines are also in manifest info.F17v3.')}


F17V4_LW_TRAIN = 2.5          # F17v4: train beta-NLL as a solid line
F17V4_EPOCH0_FONT = 16        # F17v4: text of the off-scale epoch-0 arrow
F17V4_PAD = 0.1               # F17v4: left ylim = [min - 0.1 range, max + 0.1 range] over epochs 1-79


def f17_v4(I, out):
    """F17v3 with only the left panel changed: train beta-NLL as a solid line (lw 2.5, same colour, no markers) on a
    linear left axis whose ylim is fitted to epochs 1..last ([min - 0.1 range, max + 0.1 range]); epoch 0 is off
    scale and marked by an upward arrow at x = 0 against the top edge with its value. Val MSE (right axis, log), the
    right panel, colours, size and the epoch marks as F17v3. Same keys listing and checks as F17v3."""
    T = training_logs(I)
    H, ch, ti, ri = T['H'], T['chosen'], T['info'], T['run_info']
    ep = H['epoch'].astype(int)
    opt = FIGOPT['F17v3']                                  # same size / line widths as F17v3
    keys, batch_mse, where = history_keys(I, H)
    off = ti['dropout_off_epoch_0based']
    rel_a = np.abs(H['val_mse'].mean(1) - H['val_loss']) / np.abs(H['val_loss'])
    stats = I.exists(ri['data']['label_stats_file'], 'train-only label stats (y_std)')
    with np.load(stats, allow_pickle=False) as z:
        y_std = z['y_std'].astype(np.float64)
        st_names = [str(x) for x in z['label_names']] if 'label_names' in z.files else list(LABELS)
    rmse = np.sqrt(H['val_mse'][-1]) * y_std
    stored = {r['label']: r['rmse_model'] for r in I.s5()['c1']['table']}
    exp = np.array([EXPECTED_VAL_RMSE_M14_5[lab] for lab in LABELS])
    st = np.array([stored[lab] for lab in LABELS])
    checks = dict(
        a_val_loss_equals_mean_val_mse=dict(rtol=VAL_LOSS_RTOL, max_rel_diff=float(rel_a.max()),
                                            n_epochs=len(ep), ok=bool(np.all(rel_a <= VAL_LOSS_RTOL))),
        b_last_epoch_val_rmse=dict(rmse_from_history=dict(zip(LABELS, rmse)), expected_quoted=EXPECTED_VAL_RMSE_M14_5,
                                   m14_5_summary_c1_rmse_model=dict(zip(LABELS, st)), rtol=VAL_RMSE_RTOL,
                                   ok=bool(st_names == list(LABELS) and np.all(np.abs(rmse / exp - 1) <= VAL_RMSE_RTOL)
                                           and np.all(np.abs(rmse / st - 1) <= VAL_RMSE_RTOL))))
    I.info['F17v4'] = dict(train_batch_mse_keys=batch_mse, checks=checks)
    I.check(checks['a_val_loss_equals_mean_val_mse']['ok'],
            f'F17v4 (a): val_loss = mean of the 3 val MSE at every epoch (rtol {VAL_LOSS_RTOL})')
    I.check(checks['b_last_epoch_val_rmse']['ok'],
            f'F17v4 (b): last-epoch sqrt(val MSE) x y_std reproduces the M14.5 val RMSE within {VAL_RMSE_RTOL:.1%}')
    I.check(not batch_mse, 'F17v4: left panel = train beta-NLL + val MSE (no train-batch MSE in history.npz)')
    I.check(isinstance(off, int) and int(np.argmax(H['dropout_off'])) == off,
            'history.npz dropout_off switches at run_info dropout_off_epoch')
    I.check(ep[0] == 0 and len(ep) > 2, 'F17v4: history starts at epoch 0')
    lw, ms = float(opt['line_lw']), float(opt['marker_ms'])
    c_tr, c_va = COL['train_curve'], COL['network']
    tl = np.asarray(H['train_loss'], float)
    rest = tl[ep >= 1]
    lo, hi = float(rest.min()), float(rest.max())
    rng_ = hi - lo
    ylim = (lo - F17V4_PAD * rng_, hi + F17V4_PAD * rng_)
    v0 = float(tl[ep == 0][0])
    I.check(v0 > ylim[1], f'F17v4: epoch 0 ({v0:.4f}) is above the fitted ylim {ylim}')
    print(f'F17v4: left ylim = [{ylim[0]:.6f}, {ylim[1]:.6f}] (epochs 1-{int(ep[-1])}); epoch 0 train beta-NLL = '
          f'{v0:.6f}')
    with slide_rc():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=tuple(opt['size']), layout='constrained', sharex=True)
        a1.plot(ep, tl, '-', color=c_tr, lw=F17V4_LW_TRAIN)
        a1.set_ylim(*ylim)
        a1b = a1.twinx()
        a1b.spines['right'].set_visible(True)
        a1b.plot(ep, H['val_mse'].mean(1), '-', color=c_va, lw=lw)
        a1b.set_yscale('log')
        a1b.grid(False)
        a1.set_ylabel('train $\\beta$-NLL', color=c_tr)
        a1b.set_ylabel('val MSE (std. units)', color=c_va)
        a1.tick_params(axis='y', colors=c_tr)
        a1b.tick_params(axis='y', which='both', colors=c_va)
        top = matplotlib.transforms.blended_transform_factory(a1.transData, a1.transAxes)
        a1.annotate('', xy=(0, 0.995), xytext=(0, 0.80), xycoords=top, textcoords=top, annotation_clip=False,
                    arrowprops=dict(arrowstyle='-|>', color=c_tr, lw=2.0, mutation_scale=18))
        a1.text(1.2, 0.90, f'epoch 0: {v0:.2f}', transform=top, color=c_tr, fontsize=F17V4_EPOCH0_FONT,
                ha='left', va='center')
        hr = [a2.plot(ep, H['val_mse'][:, j], '-', color=COL[f'param_{lab}'], lw=lw)[0] for j, lab in enumerate(LABELS)]
        a2.set_yscale('log')
        a2.set_ylabel('val MSE (std. units)')
        for a in (a1, a2):
            a.axvline(ch, color=COL['chosen_epoch'], lw=1.0, zorder=0.6)
            a.axvline(off, color=COL['dropout_off_epoch'], lw=1.0, ls=':', zorder=0.6)
            a.set_xlabel('epoch (0-based)')
            a.set_xlim(ep[0] - 1, ep[-1] + 1)
        mk = [Line2D([], [], color=COL['chosen_epoch'], lw=1.0),
              Line2D([], [], color=COL['dropout_off_epoch'], lw=1.0, ls=':')]
        hl = [Line2D([], [], color=c_tr, lw=F17V4_LW_TRAIN), Line2D([], [], color=c_va, lw=lw)]
        ll = ['train $\\beta$-NLL (left axis)', 'val MSE of $\\mu$ (right axis)']
        leg1 = place_legend(a1b, hl + mk, ll + [f'checkpoint (epoch {ch})', f'dropout off (epoch {off})'],
                            also=[a1], grow=4)
        leg2 = place_legend(a2, hr, [var_label(lab, unit=False) for lab in LABELS], grow=4)
        files = save_slide(fig, out, NAME['F17v4'], strict=True)
    return {'F17v4': dict(
        files=files, role='main', legend=dict(left=leg1, right=leg2), training=ti, checks=checks,
        left_ylim=list(ylim), left_ylim_rule=f'[min - {F17V4_PAD} range, max + {F17V4_PAD} range] of train_loss over '
                                             f'epochs 1-{int(ep[-1])}',
        epoch0_train_beta_nll=v0, marks=dict(checkpoint_epoch_0based=ch, dropout_off_epoch_0based=off),
        note='F17v3 with the left panel changed only: train beta-NLL as a solid line (lw 2.5), linear axis fitted to '
             'epochs 1-79, epoch 0 off scale (arrow at x = 0 with its value, 16 pt). Val MSE right axis, right panel, '
             'colours, size and epoch marks as F17v3.')}


def f18_grid(I, out):
    """F18c / F18d: the nine events in a 3 x 3 grid in the style of F12 (all events; LVK samples and 90 % contour,
    conformal 2D ellipse of the plane, intervals box, mu) in the (chirp mass, chi_eff) and (total mass, chi_eff)
    planes, with the plane radii of F18a / F18b. Legend inside one panel (zero-overlap search)."""
    ev, real, RP, R = I.events(), I.real(), pair_radii(I), I.radii()
    stored = {r['event']: r for r in real['ellipsoid_per_event']}
    order = ev['order']

    def inside(S, mu, C, r):
        dm = S - mu
        return np.einsum('ni,ni->n', dm, np.linalg.solve(C, dm.T).T) <= r

    for e in order:   # the fraction code on the stored plane reproduces the real-events JSON (all events)
        S, mu, C = ev['samples'][e][:, :2], ev['mu'][e][:2], ev['cov_phys'][e][:2, :2]
        I.check(abs(float(inside(S, mu, C, R['2D']).mean()) - stored[e]['lvk_frac_in_S_2D']) <= 1e-3,
                f'F18 {e}: LVK fraction inside the 2D ellipse reproduces the real-events JSON (chirp_mass, total_mass)')
    res = {}
    for fid, a, b in (('F18c', 'chirp_mass', 'chi_eff'), ('F18d', 'total_mass', 'chi_eff')):
        i, j = LABELS.index(a), LABELS.index(b)
        rad = RP[(a, b)]['radius']
        nums = {}
        with slide_rc():
            fig, axes = plt.subplots(3, 3, figsize=tuple(FIGOPT['F18grid']['size']), layout='constrained')
            used = []
            for k, (ax, e) in enumerate(zip(axes.flat, order)):
                S = ev['samples'][e][:, [i, j]]
                mu2, C2 = ev['mu'][e][[i, j]], ev['cov_phys'][e][np.ix_([i, j], [i, j])]
                lo, hi = ev['lo'][e][[i, j]], ev['hi'][e][[i, j]]
                sub = np.random.default_rng([SEED_PE_SUBSAMPLE, k]).choice(len(S), min(len(S), N_PE_SCATTER),
                                                                         replace=False)   # as F12
                E = ellipse_xy(mu2, C2, rad)
                pts = np.vstack([np.percentile(S, [0.5, 99.5], axis=0), E.T, lo, hi])
                p0, p1 = pts.min(0), pts.max(0)
                pad = 0.08 * (p1 - p0)
                c = EVC[e]
                ax.scatter(S[sub, 0], S[sub, 1], s=3, color=c, alpha=0.25, linewidths=0, rasterized=True)
                hpd_contour(ax, S, LEVEL, c)
                ax.plot(E[0], E[1], color=COL['ellipse'], lw=2.0)
                ax.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fill=False, ec=COL['box'], lw=1.8, ls='--'))
                ax.plot(*mu2, 'o', color=COL['prediction'], ms=5)
                ax.set(xlim=(p0[0] - pad[0], p1[0] + pad[0]), ylim=(p0[1] - pad[1], p1[1] + pad[1]))
                ax.set_title(short_event(e, order) + (' (out of support)' if e in ev['oos'] else ''))
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.grid(True, axis='both')
                if k // 3 == 2:
                    ax.set_xlabel(var_label(a))
                if k % 3 == 0:
                    ax.set_ylabel(var_label(b))
                used.append(ax)
                med = np.median(S, axis=0)
                nums[e] = dict(mu=mu2, lower=lo, upper=hi,
                               lvk_frac_in_ellipse_2D=float(inside(S, mu2, C2, rad).mean()), lvk_median=med,
                               lvk_median_in_ellipse=bool(inside(med[None], mu2, C2, rad)[0]), n_lvk_samples=len(S))
            for ax in list(axes.flat)[len(order):]:
                ax.axis('off')
            leg = place_legend(used, [Patch(color=COL['lvk_neutral'], alpha=0.5),
                                      Line2D([], [], color=COL['ellipse'], lw=2.0),
                                      Line2D([], [], color=COL['box'], lw=1.8, ls='--'),
                                      Line2D([], [], ls='', marker='o', color=COL['prediction'], ms=5)],
                               ['LVK 90 % region', 'Network ellipse (90 %)', 'Intervals', 'Network $\\mu$'],
                               grow=int(FIGOPT['F18grid']['legend_max_headroom_steps']))   # short labels (F12v4)
            files = save_slide(fig, out, NAME[fid], strict=True)
        res[fid] = dict(files=files, numbers=nums, role='appendix', plane=[a, b],
                        legend=dict(leg, panel=order[leg['axes_index']], zero_overlap=leg['overlap_score'] == 0),
                        radius=dict(RP[(a, b)], rule='same plane radius as F18a / F18b (cal, kth rank)'),
                        rows=dict(events=order, lvk_scatter=f'{N_PE_SCATTER} samples, subsample seed per event as F12'),
                        note=f'F12 (3 x 3, all events) in the plane ({a}, {b}); slide fonts and the legend inside one '
                             'panel instead of below the grid (labels shortened as in F12v4 so it fits a panel). Fraction = LVK samples with (s - mu)^T Sigma_2^-1 '
                             '(s - mu) <= radius.')
    return res


def outs_per_label(gid, base):
    return [(gid, base)] + [(f'{gid}_{lab}', f'{base}_{lab}') for lab in LABELS]


def outs_label_only(gid, base):
    return [(f'{gid}_{lab}', f'{base}_{lab}') for lab in LABELS]


# (group ID for --only, function, [(output ID, file name), ...]); a group is computed once.
FIGURES = [('F06', f06, outs_per_label('F06', 'F06_sigma_calibration')),
           ('F07', f07, outs_per_label('F07', 'F07_coverage_by_snr')),
           ('F08', f08, outs_per_label('F08', 'F08_mondrian_grid')),
           ('F09', f09, [('F09a', 'F09a_correlation_by_tau_decile'), ('F09b', 'F09b_ellipse_examples'),
                         ('F09av2', 'F09av2_correlation_by_tau_decile')]),
           ('F10', f10, outs_per_label('F10', 'F10_pred_vs_truth_test')),
           ('F11', f11, outs_per_label('F11', 'F11_real_events_violins')),
           ('F12', f12, [('F12', 'F12_real_events_ellipses'), ('F12_main', 'F12_main_real_events_ellipses')]),
           # slide versions: one variable per figure, test (F08s: validation)
           ('F10m', f10m, outs_label_only('F10m', 'F10m_pred_vs_truth_marginals_test')),
           ('F06m', f06m, outs_label_only('F06m', 'F06m_sigma_vs_abs_error_test')),
           ('F07b', f07b, outs_label_only('F07b', 'F07b_width_coverage_by_snr_test')),
           ('F08s', f08s, outs_label_only('F08s', 'F08s_config_search_val')),
           ('F09c', f09c, [('F09c', 'F09c_short_signals_box_vs_ellipse_test')]),
           ('F11v2', f11v2, outs_label_only('F11v2', 'F11v2_real_events')),
           ('F12v2', f12v2, [('F12v2', 'F12v2_real_events_ellipses_main')]),
           # v2 / v3 slide versions (compact legend inside the axes; _total_mass / _chi_eff = appendix)
           ('F10mv2', f10m_v2, outs_label_only('F10mv2', 'F10mv2_pred_vs_truth_marginals_test')),
           ('F06mv2', f06m_v2, outs_label_only('F06mv2', 'F06mv2_sigma_vs_abs_error_test')),
           ('F07bv2', f07b_v2, outs_label_only('F07bv2', 'F07bv2_width_coverage_by_snr_test')),
           ('F07q', f07q, outs_label_only('F07q', 'F07q_quantiles_by_snr_decile_cal')),
           ('F08sv2', f08s_v2, outs_label_only('F08sv2', 'F08sv2_config_search_val')),
           ('F11v3', f11v3, outs_label_only('F11v3', 'F11v3_real_events')),
           ('F12v3', f12v3, [('F12v3', 'F12v3_real_events_ellipses_main')]),
           # v3 / v4 slide versions
           ('F10mv3', f10m_v3, outs_label_only('F10mv3', 'F10mv3_pred_vs_truth_marginals_test')),
           ('F06mv3', f06m_v3, outs_label_only('F06mv3', 'F06mv3_sigma_vs_abs_error_linear_test')),
           ('F11v4', f11v4, outs_label_only('F11v4', 'F11v4_real_events')),
           ('F11h', f11h, [('F11h', 'F11h_real_events_horizontal_chirp_mass')]),
           ('F12v4', f12v4, [('F12v4', 'F12v4_real_events_ellipses_main')]),
           ('F13', f13, [('F13', 'F13_short_signals_normalised_errors_test')]),
           ('F06mv4', f06m_v4, outs_label_only('F06mv4', 'F06mv4_sigma_vs_abs_error_linear_test')),
           ('F13v2', f13_v2, [('F13v2', 'F13v2_short_signals_normalised_errors_test')]),
           ('F06c', f06c, outs_label_only('F06c', 'F06c_sigma_vs_signed_error_cone_test')),
           ('F06cv2', f06c_v2, outs_label_only('F06cv2', 'F06cv2_sigma_vs_signed_error_cone_test')),
           ('F16', f16, [('F16', 'F16_error_vs_snr_test')]),
           ('F16b', f16b, [('F16b', 'F16b_error_vs_truth_test')]),
           ('table_test', table_test, [('table_test', 'table_test')]),
           ('F17', f17, [('F17', 'F17_training_history')]),
           ('F17v2', f17_v2, [('F17v2', 'F17v2_training_history')]),
           ('F18', f18, [('F18a', 'F18a_real_events_ellipses_mc_chieff'),
                         ('F18b', 'F18b_real_events_ellipses_mtot_chieff')]),
           ('loss_definition', loss_definition, [('loss_definition', 'loss_definition')]),
           ('F17v3', f17_v3, [('F17v3', 'F17v3_training_history')]),
           ('F17v4', f17_v4, [('F17v4', 'F17v4_training_history')]),
           ('F18grid', f18_grid, [('F18c', 'F18c_real_events_ellipses_mc_chieff_all'),
                                  ('F18d', 'F18d_real_events_ellipses_mtot_chieff_all')])]
NAME = {oid: name for _, _, outs in FIGURES for oid, name in outs}
GROUP_OF = {oid: gid for gid, _, outs in FIGURES for oid, _ in outs}


# ---- numbers.json ----------------------------------------------------------------------------------------------
def count_parameters(I, ckpt):
    """numel of every tensor of the 5b state_dict (CPU load of our own checkpoint; no model, no forward)."""
    import torch  # local import: only this entry needs torch
    p = I.exists(ckpt, '5b checkpoint')
    # Trusted source (our own training run): full unpickling is needed because the checkpoint also stores
    # non-tensor metadata. weights_only=False exists from torch 1.13; the VM torch 1.9.1 never runs this script.
    obj = torch.load(p, map_location='cpu', weights_only=False)
    sd, where = None, None
    if isinstance(obj, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if isinstance(obj.get(k), dict):
                sd, where = obj[k], f"checkpoint['{k}']"
                break
        if sd is None and obj and all(torch.is_tensor(v) for v in obj.values()):
            sd, where = obj, 'checkpoint (is the state_dict)'
    I.check(sd is not None, '5b checkpoint: state_dict found')
    bn = ('running_mean', 'running_var', 'num_batches_tracked')
    tot, by_module, bn_numel, n_tensors = 0, {}, 0, 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        n = int(v.numel())
        tot, n_tensors = tot + n, n_tensors + 1
        top = k.split('.', 1)[0]
        by_module[top] = by_module.get(top, 0) + n
        if k.rsplit('.', 1)[-1] in bn:
            bn_numel += n
    return dict(n_tensor_elements=tot, n_tensors=n_tensors, by_top_level_module=by_module,
                batchnorm_statistics_numel=bn_numel, state_dict=where, checkpoint=str(p),
                checkpoint_sha256=I.sha(p), load="torch.load(map_location='cpu', weights_only=False)",
                caveat='numel of all state_dict tensors: trainable parameters plus any registered buffers '
                       '(BatchNorm statistics given separately; a fixed template bank would be included).')


# Each numbers.json entry is computed independently: a failure is stored as {"status": "ERROR", "error": ...}.
def num_units(I):
    return {lab: CFG['variables'][lab]['unit'] or 'dimensionless' for lab in LABELS}


def num_rmse(I):
    val, test = I.pred('val'), I.pred('test')
    out = {split: {lab: float(np.sqrt(np.mean((d['y'][:, j] - d['mu'][:, j]) ** 2))) for j, lab in enumerate(LABELS)}
           for split, d in (('val', val), ('test', test))}
    out['n_rows'] = dict(val=val['n'], test=test['n'])
    return out


def num_chosen(I):
    q, s7, test = I.intervals(), I.s7(), I.pred('test')
    chosen = {}
    for j, lab in enumerate(LABELS):
        h, w = hits(test, j, q[lab], 'r_sigma'), widths(test, j, q[lab], 'r_sigma')
        ref = next(g for g in s7['test'] if g['role'] == 'selected' and g['label'] == lab)
        I.check(abs(h.mean() - ref['coverage']) <= COV_ATOL and np.isclose(w.mean(), ref['mean_width_phys'], rtol=Q_RTOL),
                f'numbers {lab}: test coverage and mean width reproduce M14.7 test (selected)')
        chosen[lab] = dict(coverage=float(h.mean()), mean_width=float(w.mean()), q_r_over_sigma_cal=q[lab]['r_sigma'],
                           n_rows=test['n'])
    return dict(configuration='M14.7: single bin, symmetric, score r/sigma (cal-only quantile)', per_label=chosen)


def num_joint(I):
    q, R, s5 = I.intervals(), I.radii(), I.s5()
    val, test = I.pred('val'), I.pred('test')
    in_e = maha(test, 3) <= R['3D']
    ref = next(c for c in s5['amendment_2']['test']['coverage'] if c['method'] == 'S' and c['dim'] == '3D')
    I.check(abs(in_e.mean() - ref['coverage']) <= COV_ATOL, 'numbers: test 3D ellipsoid coverage reproduces M14.5')
    log_c = math.log(4.0 / 3.0 * math.pi) + 1.5 * math.log(R['3D'])
    log_ve = log_c + 0.5 * np.linalg.slogdet(test['cov_phys'])[1]
    log_ve_val = log_c + 0.5 * np.linalg.slogdet(val['cov_phys'])[1]
    sv = maha(val, 3) <= R['3D']
    ref_v = next(r for r in s5['descriptive']['volumes'] if r['dim'] == '3D' and r['method'] == 'S')
    I.check(np.isclose(math.exp(log_ve_val.mean()), ref_v['geometric_mean_volume'], rtol=VOL_RTOL)
            and abs(sv.mean() - ref_v['joint_coverage']) <= COV_ATOL,
            'numbers: val 3D ellipsoid geometric-mean volume and coverage reproduce M14.5 (volume convention)')
    in_b = np.all([hits(test, j, q[lab], 'r_sigma') for j, lab in enumerate(LABELS)], axis=0)
    log_vb = np.sum([np.log(widths(test, j, q[lab], 'r_sigma')) for j, lab in enumerate(LABELS)], axis=0)
    return dict(
        n_rows=test['n'], volume_units='M_sun^2 (chi_eff dimensionless)',
        ellipsoid_3d=dict(radius_cal=R['3D'], coverage=float(in_e.mean()),
                          geometric_mean_volume=float(math.exp(log_ve.mean())),
                          median_volume=float(np.exp(np.median(log_ve)))),
        box_of_three_intervals=dict(coverage=float(in_b.mean()), geometric_mean_volume=float(math.exp(log_vb.mean())),
                                    median_volume=float(np.exp(np.median(log_vb)))),
        ellipsoid_vs_box_gmean_volume_pct=float(100.0 * (math.exp(log_ve.mean() - log_vb.mean()) - 1.0)),
        note='geometric mean over test rows (M14.5 convention); box joint coverage = all three intervals hit')


def num_model(I):
    ck = {I.pred(s)['ckpt'] for s in ('cal', 'val', 'test')}
    I.check(len(ck) == 1, 'numbers: same checkpoint in cal / val / test npz')
    return count_parameters(I, ck.pop())


NUMBER_ENTRIES = [('units', num_units), ('rmse', num_rmse), ('chosen_interval_test', num_chosen),
                  ('joint_test', num_joint), ('model_5b', num_model)]


def compute_numbers(I):
    """(numbers dict, list of failed keys); never raises for a single entry."""
    num, failed = {}, []
    for key, fn in NUMBER_ENTRIES:
        try:
            num[key] = fn(I)
        except Exception as e:  # noqa: BLE001 - recorded in numbers.json, reported in the exit status
            num[key] = dict(status='ERROR', error=f'{type(e).__name__}: {e}')
            failed.append(key)
            print(f'numbers.json [{key}]: ERROR  ({type(e).__name__}: {e})')
    return num, failed


# ---- Contact sheet and driver ----------------------------------------------------------------------------------
def contact_sheet(out, entries):
    """Thumbnails of the figures listed in cpan_2026.json contact_sheet_ids (in that order)."""
    by_id = {e['id']: e for e in entries}
    ids = list(CFG['contact_sheet_ids'])
    ok = [(i, out / f) for i in ids if by_id.get(i, {}).get('status') == 'OK'
          for f in by_id[i]['files'] if f.endswith('.png')]
    bad = [f"{i} {by_id[i]['status'] if i in by_id else 'NOT_RUN'}" for i in ids
           if by_id.get(i, {}).get('status') != 'OK']
    ncol = 3
    nrow = max(1, math.ceil(len(ok) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.9 * nrow + (0.5 if bad else 0)), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (fid, p) in zip(axes.flat, ok):
        ax.imshow(plt.imread(p))
        ax.set_title(fid, fontsize=16, fontweight='bold', loc='left')
    if bad:
        fig.text(0.01, 0.005, 'Not produced: ' + ', '.join(bad), fontsize=13, color=COL['error_text'])
    fig.tight_layout()
    fig.savefig(out / 'contact_sheet.png', dpi=110, facecolor=COL['background'])
    plt.close(fig)
    return dict(file='contact_sheet.png', figures=[i for i, _ in ok], not_produced=bad)


NUMBERS_ID = 'numbers'


def run(data_root, out_base=None, only=None, resume_run=None):
    """Produce the CPAN set.

    only: None = all figures + numbers.json; else a list of group IDs (F06-F12), output IDs (e.g. F09a,
    F07_chi_eff: the whole group is regenerated) and/or 'numbers'.
    resume_run: an existing run_<UTC> directory. The selected items are regenerated in it (their files are
    replaced) and manifest.json, numbers.json and contact_sheet.png are merged with what the run already has;
    without it a new run_<UTC> directory is created.
    """
    data_root = Path(data_root)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    valid_ids = list(dict.fromkeys([g for g, _, _ in FIGURES] + list(GROUP_OF) + [NUMBERS_ID]))
    sel = valid_ids if not only else [x.strip() for x in only if x.strip()]
    bad = [x for x in sel if x not in valid_ids]
    require(not bad, f'--only: unknown IDs {bad}; valid: {valid_ids}')
    groups = [g for g, _, _ in FIGURES if g in sel or any(GROUP_OF.get(x) == g for x in sel)]
    old = None
    if resume_run is not None:
        out = Path(resume_run)
        require(out.is_dir() and out.name.startswith('run_') and (out / 'manifest.json').is_file(),
                f'--resume-run {out}: not a run_<UTC> directory with manifest.json')
        old = json.loads((out / 'manifest.json').read_text())
    else:
        base = Path(out_base) if out_base is not None else data_root / 'results' / 'presentation' / 'cpan_2026'
        out = base / f'run_{stamp}'
        require(not out.exists(), f'{out} exists; refusing to overwrite')
        out.mkdir(parents=True)
    set_style()
    I = Inputs(data_root)
    here = Path(__file__).resolve()
    script_sha = digest(here)

    def safe_entry(path):
        try:
            return I.entry(path)
        except Exception as e:  # noqa: BLE001 - an unreadable input must not prevent writing the manifest
            return dict(path=str(path), sha256=None, status='ERROR', error=f'{type(e).__name__}: {e}')

    new, numbers = {}, None
    try:
        for gid, fn, outs in FIGURES:
            if gid not in groups:
                continue
            I.begin()
            try:
                results, status, reason = fn(I, out), 'OK', None
            except Missing as e:
                results, status, reason = {}, 'MISSING', str(e)
            except FailedCheck as e:
                results, status, reason = {}, 'FAILED_CHECK', f'check failed: {e}'
            except Exception as e:  # noqa: BLE001 - recorded in the manifest, reported in the exit status
                results, status, reason = {}, 'ERROR', f'{type(e).__name__}: {e}'
            plt.close('all')
            inputs = [safe_entry(p) for p in dict.fromkeys(str(p) for p in I.used)]
            for oid, name in outs:  # same inputs for every output of the group (computed once)
                res = dict(results.get(oid) or {})
                st, why = status, reason
                if st == 'OK' and not res:
                    st, why = 'ERROR', 'not produced by its group function'
                new[oid] = dict(id=oid, group=gid, name=name, status=st, files=res.pop('files', []),
                                generated_utc=stamp, script_sha256=script_sha, inputs=inputs,
                                **({'reason': why} if why else {}), **res)
                print(f'{oid}: {st}' + (f'  ({why})' if why else ''))
        if NUMBERS_ID in sel:
            I.begin()
            numbers, _ = compute_numbers(I)
    finally:
        # Always written, also after an unexpected exception (which is re-raised after this block).
        # Old entries of a re-run group (e.g. the former combined 'F09') are moved to 'superseded' (their files stay
        # on disk); entries of groups not re-run are kept as they were.
        prev_all = list((old or {}).get('figures', []))
        rerun = set(groups)
        superseded = list((old or {}).get('superseded', [])) + [
            dict(e, superseded_utc=stamp) for e in prev_all
            if e.get('group', e['id']) in rerun and e['id'] not in new]
        prev = {e['id']: e for e in prev_all if e.get('group', e['id']) not in rerun}
        entries = [new.get(oid, prev.get(oid)) for _, _, outs in FIGURES for oid, _ in outs if oid in new or oid in prev]
        entries += [e for i, e in prev.items() if i not in NAME]
        if NUMBERS_ID in sel:
            if numbers is None:
                numbers = {'run': dict(status='ERROR', error='interrupted before numbers.json was computed')}
            n_failed = [k for k, v in numbers.items() if isinstance(v, dict) and v.get('status') == 'ERROR']
            n_status = 'ERROR' if n_failed else 'OK'
            n_inputs = [safe_entry(p) for p in dict.fromkeys(str(p) for p in I.used)]
            (out / 'numbers.json').write_text(json.dumps(to_py(dict(status=n_status, created_utc=stamp, **numbers)),
                                                         indent=2, ensure_ascii=False))
            n_entry = dict(file='numbers.json', status=n_status, failed_keys=n_failed, generated_utc=stamp,
                           script_sha256=script_sha, inputs=n_inputs)
            print(f'numbers.json: {n_status}' + (f'  (failed: {n_failed})' if n_failed else ''))
        else:  # kept as it is in the resumed run (or not produced in a new partial run)
            n_entry = (old or {}).get('numbers') or dict(file=None, status='NOT_RUN')
        try:
            sheet = contact_sheet(out, entries)
        except Exception as e:  # noqa: BLE001
            plt.close('all')
            sheet = dict(file=None, status='ERROR', error=f'{type(e).__name__}: {e}')
        problems = [f"{e['id']}: {e['status']} ({e.get('reason', '')})" for e in new.values() if e['status'] != 'OK']
        if n_entry.get('status') == 'ERROR' and NUMBERS_ID in sel:
            problems += [f"numbers.json [{k}]: ERROR ({numbers[k]['error']})" for k in n_entry['failed_keys']]
        if sheet.get('status') == 'ERROR':
            problems.append(f"contact_sheet.png: ERROR ({sheet['error']})")
        update = dict(utc=stamp, items=sel, script_sha256=script_sha, problems=problems,
                      configs=[dict(path=str(p), sha256=digest(p)) for p in (BASE_CFG, CPAN_CFG)])
        manifest = dict(
            created_utc=(old or {}).get('created_utc', stamp), updated_utc=stamp, run_dir=str(out),
            data_root=str(data_root), script=dict(path=str(here), sha256=script_sha),
            configs=update['configs'],
            artifacts=dict(model=TAG, m14_5_summary_stamp=M14_5_STAMP, m14_7_run=M14_7_STAMP,
                           real_events_run=REAL_STAMP),
            checks=list(dict.fromkeys(list((old or {}).get('checks', [])) + I.checks)),
            info={**(old or {}).get('info', {}), **to_py(I.info)}, figures=entries, numbers=n_entry,
            superseded=superseded, contact_sheet=sheet, problems_last_update=problems,
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
