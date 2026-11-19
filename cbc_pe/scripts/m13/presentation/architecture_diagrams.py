"""Architecture diagrams (paper style). Each part is saved as its own figure:
  arch_cnn_tr_overall / arch_cnn_tr_resblock / arch_cnn_tr_transformer   (baseline CNN + transformer aggregation)
  arch_dense_overall  / arch_dense_block    / arch_dense_layer          (multi-resolution DenseNet + transformer)
Shapes and parameter counts follow src/models/network.py (computed by hand; check with count_model_cost.py).
fig_tbank_overall / fig_tbank_filterbank / fig_cholesky_head (final model M14.5b: learnable filter bank + CNN +
transformer + Cholesky head) take the "model" block of a training config and compute the shapes from it; they are
drawn by scripts/m13/presentation/cpan_physics_figures.py, not by the __main__ below.
Run: python scripts/m13/presentation/architecture_diagrams.py --out DIR
"""
import argparse
from pathlib import Path
from matplotlib.transforms import Bbox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

RC = {'font.family': 'DejaVu Sans', 'font.size': 10, 'mathtext.fontset': 'dejavusans'}   # also used via rc_context
plt.rcParams.update(RC)
INK = '#1F2D3A'; MUT = '#5B6B7A'; LINE = '#3A4A5A'
C_IN = '#E8ECF1'; C_CONV = '#CFE0F0'; C_RES = '#A9C4E0'; C_PROJ = '#DCE6F0'; C_POOL = '#EADFC6'
C_TR = '#D3D1EA'; C_HEAD = '#CFE3D5'; C_OUT = '#F2F4F7'; C_ATT = '#E6D3D3'; C_DEN = '#BFD6EA'


def box(ax, x, y, w, h, title, body=None, fc=C_CONV, ec=LINE, fs=10.5, bfs=8.6, lw=1.0, r=0.06, tcol=INK, title_dy=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=2))
    if body:
        ty = y + h - (title_dy if title_dy is not None else 0.22)
        ax.text(x + w / 2, ty, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=tcol, zorder=3)
        ax.text(x + w / 2, ty - 0.2, body, ha='center', va='top', fontsize=bfs, color=INK, zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=fs, fontweight='bold', color=tcol, zorder=3, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, lw=1.1, col=LINE, ms=9, style='-|>', ls='-', cs='arc3'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, lw=lw, color=col, zorder=1, linestyle=ls, connectionstyle=cs, shrinkA=0, shrinkB=0))


def shape(ax, x, y, s, fs=8.3, col=MUT, ha='center'):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs, color=col, family='DejaVu Sans Mono', zorder=3)


def panel(ax, x, y, s):
    s = s.split(')  ', 1)[-1]
    ax.text(x, y, s, fontsize=12, fontweight='bold', color=INK, va='top')


def bracket(ax, x0, x1, y, label, col=MUT):
    ax.plot([x0, x0, x1, x1], [y - 0.08, y, y, y - 0.08], color=col, lw=0.9)
    ax.text((x0 + x1) / 2, y + 0.1, label, ha='center', va='bottom', fontsize=9, color=col, style='italic')


def oplus(ax, x, y, r=0.13):
    ax.add_patch(plt.Circle((x, y), r, fc='white', ec=LINE, lw=1.1, zorder=3))
    ax.plot([x - r * 0.6, x + r * 0.6], [y, y], color=LINE, lw=1.1, zorder=4)
    ax.plot([x, x], [y - r * 0.6, y + r * 0.6], color=LINE, lw=1.1, zorder=4)


# =====================================================================
# Figure 1: reference model (CNN encoder + transformer aggregation)
# =====================================================================
def fig_reference():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')

    panel(ax, 0.25, 8.85, '(a)  Overall architecture')
    y, h = 6.05, 1.75
    W = [1.55, 1.85, 1.85, 1.45, 1.3, 2.1, 1.55, 1.35]
    gap = 0.38
    xs = [0.3]
    for w in W[:-1]:
        xs.append(xs[-1] + w + gap)
    specs = [
        ('Input', 'whitened strain\nH1, L1, V1\n4 s @ 4096 Hz\nz-score per\ndetector', C_IN),
        ('Conv stem ×3', 'Conv1d k=16, s=2\nGroupNorm\nLeakyReLU, Dropout\nchannels\n3→16→32→64', C_CONV),
        ('Residual\ndilated ×3', 'k=7, 64 ch\ndilation 1, 2, 4\n(see b)', C_RES),
        ('Projection', 'Conv1d 1×1\n64→128\nGN, LeakyReLU', C_PROJ),
        ('Tokenise', 'AvgPool1d\nk=8, s=8', C_POOL),
        ('Transformer\nencoder ×2', '+ sinusoidal PE\n+ learned [CLS]\nd=128, 4 heads\nFFN 256, GELU\n(see c)', C_TR),
        ('MLP', 'Linear 128→64\nLeakyReLU, Dropout\nLinear 64→64\nLeakyReLU', C_HEAD),
        ('Output', '$\\mathcal{M}_c$,  $M$,  $\\chi_{\\rm eff}$\n(standardised)', C_OUT),
    ]
    out_shapes = ['3 × 16384', '64 × 2049', '64 × 2049', '128 × 2049', '128 × 256', '128 ([CLS])', '64', '3']
    for i, (x, w, (t, b, c)) in enumerate(zip(xs, W, specs)):
        tdy = 0.36 if '\n' in t else 0.24
        box(ax, x, y, w, h, t, b, fc=c, title_dy=tdy)
        shape(ax, x + w / 2, y - 0.22, out_shapes[i])
        if i < len(W) - 1:
            arrow(ax, x + w, y + h / 2, xs[i + 1], y + h / 2)
    # brackets
    bracket(ax, xs[1], xs[3] + W[3], y + h + 0.2, 'local feature extractor (CNN): receptive field 778 samples ≈ 0.19 s')
    bracket(ax, xs[4], xs[5] + W[5], y + h + 0.2, 'temporal aggregation over the whole 4 s window')
    bracket(ax, xs[6], xs[7] + W[7], y + h + 0.2, 'regression head')
    ax.text(0.3, y - 0.6, 'Tensor shapes: channels × time steps (batch omitted).   One token = 64 input samples = 15.6 ms;  256 tokens span the 4 s window.   '
            'Trainable parameters: 501,523  (CNN encoder + head 236,435;  transformer 265,088).',
            fontsize=9, color=MUT, va='center')
    ax.plot([0.25, 15.75], [4.75, 4.75], color='#C9D1DA', lw=0.8)

    # ---- (b) residual dilated block ----
    panel(ax, 0.25, 4.55, '(b)  Residual dilated block  (d = 1, 2, 4)')
    bx, bw, bh = 0.75, 1.0, 0.55
    yb = 2.75
    seq = [('Conv1d', 'k=7, dilation d'), ('GroupNorm', '8 groups'), ('LeakyReLU', '+ Dropout'), ('Conv1d', 'k=7, dilation d'), ('GroupNorm', '8 groups')]
    xx = bx + 0.55
    shape(ax, bx - 0.05, yb + bh / 2, 'x\n64×T', fs=8.3, col=INK)
    arrow(ax, bx + 0.3, yb + bh / 2, xx, yb + bh / 2)
    pos = []
    for i, (t, s_) in enumerate(seq):
        fc = C_RES if t == 'Conv1d' else ('#E3EAF2' if t == 'GroupNorm' else '#EEF1F4')
        box(ax, xx, yb, bw, bh, t, fc=fc, fs=9)
        ax.text(xx + bw / 2, yb - 0.17, s_, ha='center', va='center', fontsize=7.8, color=MUT)
        pos.append(xx)
        if i < len(seq) - 1:
            arrow(ax, xx + bw, yb + bh / 2, xx + bw + 0.15, yb + bh / 2)
        xx += bw + 0.15
    ox = xx + 0.12
    arrow(ax, xx - 0.15, yb + bh / 2, ox - 0.13, yb + bh / 2)
    oplus(ax, ox, yb + bh / 2)
    # skip
    ax.plot([bx + 0.42, bx + 0.42, ox, ox], [yb + bh / 2, yb + bh + 0.45, yb + bh + 0.45, yb + bh / 2 + 0.13], color=LINE, lw=1.1, zorder=1)
    ax.text((bx + ox) / 2, yb + bh + 0.55, 'identity (skip connection)', ha='center', fontsize=8.3, color=MUT)
    box(ax, ox + 0.35, yb, 1.0, bh, 'LeakyReLU', fc='#EEF1F4', fs=9)
    arrow(ax, ox + 0.13, yb + bh / 2, ox + 0.35, yb + bh / 2)
    shape(ax, ox + 1.68, yb + bh / 2, 'y\n64×T', fs=8.3, col=INK)
    arrow(ax, ox + 1.35, yb + bh / 2, ox + 1.47, yb + bh / 2)
    # dilation sketch
    sx, sy = 1.2, 0.85
    ax.text(sx - 0.45, sy + 1.25, 'Taps of one k=7 kernel at the CNN rate (1 step = 8 input samples ≈ 2 ms):', fontsize=8.3, color=MUT)
    for r_, d in enumerate((1, 2, 4)):
        yy = sy + 0.85 - r_ * 0.38
        ax.text(sx - 0.45, yy, f'd={d}', fontsize=8.3, color=INK, va='center')
        for t in range(25):
            ax.add_patch(Rectangle((sx + t * 0.27, yy - 0.08), 0.2, 0.16, fc='#EEF1F4', ec='#C3CCD6', lw=0.5))
        for k in range(7):
            ax.add_patch(Rectangle((sx + (12 - 3 * d + k * d) * 0.27, yy - 0.08), 0.2, 0.16, fc='#6F91B6', ec='#3F6A9A', lw=0.6))
    ax.text(sx - 0.45, sy - 0.45, 'Span 7, 13 and 25 steps per convolution; two convolutions per block.', fontsize=8.3, color=MUT, va='center')
    ax.plot([9.05, 9.05], [0.25, 4.45], color='#C9D1DA', lw=0.8)

    # ---- (c) transformer encoder layer ----
    panel(ax, 9.3, 4.55, '(c)  Transformer encoder layer  (×2, post-norm)')
    cx, cw = 10.7, 3.2
    shape(ax, cx + cw / 2, 4.22, '[CLS] + 256 tokens  →  257 × 128', fs=8.3, col=INK)
    layers = [('Multi-head self-attention', '4 heads × 32 dims', C_TR, 3.25), ('Add & LayerNorm', None, '#EEF1F4', 2.55),
              ('Feed-forward', 'Linear 128→256 → GELU → Linear 256→128', C_TR, 1.6), ('Add & LayerNorm', None, '#EEF1F4', 0.9)]
    prev = 4.08
    for t, s_, fc, yy in layers:
        hh = 0.55 if s_ else 0.4
        box(ax, cx, yy, cw, hh, t, fc=fc, fs=9.3, title_dy=0.2 if s_ else None, body=s_, bfs=8)
        arrow(ax, cx + cw / 2, prev, cx + cw / 2, yy + hh)
        prev = yy
    # residual (skip) connections
    for ys_, yt in ((3.97, 2.75), (2.4, 1.1)):
        ax.plot([cx + cw / 2, cx - 0.3, cx - 0.3], [ys_, ys_, yt], color=LINE, lw=1.0, zorder=1)
        arrow(ax, cx - 0.3, yt, cx, yt)
    arrow(ax, cx + cw / 2, 0.9, cx + cw / 2, 0.55)
    shape(ax, cx + cw / 2, 0.4, 'after layer 2: keep the [CLS] vector (128)', fs=8.3, col=INK)
    ax.text(cx + cw + 0.25, 2.6, 'Attention dropout 0;\nall other dropout is\nswitched off in the\nlast 20 % of epochs.', fontsize=8.3, color=MUT, va='center')
    return fig


# =====================================================================
# Figure 2: multi-resolution dense network
# =====================================================================
def fig_mrdense():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
    panel(ax, 0.25, 8.85, '(a)  Overall architecture: 7 resolution stages, per-scale read-out, transformer on the coarsest scale')
    T = [16384, 8192, 4096, 2048, 1024, 512, 256]
    C = [16, 24, 32, 48, 64, 96, 128]
    G = [9, 12, 15, 24, 33, 48, 63]
    OUT = [34, 48, 62, 96, 130, 192, 254]
    ys = 6.55; hs = 1.15
    box(ax, 0.3, ys, 0.9, hs, 'Input', '3 × 16384\nH1, L1, V1', fc=C_IN, fs=9.5, bfs=8)
    box(ax, 1.42, ys, 0.75, hs, '1×1\nconv', fc=C_PROJ, fs=9)
    arrow(ax, 1.2, ys + hs / 2, 1.42, ys + hs / 2)
    x = 2.45; bw = 1.0; tg = 0.36
    xsb = []
    for s_ in range(7):
        xsb.append(x)
        box(ax, x, ys, bw, hs, f'Dense\nblock {s_}', fc=C_DEN, fs=9, title_dy=0.36, body=f'\n\n{C[s_]}→{OUT[s_]} ch', bfs=8)
        shape(ax, x + bw / 2, ys + hs + 0.17, f'T={T[s_]}', fs=8)
        if s_ == 0:
            arrow(ax, 2.17, ys + hs / 2, x, ys + hs / 2)
        if s_ < 6:
            arrow(ax, x + bw, ys + hs / 2, x + bw + tg, ys + hs / 2)
            ax.text(x + bw + tg / 2, ys + hs / 2 + 0.13, '↓2', ha='center', fontsize=8, color=MUT)
        box(ax, x + 0.05, 5.2, bw - 0.1, 0.6, 'attention\npool', fc=C_ATT, fs=8)
        arrow(ax, x + bw / 2, ys, x + bw / 2, 5.8)
        arrow(ax, x + bw / 2, 5.2, x + bw / 2, 4.82)
        shape(ax, x + bw / 2, 4.68, f'{OUT[s_]}', fs=8)
        x += bw + tg
    xt = x - tg + 0.3; tw = 1.45
    arrow(ax, xsb[-1] + bw, ys + hs / 2, xt, ys + hs / 2)
    box(ax, xt, ys - 0.15, tw, hs + 0.3, 'Transformer\nencoder ×2', '1×1 conv 254→128\n+ PE + [CLS]\n(as in reference)', fc=C_TR, fs=9, bfs=7.8, title_dy=0.36)
    arrow(ax, xt + tw / 2, ys - 0.15, xt + tw / 2, 4.82)
    shape(ax, xt + tw / 2, 4.68, '128 [CLS]', fs=8)
    ax.add_patch(FancyBboxPatch((xsb[0], 3.95), xt + tw - xsb[0], 0.5, boxstyle='round,pad=0,rounding_size=0.05', fc='#F2F4F7', ec=LINE, lw=1.0))
    ax.text((xsb[0] + xt + tw) / 2, 4.2, 'concatenate   816 (per-scale) + 128 ([CLS]) = 944', ha='center', va='center', fontsize=9.3, fontweight='bold', color=INK)
    for s_ in range(7):
        arrow(ax, xsb[s_] + bw / 2, 4.56, xsb[s_] + bw / 2, 4.45)
    arrow(ax, xt + tw / 2, 4.56, xt + tw / 2, 4.45)
    hx = xt + tw + 0.3
    arrow(ax, xt + tw, 4.2, hx, 4.2)
    box(ax, hx, 3.7, 1.35, 1.0, 'MLP + linear', '944→128→64→3', fc=C_HEAD, fs=9, bfs=8, title_dy=0.3)
    ax.text(0.3, 5.5, 'Between stages:\nGN → LeakyReLU →\n1×1 conv → AvgPool(2)', fontsize=8.3, color=MUT, va='center')
    ax.text(0.3, 4.6, 'Per-scale read-out:\nsoftmax-weighted\nsum over time', fontsize=8.3, color=MUT, va='center')
    ax.text(0.3, 3.45, 'Parameters: 1,056,115.   Tensor shapes: channels × time steps.   Gradient checkpointing on stages 0–2 (memory only).', fontsize=9, color=MUT)
    ax.plot([0.25, 15.75], [3.15, 3.15], color='#C9D1DA', lw=0.8)

    # ---- (b) dense block ----
    panel(ax, 0.25, 2.95, '(b)  Dense block (2 layers, growth g)')
    by = 1.15; cy = by + 0.3
    shape(ax, 0.45, cy, 'x₀\n(c)', fs=8.6, col=INK)
    box(ax, 0.95, by, 0.9, 0.6, 'layer 1', fc=C_DEN, fs=9)
    box(ax, 2.6, by, 0.9, 0.6, 'layer 2', fc=C_DEN, fs=9)
    for cxx in (2.05, 3.75):
        ax.add_patch(FancyBboxPatch((cxx, by + 0.12), 0.35, 0.36, boxstyle='round,pad=0,rounding_size=0.04', fc='white', ec=LINE, lw=1))
        ax.text(cxx + 0.175, cy, 'C', ha='center', va='center', fontsize=9, fontweight='bold')
    arrow(ax, 0.7, cy, 0.95, cy)
    arrow(ax, 1.85, cy, 2.05, cy); ax.text(1.95, cy + 0.08, 'x₁', ha='center', fontsize=8.6, color=INK)
    arrow(ax, 2.4, cy, 2.6, cy)
    arrow(ax, 3.5, cy, 3.75, cy); ax.text(3.62, cy + 0.08, 'x₂', ha='center', fontsize=8.6, color=INK)
    # x0 skip (above) to both concatenations
    ax.plot([0.8, 0.8, 3.925], [cy, by + 1.05, by + 1.05], color=LINE, lw=1)
    arrow(ax, 2.225, by + 1.05, 2.225, by + 0.48); arrow(ax, 3.925, by + 1.05, 3.925, by + 0.48)
    ax.text(1.4, by + 1.1, 'x₀', ha='center', fontsize=8.6, color=INK)
    # x1 skip (below) to the block output
    ax.plot([1.95, 1.95, 3.925], [cy, by - 0.35, by - 0.35], color=LINE, lw=1)
    arrow(ax, 3.925, by - 0.35, 3.925, by + 0.12)
    ax.text(2.9, by - 0.3, 'x₁', ha='center', fontsize=8.6, color=INK)
    arrow(ax, 4.1, cy, 4.35, cy)
    shape(ax, 4.95, cy, '[x₀, x₁, x₂]\nc + 2g', fs=8.6, col=INK)
    ax.text(0.3, 0.3, 'C = concatenation along channels. Layer 2 sees [x₀, x₁]; the block\noutput is [x₀, x₁, x₂]: each layer adds g new channels.', fontsize=8.3, color=MUT)
    ax.plot([5.75, 5.75], [0.3, 2.9], color='#C9D1DA', lw=0.8)

    # ---- (c) dense layer ----
    panel(ax, 5.95, 2.95, '(c)  Dense layer: bottleneck, then parallel multi-kernel branches')
    ly = 1.35
    seq = [('GN, LReLU', '#EEF1F4'), ('Conv1d 1×1\n→ 2g', C_PROJ), ('GN, LReLU', '#EEF1F4')]
    x = 6.3
    shape(ax, 6.12, ly + 0.3, 'in', fs=8.3, col=INK)
    for t, fc in seq:
        box(ax, x, ly, 0.95, 0.6, t, fc=fc, fs=8.3)
        arrow(ax, x + 0.95, ly + 0.3, x + 1.12, ly + 0.3)
        x += 1.12
    for jj, k in enumerate((3, 9, 27)):
        yy = ly + 0.75 - jj * 0.75
        box(ax, x + 0.2, yy, 1.2, 0.45, f'Conv1d k={k}', fc=C_RES, fs=8.3)
        arrow(ax, x - 0.05, ly + 0.3, x + 0.2, yy + 0.225)
        arrow(ax, x + 1.4, yy + 0.225, x + 1.7, ly + 0.3)
        ax.text(x + 0.8, yy - 0.06, 'g/3 ch', ha='center', va='top', fontsize=7.3, color=MUT)
    ax.add_patch(FancyBboxPatch((x + 1.7, ly + 0.12), 0.35, 0.36, boxstyle='round,pad=0,rounding_size=0.04', fc='white', ec=LINE, lw=1))
    ax.text(x + 1.875, ly + 0.3, 'C', ha='center', va='center', fontsize=9, fontweight='bold')
    box(ax, x + 2.25, ly, 0.85, 0.6, 'Dropout', fc='#EEF1F4', fs=8.3)
    arrow(ax, x + 2.05, ly + 0.3, x + 2.25, ly + 0.3)
    arrow(ax, x + 3.1, ly + 0.3, x + 3.27, ly + 0.3)
    shape(ax, x + 3.42, ly + 0.3, 'g', fs=8.3, col=INK)
    return fig


STAGE_CHANNELS = (16, 24, 32, 48, 64, 96, 128)   # MRDenseNet config (configs/experiments/m13/train_m13_4b_G3aug_s123.json)
KERNELS, DENSE_LAYERS, FS, N_IN = (3, 9, 27), 2, 4096.0, 16384


def growth(c, n_k=3):
    return max(n_k, n_k * int(c / 2 / n_k + 0.5))   # as in MRDenseNet.__init__


def stage_rows():
    rows, rf = [], 1
    for s, c in enumerate(STAGE_CHANNELS):
        st = 2**s; g = growth(c)
        rf_s = rf + DENSE_LAYERS * (max(KERNELS) - 1) * st
        rows.append([str(s), f'{N_IN // st}', f'{st / FS * 1e3:.2f}', f'{c}', f'{g}', f'{c + DENSE_LAYERS * g}',
                     ' / '.join(f'{k * st / FS * 1e3:.2g}' if k * st / FS * 1e3 < 10 else f'{k * st / FS * 1e3:.0f}' for k in KERNELS),
                     f'{rf_s / FS * 1e3:.0f}'])
        rf = rf_s + st   # AvgPool(2) of the transition
    return rows


def fig_stage_table():
    hdr = ['Stage', '$T$', r'$\Delta t$ [ms]', r'$c_{\mathrm{in}}$', '$g$', r'$c_{\mathrm{out}} = c_{\mathrm{in}} + 2g$',
           r'$k = 3\,/\,9\,/\,27$ span [ms]', 'RF [ms]']
    rows = stage_rows()
    colx = [0.04, 0.15, 0.27, 0.39, 0.48, 0.60, 0.79, 0.95]
    with plt.rc_context({'mathtext.fontset': 'cm', 'font.family': 'serif', 'font.serif': ['DejaVu Serif']}):
        fig = plt.figure(figsize=(11, 3.6)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
        n = len(rows); top, dy = 0.88, 0.095
        ax.plot([0.01, 0.99], [top + 0.07, top + 0.07], color=INK, lw=1.4)
        for x, h in zip(colx, hdr):
            ax.text(x, top, h, ha='center', va='center', fontsize=13, color=INK)
        ax.plot([0.01, 0.99], [top - 0.055, top - 0.055], color=INK, lw=0.8)
        for r, row in enumerate(rows):
            for x, v in zip(colx, row):
                ax.text(x, top - 0.11 - r * dy, v, ha='center', va='center', fontsize=13, color=INK)
        ax.plot([0.01, 0.99], [top - 0.11 - (n - 1) * dy - 0.06] * 2, color=INK, lw=1.4)
    return fig


# =====================================================================
# Figure 3: final model (M14.5b): learnable filter bank + CNN encoder + transformer + Cholesky head
# Shapes are computed from the config "model" block; the constants below are fixed in the classes.
# =====================================================================
C_BANK = '#F0D9C4'
LOG_L_CLAMP = (-7, 3)     # SimpleCNN_ResidualDilatedTransformerAgg.LOG_SIGMA_MIN / LOG_SIGMA_MAX (clamp of log L_ii)
CNN_CHANNELS = (16, 32)   # block1 / block2 outputs of SimpleCNN_ResidualDilated (block3 = residual_channels)
D_MODEL, FFN = 128, 256   # projection / TokenTransformer width and feed-forward size (fixed in the classes)
CNN_STRIDE = 2 ** 3       # three ConvBlocks with stride 2


def conv_out_len(n, k=16, s=2):
    """Output length of ConvBlock: Conv1d(kernel k, stride s, padding k // 2)."""
    return (n + 2 * (k // 2) - k) // s + 1


def tbank_shapes(model_cfg, fs=4096.0):
    """Tensor sizes (batch omitted) of SimpleCNN_ResidualDilatedTransformerAgg with template_bank and full covariance."""
    ok = (model_cfg.get('class_name') == 'SimpleCNN_ResidualDilatedTransformerAgg' and model_cfg.get('template_bank')
          and model_cfg.get('heteroscedastic') and model_cfg.get('covariance') == 'full')
    if not ok:
        raise ValueError('needs SimpleCNN_ResidualDilatedTransformerAgg with template_bank, heteroscedastic, '
                         'covariance="full"')
    kw, tb = model_cfg['kwargs'], model_cfg['template_bank']
    d, n, k, no = kw['n_detectors'], tb['n_samples'], tb['n_templates'], kw['n_outputs']
    t = [n]
    for _ in range(3):
        t.append(conv_out_len(t[-1]))
    return dict(D=d, N=n, K=k, Lt=tb['template_length'], bank=d * k, cat=d * (1 + k), F=n // 2 + 1,
                conv=list(zip(CNN_CHANNELS + (kw['residual_channels'],), t[1:])), T=t[-1],
                rc=kw['residual_channels'], dil=[int(x) for x in kw['dilations']], rk=kw['residual_kernel_size'],
                pool=kw['agg_pool'], tok=t[-1] // kw['agg_pool'], tok_ms=CNN_STRIDE * kw['agg_pool'] / fs * 1e3,
                layers=kw['agg_layers'], heads=kw['agg_heads'], emb=kw['embedding_dim'], no=no,
                head_out=no + no * (no + 1) // 2, n_off=no * (no - 1) // 2, fs=fs, dur=n / fs)


def concat_node(ax, x, y, r=0.17):
    ax.add_patch(FancyBboxPatch((x - r, y - r), 2 * r, 2 * r, boxstyle='round,pad=0,rounding_size=0.04', fc='white',
                                ec=LINE, lw=1.0, zorder=3))
    ax.text(x, y, 'C', ha='center', va='center', fontsize=9, fontweight='bold', color=INK, zorder=4)


def on_arrow(ax, x0, x1, y, s, dy=0.2):
    """Tensor shape written above a horizontal arrow."""
    shape(ax, (x0 + x1) / 2, y + dy, s)


def fig_tbank_overall(model_cfg, source=''):
    s = tbank_shapes(model_cfg)
    (c1, t1), (c2, t2), (c3, t3) = s['conv']
    fig = plt.figure(figsize=(16, 7.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 7.0); ax.axis('off')
    h, y1, y2 = 1.75, 4.1, 1.0
    m1, m2 = y1 + h / 2, y2 + h / 2
    # ---- row 1: input, filter bank, concatenation, CNN encoder ----
    box(ax, 0.3, y1, 1.6, h, 'Input', f"whitened strain\nH1, L1, V1\n{s['dur']:.0f} s at {s['fs']:.0f} Hz\nz-score per\n"
        'detector', fc=C_IN, title_dy=0.24)
    xb, wb, xc = 3.3, 2.4, 6.75
    box(ax, xb, y1, wb, h, 'Learnable\nfilter bank',
        f"{s['K']} quadrature pairs\n" + r'$h^{I}_{k}$, $h^{Q}_{k}$' + f" of {s['Lt']} samples\nmatched filtering (FFT)\n"
        + r'$|c| = \sqrt{c_I^{2} + c_Q^{2}}$', fc=C_BANK, title_dy=0.36)
    arrow(ax, 1.9, m1, xb, m1); on_arrow(ax, 1.9, xb, m1, f"{s['D']} × {s['N']}")
    arrow(ax, xb + wb, m1, xc - 0.17, m1); on_arrow(ax, xb + wb, xc - 0.17, m1, f"{s['bank']} × {s['N']}")
    concat_node(ax, xc, m1)
    yb = y1 + h + 0.3   # identity path of the input around the bank
    ax.plot([2.55, 2.55, xc], [m1, yb, yb], color=LINE, lw=1.0, zorder=1)
    arrow(ax, xc, yb, xc, m1 + 0.17)
    ax.text((2.55 + xc) / 2, yb + 0.08, 'input unchanged', ha='center', va='bottom', fontsize=8.3, color=MUT)
    xv, wv = 8.25, 2.5
    arrow(ax, xc + 0.17, m1, xv, m1); on_arrow(ax, xc + 0.17, xv, m1, f"{s['cat']} × {s['N']}")
    box(ax, xv, y1, wv, h, 'ConvBlock ×3', f"Conv1d k=16, s=2\nGroupNorm, LeakyReLU,\nDropout\n"
        f"channels {s['cat']}→{c1}→{c2}→{c3}\nT: {s['N']}→{t1}→{t2}→{t3}", fc=C_CONV, title_dy=0.24)
    xr, wr = 12.05, 2.4
    arrow(ax, xv + wv, m1, xr, m1); on_arrow(ax, xv + wv, xr, m1, f"{c3} × {t3}")
    box(ax, xr, y1, wr, h, f"Residual\ndilated ×{len(s['dil'])}",
        f"k={s['rk']}, {s['rc']} ch\ndilation {', '.join(str(x) for x in s['dil'])}\nGroupNorm, LeakyReLU\n"
        'identity skip', fc=C_RES, title_dy=0.36)
    bracket(ax, xb, xb + wb, yb + 0.42, 'learned matched filters over the whole window')
    bracket(ax, xv, xr + wr, y1 + h + 0.2, 'local feature extractor (CNN)')
    # ---- wrap to row 2 ----
    xw, yw = xr + wr / 2, (y1 + y2 + h) / 2
    xp, wp = 0.3, 1.9
    ax.plot([xw, xw, xp + wp / 2], [y1, yw, yw], color=LINE, lw=1.1, zorder=1)
    arrow(ax, xp + wp / 2, yw, xp + wp / 2, y2 + h)
    shape(ax, xw + 0.12, (y1 + yw) / 2, f"{s['rc']} × {t3}", ha='left')
    # ---- row 2: projection, tokens, transformer, head ----
    box(ax, xp, y2, wp, h, 'Projection', f"Conv1d 1×1\n{s['rc']}→{D_MODEL}\nGroupNorm,\nLeakyReLU", fc=C_PROJ,
        title_dy=0.24)
    xa, wa = 3.6, 1.9
    arrow(ax, xp + wp, m2, xa, m2); on_arrow(ax, xp + wp, xa, m2, f'{D_MODEL} × {t3}')
    box(ax, xa, y2, wa, h, 'Tokenise', f"AvgPool1d\nk={s['pool']}, s={s['pool']}\n1 token ≈ {s['tok_ms']:.1f} ms",
        fc=C_POOL, title_dy=0.24)
    xt, wt = 6.9, 2.6
    arrow(ax, xa + wa, m2, xt, m2); on_arrow(ax, xa + wa, xt, m2, f"{D_MODEL} × {s['tok']}")
    box(ax, xt, y2, wt, h, f"Transformer\nencoder ×{s['layers']}",
        f"+ sinusoidal PE\n+ learned [CLS]\nd={D_MODEL}, {s['heads']} heads\nFFN {FFN}, GELU", fc=C_TR, title_dy=0.36)
    xh, wh = 10.9, 2.7
    arrow(ax, xt + wt, m2, xh, m2); on_arrow(ax, xt + wt, xh, m2, f'{D_MODEL} ([CLS])')
    box(ax, xh, y2, wh, h, 'Cholesky head',
        f"MLP {D_MODEL}→64→{s['emb']}\nLinear {s['emb']}→{s['head_out']}:\n" + r'$\mu$' + f" ({s['no']}), "
        + r'$\log L_{ii}$' + f" ({s['no']}),\n" + r'$L_{ij}$' + f", i > j ({s['n_off']})", fc=C_HEAD, title_dy=0.24)
    xo, wo = 14.3, 1.45
    arrow(ax, xh + wh, m2, xo, m2); on_arrow(ax, xh + wh, xo, m2, f"{s['head_out']}")
    box(ax, xo, y2, wo, h, 'Output', r'$\mu \in \mathbb{R}^{3}$' + '\n' + r'$\Sigma = L L^{\top}$' + '\nstandardised\nlabels',
        fc=C_OUT, title_dy=0.24)
    bracket(ax, xa, xt + wt, y2 + h + 0.2, 'temporal aggregation over the whole window')
    bracket(ax, xh, xo + wo, y2 + h + 0.2, 'mean and full covariance')
    note = (f"Tensor shapes: channels × time steps (batch omitted)" + (f', from {source}' if source else '')
            + f".  1 token = {CNN_STRIDE * s['pool']} input samples; {s['tok']} tokens span the {s['dur']:.0f} s window.")
    ax.text(0.3, 0.45, note, fontsize=9, color=MUT, va='center')
    return fig


def fig_tbank_filterbank(model_cfg):
    s = tbank_shapes(model_cfg)
    fig = plt.figure(figsize=(16, 5.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 5.6); ax.axis('off')
    panel(ax, 0.25, 5.45, f"Learnable filter bank (one detector; the same {s['K']} filter pairs for "
          f"{', '.join(['H1', 'L1', 'V1'][:s['D']])})")
    mid, yi, yq, hb = 2.95, 3.55, 2.35, 0.7
    shape(ax, 0.55, mid, f"$x_d$\n1 × {s['N']}", col=INK)
    box(ax, 1.2, mid - 0.35, 1.0, 0.7, 'rFFT', fc=C_PROJ, fs=9.5)
    arrow(ax, 0.9, mid, 1.2, mid)
    shape(ax, 1.7, mid - 0.62, f"{s['F']} complex\nfrequency bins")
    xm, wm = 3.0, 2.0
    for yy, q in ((yi, 'I'), (yq, 'Q')):
        arrow(ax, 2.2, mid, xm, yy)
        box(ax, xm, yy - hb / 2, wm, hb, f'× conj($\\hat h^{{{q}}}_{{k}}$)', fc=C_BANK, fs=9.5)
        arrow(ax, xm + wm, yy, 6.0, yy); on_arrow(ax, xm + wm, 6.0, yy, f"{s['K']} × {s['F']}")
        box(ax, 6.0, yy - hb / 2, 1.1, hb, 'irFFT', fc=C_PROJ, fs=9.5)
    # template inputs: h^I above the I branch, h^Q below the Q branch
    for yy, q, ytop in ((4.25, 'I', yi + hb / 2), (1.25, 'Q', yq - hb / 2)):
        box(ax, 3.2, yy, 1.6, 0.4, f"$h^{{{q}}}_{{k}}$:  {s['K']} × {s['Lt']}", fc='#F7EDE3', fs=8.6)
        if q == 'I':
            arrow(ax, 4.0, yy, 4.0, ytop)
        else:
            arrow(ax, 4.0, yy + 0.4, 4.0, ytop)
    xg, wg = 8.3, 2.5
    box(ax, xg, mid - 0.65, wg, 1.3, 'Magnitude', r'$|c_{d,k}(t)| = \sqrt{c_I^{2} + c_Q^{2} + 10^{-12}}$'
        + '\nphase-insensitive\nmatched-filter output', fc=C_BANK, fs=9.5, bfs=8.3, title_dy=0.22)
    for yy, q, dy in ((yi, 'I', 0.2), (yq, 'Q', -0.2)):
        arrow(ax, 7.1, yy, xg, mid + dy)
        shape(ax, 7.55, yy + (0.22 if q == 'I' else -0.22), f"$c_{q}$  {s['K']} × {s['N']}")
    xs_, ws_ = 11.4, 1.5
    arrow(ax, xg + wg, mid, xs_, mid); on_arrow(ax, xg + wg, xs_, mid, f"{s['K']} × {s['N']}")
    box(ax, xs_, mid - 0.5, ws_, 1.0, 'Stack\ndetectors', f"\n\n{s['D']} × {s['K']} channels", fc=C_BANK, fs=9.5,
        bfs=8.3, title_dy=0.36)
    xc = 14.05
    arrow(ax, xs_ + ws_, mid, xc - 0.17, mid); on_arrow(ax, xs_ + ws_, xc - 0.17, mid, f"{s['bank']} × {s['N']}")
    concat_node(ax, xc, mid)
    yb = 4.95
    ax.plot([0.75, 0.75, xc], [mid + 0.25, yb, yb], color=LINE, lw=1.0, zorder=1)
    arrow(ax, xc, yb, xc, mid + 0.17)
    ax.text(7.4, yb + 0.07, f"input unchanged ({s['D']} × {s['N']}, all detectors)", ha='center', va='bottom',
            fontsize=8.3, color=MUT)
    arrow(ax, xc + 0.17, mid, 14.75, mid)
    shape(ax, 15.3, mid, f"{s['cat']} × {s['N']}\nto ConvBlock 1", col=INK)
    ax.text(0.3, 0.55, f"Learnable: {s['K']} pairs " + r'$(h^{I}_{k}, h^{Q}_{k})$' + f" of {s['Lt']} samples, rescaled to unit "
            "L2 norm in every forward. Circular layout: the last template sample (merger end) sits at lag 0, so the "
            "output at t is the match of the data ending at t.\nCorrelation computed by FFT (circular, whole window); "
            "each detector keeps its own correlation series, so the delays between detectors are preserved.",
            fontsize=8.6, color=MUT, va='center', linespacing=1.4)
    return fig


def fig_cholesky_head(model_cfg, loss_note=None):
    s = tbank_shapes(model_cfg)
    no = s['no']
    fig = plt.figure(figsize=(16, 4.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0, 4.9); ax.axis('off')
    panel(ax, 0.25, 4.75, 'Cholesky head: mean and full covariance of the three labels')
    mid = 2.6
    shape(ax, 0.55, mid, f'[CLS]\n{D_MODEL}', col=INK)
    box(ax, 1.1, mid - 0.6, 2.8, 1.2, 'Embedding MLP', f"Linear {D_MODEL}→64, LeakyReLU, Dropout\n"
        f"Linear 64→{s['emb']}, LeakyReLU", fc=C_HEAD, fs=9.5, bfs=8.3, title_dy=0.25)
    arrow(ax, 0.9, mid, 1.1, mid)
    arrow(ax, 3.9, mid, 4.6, mid); on_arrow(ax, 3.9, 4.6, mid, f"{s['emb']}")
    box(ax, 4.6, mid - 0.45, 1.5, 0.9, f"Linear\n{s['emb']}→{s['head_out']}", fc=C_HEAD, fs=9.5)
    xb, wb, hb = 7.0, 2.1, 0.6
    rows = ((3.75, r'$\mu$' + f'  ({no})'), (2.6, r'$\log L_{ii}$' + f'  ({no})'),
            (1.45, r'$L_{ij}$,  i > j' + f"  ({s['n_off']})"))
    for yy, t in rows:
        arrow(ax, 6.1, mid, xb, yy)
        box(ax, xb, yy - hb / 2, wb, hb, t, fc=C_OUT, fs=9.5)
    xk, wk = 9.7, 2.0
    arrow(ax, xb + wb, 2.6, xk, 2.6)
    box(ax, xk, 2.6 - hb / 2, wk, hb, f'clamp [{LOG_L_CLAMP[0]}, {LOG_L_CLAMP[1]}], exp', fc=C_OUT, fs=9)
    # lower-triangular L
    cs, x0 = 0.42, 12.35
    y0 = 1.95 + no * cs / 2   # top edge of the matrix
    for i in range(no):
        for j in range(no):
            fc = C_RES if i == j else (C_CONV if i > j else 'white')
            ax.add_patch(Rectangle((x0 + j * cs, y0 - (i + 1) * cs), cs, cs, fc=fc, ec=LINE, lw=0.8, zorder=2))
            lab = f'$L_{{{i + 1}{j + 1}}}$' if i >= j else '0'
            ax.text(x0 + (j + 0.5) * cs, y0 - (i + 0.5) * cs, lab, ha='center', va='center', fontsize=8.3,
                    color=INK if i >= j else MUT, zorder=3)
    ax.text(x0 + no * cs / 2, y0 + 0.12, 'L (lower triangular)', ha='center', va='bottom', fontsize=8.6, color=INK)
    arrow(ax, xk + wk, 2.6, x0, y0 - cs)                  # diagonal (positive)
    arrow(ax, xb + wb, 1.45, x0, y0 - (no - 0.5) * cs)    # strictly lower part
    xs_, ws_ = 14.25, 1.5
    arrow(ax, x0 + no * cs, 1.95, xs_, 1.95)
    box(ax, xs_, 1.95 - 0.45, ws_, 0.9, r'$\Sigma = L L^{\top}$', '\npositive definite', fc=C_OUT, fs=10.5, bfs=8.3,
        title_dy=0.25)
    arrow(ax, xb + wb, 3.75, xs_, 3.75)
    box(ax, xs_, 3.75 - hb / 2, ws_, hb, r'$\mu \in \mathbb{R}^{3}$', fc=C_OUT, fs=10.5)
    note = ('Order of the labels: chirp mass, total mass, chi_eff. All outputs in standardised label units (train-only '
            'scaler); physical covariance D Σ D with D = diag(train std).\n'
            f'log L_ii is clamped to [{LOG_L_CLAMP[0]}, {LOG_L_CLAMP[1]}]; the off-diagonal L_ij are unconstrained.'
            + (f'  {loss_note}' if loss_note else ''))
    ax.text(0.3, 0.45, note, fontsize=8.6, color=MUT, va='center', linespacing=1.4)
    return fig


CROPS = {
    'reference': {'arch_cnn_tr_overall': (0.0, 4.8, 16.0, 9.0), 'arch_cnn_tr_resblock': (0.0, 0.0, 9.0, 4.7),
                  'arch_cnn_tr_transformer': (9.15, 0.0, 16.0, 4.7)},
    'mrdense': {'arch_dense_overall': (0.0, 3.2, 16.0, 9.0), 'arch_dense_block': (0.0, 0.1, 5.7, 3.05),
                'arch_dense_layer': (5.8, 0.1, 13.25, 3.05)},
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='presentation_figures'); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for key, f in (('reference', fig_reference), ('mrdense', fig_mrdense)):
        fig = f()
        for name, (x0, y0, x1, y1) in CROPS[key].items():
            for ext in ('png', 'pdf'):
                fig.savefig(out / f'{name}.{ext}', dpi=250, facecolor='white', bbox_inches=Bbox([[x0, y0], [x1, y1]]))
        plt.close(fig)
    fig = fig_stage_table()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'arch_dense_stage_table.{ext}', dpi=250, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('written to', out)
