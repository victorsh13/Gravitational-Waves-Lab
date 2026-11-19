"""Presentation figures that need no data files (physics schematics from the generation prior).

  P1 p1_band_tracks       : schematic time-frequency tracks of a light and a heavy BBH with the old/new bands.
  P3 p3_attention_schematic : how attention (with positional encoding) encodes temporal order vs average pooling.
  P4 p4_noise_aug_schematic : noise augmentation, clean signal + noise bank -> new noise every batch.
  EQ eq_*.png               : equations rendered alone (mathtext, Computer Modern), to place next to slide text.
  P2 p2_f220_coverage     : fraction of prior systems whose ringdown frequency f220 exceeds 512 / 1024 Hz,
                             in the (chirp mass, chi_eff) plane. Prior of the generation: m1, m2 ~ U[5, 90] Msun
                             (ordered), aligned spins ~ U[-1, 1]. f220 with the same protocol fits as M13.1.
Names/colours: configs/presentation/labels.json.   Run:  python scripts/m13/presentation/physics_figures.py --out DIR
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LAB = json.loads((ROOT / 'configs/presentation/labels.json').read_text())
ST = LAB['style']
T_SUN = 4.925490947e-6

# ---- f220: identical to scripts/m13/m13_1_diagnostics.py (copied to avoid its h5py/data imports) ----------------
S4, S5, T0, T2, T3 = -0.1229, 0.4537, -2.8904, -3.5171, 2.5763
A_MAX = 0.998


def r_isco(a):
    z1 = 1.0 + np.cbrt(1.0 - a**2) * (np.cbrt(1.0 + a) + np.cbrt(1.0 - a))
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    return 3.0 + z2 - np.sign(a) * np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def f220_hz(m_big, m_small, chi_big, chi_small):
    qr = m_small / m_big
    nu = qr / (1.0 + qr)**2
    a_t = (chi_big + chi_small * qr**2) / (1.0 + qr**2)
    a_f = a_t + a_t * nu * (S4 * a_t + S5 * nu + T0) + nu * (2.0 * np.sqrt(3.0) + T2 * nu + T3 * nu**2)
    a_e = (chi_big + chi_small * qr**2) / (1.0 + qr)**2
    a_f, a_e = np.clip(a_f, -A_MAX, A_MAX), np.clip(a_e, -A_MAX, A_MAX)
    e_isco = np.sqrt(1.0 - 2.0 / (3.0 * r_isco(a_e)))
    e_rad = (1.0 - e_isco) * nu + 4.0 * nu**2 * (4.0 * 0.04827 + 16.0 * 0.01707 * a_e * (a_e + 1.0) + e_isco - 1.0)
    m_f = (m_big + m_small) * (1.0 - e_rad)
    return (1.5251 - 1.1568 * (1.0 - a_f)**0.1292) / (2.0 * np.pi * m_f * T_SUN)


def style():
    plt.rcParams.update({'font.family': ST['font'], 'font.size': ST['tick_size'], 'axes.labelsize': ST['axes_size'],
                         'legend.fontsize': ST['legend_size'], 'axes.spines.top': False, 'axes.spines.right': False,
                         'mathtext.fontset': 'dejavusans'})


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=ST['dpi'], bbox_inches='tight', facecolor='white')
    plt.close(fig)


def p1_band_tracks(out):
    def track(m1, m2, a=0.69):
        M = m1 + m2; mc = (m1 * m2)**0.6 / M**0.2
        tau = np.geomspace(4.0, 1e-4, 4000)
        f = (1 / np.pi) * (5 / (256 * tau))**(3 / 8) * (T_SUN * mc)**(-5 / 8)
        fisco = 4400 / M
        f220 = (1.5251 - 1.1568 * (1 - a)**0.1292) / (2 * np.pi * T_SUN * 0.95 * M)
        t = -tau; tis = t[np.argmax(f >= fisco)]
        tr = np.linspace(tis, tis + 0.03, 200)
        tt = np.concatenate([t[t < tis], tr])
        ff = np.concatenate([f[t < tis], fisco + (f220 - fisco) * (np.tanh((tr - tis - 0.008) / 0.004) + 1) / 2])
        return tt - tis, ff, f220
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhspan(30, 512, color='#AFCBE3', alpha=0.45, lw=0)
    ax.axhspan(512, 1024, color='#9CBFA7', alpha=0.45, lw=0)
    ax.text(-2.35, 140, 'Legacy band\n30–512 Hz', color='#526D82', va='center')
    ax.text(-2.35, 730, 'W.B. adds 512–1024 Hz', color='#3F6B4C', va='center', fontweight='bold')
    for m1, m2, c in ((10, 10, '#1F2D3A'), (40, 40, '#B5676A')):
        t, f, f220 = track(m1, m2)
        ax.plot(t, f, color=c, lw=2.4, label=f'{m1}+{m2} M$_\\odot$  (ringdown ≈ {f220:.0f} Hz)')
        ax.plot([t[-1]], [f[-1]], 'o', color=c, ms=6)
    ax.set_yscale('log'); ax.set_ylim(25, 2600); ax.set_xlim(-2.5, 0.2)
    ax.set_yticks([30, 64, 128, 256, 512, 1024, 2048]); ax.set_yticklabels(['30', '64', '128', '256', '512', '1024', '2048'])
    ax.set_xlabel('time to merger [s]'); ax.set_ylabel('GW frequency [Hz]')
    ax.legend(loc='upper left', frameon=False)
    save(fig, out, 'p1_band_tracks')


def p2_f220_coverage(out, n=3_000_000, seed=20261102):
    rng = np.random.default_rng(seed)
    m = rng.uniform(5, 90, (n, 2)); s = rng.uniform(-1, 1, (n, 2))
    big = m[:, 0] >= m[:, 1]
    m1 = np.where(big, m[:, 0], m[:, 1]); m2 = np.where(big, m[:, 1], m[:, 0])
    c1 = np.where(big, s[:, 0], s[:, 1]); c2 = np.where(big, s[:, 1], s[:, 0])
    M = m1 + m2; mc = (m1 * m2)**0.6 / M**0.2; chi = (m1 * c1 + m2 * c2) / M
    f = f220_hz(m1, m2, c1, c2)
    xb = np.linspace(4, 30, 53); yb = np.linspace(-1, 1, 41)
    tot, _, _ = np.histogram2d(mc, chi, (xb, yb))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    hit, _, _ = np.histogram2d(mc[f > 512], chi[f > 512], (xb, yb))
    frac = np.where(tot > 50, hit / np.maximum(tot, 1), np.nan)
    im = ax.pcolormesh(xb, yb, frac.T, cmap='Blues', vmin=0, vmax=1, shading='flat')
    hit2, _, _ = np.histogram2d(mc[f > 1024], chi[f > 1024], (xb, yb))
    frac2 = np.where(tot > 50, hit2 / np.maximum(tot, 1), np.nan)
    xc, yc = 0.5 * (xb[1:] + xb[:-1]), 0.5 * (yb[1:] + yb[:-1])
    cs = ax.contour(xc, yc, frac2.T, levels=[0.5], colors='#B5676A', linewidths=2.0, linestyles='--')
    ax.plot([], [], color='#B5676A', ls='--', lw=2, label=f'50 % of systems above 1024 Hz')
    ax.set_xlabel(LAB['variables']['chirp_mass']['label'] + ' [' + LAB['variables']['chirp_mass']['unit'] + ']')
    ax.set_ylabel(LAB['variables']['chi_eff']['label'])
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.18), frameon=False)
    ax.text(0.98, 0.33, f'prior: {100 * np.mean(f > 512):.1f} % above 512 Hz\n{100 * np.mean(f > 1024):.1f} % above 1024 Hz', transform=ax.transAxes, ha='right', fontsize=ST['legend_size'], color='#526D82')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('fraction with $f_{220}$ > 512 Hz')
    save(fig, out, 'p2_f220_coverage')


# ---------------------------------------------------------------------------------------------------------------
def _toy_chirp(t, t_merge=0.0, f0=25.0, tau0=3.6):
    tc = np.clip(t_merge - t, 1e-3, None)
    f = f0 * (tc / tau0)**(-3 / 8)
    dt = t[1] - t[0]
    ph = 2 * np.pi * np.cumsum(f) * dt
    a = (tc / tau0)**(-1 / 4)
    post = t > t_merge
    a[post] = a[~post].max() * np.exp(-(t[post] - t_merge) / 0.015)
    return a * np.cos(ph) / a.max()


def p3_attention_schematic(out):
    C = {m: v['color'] for m, v in LAB['models'].items()}
    fs = 1024; t = np.linspace(-3.6, 0.4, int(4 * fs)); y = _toy_chirp(t)
    fig = plt.figure(figsize=(13, 6.0))
    ax = fig.add_axes([0.03, 0.40, 0.94, 0.58]); ax.axis('off')
    ax.set_xlim(t[0], t[-1] + 0.05); ax.set_ylim(-1.5, 4.0)
    ax.plot(t, y * 0.9, color='#1F2D3A', lw=0.9)
    n_tok = 32; edges = np.linspace(t[0], t[-1], n_tok + 1); mids = 0.5 * (edges[1:] + edges[:-1])
    yb = 1.2
    q = int(np.argmin(np.abs(mids - 0.0)))
    for i in range(n_tok):
        ax.add_patch(plt.Rectangle((edges[i] + 0.006, yb), edges[1] - edges[0] - 0.012, 0.22,
                                   fc=C['4a_aug'] if i == q else '#DDE6EF', ec='none'))
        ax.plot([mids[i], mids[i]], [0.95, yb], color='#C3CCD6', lw=0.6, ls=':')
    targets = [1, 4, 8, 12, 16, 20, 23, 26]
    w = np.array([0.35, 0.5, 0.45, 0.65, 0.6, 0.75, 0.85, 1.0])
    for k, wk in zip(targets, w):
        x0, x1 = mids[q], mids[k]; h = 0.35 + 2.1 * abs(x1 - x0) / 3.6
        xs = np.linspace(x1, x0, 80); ys = yb + 0.24 + h * np.sin(np.pi * (xs - x1) / (x0 - x1))
        ax.plot(xs, ys, color=C['4a_aug'], lw=0.6 + 3.0 * wk, alpha=0.30 + 0.6 * wk, solid_capstyle='round')
    ax.text(mids[q] + 0.08, yb + 0.11, 'query token\n(merger)', ha='left', va='center', fontsize=11, color=C['4a_aug'])
    ax.text(t[0], 3.85, 'Tokens of 15.6 ms (32 of the 256 drawn). Arc width = attention weight of the merger token on earlier tokens.',
            fontsize=11, color='#5B6B7A', va='top')
    ax.annotate('', xy=(t[-1], -1.3), xytext=(t[0], -1.3), arrowprops=dict(arrowstyle='->', color='#8A98A6'))
    ax.text(t[-1], -1.45, 'time', fontsize=11, color='#5B6B7A', va='top', ha='right')
    with plt.rc_context({'mathtext.fontset': 'cm'}):
        a1 = fig.add_axes([0.03, 0.02, 0.45, 0.34]); a1.axis('off'); a1.set_xlim(0, 10); a1.set_ylim(0, 4)
        a2 = fig.add_axes([0.53, 0.02, 0.45, 0.34]); a2.axis('off'); a2.set_xlim(0, 10); a2.set_ylim(0, 4)
        seqA = ['#1B9E77', '#E6AB02', '#7570B3', '#D95F02']; seqB = seqA[::-1]
        for row, seq in enumerate((seqA, seqB)):
            yy = 3.1 - row * 1.15
            for i, c in enumerate(seq):
                a1.add_patch(plt.Rectangle((0.3 + i * 0.7, yy - 0.28), 0.55, 0.55, fc=c, ec='white'))
            a1.text(3.4, yy, r'$\longrightarrow\quad h=\frac{1}{T}\sum_t f_t$', fontsize=17, va='center')
        a1.text(0.3, 0.45, 'Average pooling: identical output for any order\n("inspiral, then merger" = "merger, then inspiral")',
                fontsize=11, color='#5B6B7A')
        a2.text(0.2, 3.1, r'$\tilde f_t = f_t + \mathrm{PE}(t)$', fontsize=18, va='center')
        a2.text(0.2, 1.95, r'$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(QK^{\top}/\sqrt{d}\right)V$', fontsize=18, va='center')
        a2.text(0.2, 0.45, 'Self-attention with positional encoding: each token weighs every other one\nby content and position, so "this happens before that" can be learned',
                fontsize=11, color='#5B6B7A')
    save(fig, out, 'p3_attention_schematic')


def p4_noise_aug_schematic(out):
    C = {m: v['color'] for m, v in LAB['models'].items()}
    fs = 1024; t = np.linspace(-3.6, 0.4, int(4 * fs)); s = _toy_chirp(t) * 0.9
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(13, 4.8))
    def panel(rect, title):
        a = fig.add_axes(rect); a.axis('off'); a.set_title(title, fontsize=13, loc='left', color='#1F2D3A'); return a
    a = panel([0.02, 0.45, 0.25, 0.42], 'clean signal  $L(s)$  (processed once)')
    a.plot(t, 1.8 * s, color='#1F2D3A', lw=0.9); a.set_ylim(-3.5, 3.5)
    fig.text(0.285, 0.66, '+', fontsize=30, color='#5B6B7A', va='center')
    a = panel([0.31, 0.45, 0.30, 0.42], 'noise bank: 64 draws per PSD block')
    for i in range(6):
        n = rng.normal(0, 1, t.size) * 0.16
        a.plot(t, n + 5.2 - 1.05 * i, color=C['6a'] if i == 2 else '#B8C4D0', lw=0.5 if i != 2 else 0.7)
    a.set_ylim(-0.9, 5.8)
    a.text(t[-1] + 0.05, 5.2 - 1.05 * 2, r'$k$', color=C['6a'], fontsize=14, va='center')
    a.text(t[0], -0.85, r'random $k$,  sign $\varepsilon=\pm1$,  time reversal $t\mapsto -t$', fontsize=12, color='#5B6B7A', va='top')
    fig.text(0.625, 0.66, r'$\rightarrow$', fontsize=30, color='#5B6B7A', va='center')
    a = panel([0.66, 0.45, 0.32, 0.42], r'network input  $x=z(\ldots)$  (new every batch)')
    a.plot(t, rng.normal(0, 1, t.size) + 1.4 * s, color=C['6a'], lw=0.5); a.set_ylim(-4, 4)
    # epochs strip
    a = fig.add_axes([0.02, 0.02, 0.96, 0.33]); a.axis('off'); a.set_xlim(0, 10); a.set_ylim(0, 2.2)
    a.text(0, 1.95, 'Before: same noise in every epoch', fontsize=12, color='#B2182B')
    a.text(5.2, 1.95, 'Now: different noise in every epoch', fontsize=12, color=C['4b_aug'])
    nfix = rng.normal(0, 1, 600)
    for e in range(3):
        xx = np.linspace(0.05 + e * 1.6, 1.45 + e * 1.6, 600)
        a.plot(xx, 1.0 + 0.18 * nfix, color='#B2182B', lw=0.5)
        a.text(0.75 + e * 1.6, 0.35, f'epoch {e + 1}', ha='center', fontsize=10, color='#5B6B7A')
        a.plot(xx + 5.15, 1.0 + 0.18 * rng.normal(0, 1, 600), color=C['4b_aug'], lw=0.5)
        a.text(5.9 + e * 1.6, 0.35, f'epoch {e + 1}', ha='center', fontsize=10, color='#5B6B7A')
    save(fig, out, 'p4_noise_aug_schematic')


EQUATIONS = {
    'eq_linear': r'$x = z\left(L(s+n)\right),\qquad L(s+n) = L(s) + L(n)$',
    'eq_aug': r'$x_{b,d} = z\left(L(s_b)_d + \varepsilon_{b,d}\,R_{b,d}\left[L(n_{k_{b,d}})_d\right]\right)$',
    'eq_sets': r'$k_{b,d}\sim\mathcal{U}\{1,\ldots,64\},\qquad \varepsilon_{b,d}\in\{+1,-1\},\qquad R_{b,d}\in\{\mathrm{id},\ t\mapsto -t\}$',
    'eq_count': r'$1\ \longrightarrow\ 64\times2\times2 = 256\ \mathrm{per\ detector},\qquad 256^{3}\approx 1.7\times10^{7}\ \mathrm{per\ source}$',
    'eq_attention': r'$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\left(QK^{\top}/\sqrt{d}\right)V$',
}


def equations(out):
    with plt.rc_context({'mathtext.fontset': 'cm'}):
        for name, tex in EQUATIONS.items():
            fig = plt.figure(figsize=(0.1, 0.1))
            fig.text(0, 0, tex, fontsize=22, color='#1F2D3A')
            fig.savefig(out / f'{name}.png', dpi=ST['dpi'], bbox_inches='tight', pad_inches=0.04, transparent=True)
            fig.savefig(out / f'{name}.pdf', bbox_inches='tight', pad_inches=0.04, transparent=True)
            plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='presentation_figures'); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); style()
    p1_band_tracks(out); p2_f220_coverage(out); p3_attention_schematic(out); p4_noise_aug_schematic(out); equations(out)
    print('written to', out)
