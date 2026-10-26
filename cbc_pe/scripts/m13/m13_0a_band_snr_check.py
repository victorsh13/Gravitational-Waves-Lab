"""M13.0a: how much optimal SNR^2 lies above 512 Hz for the G1 source population?

Pre-check for a possible G2 band extension. Noise-free SEOBNRv4_opt waveforms on a
fixed (M_total, q, chi) grid are integrated against 9 raw empirical PSDs: first
block_id (alphabetical) of each validation environment (file_group_id 7, 12, 27)
x (H1, L1, V1), estimated with the frozen G1 helper (128 s off-source, Welch 8 s).
Only validation-environment noise is read; no labels, predictions or models.

SNR^2 density per PSD: 4 (|h+~|^2 + |hx~|^2) / S_n, integrated on [30, 1945] Hz.
Fractions are computed with and without a TAPER_START window on h+ and hx (pycbc
taper_timeseries) before the FFT; the abrupt start otherwise leaks broadband power.
Fixed rule, applied ONLY to the tapered version (M <= 35 points; median over points
of the per-point median fraction above 512 Hz): GO_G2 if >= 0.02, NO_G2 if < 0.005,
USER_DECISION otherwise. The untapered version and its difference are informative.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import dataset_processed_dir, resolve_data_root  # noqa: E402
# Read-only reuse of the frozen G1 PSD contract; the generator file is not modified.
import scripts.generate_m11_final_G1 as g1  # noqa: E402

CONFIG_DEFAULT = ROOT / 'configs/generation/generate_m11_final_G1_500k.json'
H5_NAME = 'm11_final_G1_500k.h5'
OUT_NAME = 'M13_0a_band_snr_check'
VAL_ENVIRONMENTS = (7, 12, 27)
M_TOTAL_GRID = (12, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 140)
Q_GRID = (1, 2, 4)
CHI_GRID = (-0.8, 0.0, 0.8)
COMPONENT_MASS_RANGE = (5.0, 90.0)
DISTANCE_MPC = 1000.0
INCLINATION = 0.0
MIN_PAD_DURATION_S = 4.0
BAND = (30.0, 1945.0)
THRESHOLDS = (256, 512, 1024)
GO_THRESHOLD = 0.02
NO_THRESHOLD = 0.005
RULE_MAX_M = 35.0
TAPER_TAGS = ('taper', 'notaper')
DECISION_TAG = 'taper'


def attr_str(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def load_val_blocks(spec, root, h5_path):
    with h5py.File(h5_path, 'r') as h:
        attrs = {k: attr_str(h.attrs[k]) for k in ('manifest_path', 'manifest_sha256', 'config_sha256')}
    config_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    g1.require(config_hash == attrs['config_sha256'], 'Config does not match canonical G1 HDF5')
    manifest_path = Path(attrs['manifest_path'])
    if not manifest_path.exists():
        # Absolute path recorded on another machine; same file name under this data root.
        manifest_path = h5_path.parent / manifest_path.name
    manifest_hash = g1.sha256_file(manifest_path)
    g1.require(manifest_hash == attrs['manifest_sha256'],
               f'Manifest sha256 mismatch: {manifest_path}')
    df = pd.read_csv(manifest_path, float_precision='round_trip',
                     usecols=['split', 'file_group_id', 'block_id', 'psd_start', 'psd_end'])
    rows = []
    for env in VAL_ENVIRONMENTS:
        env_rows = df[df.file_group_id == env]
        g1.require(len(env_rows) > 0 and env_rows['split'].eq('val').all(),
                   f'file_group_id {env} is not a validation-only environment')
        block_id = sorted(env_rows.block_id.astype(str).unique())[0]
        block = env_rows[env_rows.block_id.astype(str) == block_id]
        g1.require(block.psd_start.nunique() == 1 and block.psd_end.nunique() == 1,
                   f'Ambiguous PSD interval for {block_id}')
        rows.append(block.index[0])
    blocks = df.loc[rows].reset_index(drop=True)
    return blocks, manifest_path, manifest_hash


class PsdBank:
    """9 PSDs, re-estimated with the frozen helper at each padded length's delta_f."""

    def __init__(self, blocks, root, sampling_frequency):
        self.fs = sampling_frequency
        self.blocks = blocks
        selected, self.provenance = g1.resolve_cached_psd_files(root, blocks)
        self.strains = {row.file_group_id: g1.read_psd_intervals(row, selected)
                        for row in blocks.itertuples()}
        self.cache = {}

    def get(self, n_samples):
        if n_samples not in self.cache:
            delta_f = self.fs / n_samples
            flength = n_samples // 2 + 1
            psds = []
            for row in self.blocks.itertuples():
                for ifo in g1.DETECTORS:
                    psd = g1.estimate_raw_offsource_psd(
                        self.strains[row.file_group_id][ifo],
                        psd_start=float(row.psd_start), psd_end=float(row.psd_end),
                        delta_f=delta_f, target_flength=flength)
                    g1.require(abs(float(psd.delta_f) - delta_f) < 1e-12, 'PSD delta_f mismatch')
                    psds.append((int(row.file_group_id), ifo, psd.numpy().astype(np.float64)))
            self.cache[n_samples] = psds
        return self.cache[n_samples]


def grid_points():
    for m_total in M_TOTAL_GRID:
        for q in Q_GRID:
            m1, m2 = m_total * q / (1.0 + q), m_total / (1.0 + q)
            if m2 < COMPONENT_MASS_RANGE[0] or m1 > COMPONENT_MASS_RANGE[1]:
                continue
            for chi in CHI_GRID:
                yield m_total, q, chi, m1, m2


def band_fractions(h_plus, h_cross, psds, n_samples, fs):
    dt = 1.0 / fs
    hp = np.zeros(n_samples)
    hc = np.zeros(n_samples)
    hp[:len(h_plus)] = h_plus
    hc[:len(h_cross)] = h_cross
    # Continuous-FT convention h~(f) = dt * DFT; the overall scale cancels in fractions.
    power = np.abs(dt * np.fft.rfft(hp)) ** 2 + np.abs(dt * np.fft.rfft(hc)) ** 2
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    in_band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    out = {t: [] for t in THRESHOLDS}
    for _, _, psd in psds:
        g1.require(np.all(np.isfinite(psd[in_band])) and np.all(psd[in_band] > 0),
                   'Non-positive PSD in band')
        density = 4.0 * power[in_band] / psd[in_band]
        total = density.sum()
        for t in THRESHOLDS:
            out[t].append(density[freqs[in_band] > t].sum() / total)
    return out


def subset_median(df, column, lo, hi):
    sel = df[(df.status == 'ok') & (df.M_total > lo) & (df.M_total <= hi)]
    return (float(np.median(sel[column])) if len(sel) else None), int(len(sel))


def plot(df, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {-0.8: 'tab:blue', 0.0: 'tab:gray', 0.8: 'tab:red'}
    fig, axes = plt.subplots(1, len(Q_GRID), figsize=(13, 4.2), sharey=True)
    ok = df[df.status == 'ok']
    tag = DECISION_TAG
    for ax, q in zip(axes, Q_GRID):
        for chi in CHI_GRID:
            d = ok[(ok.q == q) & (ok.chi == chi)].sort_values('M_total')
            if d.empty:
                continue
            c = colors[chi]
            ax.plot(d.M_total, d[f'frac_gt512_median_{tag}'], '-o', ms=3, color=c,
                    label=f'chi={chi:+.1f} >512 Hz')
            ax.fill_between(d.M_total, d[f'frac_gt512_min_{tag}'], d[f'frac_gt512_max_{tag}'],
                            color=c, alpha=0.15, lw=0)
            ax.plot(d.M_total, d[f'frac_gt256_median_{tag}'], '--', color=c,
                    label=f'chi={chi:+.1f} >256 Hz')
        ax.axhline(GO_THRESHOLD, color='k', lw=0.8, ls=':')
        ax.axhline(NO_THRESHOLD, color='k', lw=0.8, ls='-.')
        ax.axvline(RULE_MAX_M, color='k', lw=0.6, alpha=0.4)
        ax.set_yscale('log')
        ax.set_title(f'q = m1/m2 = {q}')
        ax.set_xlabel('M_total [Msun]')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('fraction of SNR^2 (median over 9 PSDs; band = min-max)')
    axes[-1].legend(fontsize=7)
    fig.suptitle(f'M13.0a ({tag}): SNR^2 fraction above 512 Hz (solid) and 256 Hz (dashed); '
                 f'dotted {GO_THRESHOLD}, dash-dot {NO_THRESHOLD}', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_DEFAULT)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()

    spec = json.loads(args.config.read_text())
    root = resolve_data_root(cli_data_root=args.data_root, config_data_root=spec.get('data_root'))
    out_dir = root / 'results' / OUT_NAME
    if out_dir.exists():
        raise FileExistsError(f'Output exists; refusing to overwrite: {out_dir}')
    h5_path = dataset_processed_dir(root, spec['output']['dataset_id']) / H5_NAME

    from pycbc.psd import interpolate
    from pycbc.waveform.utils import taper_timeseries
    from src.config import SimulationConfig
    from src.parameters import CBCParameters
    from src.waveform import WaveformGenerator
    # estimate_raw_offsource_psd reads the module global set by g1.initialize_physics;
    # bind only that symbol instead of building the full G1 physics stack.
    g1.interpolate = interpolate

    config = SimulationConfig(**spec['simulation'])
    generator = WaveformGenerator(config)
    fs = config.sampling_frequency

    blocks, manifest_path, manifest_hash = load_val_blocks(spec, root, h5_path)
    bank = PsdBank(blocks, root, fs)
    print('PSD blocks:\n' + blocks[['file_group_id', 'block_id', 'psd_start', 'psd_end']].to_string(),
          flush=True)

    records = []
    for m_total, q, chi, m1, m2 in grid_points():
        rec = {'M_total': m_total, 'q': q, 'chi': chi, 'mass_1': m1, 'mass_2': m2,
               'status': 'ok', 'error': ''}
        params = CBCParameters(mass_1=m1, mass_2=m2, distance=DISTANCE_MPC, inclination=INCLINATION,
                               ra=0.0, dec=0.0, spin_1z=chi, spin_2z=chi)
        try:
            waveform = generator.generate(params)
            # Taper failures are recorded like generation failures (same point skipped).
            strains = {
                'notaper': (waveform.h_plus, waveform.h_cross),
                'taper': (taper_timeseries(waveform.h_plus, tapermethod='TAPER_START'),
                          taper_timeseries(waveform.h_cross, tapermethod='TAPER_START'))}
        except Exception as exc:  # recorded and skipped, per protocol
            rec.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            records.append(rec)
            print(f'FAILED M={m_total} q={q} chi={chi}: {rec["error"]}', flush=True)
            continue
        n_wave = len(waveform.h_plus)
        n_samples = 1 << int(np.ceil(np.log2(max(n_wave, int(round(MIN_PAD_DURATION_S * fs))))))
        rec.update(waveform_samples=n_wave, padded_samples=n_samples, delta_f=fs / n_samples)
        for tag in TAPER_TAGS:
            hp, hc = strains[tag]
            fractions = band_fractions(hp.numpy(), hc.numpy(), bank.get(n_samples), n_samples, fs)
            for t, values in fractions.items():
                rec[f'frac_gt{t}_median_{tag}'] = float(np.median(values))
                rec[f'frac_gt{t}_min_{tag}'] = float(np.min(values))
                rec[f'frac_gt{t}_max_{tag}'] = float(np.max(values))
        records.append(rec)

    df = pd.DataFrame(records)
    g1.require((df.status == 'ok').any(), 'All waveforms failed')
    rule_value, rule_n = subset_median(df, f'frac_gt512_median_{DECISION_TAG}', 0.0, RULE_MAX_M)
    g1.require(rule_value is not None, 'No successful points with M <= 35')
    if rule_value >= GO_THRESHOLD:
        decision = 'GO_G2'
    elif rule_value < NO_THRESHOLD:
        decision = 'NO_G2'
    else:
        decision = 'USER_DECISION'
    rule_notaper, _ = subset_median(df, 'frac_gt512_median_notaper', 0.0, RULE_MAX_M)
    informative = {}
    for tag in TAPER_TAGS:
        mid_value, mid_n = subset_median(df, f'frac_gt512_median_{tag}', RULE_MAX_M, 65.0)
        hi_value, hi_n = subset_median(df, f'frac_gt1024_median_{tag}', 0.0, 25.0)
        informative[tag] = {
            'frac_gt512_M_35_to_65': {'value': mid_value, 'n_points': mid_n,
                                      'subset': '35 < M_total <= 65'},
            'frac_gt1024_M_le_25': {'value': hi_value, 'n_points': hi_n,
                                    'subset': 'M_total <= 25'}}
    # Taper effect on the rule subset: per-point difference (taper - notaper) of the
    # per-point median fraction, plus the change in the rule statistic itself.
    rule_sel = df[(df.status == 'ok') & (df.M_total <= RULE_MAX_M)]
    taper_diff = {}
    for t in THRESHOLDS:
        diff = rule_sel[f'frac_gt{t}_median_taper'] - rule_sel[f'frac_gt{t}_median_notaper']
        taper_diff[f'frac_gt{t}'] = {
            'rule_statistic_taper': float(np.median(rule_sel[f'frac_gt{t}_median_taper'])),
            'rule_statistic_notaper': float(np.median(rule_sel[f'frac_gt{t}_median_notaper'])),
            'median_point_diff': float(np.median(diff)),
            'min_point_diff': float(np.min(diff)),
            'max_point_diff': float(np.max(diff)),
            'max_abs_point_diff': float(np.max(np.abs(diff)))}

    from importlib.metadata import version
    summary = {
        'experiment': 'M13.0a', 'decision': decision,
        'rule': {'subset': 'status ok and M_total <= 35',
                 'statistic': 'median over points of per-point median (over 9 PSDs) SNR^2 fraction > 512 Hz',
                 'version': f'{DECISION_TAG} (TAPER_START on h+ and hx before FFT)',
                 'GO_G2_if_ge': GO_THRESHOLD, 'NO_G2_if_lt': NO_THRESHOLD,
                 'value': rule_value, 'n_points': rule_n,
                 'value_notaper_informative': rule_notaper},
        'informative_not_decisive': informative,
        'taper_minus_notaper_M_le_35': taper_diff,
        'taper': "pycbc.waveform.utils.taper_timeseries(ts, tapermethod='TAPER_START')",
        'n_points': int(len(df)), 'n_failed': int((df.status == 'failed').sum()),
        'failures': df.loc[df.status == 'failed', ['M_total', 'q', 'chi', 'error']].to_dict('records'),
        'grid': {'M_total': list(M_TOTAL_GRID), 'q_m1_over_m2': list(Q_GRID),
                 'chi1z_eq_chi2z': list(CHI_GRID), 'component_mass_range': list(COMPONENT_MASS_RANGE),
                 'distance_mpc': DISTANCE_MPC, 'inclination': INCLINATION},
        'snr_density': '4 (|h+~|^2 + |hx~|^2) / S_n, raw empirical PSD at waveform delta_f',
        'band_hz': list(BAND), 'thresholds_hz': list(THRESHOLDS),
        'padding': f'zeros to next power of 2 >= max(len, {MIN_PAD_DURATION_S} s)',
        'simulation': spec['simulation'],
        'psd_contract': spec['empirical_psd'],
        'psd_blocks': blocks.astype({'file_group_id': int}).to_dict('records'),
        'raw_psd_files': json.loads(bank.provenance),
        'config_path': str(args.config.resolve()), 'canonical_h5': str(h5_path),
        'manifest_path': str(manifest_path), 'manifest_sha256': manifest_hash,
        'script_sha256': g1.sha256_file(__file__),
        'software_versions': {name: version(name) for name in ('numpy', 'pandas', 'pycbc', 'lalsuite')},
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    df.to_csv(out_dir / 'band_fractions.csv', index=False)
    plot(df, out_dir / 'band_fractions.png')
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))

    cols = ['M_total', 'q', 'chi', 'status', 'frac_gt256_median_taper', 'frac_gt512_median_taper',
            'frac_gt512_min_taper', 'frac_gt512_max_taper', 'frac_gt1024_median_taper',
            'frac_gt512_median_notaper']
    with pd.option_context('display.width', 220, 'display.float_format', '{:.4g}'.format):
        print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f'\nRule (taper, M <= 35, n={rule_n}): median frac>512 = {rule_value:.4g} -> {decision}')
    print(f'Notaper (informative): median frac>512 = {rule_notaper:.4g}')
    for key, d in taper_diff.items():
        print(f'Taper - notaper (M <= 35) {key}: rule stat {d["rule_statistic_taper"]:.4g} vs '
              f'{d["rule_statistic_notaper"]:.4g}; per-point diff median {d["median_point_diff"]:.3g}, '
              f'range [{d["min_point_diff"]:.3g}, {d["max_point_diff"]:.3g}]')
    for tag, info in informative.items():
        mid, hi = info['frac_gt512_M_35_to_65'], info['frac_gt1024_M_le_25']
        print(f'Info {tag} (35 < M <= 65, n={mid["n_points"]}): median frac>512 = {mid["value"]}')
        print(f'Info {tag} (M <= 25, n={hi["n_points"]}): median frac>1024 = {hi["value"]}')
    print(f'Outputs: {out_dir}')


if __name__ == '__main__':
    main()
