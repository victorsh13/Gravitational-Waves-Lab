#!/usr/bin/env python3
"""Check only waveform generation for frozen M11-final sources below 20 Msun."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from importlib.metadata import version

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import resolve_data_root, dataset_processed_dir

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT /
                        'configs/generation/generate_m11_final_G1_500k.json')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT /
                        'diagnostics/m11_final_waveform_validity')
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    root = resolve_data_root(cli_data_root=args.data_root, config_data_root=spec['data_root'])
    manifest = dataset_processed_dir(root, spec['output']['dataset_id']) / spec['output']['manifest_file']
    df = pd.read_csv(manifest, float_precision='round_trip')
    config_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not df.config_sha256.eq(config_hash).all():
        raise ValueError('Manifest/config hash mismatch')
    candidates = df[df.total_mass < 20].sort_values('source_index')
    print(f'Candidates: {len(candidates)} / {len(df)}', flush=True)
    from src.config import SimulationConfig
    from src.parameters import CBCParameters
    from src.waveform import WaveformGenerator
    generator = WaveformGenerator(SimulationConfig(**spec['simulation']))
    columns = ['source_id', 'source_index', 'mass_1', 'mass_2', 'total_mass',
               'chirp_mass', 'spin_1z', 'spin_2z', 'success', 'exception_text']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_dir / 'failed_waveforms.csv'
    summary_path = args.output_dir / 'summary.json'
    if failure_path.exists() or summary_path.exists():
        raise FileExistsError('Diagnostic output exists; use a new --output-dir')
    failures = []
    with failure_path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for tested, row in enumerate(candidates.itertuples(index=False), 1):
            try:
                parameters = CBCParameters(
                    mass_1=row.mass_1, mass_2=row.mass_2,
                    distance=row.reference_distance_mpc, inclination=row.inclination,
                    ra=row.ra, dec=row.dec, spin_1z=row.spin_1z, spin_2z=row.spin_2z,
                    polarization_angle=row.polarization_angle,
                )
                waveform = generator.generate(parameters)
                del waveform
            except Exception as exc:
                failed = {name: getattr(row, name) for name in columns[:-2]}
                failed.update(success=False, exception_text=f'{type(exc).__name__}: {exc}')
                failures.append(failed)
                writer.writerow(failed)
                handle.flush()
                print(f'FAIL {row.source_id}: {exc}', flush=True)
            if tested % 100 == 0 or tested == len(candidates):
                print(f'Tested {tested}/{len(candidates)}; failures={len(failures)}', flush=True)
    ranges = {name: {'min': min(r[name] for r in failures),
                     'max': max(r[name] for r in failures)} if failures else None
              for name in ('total_mass', 'spin_1z', 'spin_2z')}
    summary = {
        'complete': True, 'manifest': str(manifest),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'config': str(args.config), 'config_sha256': config_hash,
        'software_versions': {name: version(name) for name in ('numpy', 'pycbc', 'lalsuite')},
        'selection': 'total_mass < 20 Msun', 'manifest_rows': len(df),
        'tested': len(candidates), 'successes': len(candidates) - len(failures),
        'failures': len(failures), 'fraction_of_manifest_failed': len(failures) / len(df),
        'failure_ranges': ranges,
        'all_failures_both_spins_above_0_8':
            all(r['spin_1z'] > .8 and r['spin_2z'] > .8 for r in failures) if failures else None,
        'failed_source_ids': [r['source_id'] for r in failures],
        'scope_note': 'Sources with total_mass >= 20 were not tested.',
    }
    with summary_path.open('x') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
