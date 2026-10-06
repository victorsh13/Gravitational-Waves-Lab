#!/usr/bin/env python3
"""Explicit, separate repair and migration commands for four pending M11-final sources.

Neither command starts dataset generation. Repair performs waveform-only rejection
sampling; migrate changes four rows of manifest metadata and resume provenance.
Run with the production writer stopped and normal HDF5 file locking enabled.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts import generate_m11_final_G1 as generation
from scripts.prepare_m11_final_manifest import PARAMETERS, require

SOURCE_IDS = (
    "M11FINAL_G1_SRC_005830", "M11FINAL_G1_SRC_014431",
    "M11FINAL_G1_SRC_025600", "M11FINAL_G1_SRC_193848",
)
REPAIR_SEED = 110004
OLD_GENERATOR_HASH = "bcd1f3ea5332374f4f2f9722ad3c89fe1c33df0c5131f102a1ac831bf35831e0"
NEW_GENERATOR_HASH = "a1366e15ddd28ac886bf663877e7d222ce0f6994dab8e24851ece7fe98baa993"


def equal(a, b):
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def verify_changes(original, repaired):
    require(len(original) == len(repaired) == 500000, "Expected 500000 rows")
    require(original.source_id.is_unique, "Duplicate source IDs")
    mask = original.source_id.isin(SOURCE_IDS)
    require(mask.sum() == 4, "Missing repair source IDs")
    equal(original.loc[~mask], repaired.loc[~mask])
    preserved = [name for name in original if name not in PARAMETERS]
    equal(original[preserved], repaired[preserved])
    require(original['split'].value_counts().to_dict() ==
            repaired['split'].value_counts().to_dict(), "Split counts changed")
    for index in original.index[mask]:
        require(not original.loc[index, list(PARAMETERS)].equals(
            repaired.loc[index, list(PARAMETERS)]), "Replacement row was not changed")
    return original.index[mask].to_numpy()


def input_args(args, manifest):
    return argparse.Namespace(config=args.config, manifest=manifest, data_root=args.data_root,
                              limit=None, output=None, output_suffix=None)


def receipt_path(path):
    return path.with_suffix(path.suffix + '.repair.json')


def repair(args):
    from src.config import SimulationConfig
    from src.parameters import CBCParameters
    from src.sampling import ParameterSampler, PriorConfig
    from src.waveform import WaveformGenerator

    spec, _, original, _, attrs = generation.read_inputs(input_args(args, args.original))
    require(args.repaired.resolve() != Path(attrs['manifest_path']).resolve(),
            "Cannot overwrite original manifest")
    receipt = receipt_path(args.repaired)
    require(not args.repaired.exists() and not receipt.exists(), "Repair output already exists")
    failed = pd.read_csv(PROJECT_ROOT / 'diagnostics/m11_final_waveform_validity/failed_waveforms.csv',
                         float_precision='round_trip')
    require(len(failed) == 4 and set(failed.source_id) == set(SOURCE_IDS), "Failure list changed")
    require(failed.success.eq(False).all(), "Failure CSV includes successful rows")
    old_by_id = original.set_index('source_id')
    for row in failed.itertuples():
        for name in ('source_index', 'mass_1', 'mass_2', 'total_mass', 'chirp_mass', 'spin_1z', 'spin_2z'):
            require(old_by_id.loc[row.source_id, name] == getattr(row, name),
                    f"Failure CSV does not match original: {row.source_id}/{name}")
    config = SimulationConfig(**spec['simulation'])
    generator = WaveformGenerator(config)
    repaired = original.copy(deep=True)
    records = []
    for source_id in SOURCE_IDS:
        index = int(original.index[original.source_id == source_id][0])
        # Independent per-source streams: retry counts do not affect other rows.
        sampler = ParameterSampler(
            prior_config=PriorConfig.from_dict(spec['parameter_sampler']),
            rng=np.random.default_rng(np.random.SeedSequence([REPAIR_SEED, int(original.at[index, 'source_index'])])),
        )
        for attempt in range(1, 10001):
            p = sampler.sample_one()
            # Reference distance is immutable, even though the original prior fixes it.
            p = CBCParameters(**{name: getattr(p, name) for name in
                ('mass_1', 'mass_2', 'spin_1z', 'spin_2z', 'inclination', 'ra', 'dec', 'polarization_angle')},
                distance=float(original.at[index, 'reference_distance_mpc']))
            try:
                waveform = generator.generate(p)
            except Exception as exc:
                print(f'{source_id} rejected attempt {attempt}: {type(exc).__name__}: {exc}', flush=True)
                continue
            del waveform
            break
        else:
            raise RuntimeError(f'Rejection limit reached for {source_id}; no manifest written')
        before = {name: float(original.at[index, name]) for name in PARAMETERS}
        after = {name: float(getattr(p, name)) for name in PARAMETERS}
        for name, value in after.items():
            repaired.at[index, name] = value
        record = dict(source_id=source_id, source_index=int(original.at[index, 'source_index']),
                      attempts=attempt, rejections=attempt - 1, old=before, new=after)
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)
    verify_changes(original, repaired)
    args.repaired.parent.mkdir(parents=True, exist_ok=True)
    with args.repaired.open('x') as handle:
        repaired.to_csv(handle, index=False)
    reloaded = pd.read_csv(args.repaired, float_precision='round_trip')
    equal(repaired, reloaded)
    verify_changes(original, reloaded)
    report = dict(repair_seed=REPAIR_SEED, rng_policy='PCG64 SeedSequence([repair_seed, source_index])',
                  original_path=attrs['manifest_path'], original_sha256=attrs['manifest_sha256'],
                  repaired_path=str(args.repaired.resolve()),
                  repaired_sha256=generation.sha256_file(args.repaired),
                  config_sha256=attrs['config_sha256'], waveform_validated=True,
                  software_versions={name: version(name) for name in ('numpy', 'pycbc', 'lalsuite')},
                  rows=records)
    with receipt.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(f'Wrote {args.repaired}; equality checks passed for all 499996 untouched rows.')


def migrate(args):
    """No waveform calls; fully validate before the first HDF5 mutation."""
    require(generation.sha256_file(generation.__file__) == NEW_GENERATOR_HASH,
            "Generator differs from the reviewed checkpoint-only revision")
    _, root, original, default_output, old_attrs = generation.read_inputs(input_args(args, args.original))
    _, _, repaired, _, new_attrs = generation.read_inputs(input_args(args, args.repaired))
    indices = verify_changes(original, repaired)
    report = json.loads(receipt_path(args.repaired).read_text())
    require(report['original_sha256'] == old_attrs['manifest_sha256'] and
            report['repaired_sha256'] == new_attrs['manifest_sha256'] and
            report['config_sha256'] == old_attrs['config_sha256'], "Repair receipt hash mismatch")
    require(report['waveform_validated'] is True and report['repair_seed'] == REPAIR_SEED,
            "Missing deterministic waveform-validation receipt")
    require([r['source_id'] for r in report['rows']] == list(SOURCE_IDS), "Wrong receipt rows")
    for row in report['rows']:
        for side, frame in (('old', original), ('new', repaired)):
            actual = frame.set_index('source_id').loc[row['source_id']]
            require(all(actual[name] == row[side][name] for name in PARAMETERS),
                    f'Receipt parameter mismatch: {row["source_id"]}')
    _, raw_files = generation.resolve_cached_psd_files(root, original)
    versions = {name: version(name) for name in
                ('numpy', 'pandas', 'h5py', 'pycbc', 'lalsuite', 'scipy', 'torch')}
    require(all(versions[name] == v for name, v in report['software_versions'].items()),
            "Repair/runtime software mismatch")
    for attrs in (old_attrs, new_attrs):
        attrs['raw_psd_files'] = raw_files
        attrs['software_versions'] = json.dumps(versions, sort_keys=True)
    old_attrs['generator_sha256'] = OLD_GENERATOR_HASH
    output = args.hdf5 or default_output
    require(output.exists(), "Partial HDF5 does not exist")
    # r+ uses normal HDF5 exclusive writer locking. Never disable file locking.
    with h5py.File(output, 'r+') as h:
        generation.verify_output(h, original, old_attrs)
        status = h['status'][:]
        require((status[indices] == 0).all(), "All four repair rows must still be status=0")
        equal(original.loc[status == 1], repaired.loc[status == 1])
        # Guard resume against an interrupted migration. A failed migration remains
        # explicitly blocked rather than blessing partially updated provenance.
        h.attrs['initialization_complete'] = False
        h.attrs['repair_original_manifest_sha256'] = old_attrs['manifest_sha256']
        h.attrs['repair_original_manifest_path'] = old_attrs['manifest_path']
        h.attrs['repair_previous_generator_sha256'] = OLD_GENERATOR_HASH
        h.attrs['repair_receipt_json'] = json.dumps(report, sort_keys=True)
        h.flush()
        for name in PARAMETERS:
            h[name][indices] = repaired.loc[indices, name].to_numpy()
        h['y'][indices] = repaired.loc[indices, list(generation.LABELS)].to_numpy(dtype=np.float32)
        h.flush()
        # Only these compatibility attributes change; all other contracts were checked.
        for name in ('manifest_path', 'manifest_sha256', 'generator_sha256'):
            h.attrs[name] = new_attrs[name]
        h.flush()
        h.attrs['initialization_complete'] = True
        try:
            generation.verify_output(h, repaired, new_attrs)
            require(np.array_equal(h['status'][:], status), "Status changed during migration")
        except Exception:
            h.attrs['initialization_complete'] = False
            h.flush()
            raise
        h.flush()
    print(f'Migrated four pending metadata rows in {output}. X and generated metadata untouched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('repair', 'migrate'))
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT /
                        'configs/generation/generate_m11_final_G1_500k.json')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repaired', type=Path, required=True)
    parser.add_argument('--hdf5', type=Path, help='Migration only; defaults to final dataset path')
    args = parser.parse_args()
    if args.action == 'repair':
        require(args.hdf5 is None, '--hdf5 is only for migration')
        repair(args)
    else:
        migrate(args)


if __name__ == '__main__':
    main()
