"""Focused CPU contracts; no waveform generation, checkpoint inference or notebooks."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts.m12 import m12_5_common as c
from scripts.m12 import run_m12_5_1_full as f
from scripts.m12 import analyze_m12_5_1_full as a
from test_m12_5_repeated_noise import reference


class FullStudyTests(unittest.TestCase):
    def test_grid_metadata_and_frozen_nuisances(self):
        rows = f.conditions(reference())
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r['source_id'] for r in rows}), 25)
        self.assertEqual(100*f.REPEATS, 12800)
        self.assertEqual({(r['mass_1'], r['mass_2']) for r in rows}, set(f.MASSES))
        self.assertEqual({(r['spin_1z'], r['spin_2z']) for r in rows}, set(f.SPINS))
        for row in rows:
            for k, value in reference().items():
                self.assertEqual(row[k], value)
            m1, m2 = row['mass_1'], row['mass_2']
            self.assertEqual(row['mass_ratio'], m2/m1)
            self.assertEqual(row['chi_eff'], (m1*row['spin_1z']+m2*row['spin_2z'])/(m1+m2))
            self.assertEqual(row['target_network_snr'], c.SNRS[row['condition_id'] % 4])
        self.assertNotEqual(rows[12]['chi_eff'], rows[16]['chi_eff'])
        self.assertEqual(len(c.conditions(reference())), 36)  # Pilot defaults untouched.

    def test_seed_determinism_uniqueness_and_pilot_exclusion(self):
        seeds, rejected = f.seed_schedule()
        again, rejected_again = f.seed_schedule()
        np.testing.assert_array_equal(seeds, again)
        self.assertEqual(rejected, rejected_again)
        self.assertEqual(seeds.shape, (100, 128, 3))
        self.assertEqual(len(np.unique(seeds)), 38400)
        self.assertFalse(set(seeds.flat) & set(c.seed_schedule()[0].flat))
        self.assertGreater(seeds.min(), 0)
        self.assertLess(seeds.max(), 2**32-1)

    def test_resource_counts_and_budget(self):
        p = f.resource_plan(16)
        self.assertEqual(p['logical_array_bytes']['X'], 2516582400)
        self.assertEqual(p['logical_array_bytes']['raw_strain'], 12800*3*19712*8)
        self.assertEqual(p['logical_total_bytes'], sum(p['logical_array_bytes'].values()))
        self.assertEqual(p['recommended_free_disk_bytes'], 2*p['logical_total_bytes'])
        self.assertEqual(p['forward_calls'], 3200)
        self.assertEqual(p['predictions_rows'], 153600)
        self.assertEqual(p['model_examples'], 51200)
        self.assertEqual(p['convolution_dense_MACs_total'], p['convolution_dense_MACs_per_example']*51200)
        self.assertEqual(f.resource_plan(7)['forward_calls'], 4*((12800+6)//7))
        for batch in (0, 65):
            with self.assertRaises(ValueError):
                f.resource_plan(batch)

    def test_resource_parameters_match_frozen_architecture_without_forward(self):
        # Construction only, no checkpoint load and no forward pass.
        from src.models.network import SimpleCNN_ResidualDilated
        model = SimpleCNN_ResidualDilated(embedding_dim=64)
        self.assertEqual(sum(p.numel() for p in model.parameters()), f.resource_plan()['parameters_per_model'])

    def test_bad_gate_cannot_reach_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'gate.json'
            c.write_json(path, dict(passed=False))
            with patch.object(c, 'load_frozen', side_effect=AssertionError('physics reached')) as loader:
                with self.assertRaisesRegex(ValueError, 'blocked'):
                    f.generate(SimpleNamespace(gate=path), Path(tmp), Path(tmp))
                loader.assert_not_called()

    def test_tensor_identity_alignment_and_shared_forward(self):
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        row = dict(observation_id=129, condition_id=1, realization_id=1,
                   source_id='M1251B_SRC_00', tensor_sha256=c.array_digest(tensor))
        f.validate_observation(row, 129, tensor)
        for bad in (dict(row, realization_id=2), dict(row, source_id='wrong'),
                    dict(row, tensor_sha256='wrong')):
            with self.assertRaisesRegex(ValueError, 'alignment'):
                f.validate_observation(bad, 129, tensor)
        models = {name: Mock() for name in c.MODEL_TAGS}
        for model in models.values():
            model.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.zeros((1, 3))
        f.pilot.shared_predictions(models, tensor)
        for model in models.values():
            self.assertIs(model.call_args.args[0], tensor)

    def test_checkpoint_state_and_file_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fixture.pt'
            path.write_bytes(b'unchanged checkpoint fixture')
            records = {'M11': c.record(path, True)}
            with patch.object(f.pilot, 'state_hash', return_value={'w': 'same'}):
                f.verify_unchanged({'M11': object()}, records, {'M11': {'w': 'same'}})
                path.write_bytes(b'changed checkpoint fixture')
                with self.assertRaisesRegex(ValueError, 'file changed'):
                    f.verify_unchanged({'M11': object()}, records, {'M11': {'w': 'same'}})
            with patch.object(f.pilot, 'state_hash', return_value={'w': 'different'}):
                with self.assertRaisesRegex(ValueError, 'model state'):
                    f.verify_unchanged({'M11': object()}, records, {'M11': {'w': 'same'}})

    def test_finite_R_bootstrap_pairing_and_tails(self):
        for R in a.PREFIXES:
            noise = np.linspace(-1, 1, R)
            pred = 20+noise[:, None]+np.arange(4)[None, :]
            base = dict(condition_id=0, label='chirp_mass', R=R)
            summary, pairs = a.summarize_condition(pred, 20., base, 100)
            for row in summary:
                self.assertAlmostEqual(row['conditional_mse'], row['bias_squared']+(R-1)/R*row['sample_variance'])
                self.assertGreaterEqual(row['absolute_error_q95'], row['absolute_error_q90'])
            self.assertAlmostEqual(pairs[0]['delta_bias_ci_low'], -1.)
            self.assertAlmostEqual(pairs[0]['delta_bias_ci_high'], -1.)
            self.assertAlmostEqual(pairs[0]['delta_sample_variance_ci_high'], 0.)
            self.assertEqual(summary, a.summarize_condition(pred, 20., base, 100)[0])
        result = a.extended_moments(np.ones(128)*20, 20)
        self.assertTrue(np.isnan(result['squared_bias_fraction']))
        self.assertEqual(result['noise_std'], 0)
        self.assertEqual(a.absolute_interval(-2, 1), (0., 2))
        self.assertEqual(a.absolute_interval(-3, -1), (1, 3))

    def test_scalar_validation_and_convergence_on_one_condition_fixture(self):
        condition = f.conditions(reference())[0]
        physical = pd.DataFrame([condition]).set_index('condition_id')
        obs = pd.DataFrame([dict(observation_id=r, tensor_sha256=f'tensor_{r}') for r in range(128)])
        rows = [dict(observation_id=r, condition_id=0, realization_id=r,
                     source_id=condition['source_id'], target_snr=10., model=model,
                     label=label, unit=c.UNITS[j], truth=condition[label],
                     prediction=condition[label]+r/128+k, tensor_sha256=f'tensor_{r}')
                for r in range(128) for k, model in enumerate(c.MODEL_TAGS)
                for j, label in enumerate(c.LABELS)]
        df = pd.DataFrame(rows)
        with patch.object(f, 'N_OBSERVATIONS', 128):
            a.validate_predictions(df, physical, obs)
            wrong = df.copy()
            wrong.loc[0, 'tensor_sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'identical saved tensors'):
                a.validate_predictions(wrong, physical, obs)
            wrong = df.copy()
            wrong.loc[0, 'truth'] += 1
            with self.assertRaisesRegex(ValueError, 'truth mismatch'):
                a.validate_predictions(wrong, physical, obs)
        summary, pairs = a.analyze(df, physical, bootstrap=100)
        self.assertEqual(len(summary), 4*3*4)
        self.assertEqual(len(pairs), 6*3*4)
        self.assertEqual(set(summary.R), set(a.PREFIXES))
        self.assertLess(summary.decomposition_residual.abs().max(), 1e-10)
        self.assertEqual(summary[summary.model == 'M11'].sort_values('R').iloc[0].R, 16)

    def test_csv_json_lineage_and_metadata_without_hdf5(self):
        row = f.conditions(reference())[0]
        seeds = f.seed_schedule()[0][:1]
        physical = dict(row, actual_geocentric_time=102., psd_identity='fixed_psd',
                        snr_H1=6., snr_L1=8., snr_V1=0.,
                        final_network_snr=10., final_distance=1000.)
        observations = [dict(observation_id=r, condition_id=0, realization_id=r,
                            source_id=row['source_id'], tensor_sha256=f'tensor_{r}',
                            **{f'seed_{d}': int(seeds[0, r, j]) for j, d in enumerate(c.DETECTORS)})
                        for r in range(128)]
        predictions = [dict(observation_id=r, condition_id=0, realization_id=r,
                            source_id=row['source_id'], target_snr=10., model=model,
                            label=label, unit=c.UNITS[j], truth=row[label],
                            prediction=row[label], tensor_sha256=f'tensor_{r}')
                       for r in range(128) for model in c.MODEL_TAGS
                       for j, label in enumerate(c.LABELS)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(f, 'N_OBSERVATIONS', 128), \
                patch.object(f, 'conditions', return_value=[row]), \
                patch.object(f, 'seed_schedule', return_value=(seeds, 0)):
            study, inference = Path(tmp)/'study', Path(tmp)/'inference'
            study.mkdir()
            inference.mkdir()
            c.write_csv(study/'conditions.csv', [physical])
            c.write_csv(study/'observations.csv', observations)
            c.write_csv(study/'design.csv', [row])
            c.write_json(study/'design.json', f.design())
            c.write_csv(inference/'predictions.csv', predictions)
            generation = dict(schema=f.GEN_SCHEMA, complete=True, design=f.design(),
                              code=f.code_identity(), seed_sha256=c.array_digest(seeds),
                              provenance=dict(selected_rows={'0': reference()}),
                              artifacts={name: c.record(study/name, True) for name in
                                         ('conditions.csv', 'observations.csv', 'design.csv', 'design.json')})
            c.write_json(study/'generation.json', generation)
            receipt = dict(schema=f.INF_SCHEMA, complete=True, design=f.design(),
                           code=f.code_identity(), generation_sha256=c.digest(study/'generation.json'),
                           predictions=c.record(inference/'predictions.csv', True),
                           checkpoints={name: dict(sha256=sha) for name, sha in c.MODEL_HASHES.items()})
            c.write_json(inference/'inference.json', receipt)
            # No HDF5 or checkpoints exist in this fixture.
            df, metadata, _, _ = a.load_inputs(inference, study)
            self.assertEqual(len(df), 128*12)
            self.assertEqual(metadata.loc[0, 'spin_1z'], -.6)
            with (study/'conditions.csv').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'Scalar artifact mismatch'):
                a.load_inputs(inference, study)


if __name__ == '__main__':
    unittest.main()
