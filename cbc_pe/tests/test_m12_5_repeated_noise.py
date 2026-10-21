"""Small deterministic CPU checks; never generate waveforms or load checkpoints.

python -m unittest discover -s tests -p test_m12_5_repeated_noise.py -v
"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.m12 import m12_5_common as c
from scripts.m12.run_m12_5_1_pilot import (
    require_gate, prepare_signal, observation, shared_predictions, load_checkpoint, generate,
)
from scripts.m12.analyze_m12_5_1 import analyze


def reference():
    return dict(inclination=2.3380511530923638, ra=2.5593519730571117,
                dec=0.7795671858078682, polarization_angle=3.4573112363932053,
                processing_start=100., processing_end=104.8125,
                reference_geocentric_time=102.40625, block_id="train_block",
                file_group_id=0, noise_crop_id="train_crop", psd_start=-40., psd_end=88.,
                placement_seed=11000000)


class RepeatedNoiseTests(unittest.TestCase):
    def test_conditions_and_global_seed_identity(self):
        rows = c.conditions(reference())
        self.assertEqual(len(rows), 36)
        self.assertEqual(len({r["source_id"] for r in rows}), 9)
        for row in rows:
            for key, value in reference().items():
                self.assertEqual(row[key], value)
            self.assertEqual(row["spin_1z"], row["spin_2z"])
            self.assertIn((row["mass_1"], row["mass_2"]), c.MASSES)
            self.assertIn(row["target_network_snr"], c.SNRS)
        a, collisions = c.seed_schedule()
        b, again = c.seed_schedule()
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (36, 16, 3))
        self.assertEqual(len(np.unique(a)), 1728)
        self.assertEqual(collisions, again)

    def test_finite_R_identity_and_constant_predictions(self):
        for values in (np.arange(16)/3, np.ones(16)*7, np.array([-2., 9.])):
            result = c.moments(values, 2.)
            self.assertAlmostEqual(result["conditional_mse"],
                                   result["bias_squared"] + (len(values)-1)/len(values)
                                   * result["sample_variance"])
        self.assertEqual(c.moments(np.ones(16)*7, 2.)["sample_variance"], 0.)

    def test_comparison_nonfinite_shape_and_near_zero(self):
        x = np.zeros((3, 16384), dtype=np.float32)
        self.assertTrue(c.compare_x(x, x)["passed"])
        self.assertFalse(c.compare_x(x[:, :-1], x)["passed"])
        bad = x.copy()
        bad[0, 0] = np.nan
        self.assertFalse(c.compare_x(bad, x)["passed"])
        bad[0, 0] = 0.1
        self.assertFalse(c.compare_x(bad, x)["passed"])
        metadata = c.metadata_comparison({"actual_geocentric_time": 1e9+0.01},
                                        {"actual_geocentric_time": 1e9})
        self.assertFalse(metadata["actual_geocentric_time"]["passed"])

    def test_runtime_and_failed_stale_gate(self):
        versions = {k: "1" for k in c.PHYSICS_PACKAGES}
        c.check_versions(versions, versions)
        with self.assertRaisesRegex(ValueError, "runtime"):
            c.check_versions(versions, dict(versions, pycbc="2"))
        with self.assertRaisesRegex(ValueError, "blocked"):
            require_gate(dict(passed=False))
        gate = dict(schema="M12.5.1-gate-v1", passed=True, rows=list(c.ROWS),
                    code=c.code_identity(),
                    tolerances=dict(atol=c.ATOL, rtol=c.RTOL, relative_l2=c.REL_RMSE),
                    checks=[dict(row_id=i, passed=True) for i in c.ROWS])
        require_gate(gate)
        gate["code"] = {}
        with self.assertRaisesRegex(ValueError, "Stale"):
            require_gate(gate)

    def test_output_no_overwrite_and_repo_guard(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            out = root / "results" / c.DATASET / "M12_5_1" / "new"
            self.assertEqual(c.output_directory(root, out), out)
            with self.assertRaises(FileExistsError):
                c.output_directory(root, out)
            with self.assertRaises(ValueError):
                c.output_directory(root, root / "processed" / "bad")
            with self.assertRaises(ValueError):
                c.output_directory(c.ROOT, c.ROOT / "results" / c.DATASET / "M12_5_1" / "bad")

    def test_failed_gate_cannot_enter_generation(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "gate.json"
            c.write_json(path, dict(passed=False))
            with patch.object(c, "load_frozen", side_effect=AssertionError("Generation reached")) as loader:
                with self.assertRaisesRegex(ValueError, "blocked"):
                    generate(SimpleNamespace(gate=path), Path(t), Path(t))
                loader.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Missing required input"):
                c.record(Path(t) / "missing.h5")

    def test_shared_tensor_and_old_torch_load_signature(self):
        batch = object()
        models = {k: Mock() for k in c.MODEL_TAGS}
        for model in models.values():
            model.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.zeros((2, 3))
        with patch("numpy.random.default_rng", side_effect=AssertionError("Noise regenerated")):
            result = shared_predictions(models, batch)
        self.assertEqual(list(result), list(c.MODEL_TAGS))
        for model in models.values():
            model.assert_called_once_with(batch)
        calls = []
        def old_load(path, map_location):
            calls.append((path, map_location))
            return {"frozen": True}
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "checkpoint"
            path.write_bytes(b"immutable fixture")
            before = c.record(path, True)
            self.assertTrue(load_checkpoint(SimpleNamespace(load=old_load), path)["frozen"])
            self.assertEqual(c.record(path, True), before)
        self.assertEqual(calls[0][1], "cpu")

    def test_noise_context_order_and_normalization(self):
        # Real shared normalization, deterministic fake physics; no PyCBC call.
        from src.models.dataset import normalize_input_per_sample_per_detector_zscore
        arrays = {d: np.linspace(-3, 5, 16384)**(j+1) for j, d in enumerate(c.DETECTORS)}
        processor = Mock()
        processor.process_network.return_value = arrays
        g = SimpleNamespace(
            build_gaussian_processing_noise=Mock(return_value={"noise": "only once"}),
            PROCESSOR=processor,
            stack_and_normalize_network=lambda net: (
                None, normalize_input_per_sample_per_detector_zscore(
                    np.stack([net[d] for d in c.DETECTORS]))))
        injector = Mock()
        injector.inject_network.return_value = {
            d: SimpleNamespace(strain=f"raw_{d}") for d in c.DETECTORS}
        X, raw = observation(g, reference(), {"proc": "PSD"}, injector,
                             {"full_projection": True}, [11, 22, 33])
        self.assertEqual(list(raw), list(c.DETECTORS))
        np.testing.assert_allclose(X.mean(axis=1), 0, atol=1e-6)
        np.testing.assert_allclose(X.std(axis=1), 1, atol=2e-6)
        self.assertFalse(np.array_equal(X[0], X[1]))
        self.assertEqual(g.build_gaussian_processing_noise.call_count, 1)
        self.assertEqual(g.build_gaussian_processing_noise.call_args.kwargs["detector_seeds"],
                         dict(H1=11, L1=22, V1=33))
        self.assertEqual(injector.inject_network.call_args.kwargs["signals"],
                         {"full_projection": True})

    def test_signal_preparation_no_noisy_SNR_and_fixed_placement(self):
        # This checks orchestration. Physical numerical agreement is the opt-in gate.
        class Params:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            def with_distance(self, distance):
                return Params(**dict(self.__dict__, distance=distance))
        cfg = SimpleNamespace(duration=4., delta_t=1/4096,
                              processing_context_start_seconds=.40625,
                              processing_duration=4.8125, safe_margin_start=0.,
                              safe_margin_end=0., length=16384, snr_relative_tolerance=.05)
        injector = Mock()
        injector.choose_segment_placement_containing_network.return_value = SimpleNamespace(
            segment_start_time=100.40625)
        injector.inject_network.return_value = {d: SimpleNamespace(strain="signal_only")
                                                for d in c.DETECTORS}
        window = SimpleNamespace(strains={"window": True}, metadata=SimpleNamespace(
            used_window_start_time=100.5, used_window_end_time=103.5,
            full_network_duration=3., required_available_final_duration=1.,
            is_truncated=False))
        g = SimpleNamespace(CONFIG=cfg, SignalInjector=Mock(return_value=injector),
                            CBCParameters=Params, WAVEFORM_GENERATOR=Mock(),
                            DETECTOR_PROJECTOR=Mock(), WINDOW_SELECTOR=Mock(),
                            compute_network_optimal_snr=Mock(),
                            rescale_distance_for_target_network_snr=Mock(
                                side_effect=lambda d, rho, target: d*rho/target),
                            build_signal_network_for_params=Mock(),
                            validate_snr_rescaling=Mock())
        g.WINDOW_SELECTOR.select.return_value = window
        g.build_signal_network_for_params.return_value = {
            "segments": {"final": True}, "projection": SimpleNamespace(strains={"full": True})}
        gps = []
        for row in c.conditions(reference())[:4]:
            target = row["target_network_snr"]
            g.compute_network_optimal_snr.side_effect = [
                ({d: 1. for d in c.DETECTORS}, 30.),
                ({d: target/np.sqrt(3) for d in c.DETECTORS}, target)]
            _, signal, meta = prepare_signal(g, row, {"snr": "raw PSD"})
            self.assertEqual(signal, {"full": True})
            self.assertEqual(meta["final_distance"], 1000.*30./target)
            self.assertEqual(meta["final_network_snr"], target)
            gps.append(meta["actual_geocentric_time"])
        self.assertEqual(len(set(gps)), 1)

    def test_historical_optimal_SNR_on_analytic_cpu_fixture(self):
        # A bin-centered sinusoid of amplitude A and duration 1 s in S(f)=1
        # has rho=A under the one-sided G1 norm. No noise/waveform generation.
        from pycbc.types import TimeSeries, FrequencySeries
        from src.snr import compute_network_optimal_snr, rescale_distance_for_target_network_snr
        t = np.arange(4096) / 4096
        cfg = SimpleNamespace(length=4096, delta_t=1/4096, flength=2049,
                              delta_f=1., low_frequency_cutoff=30.,
                              snr_high_frequency_cutoff=512.)
        signals = {d: TimeSeries(a*np.sin(2*np.pi*64*t), delta_t=1/4096)
                   for d, a in zip(c.DETECTORS, (1., 2., 3.))}
        psds = {d: FrequencySeries(np.ones(2049), delta_f=1.) for d in c.DETECTORS}
        detector, network = compute_network_optimal_snr(signals, psds, cfg)
        np.testing.assert_allclose(list(detector.values()), [1., 2., 3.], atol=1e-10)
        self.assertAlmostEqual(network, np.sqrt(14), places=10)
        self.assertEqual(compute_network_optimal_snr(signals, psds, cfg)[1], network)
        self.assertAlmostEqual(rescale_distance_for_target_network_snr(1000., network, 20.),
                               1000.*np.sqrt(14)/20.)

    def test_scalar_analysis_pairs_units_and_alignment(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "predictions.csv"
            rows = []
            for condition in c.conditions(reference()):
                cid = condition["condition_id"]
                for r in range(16):
                    for k, model in enumerate(c.MODEL_TAGS):
                        for j, label in enumerate(c.LABELS):
                            rows.append(dict(
                                observation_id=16*cid+r, condition_id=cid, realization_id=r,
                                source_id=condition["source_id"], target_snr=condition["target_network_snr"],
                                model=model, label=label, unit=c.UNITS[j], truth=condition[label],
                                prediction=condition[label] + (r-7.5)*0.1+k,
                                tensor_sha256=f"fixture_{16*cid+r}"))
            c.write_csv(path, rows)
            receipt = dict(schema="M12.5.1-inference-v1", complete=True,
                           predictions=dict(sha256=c.digest(path)))
            summary, pairs = analyze(path, receipt, bootstrap=100)
            self.assertEqual(len(summary), 432)
            self.assertEqual(len(pairs), 648)
            first = pairs.iloc[0]
            self.assertAlmostEqual(first.delta_mean_prediction, -1.)
            self.assertAlmostEqual(first.delta_mean_prediction_ci_low, -1.)
            self.assertAlmostEqual(first.delta_mean_prediction_ci_high, -1.)
            self.assertLess(summary.decomposition_residual.abs().max(), 1e-10)
            rows[0]["tensor_sha256"] = "wrong_tensor"
            other = Path(t) / "bad.csv"
            c.write_csv(other, rows)
            receipt["predictions"]["sha256"] = c.digest(other)
            with self.assertRaisesRegex(ValueError, "identical tensors"):
                analyze(other, receipt, bootstrap=100)


if __name__ == "__main__":
    unittest.main()
