#!/usr/bin/env python3
"""Opt-in 576-observation diagnostic, not a training dataset.

generate requires a passing, current reproduction gate and historical physics
versions. infer reads the saved tensors once in bounded batches and never
imports/executes generation physics. No historical artifact is opened writable.
"""
import argparse
import csv
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.m12 import m12_5_common as c
from src.paths import resolve_data_root


def prepare_signal(g, row, psds):
    """G1 orchestration using its unchanged physics helpers.

    This adapter is checked against build_g1_row by the reproduction gate.
    Placement is deterministic from the fixed train reference seed and depends
    on waveform duration, not target SNR or noise. No new placement policy.
    """
    cfg = g.CONFIG
    injector = g.SignalInjector(config=cfg, rng=np.random.default_rng(int(row["placement_seed"])))
    params = g.CBCParameters(**{k: float(row[k]) for k in (
        "mass_1", "mass_2", "spin_1z", "spin_2z", "inclination", "ra", "dec",
        "polarization_angle")}, distance=float(row["reference_distance_mpc"]))
    final_start = float(row["processing_start"]) + cfg.processing_context_start_seconds
    center = final_start + cfg.duration / 2
    c.require(abs(center - float(row["reference_geocentric_time"])) < cfg.delta_t,
              "Reference GPS mismatch")
    c.require(abs(float(row["processing_end"]) - float(row["processing_start"])
                  - cfg.processing_duration) < 1e-6, "Invalid processing context")
    waveform = g.WAVEFORM_GENERATOR.generate(params)
    provisional = g.DETECTOR_PROJECTOR.project(
        waveform.h_plus, waveform.h_cross, params, center)
    window = g.WINDOW_SELECTOR.select(provisional.strains, max_duration=cfg.duration)
    placement = injector.choose_segment_placement_containing_network(
        signals=window.strains, placement_policy="random_contained",
        safe_margin_start=float(cfg.safe_margin_start),
        safe_margin_end=float(cfg.safe_margin_end), enforce_safe_margins=True)
    # Preserve historical operation order at large absolute GPS values.
    placement_shift = final_start - float(placement.segment_start_time)
    gps = center + placement_shift
    projection = g.DETECTOR_PROJECTOR.project(
        waveform.h_plus, waveform.h_cross, params, gps)
    window = g.WINDOW_SELECTOR.select(projection.strains, max_duration=cfg.duration)
    c.require(window.metadata.used_window_start_time >= final_start - 2 * cfg.delta_t
              and window.metadata.used_window_end_time <= final_start + cfg.duration + 2 * cfg.delta_t,
              "Reprojected source does not fit final window")
    zeros = {d: injector.build_zero_strain(start_time=final_start, length=cfg.length)
             for d in c.DETECTORS}
    reference = injector.inject_network(noises=zeros, signals=window.strains)
    _, rho_ref = g.compute_network_optimal_snr(
        {d: reference[d].strain for d in c.DETECTORS}, psds["snr"], cfg)
    distance = g.rescale_distance_for_target_network_snr(
        params.distance, rho_ref, float(row["target_network_snr"]))
    signal = g.build_signal_network_for_params(
        params.with_distance(distance), geocentric_time=gps,
        final_start=final_start, injector=injector)
    snrs, rho = g.compute_network_optimal_snr(signal["segments"], psds["snr"], cfg)
    g.validate_snr_rescaling(rho, float(row["target_network_snr"]), cfg.snr_relative_tolerance)
    metadata = dict(reference_network_snr=rho_ref, final_network_snr=rho,
                    final_distance=distance, actual_geocentric_time=gps,
                    placement_offset_s=gps-center,
                    full_network_duration=window.metadata.full_network_duration,
                    required_final_duration=window.metadata.required_available_final_duration,
                    is_truncated=bool(window.metadata.is_truncated))
    metadata.update({f"snr_{d}": snrs[d] for d in c.DETECTORS})
    return injector, signal["projection"].strains, metadata


def observation(g, row, psds, injector, signal, detector_seeds):
    noise = g.build_gaussian_processing_noise(
        psds_proc=psds["proc"], processing_start=float(row["processing_start"]),
        detector_seeds=dict(zip(c.DETECTORS, map(int, detector_seeds))), injector=injector)
    injected = injector.inject_network(noises=noise, signals=signal)
    raw = {d: injected[d].strain for d in c.DETECTORS}
    processed = g.PROCESSOR.process_network(strains=raw, psds=psds["proc"])
    _, X = g.stack_and_normalize_network(processed)
    c.require(X.shape == (3, 16384) and np.isfinite(X).all(), "Invalid pilot tensor")
    return X, raw


def require_gate(gate):
    c.require(gate.get("schema") == "M12.5.1-gate-v1" and gate.get("passed") is True,
              "Missing/failed reproduction gate: generation blocked")
    c.require(gate.get("rows") == list(c.ROWS) and gate.get("code") == c.code_identity(),
              "Stale gate rows or code; rerun reproduction")
    c.require(gate.get("tolerances") ==
              dict(atol=c.ATOL, rtol=c.RTOL, relative_l2=c.REL_RMSE),
              "Gate numerical criteria changed")
    checks = gate.get("checks", [])
    c.require([x.get("row_id") for x in checks] == list(c.ROWS)
              and all(x.get("passed") is True for x in checks), "Incomplete gate")


def generate(args, root, out):
    import h5py
    gate = c.read_json(args.gate)
    require_gate(gate)
    g, spec, rows, _, files, provenance = c.load_frozen(root)
    c.require(provenance == gate["provenance"], "Gate inputs/runtime changed; rerun gate")
    reference = rows[0]
    g.initialize_physics(spec)
    strains = g.read_psd_intervals(reference, files)
    interval_hashes = {d: c.array_digest(strains[d].numpy()) for d in c.DETECTORS}
    c.require(interval_hashes == gate["psd_interval_hashes"]["0"], "PSD interval changed")
    psds = g.get_block_psds(reference, long_strains=strains, psd_cache={})
    seeds, collisions = c.seed_schedule()
    schedule = c.conditions(reference)
    observations, condition_rows, placements = [], [], {}
    path = out / "observations.h5"
    with h5py.File(path, "x") as h:
        h.attrs.update(schema="M12.5.1-pilot-v1", complete=False, purpose="diagnostic_only",
                       detector_order=",".join(c.DETECTORS), labels=",".join(c.LABELS),
                       units=",".join(c.UNITS), sampling_frequency_hz=4096, duration_s=4.,
                       raw_duration_s=4.8125, input_normalization="per_sample_per_detector_zscore",
                       epsilon=1e-6, phase=0., transverse_spins=0.,
                       gate_sha256=c.digest(args.gate))
        x = h.create_dataset("X", (576, 3, 16384), dtype="f4", chunks=(1, 3, 16384), compression="lzf")
        raw_data = (h.create_dataset("raw_strain", (576, 3, 19712), dtype="f8",
                                    chunks=(1, 3, 19712), compression="lzf") if args.preserve_raw else None)
        clean = (h.create_dataset("signal_context", (36, 3, 19712), dtype="f8",
                                 chunks=(1, 3, 19712), compression="lzf") if args.preserve_raw else None)
        h.create_dataset("detector_seeds", data=seeds.reshape(576, 3))
        h.create_dataset("condition_id", data=np.repeat(np.arange(36), c.REPEATS))
        h.create_dataset("realization_id", data=np.tile(np.arange(c.REPEATS), 36))
        h.create_dataset("observation_id", data=np.arange(576))
        y = h.create_dataset("y", (576, 3), dtype="f8")
        for kind in ("snr", "proc"):
            for d in c.DETECTORS:
                ds = h.create_dataset(f"psd/{kind}/{d}", data=psds[kind][d].numpy())
                ds.attrs.update(delta_f=float(psds[kind][d].delta_f), units="strain^2/Hz",
                                conditioning="raw empirical PSD, no inverse truncation")
        for row in schedule:
            cid = row["condition_id"]
            injector, signal, meta = prepare_signal(g, row, psds)
            gps = meta["actual_geocentric_time"]
            previous = placements.setdefault(row["source_id"], gps)
            c.require(gps == previous, "Placement changed with SNR; no fallback permitted")
            condition = dict(row, **meta, coa_phase=0., spin_1x=0., spin_1y=0.,
                             spin_2x=0., spin_2y=0., eccentricity=0., reference_train_row=0,
                             distance_units="Mpc", mass_units="Msun_detector_frame",
                             angles_units="rad", gps_units="s",
                             psd_identity=json.dumps(interval_hashes, sort_keys=True))
            condition_rows.append(condition)
            if clean is not None:
                zero = {d: injector.build_zero_strain(
                    start_time=float(row["processing_start"]), length=g.CONFIG.processing_length)
                        for d in c.DETECTORS}
                injected = injector.inject_network(noises=zero, signals=signal)
                clean[cid] = np.stack([injected[d].strain.numpy() for d in c.DETECTORS])
            for r in range(c.REPEATS):
                oid = cid * c.REPEATS + r
                X, raw = observation(g, row, psds, injector, signal, seeds[cid, r])
                x[oid] = X
                y[oid] = [row[k] for k in c.LABELS]
                if raw_data is not None:
                    raw_data[oid] = np.stack([raw[d].numpy() for d in c.DETECTORS])
                observations.append(dict(observation_id=oid, condition_id=cid, realization_id=r,
                                         source_id=row["source_id"], tensor_sha256=c.array_digest(X),
                                         **{f"seed_{d}": int(seeds[cid, r, j])
                                            for j, d in enumerate(c.DETECTORS)}))
            print(f"Condition {cid+1}/36 complete", flush=True)
        h.attrs["complete"] = True
    c.write_csv(out / "conditions.csv", condition_rows)
    c.write_csv(out / "observations.csv", observations)
    return dict(schema="M12.5.1-generation-v1", complete=True,
                gate=c.record(args.gate, True), provenance=provenance, code=c.code_identity(),
                master_seed=c.MASTER_SEED, rejected_seed_collisions=collisions,
                seed_sha256=c.array_digest(seeds), independent_detector_seeds=1728,
                tensors=576, reference_train_row=0, preserve_raw=args.preserve_raw,
                placement_policy="historical random_contained; fixed seed, per-source duration",
                interpretation="conditional estimator bias/noise sensitivity; not posterior uncertainty",
                artifacts={name: c.record(out / name, True) for name in
                           ("observations.h5", "conditions.csv", "observations.csv")})


def load_checkpoint(torch, path):
    # weights_only is absent in PyTorch 1.9.1. Files are hash-pinned before loading.
    kwargs = dict(map_location="cpu")
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def state_hash(model):
    return {k: c.array_digest(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}


def shared_predictions(models, batch):
    """One already-created tensor, unchanged order, no RNG or generation here."""
    return {name: model(batch).detach().cpu().numpy() for name, model in models.items()}


def infer(args, root, out):
    import h5py
    import torch
    from src.models.network import SimpleCNN_ResidualDilated
    run = c.read_json(args.pilot / "generation.json")
    c.require(run.get("schema") == "M12.5.1-generation-v1" and run.get("complete") is True,
              "Pilot generation incomplete")
    c.require(run.get("code") == c.code_identity(), "Pilot implementation changed")
    c.require(c.digest(ROOT / "src/models/network.py") ==
              run["provenance"]["physics_hashes"]["src/models/network.py"],
              "Frozen model implementation changed")
    # Allow relocation to the VM, but verify the complete small pilot artifact.
    for name in ("observations.h5", "conditions.csv", "observations.csv"):
        c.require(c.digest(args.pilot / name) == run["artifacts"][name]["sha256"],
                  f"Pilot artifact changed: {name}")
    c.require(1 <= args.batch_size <= 64, "Inference batch size must be 1..64")
    if args.device == "cuda":
        c.require(torch.cuda.is_available(), "CUDA requested but unavailable")
    with (args.pilot / "observations.csv").open() as f:
        obs = list(csv.DictReader(f))
    with (args.pilot / "conditions.csv").open() as f:
        conditions_by_id = {int(r["condition_id"]): r for r in csv.DictReader(f)}
    c.require(len(obs) == 576 and len(conditions_by_id) == 36, "Incomplete pilot table")
    models, records, scalers, states = {}, {}, {}, {}
    expected_scaler = None
    for name, tag in c.MODEL_TAGS.items():
        path = root / "models" / "checkpoints" / c.DATASET / (
            f"{c.DATASET}_SimpleCNN_ResidualDilated_{tag}_MSELoss_seed123_checkpoint.pt")
        records[name] = c.record(path, True)
        c.require(records[name]["sha256"] == c.MODEL_HASHES[name], f"Unrecognized frozen checkpoint: {name}")
        checkpoint = load_checkpoint(torch, path)
        cfg = checkpoint["model_config"]
        c.require(cfg["label_names"] == list(c.LABELS) and cfg["n_detectors"] == 3
                  and cfg["signal_length"] == 16384 and cfg["dataset_id"] == c.DATASET,
                  f"Checkpoint scientific contract mismatch: {name}")
        c.require(cfg["input_normalization"] == dict(
            enabled=False, already_normalized=True, mode="per_sample_per_detector_zscore", eps=1e-6),
            "Checkpoint normalization mismatch; refusing to normalize twice")
        mean, std = (np.asarray(checkpoint[k], dtype=np.float64) for k in ("y_mean", "y_std"))
        c.require(mean.shape == std.shape == (3,) and np.isfinite(mean).all()
                  and np.isfinite(std).all() and (std > 0).all(), "Invalid frozen scaler")
        if expected_scaler is None:
            expected_scaler = (mean, std)
        c.require(all(np.array_equal(a, b) for a, b in zip((mean, std), expected_scaler)),
                  "Checkpoints use different scalers")
        model = SimpleCNN_ResidualDilated(**cfg["model_kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(args.device).eval()
        models[name], scalers[name], states[name] = model, (mean, std), state_hash(model)
        records[name].update(epoch=checkpoint["epoch"], model_config=cfg,
                             y_mean=mean.tolist(), y_std=std.tolist())
        del checkpoint
    fields = ["observation_id", "condition_id", "realization_id", "source_id", "target_snr",
              "model", "label", "unit", "truth", "prediction", "prediction_standardized", "tensor_sha256"]
    with h5py.File(args.pilot / "observations.h5", "r") as h, (
            out / "predictions.csv").open("x", newline="") as f:
        c.require(bool(h.attrs["complete"]) and h["X"].shape == (576, 3, 16384)
                  and h.attrs["detector_order"] == ",".join(c.DETECTORS), "Invalid tensor contract")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for start in range(0, 576, args.batch_size):
                stop = min(576, start + args.batch_size)
                X = h["X"][start:stop]  # Exactly one read/shared tensor per batch.
                truth = h["y"][start:stop]
                for offset, array in enumerate(X):
                    oid = start + offset
                    c.require(int(obs[oid]["observation_id"]) == oid
                              and int(obs[oid]["condition_id"]) == oid // 16
                              and int(obs[oid]["realization_id"]) == oid % 16
                              and c.array_digest(array) == obs[oid]["tensor_sha256"],
                              "Tensor/row alignment mismatch")
                batch = torch.from_numpy(X).to(args.device)
                for name, standardized in shared_predictions(models, batch).items():
                    mean, std = scalers[name]
                    physical = standardized.astype(np.float64) * std + mean
                    c.require(physical.shape == (stop-start, 3) and np.isfinite(physical).all(),
                              f"Nonfinite/invalid prediction: {name}")
                    for k in range(stop-start):
                        o = obs[start+k]
                        for j, label in enumerate(c.LABELS):
                            writer.writerow(dict(
                                **{key: o[key] for key in ("observation_id", "condition_id", "realization_id", "source_id")},
                                target_snr=conditions_by_id[int(o["condition_id"])]["target_network_snr"],
                                model=name, label=label, unit=c.UNITS[j], truth=truth[k, j],
                                prediction=physical[k, j], prediction_standardized=standardized[k, j],
                                tensor_sha256=o["tensor_sha256"]))
    for name, model in models.items():
        c.require(state_hash(model) == states[name], "Inference modified model state")
        before = records[name]
        after = c.record(before["path"], True)
        c.require(all(after[k] == before[k] for k in after), "Checkpoint file changed")
    return dict(schema="M12.5.1-inference-v1", complete=True,
                generation_sha256=c.digest(args.pilot / "generation.json"),
                checkpoints=records, device=args.device, torch_version=torch.__version__,
                batch_size=args.batch_size, tensor_order="observation_id 0..575, identical for all models",
                predictions=c.record(out / "predictions.csv", True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "infer"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--preserve-raw", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not args.run or (args.phase == "generate" and not args.gate) or (
            args.phase == "infer" and not args.pilot):
        parser.error("Add --run and --gate for generate, or --pilot for infer")
    root = resolve_data_root(cli_data_root=args.data_root)
    out = c.output_directory(root, args.output_dir)
    result = dict(complete=False, phase=args.phase)
    try:
        result = generate(args, root, out) if args.phase == "generate" else infer(args, root, out)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        c.write_json(out / ("generation.json" if args.phase == "generate" else "inference.json"), result)


if __name__ == "__main__":
    main()
