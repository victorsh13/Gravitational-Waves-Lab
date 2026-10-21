#!/usr/bin/env python3
"""M12.5.1-B: explicit opt-in generation/inference; plan performs neither.

The gate-hashed pilot modules remain unchanged. Physics, checkpoint deserialization,
shared-tensor forwarding and immutable-state hashing use their proven helpers.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.m12 import m12_5_common as c
from scripts.m12 import run_m12_5_1_pilot as pilot
from src.paths import resolve_data_root

MASSES = (*c.MASSES, (30., 10.), (85., 70.))
SPINS = ((-.6, -.6), (0., 0.), (.6, .6), (.6, -.6), (-.6, .6))
REPEATS, N_CONDITIONS, N_OBSERVATIONS = 128, 100, 12800
MASTER_SEED = 125101
GEN_SCHEMA = "M12.5.1-B-generation-v1"
INF_SCHEMA = "M12.5.1-B-inference-v1"
H5_SCHEMA = "M12.5.1-B-observations-v1"


def design():
    return dict(masses=[list(x) for x in MASSES], spins=[list(x) for x in SPINS],
                snrs=list(c.SNRS), repeats=REPEATS, sources=25,
                conditions=N_CONDITIONS, observations=N_OBSERVATIONS,
                master_seed=MASTER_SEED, reference_train_row=0)


def code_identity():
    return dict(pilot=c.code_identity(), full=c.digest(__file__))


def conditions(reference):
    # Reuse the pilot's fixed nuisance/environment contract, replace only grid.
    template = c.conditions(reference)[0]
    rows = []
    for m1, m2 in MASSES:
        for s1, s2 in SPINS:
            source = len(rows) // len(c.SNRS)
            for snr in c.SNRS:
                rows.append(dict(template, source_id=f"M1251B_SRC_{source:02d}",
                                 condition_id=len(rows), mass_1=m1, mass_2=m2,
                                 spin_1z=s1, spin_2z=s2, mass_ratio=m2/m1,
                                 chirp_mass=(m1*m2)**.6/(m1+m2)**.2,
                                 total_mass=m1+m2, chi_eff=(m1*s1+m2*s2)/(m1+m2),
                                 target_network_snr=snr))
    return rows


def seed_schedule():
    """Unique detector seeds, also disjoint from every pilot detector seed.

    This guarantees no reuse within A/B, not independence from every historical
    G1 seed (checking that would require an unauthorized full-manifest scan).
    Different integer seeds are the PRNG independence assumption, not a proof.
    """
    old, _ = c.seed_schedule()
    used = set(map(int, old.flat))
    rng = np.random.default_rng(MASTER_SEED)
    seeds, rejected = [], 0
    while len(seeds) < N_OBSERVATIONS * 3:
        seed = int(rng.integers(1, 2**32-1))
        if seed in used:
            rejected += 1
            continue
        used.add(seed)
        seeds.append(seed)
    return np.asarray(seeds, dtype=np.uint32).reshape(N_CONDITIONS, REPEATS, 3), rejected


def resource_plan(batch_size=16):
    c.require(1 <= batch_size <= 64, "Batch size must be 1..64")
    arrays = dict(X=N_OBSERVATIONS*3*16384*4,
                  raw_strain=N_OBSERVATIONS*3*19712*8,
                  signal_context=N_CONDITIONS*3*19712*8,
                  detector_seeds=N_OBSERVATIONS*3*4,
                  ids=3*N_OBSERVATIONS*8, y=N_OBSERVATIONS*3*8,
                  psds=3*(8193+9857)*8)
    # Exact Conv1d lengths for the frozen architecture (padding=k//2).
    lengths = (8193, 4097, 2049)
    convs = [(3, 16, 16, lengths[0]), (16, 32, 16, lengths[1]),
             (32, 64, 16, lengths[2])] + [(64, 64, 7, lengths[2])]*6 + [(64, 128, 1, lengths[2])]
    dense = [(128, 64), (64, 64), (64, 3)]
    parameters = sum(ci*co*k+co+2*co for ci, co, k, _ in convs)
    parameters += sum(ci*co+co for ci, co in dense)
    macs = sum(ci*co*k*n for ci, co, k, n in convs) + sum(ci*co for ci, co in dense)
    weights = 4*parameters*4
    input_bytes = batch_size*3*16384*4
    largest_activation = batch_size*128*2049*4
    return dict(design=design(), logical_array_bytes=arrays,
                logical_total_bytes=sum(arrays.values()),
                recommended_free_disk_bytes=2*sum(arrays.values()),
                disk_note="Logical uncompressed arrays; LZF ratio and CSV/HDF5 overhead unknown. 2x is headroom, not an upper bound.",
                batch_size=batch_size, input_batch_bytes=input_bytes,
                parameters_per_model=parameters, four_model_weight_bytes=weights,
                largest_single_activation_bytes=largest_activation,
                resident_weights_input_activation_floor_bytes=weights+input_bytes+largest_activation,
                gpu_note="Floor only, NOT peak: simultaneous intermediates, CUDA context, allocator and cuDNN workspace are additional; no optimizer or gradients.",
                model_examples=4*N_OBSERVATIONS,
                forward_calls=4*((N_OBSERVATIONS+batch_size-1)//batch_size),
                convolution_dense_MACs_per_example=macs,
                convolution_dense_MACs_total=4*N_OBSERVATIONS*macs,
                operations_note="MAC excludes normalization, activations, pooling, transfers and kernel overhead. Rough arithmetic FLOPs=2*MAC, not a runtime prediction.",
                generation_signal_preparations=N_CONDITIONS,
                waveform_calls=2*N_CONDITIONS, detector_noise_draws=3*N_OBSERVATIONS,
                detector_processing_passes=3*N_OBSERVATIONS,
                predictions_rows=N_OBSERVATIONS*4*3,
                runtime_note="No timing measured or extrapolated; waveform cost depends on duration. Whole-artifact SHA256 reads add sequential I/O.")


def generate(args, root, out):
    import h5py
    gate = c.read_json(args.gate)
    pilot.require_gate(gate)  # Exact original code identities and tolerances.
    g, spec, rows, _, files, provenance = c.load_frozen(root)
    c.require(provenance == gate["provenance"], "Gate inputs/runtime changed; rerun gate")
    plan = resource_plan(args.batch_size)
    c.require(shutil.disk_usage(out).free >= plan["recommended_free_disk_bytes"],
              "Insufficient free disk for documented 2x logical-array headroom")
    g.initialize_physics(spec)
    reference = rows[0]
    strains = g.read_psd_intervals(reference, files)
    hashes = {d: c.array_digest(strains[d].numpy()) for d in c.DETECTORS}
    c.require(hashes == gate["psd_interval_hashes"]["0"], "PSD interval changed")
    psds = g.get_block_psds(reference, long_strains=strains, psd_cache={})
    seeds, rejected = seed_schedule()
    schedule = conditions(reference)
    # Freeze design before any waveform generation or predictions; failures leave
    # a complete=False receipt and an immutable partial directory, never fallback.
    c.write_csv(out / "design.csv", schedule)
    c.write_json(out / "design.json", dict(design=design(), code=code_identity(),
                 seed_sha256=c.array_digest(seeds), rejected_seed_collisions=rejected,
                 pilot_seed_sha256=c.array_digest(c.seed_schedule()[0]), resource_plan=plan))
    obs, physical, placements = [], [], {}
    with h5py.File(out / "observations.h5", "x") as h:
        h.attrs.update(schema=H5_SCHEMA, complete=False, purpose="designed_conditional_diagnostic",
                       detector_order=",".join(c.DETECTORS), labels=",".join(c.LABELS),
                       units=",".join(c.UNITS), sampling_frequency_hz=4096,
                       duration_s=4., raw_duration_s=4.8125,
                       input_normalization="per_sample_per_detector_zscore", epsilon=1e-6,
                       signal_context_mode="full_projection", gate_sha256=c.digest(args.gate))
        x = h.create_dataset("X", (N_OBSERVATIONS, 3, 16384), dtype="f4", chunks=(1, 3, 16384), compression="lzf")
        raw = h.create_dataset("raw_strain", (N_OBSERVATIONS, 3, 19712), dtype="f8", chunks=(1, 3, 19712), compression="lzf")
        clean = h.create_dataset("signal_context", (N_CONDITIONS, 3, 19712), dtype="f8", chunks=(1, 3, 19712), compression="lzf")
        h.create_dataset("detector_seeds", data=seeds.reshape(-1, 3))
        h.create_dataset("condition_id", data=np.repeat(np.arange(N_CONDITIONS, dtype=np.int64), REPEATS))
        h.create_dataset("realization_id", data=np.tile(np.arange(REPEATS, dtype=np.int64), N_CONDITIONS))
        h.create_dataset("observation_id", data=np.arange(N_OBSERVATIONS, dtype=np.int64))
        y = h.create_dataset("y", (N_OBSERVATIONS, 3), dtype="f8")
        for kind in ("snr", "proc"):
            for d in c.DETECTORS:
                ds = h.create_dataset(f"psd/{kind}/{d}", data=psds[kind][d].numpy())
                ds.attrs.update(delta_f=float(psds[kind][d].delta_f), units="strain^2/Hz",
                                conditioning="raw empirical PSD, no inverse truncation")
        for row in schedule:
            cid = row["condition_id"]
            injector, signal, meta = pilot.prepare_signal(g, row, psds)
            gps = meta["actual_geocentric_time"]
            c.require(placements.setdefault(row["source_id"], gps) == gps,
                      "Source placement changed with SNR; no fallback")
            physical.append(dict(row, **meta, coa_phase=0., spin_1x=0., spin_1y=0.,
                                 spin_2x=0., spin_2y=0., eccentricity=0., reference_train_row=0,
                                 mass_units="Msun_detector_frame", distance_units="Mpc",
                                 angles_units="rad", gps_units="s",
                                 psd_identity=json.dumps(hashes, sort_keys=True)))
            zero = {d: injector.build_zero_strain(start_time=float(row["processing_start"]),
                    length=g.CONFIG.processing_length) for d in c.DETECTORS}
            injected = injector.inject_network(noises=zero, signals=signal)
            clean[cid] = np.stack([injected[d].strain.numpy() for d in c.DETECTORS])
            for r in range(REPEATS):
                oid = cid*REPEATS+r
                X, D = pilot.observation(g, row, psds, injector, signal, seeds[cid, r])
                D = np.stack([D[d].numpy() for d in c.DETECTORS])
                c.require(D.shape == (3, 19712) and np.isfinite(D).all(), "Invalid raw context")
                x[oid], raw[oid], y[oid] = X, D, [row[k] for k in c.LABELS]
                obs.append(dict(observation_id=oid, condition_id=cid, realization_id=r,
                                source_id=row["source_id"], tensor_sha256=c.array_digest(X),
                                raw_sha256=c.array_digest(D),
                                **{f"seed_{d}": int(seeds[cid, r, j]) for j, d in enumerate(c.DETECTORS)}))
            print(f"Condition {cid+1}/{N_CONDITIONS} complete", flush=True)
        h.attrs["complete"] = True
    c.write_csv(out / "conditions.csv", physical)
    c.write_csv(out / "observations.csv", obs)
    return dict(schema=GEN_SCHEMA, complete=True, design=design(), code=code_identity(),
                gate=c.record(args.gate, True), provenance=provenance, preserve_raw=True,
                seed_sha256=c.array_digest(seeds), independent_detector_seeds=seeds.size,
                rejected_seed_collisions=rejected, resource_plan=plan,
                interpretation="25 designed sources; noise repetitions are not BBH population draws",
                artifacts={name: c.record(out / name, True) for name in
                           ("observations.h5", "conditions.csv", "observations.csv", "design.csv", "design.json")})


def load_models(torch, root, device):
    """Same frozen configuration/scaler checks as pilot; old torch load helper."""
    from src.models.network import SimpleCNN_ResidualDilated
    models, records, scalers, states = {}, {}, {}, {}
    expected = None
    for name, tag in c.MODEL_TAGS.items():
        path = root / "models" / "checkpoints" / c.DATASET / (
            f"{c.DATASET}_SimpleCNN_ResidualDilated_{tag}_MSELoss_seed123_checkpoint.pt")
        records[name] = c.record(path, True)
        c.require(records[name]["sha256"] == c.MODEL_HASHES[name], f"Unrecognized checkpoint: {name}")
        ckpt = pilot.load_checkpoint(torch, path)
        cfg = ckpt["model_config"]
        c.require(cfg["label_names"] == list(c.LABELS) and cfg["n_detectors"] == 3
                  and cfg["signal_length"] == 16384 and cfg["dataset_id"] == c.DATASET,
                  "Checkpoint scientific contract mismatch")
        c.require(cfg["input_normalization"] == dict(enabled=False, already_normalized=True,
                  mode="per_sample_per_detector_zscore", eps=1e-6), "Checkpoint normalization mismatch")
        mean, std = (np.asarray(ckpt[k], dtype=np.float64) for k in ("y_mean", "y_std"))
        c.require(mean.shape == std.shape == (3,) and np.isfinite(mean).all()
                  and np.isfinite(std).all() and (std > 0).all(), "Invalid frozen scaler")
        if expected is None:
            expected = mean, std
        c.require(all(np.array_equal(a, b) for a, b in zip((mean, std), expected)), "Scalers differ")
        model = SimpleCNN_ResidualDilated(**cfg["model_kwargs"])
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device).eval()
        models[name], scalers[name], states[name] = model, (mean, std), pilot.state_hash(model)
        records[name].update(epoch=ckpt["epoch"], model_config=cfg, y_mean=mean.tolist(), y_std=std.tolist())
    return models, records, scalers, states


def verify_unchanged(models, records, states):
    for name, model in models.items():
        c.require(pilot.state_hash(model) == states[name], "Inference modified model state")
        after = c.record(records[name]["path"], True)
        c.require(all(after[k] == records[name][k] for k in after), "Checkpoint file changed")


def validate_observation(o, oid, array):
    c.require(int(o["observation_id"]) == oid and int(o["condition_id"]) == oid//REPEATS
              and int(o["realization_id"]) == oid % REPEATS
              and o["source_id"] == f"M1251B_SRC_{oid//(REPEATS*4):02d}"
              and c.array_digest(array) == o["tensor_sha256"], "Tensor/row alignment mismatch")


def infer(args, root, out):
    import h5py
    import torch
    run = c.read_json(args.study / "generation.json")
    c.require(run.get("schema") == GEN_SCHEMA and run.get("complete") is True, "Incomplete full study")
    c.require(run["design"] == design() and run["code"] == code_identity(), "Study code/design changed")
    c.require(c.digest(ROOT / "src/models/network.py") ==
              run["provenance"]["physics_hashes"]["src/models/network.py"], "Frozen model implementation changed")
    for name, record in run["artifacts"].items():
        c.require(c.digest(args.study / name) == record["sha256"], f"Study artifact changed: {name}")
    resource_plan(args.batch_size)
    if args.device == "cuda":
        c.require(torch.cuda.is_available(), "CUDA requested but unavailable")
    with (args.study / "observations.csv").open() as f:
        obs = list(csv.DictReader(f))
    with (args.study / "conditions.csv").open() as f:
        conditions_by_id = {int(r["condition_id"]): r for r in csv.DictReader(f)}
    c.require(len(obs) == N_OBSERVATIONS and set(conditions_by_id) == set(range(N_CONDITIONS)), "Incomplete study tables")
    models, records, scalers, states = load_models(torch, root, args.device)
    fields = ["observation_id", "condition_id", "realization_id", "source_id", "target_snr",
              "model", "label", "unit", "truth", "prediction", "prediction_standardized", "tensor_sha256"]
    with h5py.File(args.study / "observations.h5", "r") as h, (out / "predictions.csv").open("x", newline="") as f:
        c.require(h.attrs["schema"] == H5_SCHEMA and bool(h.attrs["complete"])
                  and h["X"].shape == (N_OBSERVATIONS, 3, 16384) and h["X"].dtype == np.dtype("float32")
                  and h.attrs["detector_order"] == ",".join(c.DETECTORS)
                  and h.attrs["labels"] == ",".join(c.LABELS), "Invalid tensor contract")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for start in range(0, N_OBSERVATIONS, args.batch_size):
                stop = min(N_OBSERVATIONS, start+args.batch_size)
                X, truths = h["X"][start:stop], h["y"][start:stop]
                for k, array in enumerate(X):
                    oid = start+k
                    validate_observation(obs[oid], oid, array)
                    c.require(all(int(h[key][oid]) == int(obs[oid][key]) for key in
                                  ("observation_id", "condition_id", "realization_id")), "HDF5 ID mismatch")
                    row = conditions_by_id[oid//REPEATS]
                    c.require(np.allclose(truths[k], [float(row[label]) for label in c.LABELS],
                                          rtol=1e-12, atol=1e-12), "HDF5 truth mismatch")
                batch = torch.from_numpy(X).to(args.device)
                for name, standardized in pilot.shared_predictions(models, batch).items():
                    mean, std = scalers[name]
                    physical = standardized.astype(np.float64)*std+mean
                    c.require(physical.shape == (stop-start, 3) and np.isfinite(physical).all(), "Invalid prediction")
                    for k in range(stop-start):
                        o = obs[start+k]
                        for j, label in enumerate(c.LABELS):
                            writer.writerow(dict(**{key: o[key] for key in fields[:4]},
                                target_snr=conditions_by_id[int(o["condition_id"])]["target_network_snr"],
                                model=name, label=label, unit=c.UNITS[j], truth=truths[k, j],
                                prediction=physical[k, j], prediction_standardized=standardized[k, j],
                                tensor_sha256=o["tensor_sha256"]))
    verify_unchanged(models, records, states)
    return dict(schema=INF_SCHEMA, complete=True, design=design(), code=code_identity(),
                generation_sha256=c.digest(args.study / "generation.json"), checkpoints=records,
                device=args.device, torch_version=torch.__version__, batch_size=args.batch_size,
                tensor_order="observation_id 0..12799; shared batch across all models",
                predictions=c.record(out / "predictions.csv", True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("plan", "generate", "infer"))
    p.add_argument("--run", action="store_true")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--gate", type=Path)
    p.add_argument("--study", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.phase == "plan":
        print(json.dumps(resource_plan(args.batch_size), indent=2))
        return
    if not args.run or not args.output_dir or (args.phase == "generate" and not args.gate) or (args.phase == "infer" and not args.study):
        p.error("Explicit --run, --output-dir, and --gate (generate) / --study (infer) required")
    root = Path(resolve_data_root(cli_data_root=args.data_root))
    out = c.output_directory(root, args.output_dir)
    result = dict(complete=False, phase=args.phase, design=design(), code=code_identity())
    try:
        result = generate(args, root, out) if args.phase == "generate" else infer(args, root, out)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        c.write_json(out / ("generation.json" if args.phase == "generate" else "inference.json"), result)


if __name__ == "__main__":
    main()
