"""Contracts shared by the opt-in M12.5.1 gate, pilot and scalar analysis.

Imports do not initialize waveform libraries, load checkpoints or open data.
Generation requires historical physics versions; inference is a separate phase.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = "M11_final_G1_500k"
ROWS = (0, 1, 5830)  # Includes one repaired source; never selected from predictions.
LABELS = ("chirp_mass", "total_mass", "chi_eff")
DETECTORS = ("H1", "L1", "V1")
UNITS = ("Msun_detector_frame", "Msun_detector_frame", "dimensionless")
MASSES = ((20., 15.), (40., 30.), (70., 50.))
SPINS = (-0.6, 0., 0.6)
SNRS = (10., 15., 20., 24.5)
REPEATS = 16
MASTER_SEED = 125001
ATOL, RTOL, REL_RMSE = 1e-5, 1e-5, 1e-6
PHYSICS_PACKAGES = ("numpy", "scipy", "pycbc", "lalsuite")
PHYSICS_FILES = (
    "src/config.py", "src/parameters.py", "src/sampling.py", "src/waveform.py",
    "src/detectors.py", "src/windowing.py", "src/injection.py", "src/snr.py",
    "src/processing.py", "src/models/dataset.py", "src/models/network.py",
)
CONFIG_HASH = "cf9562725bb88ee68b325beb3e44b41bfb5327a6d1a946cd1d9577bab6bd9467"
MANIFEST_HASH = "544579ef7270675b80caee34cfe440d9b767b02ada3fd5f4de2e8e0f066b864e"
MODEL_TAGS = {
    "M11": "M11_final_G1_inputzscore_resdilated_emb64_d124_train400k",
    "M12.1": "M12_1a_G1_mahalanobis_alpha0p5_meanone_seed123",
    "M12.2": "M12_2_G1_marginal_alpha0p5_meanone_seed123",
    "M12.3-S1": "M12_3_S1_G1_balanced_sampling_p035_seed123",
}
MODEL_HASHES = dict(zip(MODEL_TAGS, (
    "3670f0dbe985cf4b3405ee47b970eedd56f573d0cd5741804a0626e336ccd4dd",
    "2b573cc67b3bcee54881f174381706286db6a41ef20c5234c5508154ba553670",
    "8d33c89ebc670a0570aac7f88da012ac6975900f2ebcacd1d01d2d717efeffc1",
    "45826a25ac854a8b85e2b65b186e810468e6646fe4b671b6ef5914a05bb76bee",
)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode()
                          + a.tobytes()).hexdigest()


def record(path, content=False):
    p = Path(path).resolve()
    require(p.is_file(), f"Missing required input: {p}")
    st = p.stat()
    out = dict(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns)
    if content:
        out["sha256"] = digest(p)
    return out


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    require(bool(rows), "Empty scalar export")
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def output_directory(root, path):
    """Exclusive new directory under external results; never accepts repo paths."""
    p = Path(path).resolve()
    allowed = (Path(root) / "results" / DATASET / "M12_5_1").resolve()
    require(ROOT != p and ROOT not in p.parents, "Outputs must be outside Git")
    require(allowed in p.parents, f"Output must be a child of {allowed}")
    p.mkdir(parents=True, exist_ok=False)
    return p


def code_identity():
    files = ("m12_5_common.py", "reproduce_m12_5_1_g1.py",
             "run_m12_5_1_pilot.py")
    return {name: digest(Path(__file__).parent / name) for name in files}


def scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_versions(historical, runtime):
    bad = {k: dict(expected=historical.get(k), actual=runtime.get(k))
           for k in PHYSICS_PACKAGES if historical.get(k) != runtime.get(k)}
    require(not bad, f"Incompatible physics runtime; no override: {bad}")


def load_frozen(root):
    """Read headers and only predetermined train rows, not full HDF5 columns.

    CSV parsing stops at row 5830. The historical whole-manifest digest is
    recorded, not recomputed: no held-out manifest rows are read.
    """
    import h5py
    import pandas as pd
    from scripts import generate_m11_final_G1 as g

    directory = Path(root) / "processed" / DATASET
    dataset = directory / "m11_final_G1_500k.h5"
    manifest = directory / "m11_final_G1_500k_manifest_repaired.csv"
    receipt = directory / "m11_final_G1_500k_manifest_repaired.csv.repair.json"
    provenance = dict(dataset=record(dataset), manifest=record(manifest),
                      repair=record(receipt, True))
    runtime = {k: version(k) for k in
               (*PHYSICS_PACKAGES, "h5py", "pandas", "torch")}
    with h5py.File(dataset, "r") as h:
        spec = json.loads(h.attrs["config_json"])
        canonical = hashlib.sha256(json.dumps(
            spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        require(canonical == h.attrs["config_sha256"] == CONFIG_HASH,
                "Not the audited G1 configuration")
        cfg = ROOT / "configs/generation/generate_m11_final_G1_500k.json"
        require(read_json(cfg) == spec, "Repository/config_json disagreement")
        require(h.attrs["manifest_sha256"] == MANIFEST_HASH,
                "Not the repaired frozen manifest")
        require(Path(h.attrs["manifest_path"]).name == manifest.name,
                "Wrong manifest identity")
        repair = read_json(receipt)
        require(repair == json.loads(h.attrs["repair_receipt_json"]),
                "Repair receipt disagrees with HDF5")
        require(h["X"].shape == (500000, 3, 16384)
                and h["y"].shape == (500000, 3)
                and h["X"].dtype == np.dtype("float32"), "Wrong G1 shape/dtype")
        require(h.attrs["detector_order"] == ",".join(DETECTORS)
                and h.attrs["labels"] == ",".join(LABELS), "Wrong label/detector order")
        require(h.attrs["input_normalization"] == "per_sample_per_detector_zscore"
                and h.attrs["input_zscore_epsilon"] == 1e-6
                and h.attrs["signal_context_mode"] == "full_projection",
                "G1 preprocessing attributes disagree")
        historical = json.loads(h.attrs["software_versions"])
        check_versions(historical, runtime)
        hashes = json.loads(h.attrs["source_module_sha256"])
        for name in PHYSICS_FILES:
            require(digest(ROOT / name) == hashes[name], f"Changed physics: {name}")
        require(digest(Path(g.__file__)) == h.attrs["generator_sha256"],
                "Historical generator hash mismatch")
        provenance.update(config_sha256=canonical, manifest_sha256=MANIFEST_HASH,
                          generator_sha256=h.attrs["generator_sha256"],
                          physics_hashes={k: hashes[k] for k in PHYSICS_FILES},
                          historical_versions=historical, runtime_versions=runtime,
                          platform=platform.platform(), python=platform.python_version())
        stored = {}
        for i in ROWS:
            require(scalar(h["split"][i]) == "train" and h["status"][i] == 1,
                    f"Row {i} is not completed train data")
            stored[i] = {k: scalar(h[k][i]) for k in h if k not in ("X", "y")}
            require(np.array_equal(h["y"][i], np.asarray(
                [stored[i][k] for k in LABELS], dtype=np.float32)), f"Labels disagree at row {i}")
        raw_files = json.loads(h.attrs["raw_psd_files"])
    selected = {}
    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader):
            if i in ROWS:
                # Use HDF5 scalar types to preserve integers and round-trip floats.
                converted = {}
                for k, value in raw.items():
                    expected = stored[i][k]
                    actual = (int(value) if isinstance(expected, int)
                              else float(value) if isinstance(expected, float) else value)
                    require(actual == expected, f"Manifest/HDF5 mismatch row {i}: {k}")
                    converted[k] = actual
                selected[i] = pd.Series(converted)
            if i == max(ROWS):
                break
    require(set(selected) == set(ROWS), "Predetermined train rows missing")
    frames = pd.DataFrame(list(selected.values()))
    psd_files, _ = g.resolve_cached_psd_files(Path(root), frames)
    provenance["psd_files"] = {}
    for (block, detector), (path, start) in psd_files.items():
        originals = [v for k, v in raw_files.items() if Path(k).name == path.name]
        require(len(originals) == 1 and originals[0]["size"] == path.stat().st_size,
                f"PSD file identity/size mismatch: {path}")
        provenance["psd_files"][f"{block}/{detector}"] = record(path)
    provenance["selected_rows"] = {str(i): selected[i].to_dict() for i in ROWS}
    return g, spec, selected, stored, psd_files, provenance


def compare_x(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    shape = a.shape == b.shape == (3, 16384)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    out = dict(shape=list(a.shape), shape_ok=shape, finite=finite, passed=False)
    if not shape or not finite:
        return out
    delta = a.astype(float) - b.astype(float)
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    relative = rmse / max(float(np.sqrt(np.mean(b.astype(float) ** 2))), 1e-30)
    out.update(max_abs=float(np.max(np.abs(delta))), rmse=rmse,
               relative_l2=relative,
               max_relative_floor_atol=float(np.max(
                   np.abs(delta) / np.maximum(np.abs(b), ATOL))),
               bitwise_equal=bool(np.array_equal(a, b)),
               passed=bool(np.allclose(a, b, atol=ATOL, rtol=RTOL)
                           and relative <= REL_RMSE))
    return out


def metadata_comparison(actual, stored):
    results = {}
    for k, value in actual.items():
        expected = stored[k]
        if isinstance(value, (bool, int, np.integer, np.bool_)):
            ok = value == expected
        else:
            # GPS uses an absolute, sub-sample tolerance, never relative to 1e9.
            tol = 1e-6 if "time" in k or "offset" in k else 1e-8
            ok = np.isclose(value, expected, rtol=0 if "time" in k else 1e-8,
                            atol=tol)
        def finite_scalar(v):
            v = scalar(v)
            return str(v) if isinstance(v, float) and not np.isfinite(v) else v
        results[k] = dict(actual=finite_scalar(value), expected=finite_scalar(expected), passed=bool(ok))
    return results


def seed_schedule():
    """Collision rejection is deterministic and audited across all 1728 seeds."""
    rng = np.random.default_rng(MASTER_SEED)
    used, seeds, collisions = set(), [], 0
    for _ in range(36 * REPEATS * 3):
        while True:
            seed = int(rng.integers(1, 2**32 - 1))
            if seed not in used:
                used.add(seed)
                seeds.append(seed)
                break
            collisions += 1
    return np.asarray(seeds, dtype=np.uint32).reshape(36, REPEATS, 3), collisions


def conditions(reference):
    out = []
    for mass1, mass2 in MASSES:
        for spin in SPINS:
            source = len(out) // len(SNRS)
            for snr in SNRS:
                row = dict(source_id=f"M1251_SRC_{source:02d}", condition_id=len(out),
                           mass_1=mass1, mass_2=mass2, spin_1z=spin, spin_2z=spin,
                           chirp_mass=(mass1 * mass2)**0.6 / (mass1 + mass2)**0.2,
                           total_mass=mass1 + mass2, chi_eff=spin,
                           target_network_snr=snr, reference_distance_mpc=1000.)
                for k in ("inclination", "ra", "dec", "polarization_angle",
                          "processing_start", "processing_end", "reference_geocentric_time",
                          "block_id", "file_group_id", "noise_crop_id", "psd_start", "psd_end",
                          "placement_seed"):
                    row[k] = scalar(reference[k])
                out.append(row)
    return out


def moments(prediction, truth):
    """MSE = bias² + (R-1)/R * unbiased sample variance, exactly at finite R."""
    p = np.asarray(prediction, dtype=float)
    require(p.ndim == 1 and len(p) >= 2 and np.isfinite(p).all(), "Invalid repetitions")
    mean = float(p.mean())
    bias = mean - float(truth)
    variance = float(p.var(ddof=1))
    mse = float(np.mean((p - truth)**2))
    component = (len(p) - 1) / len(p) * variance
    require(np.isclose(mse, bias*bias + component, rtol=1e-10, atol=1e-12),
            "Finite-R identity failed")
    return dict(mean_prediction=mean, bias=bias, sample_variance=variance,
                conditional_mse=mse, bias_squared=bias*bias,
                finite_R_variance_component=component,
                decomposition_residual=mse - bias*bias - component)
