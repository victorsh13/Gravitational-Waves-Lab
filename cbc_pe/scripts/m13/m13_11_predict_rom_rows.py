#!/usr/bin/env python3
"""M13.11: mu, sigma, Sigma of the three compared networks on the ROM rows and on the same original val G3T rows (VM).

Protocol: docs/experiments/M13_11_bayesian_reference.md (amendment 2: gate G-NET; amendments 6 and 7: networks).
Input: rom_rows_<UTC>/fp16_memmap of scripts/m13/m13_11_build_rom_rows.py (180 val rows, signal SEOBNRv4_ROM, X
already z-scored as the G3T fp16_memmap) and the val G3T fp16_memmap (same rows, signal S15).

Networks (amendments 2, 6, 7): 4a-G3T s123 (primary), M14.5a (descriptive), M14.5b (final model of M14).
Model construction, label scaler and the mu / sigma / cov split are those of scripts/m14/predict_events_npz.py
(itself predict_cnn_hdf5.py): checkpoint loaded on CPU, train-only label stats of the dataset of the checkpoint.

Reproduction check (before anything is saved): the predictions on the original val G3T rows must reproduce those of
the stored val npz of the same checkpoint (preds_val_G3T) at max |dmu|, max |dcov| <= 1e-5 (standardized units),
aligned by row id.

Gate G-NET (amendments 2 and 8, point 1), per cell = network, regime and label, on the 180 selected rows (no PE),
physical label units:
    |RMSE_ROM - RMSE_S15| / RMSE_S15 <= 0.02   (relative)
RMSE_ROM: predictions on the ROM X; RMSE_S15: predictions on the val G3T X of the same rows; same truth.
A failed cell does not stop the script: it is recorded (gnet_json) and printed, and only that network / regime /
label is reported as not valid.

Output, <data_root>/results/M13_3/M13_11/gnet_<UTC>.npz (never overwritten; standardized label units, as every
prediction npz of the repository; physical = x * y_std + y_mean, Sigma_phys = D Sigma D, D = diag(y_std)):
    row_ids (180, sorted), regime, y_true (physical), y_mean, y_std, label_names, nets
    mu_rom_<net>, mu_s15_<net>                (180, 3)
    sigma_rom_<net>, sigma_s15_<net>, cov_rom_<net>, cov_s15_<net>   only for heteroscedastic full-covariance nets
    checkpoint_<net>, checkpoint_sha256_<net>, reference_<net>
    val_check_json, gnet_json, provenance_json

PyTorch 1.9.1 compatible; no new dependencies.

Usage (VM, from cbc_pe):
    python scripts/m13/m13_11_predict_rom_rows.py [--rom-dir <data_root>/results/M13_3/M13_11/rom_rows_<UTC>]
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import resolve_data_root, resolve_processed_artifact
from scripts.m14.predict_events_npz import LABELS, file_sha256, predict, require, split_output
from scripts.predict_cnn_hdf5 import import_model_class

ROM_STAMP = "rom_rows_20261106T173402Z"
SPLIT = "val"
G3T_DATASET_ID = "M13_G3_band1024_w16_500k/m13_13_g3t"
G3T_MEMMAP = "fp16_memmap"
REPRO_TOL = 1e-5          # max |dmu|, |dcov| (standardized), as VAL_CHECK_TOL of predict_events_npz.py
GNET_MAX_REL = 0.02       # amendments 2 and 8: relative |dRMSE| <= 0.02 per network, regime and label
REGIMES = ("short", "medium", "long")

# Checkpoints relative to <data_root>/models/checkpoints/; reference val npz relative to <data_root>.
NETS = {
    "4a_G3T": dict(role="primary (amendment 2)",
                   checkpoint="M13_3/M13_13_4a_G3T_s123_final.pt",
                   reference="results/M13_3/M13_13_4a_G3T_s123/val_predictions_embeddings.npz"),
    "M14_5a": dict(role="descriptive (amendment 6)",
                   checkpoint="M13_3/M14_5a_4a_G3T_fullcov_s123_final.pt",
                   reference="results/M13_3/M14_5a_4a_G3T_fullcov_s123/preds_val_G3T/val_predictions_embeddings.npz"),
    "M14_5b": dict(role="final model of M14 (amendment 7)",
                   checkpoint="M13_3/M14_5b_4a_G3T_tbank_fullcov_s123_final.pt",
                   reference="results/M13_3/M14_5b_4a_G3T_tbank_fullcov_s123/preds_val_G3T/val_predictions_embeddings.npz"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rom-dir", type=Path, default=None,
                   help=f"rom_rows_<UTC> directory (default <data_root>/results/M13_3/M13_11/{ROM_STAMP})")
    p.add_argument("--output", type=Path, default=None,
                   help="New output npz (default <data_root>/results/M13_3/M13_11/gnet_<UTC>.npz; must not exist)")
    p.add_argument("--nets", nargs="+", choices=list(NETS), default=list(NETS))
    p.add_argument("--data-root", type=Path, default=None, help="Precedence: CLI, CBC_PE_DATA_ROOT, default")
    return p.parse_args()


def load_model(checkpoint_path, data_root, device):
    """As predict_events_npz.py: CPU load, model to device, scaler of the dataset of the checkpoint."""
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False   # trusted local checkpoint (own training)
    checkpoint = torch.load(checkpoint_path, **load_kwargs)
    model_config = checkpoint["model_config"]
    model_kwargs = dict(model_config["model_kwargs"])
    model_kwargs.setdefault("n_detectors", 3)
    model_kwargs.setdefault("n_outputs", len(LABELS))
    model = import_model_class(model_config["class_name"])(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    ds = checkpoint["config"]["dataset"]
    require(ds["dataset_id"] == G3T_DATASET_ID, f"{checkpoint_path.name}: dataset {ds['dataset_id']}")
    stats_path = resolve_processed_artifact(data_root=data_root, dataset_id=ds["dataset_id"],
                                            file_name=ds["label_stats_file"])
    with np.load(stats_path) as stats:
        y_mean, y_std = stats["y_mean"].astype(np.float32), stats["y_std"].astype(np.float32)
        names = [str(x) for x in stats["label_names"].tolist()] if "label_names" in stats.files else LABELS
    require(names == LABELS, f"Label order {names}")
    require(np.array_equal(np.asarray(checkpoint["y_mean"], np.float32), y_mean)
            and np.array_equal(np.asarray(checkpoint["y_std"], np.float32), y_std),
            "Checkpoint scaler differs from label stats")
    norm = model_config.get("input_normalization", {"enabled": False})
    # fp16_memmap X (G3T and ROM rows) is already z-scored: the model must not normalize again.
    require(not norm.get("enabled", False) and norm.get("already_normalized", False),
            f"{checkpoint_path.name}: input_normalization {norm}")
    return model, checkpoint, y_mean, y_std, norm, stats_path


def load_reference(path, checkpoint_path, y_mean, y_std, norm, row_ids):
    """Stored val predictions of the checkpoint, at row_ids (aligned by idx_val, whatever its order)."""
    with np.load(path, allow_pickle=True) as r:
        if "checkpoint_file" in r.files:
            require(Path(str(r["checkpoint_file"])).name == checkpoint_path.name, f"{path}: other checkpoint")
        require([str(x) for x in r["label_names"].tolist()] == LABELS, f"{path}: label order")
        require(np.array_equal(r["y_mean"], y_mean) and np.array_equal(r["y_std"], y_std), f"{path}: scaler")
        if "input_normalization" in r.files:
            ref_norm = r["input_normalization"].item()
            require(not ref_norm.get("enabled", False) and ref_norm.get("mode") == norm.get("mode")
                    and float(ref_norm.get("eps")) == float(norm.get("eps")),
                    f"{path}: input normalization {ref_norm} vs model {norm}")
        memmap_dir = Path(str(r["memmap_dir"])) if "memmap_dir" in r.files else None
        idx = np.asarray(r[f"idx_{SPLIT}"], dtype=np.int64)
        order = np.argsort(idx, kind="stable")
        pos = order[np.searchsorted(idx[order], row_ids)]
        require(np.array_equal(idx[pos], row_ids), f"{path}: rows of rom_rows missing from idx_{SPLIT}")
        mu = np.asarray(r[f"pred_{SPLIT}"][pos], dtype=np.float64)
        cov = np.asarray(r[f"cov_{SPLIT}"][pos], dtype=np.float64) if f"cov_{SPLIT}" in r.files else None
        y_std_units = np.asarray(r[f"y_{SPLIT}"][pos], dtype=np.float64)
    return mu, cov, y_std_units, memmap_dir


def rmse(pred, y):
    return np.sqrt(np.mean((pred - y) ** 2, axis=0))


def main():
    args = parse_args()
    data_root = resolve_data_root(cli_data_root=args.data_root)
    m13_11 = data_root / "results" / "M13_3" / "M13_11"
    rom_dir = args.rom_dir if args.rom_dir is not None else m13_11 / ROM_STAMP
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output if args.output is not None else m13_11 / f"gnet_{stamp}.npz"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| torch", torch.__version__)
    print("rom_dir:", rom_dir)

    # ROM rows: fp16_memmap of m13_11_build_rom_rows.py (sorted row ids, y unstandardized float32).
    rom_mm = rom_dir / "fp16_memmap"
    manifest = json.loads((rom_mm / "manifest.json").read_text())
    require(manifest.get("kind") == "m13_11_rom_rows", f"{rom_mm}: kind {manifest.get('kind')}")
    require(file_sha256(rom_mm / f"row_ids_{SPLIT}.npy") == manifest["splits"][SPLIT]["row_ids_sha256"],
            f"{rom_mm}: row_ids sha256 differs from manifest")
    row_ids = np.load(rom_mm / f"row_ids_{SPLIT}.npy").astype(np.int64)
    require(np.all(np.diff(row_ids) > 0), "ROM row ids not strictly increasing")
    selection = json.loads((rom_dir / "selection.json").read_text())
    regime_of = {int(r["row_index"]): r["regime"] for r in selection["rows"]}
    require(sorted(regime_of) == row_ids.tolist(), "ROM row ids differ from selection.json")
    regime = np.array([regime_of[int(i)] for i in row_ids])
    y_true = np.asarray(np.load(rom_mm / f"y_{SPLIT}.npy"), dtype=np.float64)
    X_rom = np.asarray(np.load(rom_mm / f"X_{SPLIT}.npy", mmap_mode="r"), dtype=np.float32)  # fp16 -> fp32 exact
    require(X_rom.shape == (row_ids.size, 3, 16384) and np.isfinite(X_rom).all(), f"ROM X {X_rom.shape}")

    # Same rows, original val G3T X (signal S15).
    g3t_mm = resolve_processed_artifact(data_root=data_root, dataset_id=G3T_DATASET_ID, file_name=G3T_MEMMAP)
    g3t_ids = np.load(g3t_mm / f"row_ids_{SPLIT}.npy", mmap_mode="r")
    pos = np.searchsorted(g3t_ids, row_ids)
    require(np.all(pos < g3t_ids.size) and np.array_equal(np.asarray(g3t_ids)[pos], row_ids),
            f"ROM rows missing from {g3t_mm}/row_ids_{SPLIT}.npy")
    X_s15 = np.asarray(np.load(g3t_mm / f"X_{SPLIT}.npy", mmap_mode="r")[pos], dtype=np.float32)
    y_g3t = np.asarray(np.load(g3t_mm / f"y_{SPLIT}.npy", mmap_mode="r")[pos], dtype=np.float64)
    require(np.allclose(y_g3t, y_true, rtol=1e-6, atol=0.0), "y of ROM rows differs from val G3T memmap")
    print(f"rows: {row_ids.size} ({', '.join(f'{k} {int(np.sum(regime == k))}' for k in REGIMES)}); g3t memmap {g3t_mm}")

    payload = dict(row_ids=row_ids, regime=regime, y_true=y_true, label_names=np.array(LABELS),
                   nets=np.array(args.nets))
    val_check, gnet, provenance = {}, dict(threshold_rel=GNET_MAX_REL,
                                           definition="|RMSE_ROM - RMSE_S15| / RMSE_S15, physical units, "
                                                      "per net, regime and label, on the rows of rom_rows"), {}
    scaler = None
    for net in args.nets:
        spec = NETS[net]
        ckpt_path = data_root / "models" / "checkpoints" / spec["checkpoint"]
        ref_path = data_root / spec["reference"]
        print(f"\n== {net} ({spec['role']}): {ckpt_path}")
        model, checkpoint, y_mean, y_std, norm, stats_path = load_model(ckpt_path, data_root, device)
        if scaler is None:
            scaler = (y_mean, y_std)
            payload.update(y_mean=y_mean, y_std=y_std)
        require(np.array_equal(scaler[0], y_mean) and np.array_equal(scaler[1], y_std), f"{net}: other scaler")

        ref_mu, ref_cov, ref_y, ref_mm = load_reference(ref_path, ckpt_path, y_mean, y_std, norm, row_ids)
        if ref_mm is not None:
            require(ref_mm.resolve() == Path(g3t_mm).resolve(), f"{ref_path}: memmap {ref_mm} is not {g3t_mm}")
        require(np.allclose(ref_y * y_std + y_mean, y_true, rtol=1e-5, atol=1e-6), f"{ref_path}: y misaligned")

        with torch.no_grad():
            out_s15, _ = predict(model, X_s15, device)
            out_rom, _ = predict(model, X_rom, device)
        mu_s15, sig_s15, cov_s15 = split_output(model, out_s15, len(LABELS))
        mu_rom, sig_rom, cov_rom = split_output(model, out_rom, len(LABELS))
        require(np.isfinite(mu_s15).all() and np.isfinite(mu_rom).all(), f"{net}: nonfinite output")
        require((cov_s15 is None) == (ref_cov is None), f"{net}: covariance in model and reference differ")

        dmu = float(np.abs(mu_s15 - ref_mu).max())
        dcov = float(np.abs(cov_s15 - ref_cov).max()) if cov_s15 is not None else 0.0
        print(f"reproduction of {ref_path.name} on the {row_ids.size} rows: max|dmu| {dmu:.3e}, "
              f"max|dcov| {dcov:.3e} (tol {REPRO_TOL:g})")
        require(dmu <= REPRO_TOL and dcov <= REPRO_TOL, f"{net}: REPRODUCTION FAILED, nothing saved")
        val_check[net] = dict(n=int(row_ids.size), passed=True, max_abs_dmu=dmu, max_abs_dcov=dcov, tol=REPRO_TOL,
                              reference=str(ref_path), memmap_dir=str(g3t_mm))

        # G-NET, physical units.
        p_rom, p_s15 = mu_rom * y_std + y_mean, mu_s15 * y_std + y_mean
        gnet[net] = {}
        print(f"G-NET {net}: |RMSE_ROM - RMSE_S15| / RMSE_S15 (max {GNET_MAX_REL:.0%})")
        for name in REGIMES:
            sel = regime == name
            r_rom, r_s15 = rmse(p_rom[sel], y_true[sel]), rmse(p_s15[sel], y_true[sel])
            gnet[net][name] = {}
            for j, lab in enumerate(LABELS):
                rel = float(abs(r_rom[j] - r_s15[j]) / r_s15[j])
                gnet[net][name][lab] = dict(n=int(sel.sum()), rmse_rom=float(r_rom[j]), rmse_s15=float(r_s15[j]),
                                            rel_diff=rel, passed=bool(rel <= GNET_MAX_REL))
                print(f"  {name:<7}{lab:<11} ROM {r_rom[j]:.4g}  S15 {r_s15[j]:.4g}  rel {rel:.2%}  "
                      f"{'PASS' if rel <= GNET_MAX_REL else 'FAIL'}")
        gnet[net]["passed"] = all(gnet[net][k][lab]["passed"] for k in REGIMES for lab in LABELS)
        bad = [(k, lab) for k in REGIMES for lab in LABELS if not gnet[net][k][lab]["passed"]]
        print(f"G-NET {net}: " + ("PASS in every cell" if not bad else f"not valid (only these cells): {bad}"))

        payload[f"mu_rom_{net}"], payload[f"mu_s15_{net}"] = mu_rom, mu_s15
        if cov_rom is not None:
            payload.update({f"sigma_rom_{net}": sig_rom, f"sigma_s15_{net}": sig_s15,
                            f"cov_rom_{net}": cov_rom, f"cov_s15_{net}": cov_s15})
        payload[f"checkpoint_{net}"] = np.array(str(ckpt_path))
        payload[f"checkpoint_sha256_{net}"] = np.array(file_sha256(ckpt_path))
        payload[f"reference_{net}"] = np.array(str(ref_path))
        provenance[net] = dict(role=spec["role"], checkpoint_tag=str(checkpoint.get("checkpoint_tag")),
                               epoch=int(checkpoint.get("epoch", -1)), label_stats=str(stats_path),
                               input_normalization=norm, model_config=checkpoint["model_config"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    provenance.update(rom_dir=str(rom_dir.resolve()), rom_manifest=manifest, g3t_memmap=str(g3t_mm),
                      script_sha256=file_sha256(__file__), created_utc=datetime.now(timezone.utc).isoformat(),
                      torch_version=torch.__version__, device=str(device))
    payload.update(val_check_json=np.array(json.dumps(val_check)), gnet_json=np.array(json.dumps(gnet)),
                   provenance_json=np.array(json.dumps(provenance, default=str)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        np.savez(f, **payload)
    print("\nG-NET, all cells passed:", {net: gnet[net]["passed"] for net in args.nets})
    print("Saved:", output)


if __name__ == "__main__":
    main()
