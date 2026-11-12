"""Predict mu, Sigma and embeddings of a checkpoint on the real-event inputs of an events_X_<UTC>.npz (VM).

Model construction, label scaler and the mu / sigma / cov split are those of scripts/predict_cnn_hdf5.py
(import_model_class, label stats from the dataset of the checkpoint, full_covariance_from_output).
The events X is already the network input (z-scored once in the M13.7 processing). As in the fp16_memmap
mode of predict_cnn_hdf5.py, the model input normalization must therefore be disabled
(already_normalized), and its mode and eps must match those recorded in the events npz.

Outputs (standardized label units): pred (mu), sigma = sqrt(diag Sigma), cov (Sigma), emb, plus y_mean,
y_std, event names and checkpoint. Never overwrites.

--val-check N: also predicts the first N rows of the val fp16 memmap used by preds_val_G3T of the same
checkpoint and requires max |dmu|, max |dcov| <= 1e-5 against that npz (same tolerance as M14.5).

PyTorch 1.9.1 compatible; no new dependencies.

Usage (from cbc_pe):
    python scripts/m14/predict_events_npz.py --checkpoint M13_3/<tag>_final.pt \
        --input <events_X.npz> --output <new.npz> [--val-check 64]
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
from src.models.evaluate import extract_predictions_and_embeddings
from scripts.predict_cnn_hdf5 import import_model_class

LABELS = ["chirp_mass", "total_mass", "chi_eff"]
VAL_CHECK_TOL = 1e-5      # max |dmu|, |dcov| (standardized), as REPRO_TOL in M14.5
BATCH_SIZE = 64


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def x_sha256(X):
    """Same definition as scripts/m14/export_real_events_X.py."""
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float32).tobytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="Checkpoint relative to <data_root>/models/checkpoints/")
    p.add_argument("--input", type=Path, required=True, help="events_X_<UTC>.npz")
    p.add_argument("--output", type=Path, required=True, help="New output npz (must not exist)")
    p.add_argument("--val-check", type=int, default=0, help="N first val memmap rows to compare with preds_val_G3T")
    p.add_argument("--data-root", type=Path, default=None, help="Precedence: CLI, CBC_PE_DATA_ROOT, default")
    return p.parse_args()


def split_output(model, out, k):
    """mu, sigma, cov as in predict_cnn_hdf5.py (standardized)."""
    if getattr(model, "heteroscedastic", False) and getattr(model, "covariance", "diag") == "full":
        from src.models.network import full_covariance_from_output
        mu_t, _, cov_t = full_covariance_from_output(torch.from_numpy(np.asarray(out, dtype=np.float32)), k)
        cov = cov_t.numpy().astype(np.float32)
        return mu_t.numpy(), np.sqrt(np.diagonal(cov, axis1=1, axis2=2)).astype(np.float32), cov
    if getattr(model, "heteroscedastic", False):
        return out[:, :k], np.exp(out[:, k:]), None
    return out, None, None


def predict(model, X, device):
    """Sequential batches of float32 X, through the same extractor as predict_cnn_hdf5.py."""
    loader = ((torch.from_numpy(np.ascontiguousarray(X[i:i + BATCH_SIZE], dtype=np.float32)),
               torch.zeros(len(X[i:i + BATCH_SIZE]), len(LABELS)))
              for i in range(0, len(X), BATCH_SIZE))
    out, emb, _ = extract_predictions_and_embeddings(model=model, loader=loader, device=device)
    return out, emb


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    data_root = resolve_data_root(cli_data_root=args.data_root)
    checkpoint_path = data_root / "models" / "checkpoints" / args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| torch", torch.__version__)
    print("checkpoint:", checkpoint_path)

    # Trusted local checkpoint (own training). torch >= 2.6 needs weights_only=False; 1.9.1 has no such argument.
    # Loaded on CPU: metadata tensors (y_mean, y_std) are compared as numpy; only the model goes to `device`.
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    checkpoint = torch.load(checkpoint_path, **load_kwargs)
    model_config = checkpoint["model_config"]
    model_kwargs = dict(model_config["model_kwargs"])
    model_kwargs.setdefault("n_detectors", 3)
    model_kwargs.setdefault("n_outputs", len(LABELS))
    model = import_model_class(model_config["class_name"])(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Train-only label scaler of the dataset of the checkpoint (as predict_cnn_hdf5.py), checked against the checkpoint.
    ds = checkpoint["config"]["dataset"]
    stats_path = resolve_processed_artifact(data_root=data_root, dataset_id=ds["dataset_id"], file_name=ds["label_stats_file"])
    with np.load(stats_path) as stats:
        y_mean, y_std = stats["y_mean"].astype(np.float32), stats["y_std"].astype(np.float32)
        label_names = [str(x) for x in stats["label_names"].tolist()] if "label_names" in stats.files else LABELS
    require(label_names == LABELS, f"Label order {label_names}")
    require(np.array_equal(np.asarray(checkpoint["y_mean"], np.float32), y_mean)
            and np.array_equal(np.asarray(checkpoint["y_std"], np.float32), y_std), "Checkpoint scaler differs from label stats")

    with np.load(args.input, allow_pickle=False) as z:
        X = np.asarray(z["X"], dtype=np.float32)
        event_names = z["event_names"].astype(str)
        gps = np.asarray(z["gps"], dtype=float)
        stored_x_sha = str(z["x_sha256"])
        meta = json.loads(str(z["metadata_json"]))
    require(x_sha256(X) == stored_x_sha == meta["x_sha256"], "X sha256 differs from the one stored in the input npz")
    require(X.ndim == 3 and X.shape[1:] == (3, 16384) and np.isfinite(X).all(), f"Unexpected X {X.shape}")

    # Input normalization: X is already z-scored (as the fp16_memmap mode of predict_cnn_hdf5.py).
    norm = model_config.get("input_normalization", {"enabled": False})
    ev_norm = meta["input_normalization"]
    require(not norm.get("enabled", False), "Model input_normalization.enabled must be false: events X is already normalized")
    require(norm.get("already_normalized", False) and ev_norm["applied"]
            and norm.get("mode") == ev_norm["mode"] and float(norm.get("eps")) == float(ev_norm["eps"]),
            f"Input normalization contract differs: model {norm}, events {ev_norm}")
    print("input_normalization:", norm)

    out, emb = predict(model, X, device)
    pred, sigma, cov = split_output(model, out, len(LABELS))
    require(np.isfinite(pred).all() and np.isfinite(emb).all(), "Nonfinite event output")
    print("events:", list(event_names))
    print("pred (physical):\n", pred * y_std + y_mean)

    payload = dict(pred=pred, emb=emb, y_mean=y_mean, y_std=y_std, label_names=np.array(LABELS),
                   event_names=event_names, gps=gps, checkpoint_file=np.array(str(checkpoint_path)),
                   checkpoint_sha256=np.array(file_sha256(checkpoint_path)),
                   checkpoint_tag=np.array(str(checkpoint.get("checkpoint_tag"))),
                   checkpoint_epoch=np.array(int(checkpoint.get("epoch", -1))),
                   model_config_json=np.array(json.dumps(model_config, default=str)),
                   input_file=np.array(str(args.input.resolve())), input_sha256=np.array(file_sha256(args.input)),
                   x_sha256=np.array(stored_x_sha), label_stats_path=np.array(str(stats_path)),
                   created_utc=np.array(datetime.now(timezone.utc).isoformat()),
                   torch_version=np.array(torch.__version__), device=np.array(str(device)))
    if sigma is not None:
        payload["sigma"] = sigma
    if cov is not None:
        payload["cov"] = cov

    val_check = dict(n=0, passed=False)
    if args.val_check > 0:
        n = int(args.val_check)
        tag = str(checkpoint["checkpoint_tag"])
        ref_path = data_root / "results/M13_3" / tag / "preds_val_G3T" / "val_predictions_embeddings.npz"
        with np.load(ref_path, allow_pickle=True) as r:
            require(Path(str(r["checkpoint_file"])).name == checkpoint_path.name, f"{ref_path}: other checkpoint")
            require(r["label_names"].tolist() == LABELS, f"{ref_path}: label order")
            require(np.array_equal(r["y_mean"], y_mean) and np.array_equal(r["y_std"], y_std), f"{ref_path}: scaler")
            ref_norm = r["input_normalization"].item()
            memmap_dir = Path(str(r["memmap_dir"]))
            ref_idx = np.asarray(r["idx_val"][:n], dtype=np.int64)
            ref_pred = np.asarray(r["pred_val"][:n], dtype=np.float64)
            ref_cov = np.asarray(r["cov_val"][:n], dtype=np.float64) if cov is not None else None
        require(not ref_norm.get("enabled", False), "preds_val_G3T used on-the-fly input normalization")
        row_ids = np.load(memmap_dir / "row_ids_val.npy", mmap_mode="r")
        require(np.array_equal(np.asarray(row_ids[:n], dtype=np.int64), ref_idx), "val memmap rows != preds_val_G3T idx")
        X_val = np.asarray(np.load(memmap_dir / "X_val.npy", mmap_mode="r")[:n], dtype=np.float32)  # fp16 -> fp32 exact
        out_v, _ = predict(model, X_val, device)
        mu_v, _, cov_v = split_output(model, out_v, len(LABELS))
        dmu = float(np.abs(mu_v - ref_pred).max())
        dcov = float(np.abs(cov_v - ref_cov).max()) if cov_v is not None else 0.0
        print(f"val check (n = {n}): max |dmu| = {dmu:.3e}, max |dcov| = {dcov:.3e} (tol {VAL_CHECK_TOL:g})")
        require(dmu <= VAL_CHECK_TOL and dcov <= VAL_CHECK_TOL, "VAL CHECK FAILED: preds_val_G3T not reproduced")
        val_check = dict(n=n, passed=True, max_abs_dmu=dmu, max_abs_dcov=dcov, tol=VAL_CHECK_TOL,
                         reference=str(ref_path), memmap_dir=str(memmap_dir))
    payload["val_check_json"] = np.array(json.dumps(val_check))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        np.savez(f, **payload)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
