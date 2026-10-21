"""One X-only patch-MLP reference; user-operated train / validation inference.

No physics imports, M11 features, re-normalization, augmentation or new scaler.
The historical trainer/loader remain unchanged. See the accompanying protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import resolve_data_root, resolve_processed_artifact
from src.models.hdf5_batch_dataset import HDF5BatchIterableDataset
from src.models.train import train_model
from src.models.evaluate import predict_on_loader

LABELS = ["chirp_mass", "total_mass", "chi_eff"]
DATASET = "M11_final_G1_500k"
VERSION = "M12.5.2-patch-mlp-v1"
BASE_CONFIG = ROOT / "configs/experiments/train_m11_final_G1_500k.json"
BASE_PRED = "m11_final_G1_500k_val_predictions_embeddings.npz"
PINNED = {
    "split_file": "1abf607c44cc4c55ca196de046874137e274c88cf221143cd936d645e97b97d2",
    "label_stats_file": "d0891bf1c130186fb4e7af442e74952a03e4d7d5e0cc240f78d8772e4421855f",
    "baseline": "586b95f2cb2cfaf150fbdd05760f50f47a95a555451fcb46d89492b7d00ed88b",
}
MODEL = dict(patch_size=128, width=128, depth=4, token_hidden=128, channel_hidden=256)
TRAIN = dict(seed=123, batch_size=256, max_epochs=100, patience=25,
             learning_rate=3e-4, weight_decay=3e-4)
NORMALIZATION = dict(enabled=False, already_normalized=True,
                     mode="per_sample_per_detector_zscore", eps=1e-6)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def code_identity():
    names = ["scripts/m12/run_m12_5_2_reference.py", "src/models/train.py",
             "src/models/hdf5_batch_dataset.py", "src/models/evaluate.py",
             "src/paths.py", "configs/experiments/train_m11_final_G1_500k.json"]
    return {name: digest(ROOT / name) for name in names}


class MixerBlock(nn.Module):
    def __init__(self, tokens, width, token_hidden, channel_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.token = nn.Sequential(nn.Linear(tokens, token_hidden), nn.GELU(),
                                   nn.Linear(token_hidden, tokens))
        self.norm2 = nn.LayerNorm(width)
        self.channel = nn.Sequential(nn.Linear(width, channel_hidden), nn.GELU(),
                                     nn.Linear(channel_hidden, width))

    def forward(self, x):
        x = x + self.token(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel(self.norm2(x))


class PatchMLP(nn.Module):
    """All 49152 X values enter a learned projection; patching is a permutation.

    Learned compression is part of the estimator, not a new input observable.
    Fixed token positions retain temporal order without auxiliary GPS information.
    """
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3 * MODEL["patch_size"], MODEL["width"])
        self.blocks = nn.Sequential(*[
            MixerBlock(128, MODEL["width"], MODEL["token_hidden"], MODEL["channel_hidden"])
            for _ in range(MODEL["depth"])])
        self.norm = nn.LayerNorm(MODEL["width"])
        self.head = nn.Linear(MODEL["width"], 3)

    def forward(self, x):
        require(tuple(x.shape[1:]) == (3, 16384), "X contract changed")
        x = x.reshape(-1, 3, 128, 128).permute(0, 2, 1, 3).reshape(-1, 128, 384)
        pred = self.head(self.norm(self.blocks(self.project(x))).mean(dim=1))
        require(bool(torch.isfinite(pred).all()), "Nonfinite prediction; abort")
        return pred


class CheckedLoader:
    """Finite checks at consumption; forwards epoch scheduling to shared loader."""
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __iter__(self):
        for x, y in self.loader:
            require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                    "Nonfinite X/labels; no rows may be silently excluded")
            yield x, y


def load_contract(root):
    cfg = json.loads(BASE_CONFIG.read_text())
    require(cfg["dataset"]["dataset_id"] == DATASET and
            cfg["input_normalization"] == NORMALIZATION, "Frozen config changed")
    paths = {k: resolve_processed_artifact(data_root=root, dataset_id=DATASET,
             file_name=cfg["dataset"][k]) for k in
             ("dataset_file", "split_file", "label_stats_file")}
    for k in ("split_file", "label_stats_file"):
        require(digest(paths[k]) == PINNED[k], f"Frozen {k} identity mismatch")
    with np.load(paths["split_file"]) as s:
        splits = {k: np.sort(s[k + "_idx"]) for k in ("train", "val", "cal", "test")}
    for k, count in dict(train=400000, val=40000, cal=30000, test=30000).items():
        ids = splits[k]
        require(ids.dtype.kind in "iu" and ids.shape == (count,) and
                len(np.unique(ids)) == count, f"Invalid {k} IDs")
    require(np.array_equal(np.sort(np.concatenate(list(splits.values()))),
                           np.arange(500000)), "Overlapping or incomplete splits")
    with np.load(paths["label_stats_file"]) as s:
        require(s["label_names"].tolist() == LABELS and
                np.array_equal(np.sort(s["train_idx"]), splits["train"]),
                "Scaler not fitted on frozen train membership")
        mean, std = s["y_mean"].astype(np.float32), s["y_std"].astype(np.float32)
    require(mean.shape == std.shape == (3,) and np.isfinite(mean).all() and
            np.isfinite(std).all() and (std > 0).all(), "Invalid label scaler")
    with h5py.File(paths["dataset_file"], "r") as h:
        require(h["X"].shape == (500000, 3, 16384) and
                h["X"].dtype == np.dtype("float32") and h["y"].shape == (500000, 3),
                "G1 tensor contract mismatch")
        attrs = dict(config_sha256="cf9562725bb88ee68b325beb3e44b41bfb5327a6d1a946cd1d9577bab6bd9467",
                     manifest_sha256="544579ef7270675b80caee34cfe440d9b767b02ada3fd5f4de2e8e0f066b864e",
                     labels=",".join(LABELS), detector_order="H1,L1,V1",
                     input_normalization=NORMALIZATION["mode"], input_zscore_epsilon=1e-6,
                     storage_layout_variant="trainio")
        require(all(h.attrs.get(k) == v for k, v in attrs.items()), "G1 header mismatch")
        # Only compact membership metadata, never cal/test X or labels.
        membership = h["split"].asstr()[:]
        sources = h["source_id"].asstr()[:]
        require(len(np.unique(sources)) == 500000, "Source leakage / duplicate source IDs")
        for k, ids in splits.items():
            require(np.all(membership[ids] == k), f"HDF5/{k} membership mismatch")
    st = paths["dataset_file"].stat()
    identity = dict(version=VERSION, pinned=PINNED, header=attrs,
                    dataset_size=st.st_size, labels=LABELS, model=MODEL,
                    training=TRAIN, normalization=NORMALIZATION, code=code_identity())
    return paths, splits, mean, std, identity


def loader(path, ids, mean, std, train, workers):
    ds = HDF5BatchIterableDataset(path, ids, mean, std, NORMALIZATION,
        batch_size=TRAIN["batch_size"], drop_last=train, seed=TRAIN["seed"],
        shuffle_batches=train, shuffle_within_batch=False, max_slice_overread=4.0)
    # Inference uses one iterator: output IDs are provably sorted, independent
    # of multi-worker IterableDataset scheduling in different PyTorch versions.
    options = dict(batch_size=None, num_workers=workers, pin_memory=True)
    if workers:
        options.update(prefetch_factor=2, persistent_workers=False)
    return CheckedLoader(DataLoader(ds, **options))


def new_output(root, tag):
    require(tag and all(c.isalnum() or c in "_-" for c in tag), "Use a simple unique tag")
    out = (root / "results" / DATASET / "M12_5_2" / tag).resolve()
    require(ROOT not in out.parents, "Artifacts must be outside Git")
    out.mkdir(parents=True, exist_ok=False)
    return out


def finite_gradient(grad):
    require(bool(torch.isfinite(grad).all()), "Nonfinite gradient; abort")
    return grad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "infer"])
    p.add_argument("--data-root")
    p.add_argument("--tag", required=True, help="New exclusive external result directory")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=2, choices=[0, 1, 2, 4])
    args = p.parse_args()
    require((args.mode == "infer") == (args.checkpoint is not None),
            "--checkpoint is required only for inference")
    cfg = json.loads(BASE_CONFIG.read_text())
    root = resolve_data_root(cli_data_root=args.data_root, config_data_root=cfg.get("data_root"))
    paths, splits, mean, std, identity = load_contract(root)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable")
    random.seed(TRAIN["seed"])
    np.random.seed(TRAIN["seed"])
    torch.manual_seed(TRAIN["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = PatchMLP().to(device)
    out = new_output(root, args.tag)
    write_json(out / "started.json", dict(complete=False, identity=identity,
        paths={k: str(v) for k, v in paths.items()}, torch=torch.__version__,
        device=str(device), workers=args.workers, parameters=sum(p.numel() for p in model.parameters())))
    if args.mode == "train":
        for param in model.parameters():
            param.register_hook(finite_gradient)
        train_loader = loader(paths["dataset_file"], splits["train"], mean, std, True, args.workers)
        val_loader = loader(paths["dataset_file"], splits["val"], mean, std, False, args.workers)
        checkpoint, history = train_model(model, train_loader, val_loader, device,
            mean, std, model_config=dict(class_name="PatchMLP", kwargs=MODEL), **TRAIN)
        require("model_state_dict" in checkpoint and
                all(np.isfinite(history[k]).all() for k in ("train_loss", "val_loss")),
                "Incomplete/nonfinite training")
        checkpoint["reference_identity"] = identity
        with (out / "checkpoint.pt").open("xb") as f:
            torch.save(checkpoint, f)
        write_json(out / "history.json", history)
        write_json(out / "complete.json", dict(complete=True, identity=identity,
            checkpoint_sha256=digest(out / "checkpoint.pt"), epoch=checkpoint["epoch"]))
        return
    receipt = json.loads((args.checkpoint.parent / "complete.json").read_text())
    require(receipt["complete"] and digest(args.checkpoint) == receipt["checkpoint_sha256"],
            "Incomplete or changed reference checkpoint")
    # Trusted local checkpoint, old torch-compatible API (no weights_only).
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    require(checkpoint["reference_identity"] == identity == receipt["identity"],
            "Checkpoint/code/data contract mismatch; use the exact training code")
    require(np.array_equal(checkpoint["y_mean"].numpy(), mean) and
            np.array_equal(checkpoint["y_std"].numpy(), std), "Checkpoint scaler mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
            "Nonfinite checkpoint")
    baseline_path = root / "results" / DATASET / BASE_PRED
    require(digest(baseline_path) == PINNED["baseline"], "Frozen M11 predictions changed")
    ids = splits["val"]
    pred, truth = predict_on_loader(model, loader(paths["dataset_file"], ids, mean, std, False, 0), device)
    require(pred.shape == truth.shape == (40000, 3) and np.isfinite(pred).all(), "Incomplete inference")
    with np.load(baseline_path) as b:
        order = np.argsort(b["idx_val"])
        require(np.array_equal(b["idx_val"][order], ids) and
                b["label_names"].tolist() == LABELS and np.array_equal(b["y_mean"], mean) and
                np.array_equal(b["y_std"], std) and
                np.allclose(b["y_val"][order], truth, rtol=0, atol=2e-6),
                "M11/reference row, truth, or scaling mismatch")
        baseline = b["pred_val"][order]
    # Diagnostic labels/metadata never enter model.forward or checkpoint selection.
    with h5py.File(paths["dataset_file"], "r") as h:
        y_train = np.asarray(h["y"][splits["train"]], dtype=np.float64)
        require(np.isfinite(y_train).all(), "Nonfinite train diagnostic labels")
        z_train = (y_train - mean) / (std + np.float32(1e-8))
        center = z_train.mean(axis=0)
        precision = np.linalg.pinv(np.cov(z_train, rowvar=False))
        distance = np.einsum("ni,ij,nj->n", z_train-center, precision, z_train-center)
        payload = dict(idx_val=ids, pred_val=pred, y_val=truth, m11_pred_val=baseline,
            label_names=np.array(LABELS), y_mean=mean, y_std=std,
            source_id=h["source_id"].asstr()[ids], environment=h["file_group_id"][ids].astype(str),
            snr=h["target_network_snr"][ids], train_center=center, train_precision=precision,
            train_distance_cuts=np.quantile(distance, [.8, .95, .99]),
            train_label_cuts=np.quantile(y_train, [.1, .9], axis=0))
    require(np.isfinite(baseline).all(), "Nonfinite M11 predictions")
    with (out / "val_predictions.npz").open("xb") as f:
        np.savez_compressed(f, **payload)
    write_json(out / "complete.json", dict(complete=True, identity=identity,
        checkpoint_sha256=digest(args.checkpoint), baseline_sha256=digest(baseline_path),
        predictions_sha256=digest(out / "val_predictions.npz"),
        epoch=checkpoint["epoch"], split="val", inference_workers=0))


if __name__ == "__main__":
    main()
