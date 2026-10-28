#!/usr/bin/env python3
"""M13.5: select the train/val rows to regenerate with a new noise realisation.

Train: 10000 ids drawn with default_rng(135002) from idx_train of
results/M13_3/M13_3b_G3_s123/train40k_predictions.npz; that id set must be identical for
M13_3b_G3_s124 and M13_4a_G3_s123. Val: 2000 ids drawn with default_rng(135003) from idx_val
of results/M13_3/M13_3b_G3_s123/val_predictions_embeddings.npz. All ids are original HDF5 row ids
(G1 manifest positions), stored ascending.

Writes (exclusive) to <data_root>/processed/M13_G3_band1024_w16_500k/m13_5_renoise/:
    rows_train10k.npy, rows_val2k.npy, rows_all12k.npy, rows_control16.npy, selection.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import dataset_processed_dir, resolve_data_root

DATASET_ID = "M13_G3_band1024_w16_500k"
REFERENCE_TAG = "M13_3b_G3_s123"
TAGS = ("M13_3b_G3_s123", "M13_3b_G3_s124", "M13_4a_G3_s123")
N_TRAIN, SEED_TRAIN = 10000, 135002
N_VAL, SEED_VAL = 2000, 135003
N_CONTROL = 16


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_idx(path, split):
    with np.load(path, allow_pickle=False) as z:
        idx = z[f"idx_{split}"].astype(np.int64)
    if not (idx.ndim == 1 and np.all(np.diff(idx) > 0)):
        raise SystemExit(f"idx_{split} is not strictly ascending in {path}")
    return idx


def save_exclusive(path, array):
    with open(path, "xb") as fh:
        np.save(fh, array)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = resolve_data_root(cli_data_root=args.data_root)
    res = root / "results" / "M13_3"
    out_dir = dataset_processed_dir(root, DATASET_ID) / "m13_5_renoise"

    train_npz = {tag: res / tag / "train40k_predictions.npz" for tag in TAGS}
    val_npz = res / REFERENCE_TAG / "val_predictions_embeddings.npz"
    train_idx = {tag: load_idx(p, "train") for tag, p in train_npz.items()}
    for tag in TAGS:
        if not np.array_equal(train_idx[tag], train_idx[REFERENCE_TAG]):
            raise SystemExit(f"train40k idx of {tag} differs from {REFERENCE_TAG}")
    pool_train = train_idx[REFERENCE_TAG]
    pool_val = load_idx(val_npz, "val")
    if np.intersect1d(pool_train, pool_val).size:
        raise SystemExit("train40k and val ids overlap")

    rows_train = np.sort(np.random.default_rng(SEED_TRAIN).choice(pool_train, N_TRAIN, replace=False))
    rows_val = np.sort(np.random.default_rng(SEED_VAL).choice(pool_val, N_VAL, replace=False))
    rows_all = np.sort(np.concatenate([rows_train, rows_val]))
    rows_control = rows_train[:N_CONTROL]
    outputs = {"rows_train10k.npy": rows_train, "rows_val2k.npy": rows_val,
               "rows_all12k.npy": rows_all, "rows_control16.npy": rows_control}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in list(outputs) + ["selection.json"]:
        if (out_dir / name).exists():
            raise SystemExit(f"Output exists, refusing to overwrite: {out_dir / name}")
    for name, array in outputs.items():
        save_exclusive(out_dir / name, array.astype(np.int64))
    info = {
        "dataset_id": DATASET_ID,
        "row_id_meaning": "original HDF5 row id (G1 manifest position = G3 trainio row)",
        "train": {"source": str(train_npz[REFERENCE_TAG]), "pool_size": int(pool_train.size),
                  "n": N_TRAIN, "rng": f"np.random.default_rng({SEED_TRAIN}).choice(replace=False), sorted",
                  "pool_identical_in": list(TAGS)},
        "val": {"source": str(val_npz), "pool_size": int(pool_val.size), "n": N_VAL,
                "rng": f"np.random.default_rng({SEED_VAL}).choice(replace=False), sorted"},
        "control": f"first {N_CONTROL} of rows_train10k.npy",
        "source_sha256": {str(p): sha256(p) for p in list(train_npz.values()) + [val_npz]},
        "output_sha256": {name: sha256(out_dir / name) for name in outputs},
        "script_sha256": sha256(__file__),
    }
    with open(out_dir / "selection.json", "x") as fh:
        json.dump(info, fh, indent=2, sort_keys=True)
    print(f"Wrote {', '.join(outputs)} and selection.json to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
