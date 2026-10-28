#!/usr/bin/env python3
"""M13.3b: copy the frozen G1 split and train-only label-stats npz files into the G3 folder.

G3 keeps the G1 manifest, source IDs, labels and train/val/cal/test membership row by row, so the
G1 artifacts apply unchanged; they are copied, never recomputed. Before copying, G3 trainio `y`
must equal G1 trainio `y` exactly, `file_group_id` must match, and the G3 `split` column must give
the same membership as the G1 split npz. Any difference aborts before anything is written.
Destination files keep the G1 names and are opened in exclusive mode; g3_split_artifacts.json
records source and destination SHA256.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import dataset_processed_dir, resolve_data_root  # noqa: E402

G1_DATASET_ID = "M11_final_G1_500k"
G1_TRAINIO = "m11_final_G1_500k_trainio.h5"
SPLITS = ("train", "val", "cal", "test")
LABELS = ["chirp_mass", "total_mass", "chi_eff"]
REPORT = "g3_split_artifacts.json"


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def require(cond, msg):
    if not cond:
        raise SystemExit(f"ABORT: {msg}")


def read_col(ds):
    # Same rule as merge_m13_G3_shards.read_col (not imported: that module pulls in the generator).
    return ds.asstr()[:] if h5py.check_string_dtype(ds.dtype) else ds[:]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True, help="G3 training config (M13.3b)")
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    dcfg = cfg["dataset"]
    require(dcfg["dataset_id"] != G1_DATASET_ID, "config dataset_id is G1; expected the G3 dataset")
    data_root = resolve_data_root(cli_data_root=args.data_root, config_data_root=cfg.get("data_root"))
    g1_dir = dataset_processed_dir(data_root, G1_DATASET_ID)
    g3_dir = dataset_processed_dir(data_root, dcfg["dataset_id"])
    g1_h5, g3_h5 = g1_dir / G1_TRAINIO, g3_dir / dcfg["dataset_file"]
    names = {"split_file": dcfg["split_file"], "label_stats_file": dcfg["label_stats_file"]}
    src = {k: g1_dir / v for k, v in names.items()}
    dst = {k: g3_dir / v for k, v in names.items()}
    report_path = g3_dir / REPORT
    for p in (g1_h5, g3_h5, *src.values()):
        require(p.exists(), f"missing {p}")
    for p in (*dst.values(), report_path):
        require(not p.exists(), f"destination exists, refusing to overwrite: {p}")

    with h5py.File(g3_h5, "r") as f:
        require(bool(f.attrs.get("merge_complete", False)), f"{g3_h5} lacks merge_complete")
        y3 = f["y"][:]
        split3 = f["split"].asstr()[:]
        g3_group = f["file_group_id"]
        group3_dtype, group3_str = g3_group.dtype, h5py.check_string_dtype(g3_group.dtype) is not None
        group3 = read_col(g3_group)
    with h5py.File(g1_h5, "r") as f:
        y1 = f["y"][:]
        g1_group = f["file_group_id"]
        group1_dtype, group1_str = g1_group.dtype, h5py.check_string_dtype(g1_group.dtype) is not None
        group1 = read_col(g1_group)
    require(y3.dtype == y1.dtype and y3.shape == y1.shape and np.array_equal(y3, y1),
            "G3 y differs from G1 trainio y")
    require(group3_str == group1_str,
            f"file_group_id text/non-text mismatch (G3 {group3_dtype} vs G1 {group1_dtype})")
    require(np.array_equal(group3, group1), "G3 file_group_id differs from G1 trainio")
    require(np.isin(split3, SPLITS).all(), "unknown labels in G3 split column")

    with np.load(src["split_file"]) as sp:
        idx = {s: np.sort(sp[f"{s}_idx"].astype(np.int64)) for s in SPLITS}
    with np.load(src["label_stats_file"]) as st:
        stats_train = np.sort(st["train_idx"].astype(np.int64))
        stats_labels = [str(x) for x in st["label_names"].tolist()]
    counts = {}
    for s in SPLITS:
        rows = np.flatnonzero(split3 == s).astype(np.int64)
        require(np.array_equal(rows, idx[s]), f"G3 {s} membership differs from the G1 split npz")
        counts[s] = int(rows.size)
    require(sum(counts.values()) == y3.shape[0], "G1 split npz does not cover every G3 row")
    require(np.array_equal(stats_train, idx["train"]), "label stats train_idx differs from the split npz")
    require(stats_labels == LABELS, f"unexpected label order in label stats: {stats_labels}")

    files = {}
    for k in names:
        h_src = sha256_file(src[k])
        with open(src[k], "rb") as fi, open(dst[k], "xb") as fo:
            for part in iter(lambda: fi.read(1 << 24), b""):
                fo.write(part)
        h_dst = sha256_file(dst[k])
        require(h_dst == h_src, f"SHA256 mismatch after copy: {dst[k]}")
        files[k] = {"source": str(src[k]), "source_sha256": h_src,
                    "destination": str(dst[k]), "destination_sha256": h_dst}
        print(f"copied {src[k].name} -> {g3_dir} ({h_src})", flush=True)

    report = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(args.config),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "g1_dataset_id": G1_DATASET_ID,
        "g3_dataset_id": dcfg["dataset_id"],
        "g1_trainio": str(g1_h5),
        "g3_trainio": str(g3_h5),
        "checks": {"y_identical": True, "file_group_id_identical": True,
                   "split_membership_identical": True, "label_stats_train_idx_matches": True,
                   "counts": counts},
        "files": files,
    }
    with open(report_path, "x") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
