"""M13.3: export train/val rows of the frozen G1 trainio HDF5 to float16 .npy memmaps.

Output (never overwritten): $CBC_PE_DATA_ROOT/processed/<dataset_id>/fp16_memmap/
    X_<split>.npy (float16, (N, C, T)), y_<split>.npy (float32, unstandardized),
    row_ids_<split>.npy (int64, ascending HDF5 row IDs), manifest.json
Only train/val are exportable; cal/test are never written. The HDF5 is scanned in contiguous
1024-row blocks, so rows of other splits pass transiently through memory and are discarded
before any statistic is computed. X is stored as the HDF5 holds it (already z-scored per sample
and detector); configs that normalize on the fly are rejected.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import dataset_processed_dir, resolve_data_root, resolve_processed_artifact  # noqa: E402

BLOCK_ROWS = 1024  # multiple of the 64-row HDF5 chunk
FP16_ABS_LIMIT = 6e4  # float16 max is 65504
PROGRESS_EVERY = 20000
ZSCORE_SAMPLE_ROWS = 1000
ZSCORE_SEED = 133300
H5_ATTRS = ("input_normalization", "labels", "label_names", "detector_order")


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def jsonable(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"])
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def main():
    t_start = time.time()
    args = parse_args()
    splits_req = list(dict.fromkeys(args.splits))
    cfg = json.loads(args.config.read_text())
    dcfg = cfg["dataset"]
    dataset_id = dcfg["dataset_id"]
    if cfg.get("input_normalization", {}).get("enabled", False):
        raise SystemExit("Config normalizes inputs on the fly; the export would not match the HDF5 loader")

    data_root = resolve_data_root(cli_data_root=args.data_root, config_data_root=cfg.get("data_root"))
    h5_path = resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id,
                                         file_name=dcfg["dataset_file"])
    split_path = resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id,
                                            file_name=dcfg["split_file"])
    for p in (h5_path, split_path):
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir = dataset_processed_dir(data_root, dataset_id) / "fp16_memmap"
    tmp_dir = out_dir.with_name(f"fp16_memmap.tmp_{os.getpid()}")
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists, refusing to overwrite: {out_dir}")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    print(f"HDF5:   {h5_path}\nsplits: {split_path}\ntmp:    {tmp_dir}\nfinal:  {out_dir}", flush=True)

    splits = np.load(split_path)
    ids = {}
    for s in splits_req:
        r = np.sort(splits[f"{s}_idx"].astype(np.int64))
        if np.unique(r).size != r.size:
            raise SystemExit(f"Duplicate row IDs in split {s}")
        ids[s] = r

    h5_stat = h5_path.stat()
    with h5py.File(h5_path, "r") as f:
        Xd, yd = f["X"], f["y"]
        n_rows, C, T = Xd.shape
        D = yd.shape[1]
        attrs = {k: (jsonable(f.attrs[k]) if k in f.attrs else None) for k in H5_ATTRS}
        for s, r in ids.items():
            if r[0] < 0 or r[-1] >= n_rows:
                raise SystemExit(f"Split {s} has row IDs outside [0, {n_rows})")

        mm = {}
        for s, r in ids.items():
            np.save(tmp_dir / f"row_ids_{s}.npy", r)
            mm[s] = (
                np.lib.format.open_memmap(tmp_dir / f"X_{s}.npy", mode="w+", dtype=np.float16,
                                          shape=(r.size, C, T)),
                np.lib.format.open_memmap(tmp_dir / f"y_{s}.npy", mode="w+", dtype=np.float32,
                                          shape=(r.size, D)),
            )
        acc = {s: {"max_abs": np.zeros(C), "max_err": np.zeros(C),
                   "sum_err2": np.zeros(C), "sum_x2": np.zeros(C)} for s in ids}
        written = {s: 0 for s in ids}
        next_report = PROGRESS_EVERY

        for start in range(0, n_rows, BLOCK_ROWS):
            stop = min(start + BLOCK_ROWS, n_rows)
            sel = {}
            for s, r in ids.items():
                lo, hi = np.searchsorted(r, [start, stop])
                if hi > lo:
                    sel[s] = (lo, hi)
            if sel:
                X_block = np.asarray(Xd[start:stop], dtype=np.float32)
                y_block = np.asarray(yd[start:stop], dtype=np.float32)
                for s, (lo, hi) in sel.items():
                    local = ids[s][lo:hi] - start
                    x = X_block[local]
                    y = y_block[local]
                    if not (np.isfinite(x).all() and np.isfinite(y).all()):
                        raise SystemExit(f"Non-finite values in split {s}, rows {ids[s][lo]}..{ids[s][hi - 1]}")
                    max_abs = np.abs(x).max(axis=(0, 2))
                    if max_abs.max() > FP16_ABS_LIMIT:
                        raise SystemExit(f"|x| = {max_abs.max():.4g} > {FP16_ABS_LIMIT:g} in split {s}")
                    x16 = x.astype(np.float16)
                    err = np.abs(x16.astype(np.float32) - x)
                    a = acc[s]
                    a["max_abs"] = np.maximum(a["max_abs"], max_abs)
                    a["max_err"] = np.maximum(a["max_err"], err.max(axis=(0, 2)))
                    for c in range(C):
                        a["sum_err2"][c] += np.square(err[:, c], dtype=np.float64).sum()
                        a["sum_x2"][c] += np.square(x[:, c], dtype=np.float64).sum()
                    mm[s][0][lo:hi] = x16
                    mm[s][1][lo:hi] = y
                    written[s] += hi - lo
                del X_block, y_block
            if stop >= next_report or stop == n_rows:
                print(f"  scanned {stop}/{n_rows} rows | written "
                      + " ".join(f"{s}={written[s]}" for s in ids)
                      + f" | {time.time() - t_start:.0f} s", flush=True)
                next_report = (stop // PROGRESS_EVERY + 1) * PROGRESS_EVERY

        for s in ids:
            if written[s] != ids[s].size:
                raise SystemExit(f"Split {s}: wrote {written[s]} of {ids[s].size} rows")
            mm[s][0].flush()
            mm[s][1].flush()
        del mm

    # z-score check on the exported float16 values (sample of rows per split).
    rng = np.random.default_rng(ZSCORE_SEED)
    per_split = {}
    for s, r in ids.items():
        X = np.load(tmp_dir / f"X_{s}.npy", mmap_mode="r")
        pos = np.sort(rng.choice(r.size, size=min(ZSCORE_SAMPLE_ROWS, r.size), replace=False))
        xs = np.asarray(X[pos], dtype=np.float32)
        m, sd = xs.mean(axis=2), xs.std(axis=2)
        del X, xs
        a = acc[s]
        per_split[s] = {
            "n_rows": int(r.size),
            "shape_X": [int(r.size), int(C), int(T)],
            "row_id_min": int(r[0]),
            "row_id_max": int(r[-1]),
            "row_ids_sha256": sha256_file(tmp_dir / f"row_ids_{s}.npy"),
            "per_detector": [
                {
                    "detector_index": c,
                    "max_abs_x": float(a["max_abs"][c]),
                    "max_abs_err_fp16_vs_fp32": float(a["max_err"][c]),
                    "rel_sq_err_sum_err2_over_sum_x2": float(a["sum_err2"][c] / a["sum_x2"][c]),
                    "zscore_sample": {
                        "n_rows": int(pos.size),
                        "seed": ZSCORE_SEED,
                        "row_mean_mean": float(m[:, c].mean()),
                        "row_mean_max_abs": float(np.abs(m[:, c]).max()),
                        "row_std_mean": float(sd[:, c].mean()),
                        "row_std_min": float(sd[:, c].min()),
                        "row_std_max": float(sd[:, c].max()),
                    },
                }
                for c in range(C)
            ],
        }

    manifest = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(args.config),
        "dataset_id": dataset_id,
        "source_hdf5": {"path": str(h5_path), "size_bytes": int(h5_stat.st_size),
                        "mtime": float(h5_stat.st_mtime),
                        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(h5_stat.st_mtime)),
                        "shape_X": [int(n_rows), int(C), int(T)], "attrs": attrs},
        "split_file": {"path": str(split_path), "sha256": sha256_file(split_path)},
        "splits_exported": splits_req,
        "x_dtype": "float16",
        "y_dtype": "float32 (unstandardized)",
        "block_rows": BLOCK_ROWS,
        "fp16_abs_limit": FP16_ABS_LIMIT,
        "splits": per_split,
        "total_time_s": time.time() - t_start,
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if out_dir.exists():
        raise SystemExit(f"Output directory appeared during export; result left in {tmp_dir}")
    os.rename(tmp_dir, out_dir)
    print(f"done in {manifest['total_time_s']:.0f} s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
