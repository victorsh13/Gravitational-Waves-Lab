#!/usr/bin/env python3
"""M13.6 step 1 (merge): join generate_m13_6_clean.py shards into float16 .npy memmaps.

Input:  <data_root>/processed/<dataset_id>/m13_6_aug/{signal,noise_bank}/*_shardIIofNN.h5
        <data_root>/processed/<dataset_id>/fp16_memmap/{row_ids_train,y_train}.npy (G3 memmap)
Output (never overwritten): <data_root>/processed/<dataset_id>/m13_6_aug/memmap/
    X_clean_train.npy  float16 (N, 3, 16384), L(s) before z-score, rows in the order of the G3
                       memmap row_ids_train (checked: same row set; y checked against y_train)
    psd_key_train.npy  unicode (N,), psd_key of each row
    bank_X.npy         float16 (n_keys * K, 3, 16384), L(n) before z-score, sorted by (psd_key, k)
    bank_psd_key.npy   unicode (n_keys * K,)
    manifest.json      statistics (max|x|, std per detector, keys, rows per key) and hashes
K is the bank_k attribute of the noise_bank shards (must agree across shards); every train psd_key
must have exactly k = 0..K-1 in the bank.
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
from src.paths import dataset_processed_dir, resolve_data_root  # noqa: E402

BLOCK_ROWS = 1024
# Attributes that must agree across all shards (same generator, config, G3 inputs).
SHARED_ATTRS = ("m13_6_generator_sha256", "m13_config_file_sha256", "g3_generator_sha256",
                "g1_generator_sha256", "g3_trainio_path", "g3_trainio_size", "g3_trainio_mtime_ns",
                "config_sha256", "manifest_sha256")


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def jsonable(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


class Stats:
    """Per-detector max|x| and global mean/std, accumulated in float64."""

    def __init__(self, C):
        self.max_abs = np.zeros(C)
        self.s1 = np.zeros(C)
        self.s2 = np.zeros(C)
        self.n = 0

    def add(self, x16):
        x = x16.astype(np.float32)
        if not np.isfinite(x).all():
            raise SystemExit("Non-finite values in a shard")
        self.max_abs = np.maximum(self.max_abs, np.abs(x).max(axis=(0, 2)))
        for c in range(x.shape[1]):
            self.s1[c] += x[:, c].sum(dtype=np.float64)
            self.s2[c] += np.square(x[:, c], dtype=np.float64).sum()
        self.n += x.shape[0] * x.shape[2]

    def report(self):
        mean = self.s1 / self.n
        return [{"detector_index": c, "max_abs_x": float(self.max_abs[c]), "mean": float(mean[c]),
                 "std": float(np.sqrt(max(self.s2[c] / self.n - mean[c] ** 2, 0.0)))}
                for c in range(len(mean))]


def shard_paths(directory, prefix, num_shards):
    paths = [directory / f"{prefix}_shard{i:02d}of{num_shards:02d}.h5" for i in range(num_shards)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing shards: {missing}")
    extra = sorted(set(directory.glob(f"{prefix}_shard*.h5")) - set(paths))
    if extra:
        raise SystemExit(f"Unexpected shard files (other num_shards?): {[str(p) for p in extra]}")
    return paths


def check_shard(h, path, mode, reference):
    if h.attrs.get("m13_6_mode") != mode:
        raise SystemExit(f"{path}: m13_6_mode != {mode}")
    if not bool((h["status"][:] == 1).all()):
        raise SystemExit(f"{path}: incomplete shard (status != 1)")
    attrs = {k: jsonable(h.attrs[k]) for k in SHARED_ATTRS if k in h.attrs}
    if reference and attrs != reference:
        raise SystemExit(f"{path}: provenance attrs differ from the first shard")
    return attrs


def stat_entry(path):
    st = path.stat()
    return {"path": str(path.resolve()), "size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def main():
    t_start = time.time()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True, help="G3 generation config")
    p.add_argument("--num-shards", type=int, required=True, help="number of signal shards")
    p.add_argument("--bank-num-shards", type=int, default=1, help="number of noise_bank shards")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args()

    dataset_id = json.loads(args.config.read_text())["dataset_id"]
    data_root = resolve_data_root(cli_data_root=args.data_root)
    base = dataset_processed_dir(data_root, dataset_id)
    aug = base / "m13_6_aug"
    g3_mm = base / "fp16_memmap"
    out_dir = aug / "memmap"
    tmp_dir = out_dir.with_name(f"memmap.tmp_{os.getpid()}")
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists, refusing to overwrite: {out_dir}")
    signal_paths = shard_paths(aug / "signal", "signal", args.num_shards)
    bank_paths = shard_paths(aug / "noise_bank", "noise_bank", args.bank_num_shards)
    row_ids = np.load(g3_mm / "row_ids_train.npy")
    y_train = np.load(g3_mm / "y_train.npy", mmap_mode="r")
    if np.unique(row_ids).size != row_ids.size or not (np.diff(row_ids) > 0).all():
        raise SystemExit("G3 memmap row_ids_train is not strictly increasing")
    tmp_dir.mkdir(parents=False, exist_ok=False)
    print(f"signal shards: {len(signal_paths)} | bank shards: {len(bank_paths)}\n"
          f"G3 memmap: {g3_mm}\ntmp: {tmp_dir}\nfinal: {out_dir}", flush=True)

    # Signal: map each shard row to its position in row_ids_train.
    reference, seen, keys_train = {}, np.zeros(row_ids.size, dtype=bool), np.empty(row_ids.size, dtype=object)
    with h5py.File(signal_paths[0], "r") as h:
        _, C, T = h["X_clean"].shape
    X = np.lib.format.open_memmap(tmp_dir / "X_clean_train.npy", mode="w+", dtype=np.float16,
                                  shape=(row_ids.size, C, T))
    sig_stats = Stats(C)
    for path in signal_paths:
        with h5py.File(path, "r") as h:
            reference = check_shard(h, path, "signal", reference)
            ri = h["row_index"][:]
            pos = np.searchsorted(row_ids, ri)
            if (pos >= row_ids.size).any() or not np.array_equal(row_ids[np.minimum(pos, row_ids.size - 1)], ri):
                raise SystemExit(f"{path}: row_index not in G3 row_ids_train")
            if seen[pos].any() or not (np.diff(pos) > 0).all():
                raise SystemExit(f"{path}: duplicated or unordered rows")
            if not np.array_equal(h["y"][:], y_train[pos]):
                raise SystemExit(f"{path}: y differs from G3 memmap y_train")
            seen[pos] = True
            keys_train[pos] = h["psd_key"].asstr()[:]
            for lo in range(0, ri.size, BLOCK_ROWS):
                hi = min(lo + BLOCK_ROWS, ri.size)
                x = h["X_clean"][lo:hi]
                sig_stats.add(x)
                X[pos[lo:hi]] = x
        print(f"  merged {path.name}: {ri.size} rows | {time.time() - t_start:.0f} s", flush=True)
    if not seen.all():
        raise SystemExit(f"{int((~seen).sum())} G3 train rows missing from the signal shards")
    X.flush()
    del X
    keys_train = keys_train.astype(str)
    np.save(tmp_dir / "psd_key_train.npy", keys_train)

    # Noise bank: sort by (psd_key, k); every train key needs k = 0..K-1.
    bank_keys, bank_k, bank_src, shard_K = [], [], [], set()
    for path in bank_paths:
        with h5py.File(path, "r") as h:
            check_shard(h, path, "noise_bank", reference)
            shard_K.add(int(h.attrs["bank_k"]))
            bank_keys.append(h["psd_key"].asstr()[:].astype(str))
            bank_k.append(h["k"][:].astype(np.int64))
            bank_src.append(np.full(h["k"].shape[0], len(bank_src)))
    if len(shard_K) != 1:
        raise SystemExit(f"bank_k differs across noise_bank shards: {sorted(shard_K)}")
    K = shard_K.pop()
    bank_keys, bank_k, bank_src = map(np.concatenate, (bank_keys, bank_k, bank_src))
    local = np.concatenate([np.arange((bank_src == s).sum()) for s in range(len(bank_paths))])
    order = np.lexsort((bank_k, bank_keys))
    keys_sorted, k_sorted = bank_keys[order], bank_k[order]
    uniq = np.unique(keys_sorted)
    if keys_sorted.size != uniq.size * K or not np.array_equal(k_sorted, np.tile(np.arange(K), uniq.size)):
        raise SystemExit(f"Noise bank does not hold exactly k = 0..{K - 1} per psd_key")
    if not np.array_equal(uniq, np.unique(keys_train)):
        raise SystemExit("Noise bank psd_keys differ from the train psd_keys")
    B = np.lib.format.open_memmap(tmp_dir / "bank_X.npy", mode="w+", dtype=np.float16,
                                  shape=(keys_sorted.size, C, T))
    bank_stats = Stats(C)
    for s, path in enumerate(bank_paths):
        dest = np.flatnonzero(bank_src[order] == s)  # output positions of shard s rows
        src = local[order][dest]                      # their rows in shard s
        # Shards are written sorted by (psd_key, k), so src must be increasing (h5py fancy read).
        if not (np.diff(src) > 0).all():
            raise SystemExit(f"{path}: bank rows are not sorted by (psd_key, k)")
        with h5py.File(path, "r") as h:
            for lo in range(0, dest.size, BLOCK_ROWS):
                hi = min(lo + BLOCK_ROWS, dest.size)
                x = h["N_bank"][src[lo:hi]]
                bank_stats.add(x)
                B[dest[lo:hi]] = x
    B.flush()
    del B
    np.save(tmp_dir / "bank_psd_key.npy", keys_sorted)

    _, rows_per_key = np.unique(keys_train, return_counts=True)
    manifest = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "dataset_id": dataset_id,
        "content": "L(s) and L(n) before the per-sample per-detector z-score; X = zscore(L(s) + L(n))",
        "shard_provenance": reference,
        "signal_shards": [stat_entry(p) for p in signal_paths],
        "noise_bank_shards": [stat_entry(p) for p in bank_paths],
        "g3_memmap": {"dir": str(g3_mm), "row_ids_train_sha256": sha256_file(g3_mm / "row_ids_train.npy"),
                      "manifest_sha256": (sha256_file(g3_mm / "manifest.json")
                                          if (g3_mm / "manifest.json").exists() else None)},
        "x_dtype": "float16",
        "X_clean_train": {"shape": [int(row_ids.size), int(C), int(T)],
                          "row_order": "G3 fp16_memmap row_ids_train",
                          "per_detector": sig_stats.report()},
        "bank_X": {"shape": [int(keys_sorted.size), int(C), int(T)], "k_per_key": K,
                   "row_order": f"sorted by (psd_key, k); k = position % {K}",
                   "per_detector": bank_stats.report()},
        "n_psd_keys": int(uniq.size),
        "train_rows_per_key": {"min": int(rows_per_key.min()), "median": float(np.median(rows_per_key)),
                               "max": int(rows_per_key.max()), "mean": float(rows_per_key.mean())},
        "outputs_sha256": {},
    }
    for name in ("X_clean_train.npy", "psd_key_train.npy", "bank_X.npy", "bank_psd_key.npy"):
        manifest["outputs_sha256"][name] = sha256_file(tmp_dir / name)
    manifest["total_time_s"] = time.time() - t_start
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if out_dir.exists():
        raise SystemExit(f"Output directory appeared during merge; result left in {tmp_dir}")
    os.rename(tmp_dir, out_dir)
    print(f"done in {manifest['total_time_s']:.0f} s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
