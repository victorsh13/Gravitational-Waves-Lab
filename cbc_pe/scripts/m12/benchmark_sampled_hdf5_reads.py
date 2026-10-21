"""Bounded, single-process CPU comparison; no training or artifact writes.

Run from cbc_pe with the existing environment and optional --data-root override.
Physical disk bytes are NOT inferred from HDF5 selection sizes or chunk counts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np

from src.models.balanced_sampling import load_train_marginal_sampler
from src.models.sample_weighting import array_hash
from src.paths import resolve_data_root, resolve_processed_artifact, resolve_project_root


def grouped_spans(unique):
    """Greedy within-chunk spans: <=64 rows and <=4x logical overread.

    Each tuple maps one source slice to consecutive positions in unique order.
    No entire-chunk materialization or cross-batch cache is used.
    """
    groups = []
    start = 0
    while start < len(unique):
        end = start + 1
        while end < len(unique):
            if unique[end] // 64 != unique[start] // 64:
                break
            if unique[end] - unique[start] + 1 > 4 * (end - start + 1):
                break
            end += 1
        groups.append((start, end, int(unique[start]), int(unique[end - 1]) + 1))
        start = end
    return groups


def measure(method, rows, X, y):
    t = perf_counter()
    unique, inverse = np.unique(rows, return_inverse=True)
    groups = grouped_spans(unique) if method == "B" else None
    preparation = perf_counter() - t

    t = perf_counter()
    if method == "A":
        xu = X[unique]
        fetched = len(unique)
        calls = 1
    else:
        xu = np.empty((len(unique), *X.shape[1:]), dtype=X.dtype)
        for start, end, lo, hi in groups:
            block = X[lo:hi]
            xu[start:end] = block[unique[start:end] - lo]
            del block
        fetched = sum(hi - lo for _, _, lo, hi in groups)
        calls = len(groups)
    x_seconds = perf_counter() - t
    # Keep y access identical and strictly restricted to selected train labels.
    t = perf_counter()
    yu = y[unique]
    y_seconds = perf_counter() - t
    t = perf_counter()
    xb = np.asarray(xu[inverse], dtype=np.float32)
    yb = np.asarray(yu[inverse], dtype=np.float32)
    reconstruction = perf_counter() - t
    record = dict(method=method, preparation=preparation, x_read=x_seconds,
                  y_read=y_seconds, reconstruction=reconstruction,
                  total=preparation + x_seconds + y_seconds + reconstruction,
                  unique=len(unique), x_rows=fetched, y_rows=len(unique),
                  x_calls=calls, chunks=len(np.unique(unique // 64)))
    return record, xb, yb, xu, yu, inverse


def summary(records, title):
    print("\n" + title)
    for method in ("A", "B"):
        selected = [r for r in records if r["method"] == method]
        if not selected:
            continue
        totals = np.array([r["total"] for r in selected])
        print(f"{method}: n={len(selected)} total={totals.sum():.6f}s "
              f"mean={totals.mean():.6f}s/batch median={np.median(totals):.6f}s "
              f"P90={np.percentile(totals, 90):.6f}s")
        print("  mean stage seconds: " + ", ".join(
            f"{key}={np.mean([r[key] for r in selected]):.6f}"
            for key in ("preparation", "x_read", "y_read", "reconstruction")))
        requested = sum(r["unique"] for r in selected)
        fetched = sum(r["x_rows"] for r in selected)
        print(f"  X logical rows fetched={fetched}, requested unique rows={requested}, "
              f"amplification={fetched / requested:.4f}; "
              f"y rows fetched={sum(r['y_rows'] for r in selected)} (amplification=1)")
    a = [r for r in records if r["method"] == "A"]
    b = [r for r in records if r["method"] == "B"]
    if a and b:
        for key in ("total", "x_read"):
            ratio = np.mean([r[key] for r in a]) / np.mean([r[key] for r in b])
            print(f"  A/B {key} time ratio={ratio:.3f} (>1 favors B)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batches", type=int, choices=range(12, 21), default=16)
    args = parser.parse_args()
    root = resolve_project_root()
    config_path = root / "configs/experiments/train_m12_3_S1_balanced_sampling.json"
    cfg = json.loads(config_path.read_text())
    dc, tc = cfg["dataset"], cfg["training"]
    policy = tc["balanced_sampling"]
    if (dc["dataset_id"] != "M11_final_G1_500k" or tc["batch_size"] != 256
            or tc["drop_last"] is not True or tc.get("sample_weighting") is not None
            or policy != dict(enabled=True, mode="marginal_mixture", threshold=0.8,
                              p_peripheral=0.35, seed=123)):
        raise ValueError("Expected frozen M12.3-S1 configuration")
    data_root = resolve_data_root(cli_data_root=args.data_root,
                                  config_data_root=cfg.get("data_root"))
    paths = {key: resolve_processed_artifact(data_root=data_root,
             dataset_id=dc["dataset_id"], file_name=dc[key])
             for key in ("dataset_file", "split_file")}
    with np.load(paths["split_file"], allow_pickle=False) as splits:
        ids = splits["train_idx"].astype(np.int64)
    if ids.shape != (400000,):
        raise ValueError("Expected 400000 frozen training IDs")
    t = perf_counter()
    sampler = load_train_marginal_sampler(paths["dataset_file"], ids,
                                         seed=policy["seed"], threshold=policy["threshold"],
                                         p_peripheral=policy["p_peripheral"])
    # Generate the FULL existing schedule: requesting fewer draws would alter it.
    schedule = sampler.schedule(0)
    print(f"Reference/schedule setup: {perf_counter() - t:.3f}s (excluded)")
    print("Config:", config_path, "\nHDF5:", paths["dataset_file"])
    print("Schedule SHA256:", array_hash(schedule))
    print(f"numpy={np.__version__}, h5py={h5py.__version__}, HDF5={h5py.version.hdf5_version}")
    print(f"Fixed epoch-0 batch IDs: 20..{20 + args.batches - 1}; no earlier strain reads")
    print("Single process, no DataLoader/CUDA. Two bounded passes; no cache flush.")
    print("Pass 1: existing OS-cache state UNKNOWN, second method potentially warmed.")
    print("Pass 2: repeat/cache-warmed workload, residency NOT guaranteed; neither is cold I/O.")
    print("Rows fetched = logical HDF5 selection rows; physical disk bytes are unmeasured.")
    print("Chunk coverage = distinct chunks * 64 rows, NOT measured read amplification.")
    print("B X timing includes allocation, slice reads and packing into unique order.")
    print("C repeats inverse reconstruction 3 times on resident arrays, without HDF5 reads.")
    records, c_times = [], []
    # The existing sampler opens/closes its own train-label reader during setup.
    # All timed strain operations share this one file and these dataset handles.
    with h5py.File(paths["dataset_file"], "r") as h:
        X, y = h["X"], h["y"]
        if (X.shape != (500000, 3, 16384) or X.dtype != np.dtype("float32")
                or X.chunks != (64, 3, 16384) or X.compression is not None
                or y.shape != (500000, 3) or y.dtype != np.dtype("float32")
                or y.chunks != (1024, 3) or y.compression is not None):
            raise ValueError("HDF5 layout differs from the specified benchmark contract")
        print("X chunk cache (slots, bytes, w0):", X.id.get_access_plist().get_chunk_cache())
        for pass_id in (1, 2):
            for offset in range(args.batches):
                batch_id = 20 + offset
                rows = schedule[batch_id * 256:(batch_id + 1) * 256]
                order = ("A", "B") if (offset + pass_id) % 2 else ("B", "A")
                reference = None
                for position, method in enumerate(order):
                    record, xb, yb, xu, yu, inverse = measure(method, rows, X, y)
                    record.update(pass_id=pass_id, position=position, batch=batch_id)
                    records.append(record)
                    if reference is None:
                        reference = (xb, yb)
                    else:
                        # Byte views enforce signed-zero/NaN-payload identity too.
                        if not (np.array_equal(xb.view(np.uint8), reference[0].view(np.uint8))
                                and np.array_equal(yb.view(np.uint8), reference[1].view(np.uint8))):
                            raise AssertionError(f"A/B bitwise mismatch at batch {batch_id}")
                    print(f"pass={pass_id} batch={batch_id} method={method} position={position + 1} "
                          f"unique={record['unique']} duplicates={256 - record['unique']} "
                          f"X_chunks={record['chunks']} chunk_coverage_rows={record['chunks'] * 64} "
                          f"X_rows={record['x_rows']} X_amp={record['x_rows'] / record['unique']:.3f} "
                          f"y_rows={record['y_rows']} y_amp=1 X_calls={record['x_calls']} "
                          + " ".join(f"{key}={record[key]:.6f}s" for key in
                                     ("preparation", "x_read", "y_read", "reconstruction")), flush=True)
                    if position == 1:
                        for _ in range(3):
                            t = perf_counter()
                            cx = np.asarray(xu[inverse], dtype=np.float32)
                            cy = np.asarray(yu[inverse], dtype=np.float32)
                            c_times.append(perf_counter() - t)
                            if not (np.array_equal(cx.view(np.uint8), xb.view(np.uint8))
                                    and np.array_equal(cy.view(np.uint8), yb.view(np.uint8))):
                                raise AssertionError("C reconstruction mismatch")
                            del cx, cy
                    del xb, yb, xu, yu, inverse
                del reference
    for pass_id in (1, 2):
        subset = [r for r in records if r["pass_id"] == pass_id]
        summary(subset, f"Pass {pass_id}: mixed method-order timings")
        for position in (0, 1):
            summary([r for r in subset if r["position"] == position],
                    f"Pass {pass_id}, method position {position + 1} (unpaired order stratum)")
    c = np.array(c_times)
    print(f"\nC resident reconstruction: n={len(c)} mean={c.mean():.6f}s "
          f"median={np.median(c):.6f}s P90={np.percentile(c, 90):.6f}s")
    for method in ("A", "B"):
        elapsed = np.mean([r["total"] for r in records if r["method"] == method])
        print(f"C mean / {method} mean total = {c.mean() / elapsed:.3f} (context, not a speedup)")
    print("All returned batches bitwise equal, including duplicates/order. No artifacts written.")
    print("These single-process timings are not comparable directly to four-worker seconds/batch.")


if __name__ == "__main__":
    main()
