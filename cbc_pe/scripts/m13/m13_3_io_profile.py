"""M13.3: training-step I/O profile of the M11 model (compute / hdf5 / memmap input paths).

Randomly initialized M11 model (fixed seed), AdamW(lr=3e-4, wd=3e-4), MSE, batch 256, train
mode. Weights and checkpoints are never saved. Per step:
    t_wait = time spent in next(iterator)
    t_h2d  = host-to-device copy (+ float16->float32 cast in memmap mode), synchronized
    t_step = forward + backward + optimizer, torch.cuda.synchronize() before and after
Warmup steps are excluded. Only train rows are read (hdf5/memmap); labels use train-only stats.
Output: stdout and $CBC_PE_DATA_ROOT/results/M13_3_io/<tag>.json (exclusive, never overwritten).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import dataset_processed_dir, resolve_data_root, resolve_processed_artifact  # noqa: E402
from src.models import network  # noqa: E402
from src.models.hdf5_batch_dataset import HDF5BatchIterableDataset  # noqa: E402
from src.models.memmap_dataset import FP16MemmapBatchDataset, RandomBatchSampler  # noqa: E402

BATCH = 256
LR = 3e-4
WD = 3e-4
EPOCH_BATCHES = 1562  # 400000 // 256, M11 train epoch with drop_last


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--mode", choices=["compute", "hdf5", "memmap"], required=True)
    p.add_argument("--cudnn", choices=["deterministic", "benchmark"], required=True)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--n-batches", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=None, help="default: config training.num_workers")
    p.add_argument("--tag", required=True)
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {"median_s": float(np.median(x)), "mean_s": float(x.mean()), "p90_s": float(np.quantile(x, 0.9)),
            "max_s": float(x.max()), "total_s": float(x.sum())}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cfg = json.loads(args.config.read_text())
    dcfg, tcfg, mcfg = cfg["dataset"], cfg["training"], cfg["model"]
    dataset_id = dcfg["dataset_id"]
    seed = int(tcfg["seed"])
    if int(tcfg["batch_size"]) != BATCH:
        raise SystemExit(f"Config batch_size {tcfg['batch_size']} != {BATCH}")
    num_workers = int(tcfg.get("num_workers", 0) if args.num_workers is None else args.num_workers)
    prefetch_factor = int(tcfg.get("prefetch_factor", 2))
    n_total = args.warmup + args.n_batches
    if n_total > EPOCH_BATCHES:
        raise SystemExit(f"warmup + n_batches = {n_total} exceeds one epoch ({EPOCH_BATCHES})")

    data_root = resolve_data_root(cli_data_root=args.data_root, config_data_root=cfg.get("data_root"))
    out_path = data_root / "results" / "M13_3_io" / f"{args.tag}.json"
    if out_path.exists():
        raise SystemExit(f"Output exists, refusing to overwrite: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.deterministic = args.cudnn == "deterministic"
    torch.backends.cudnn.benchmark = args.cudnn == "benchmark"
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = getattr(network, mcfg["class_name"])(**mcfg["kwargs"]).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    provenance = {}
    if args.mode == "compute":
        X_gpu = torch.randn(BATCH, 3, 16384, device=device)
        y_gpu = torch.randn(BATCH, int(mcfg["kwargs"].get("n_outputs", 3)), device=device)

        def synthetic():
            while True:
                yield X_gpu, y_gpu
        it = synthetic()
        loader = None
    else:
        split_path = resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id,
                                                file_name=dcfg["split_file"])
        stats_path = resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id,
                                                file_name=dcfg["label_stats_file"])
        train_idx = np.load(split_path)["train_idx"].astype(np.int64)
        stats = np.load(stats_path)
        y_mean = stats["y_mean"].astype(np.float32)
        y_std = stats["y_std"].astype(np.float32)
        provenance.update(split_file=str(split_path), label_stats_file=str(stats_path),
                          n_train=int(train_idx.size))
        loader_kwargs = dict(batch_size=None, num_workers=num_workers, pin_memory=True)
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=False, prefetch_factor=prefetch_factor)

        if args.mode == "hdf5":
            # Same construction as scripts/train_cnn_hdf5.py (hdf5_batch_slices, M11 config).
            h5_path = resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id,
                                                 file_name=dcfg["dataset_file"])
            ds = HDF5BatchIterableDataset(
                h5_path=h5_path, indices=train_idx, y_mean=y_mean, y_std=y_std,
                input_normalization=cfg.get("input_normalization", {"enabled": False}),
                batch_size=BATCH, drop_last=bool(tcfg.get("drop_last", True)), seed=seed,
                shuffle_batches=bool(tcfg.get("shuffle_batches", True)),
                shuffle_within_batch=bool(tcfg.get("shuffle_within_batch", False)),
                max_slice_overread=float(tcfg.get("max_slice_overread", 4.0)),
            )
            ds.set_epoch(0)
            loader = DataLoader(ds, **loader_kwargs)
            provenance.update(source=str(h5_path), data_loading_mode=tcfg.get("data_loading_mode"))
        else:
            if cfg.get("input_normalization", {}).get("enabled", False):
                raise SystemExit("memmap export holds HDF5 values; config normalizes on the fly")
            mm_dir = dataset_processed_dir(data_root, dataset_id) / "fp16_memmap"
            row_ids = np.load(mm_dir / "row_ids_train.npy")
            if not np.array_equal(row_ids, np.sort(train_idx)):
                raise SystemExit("row_ids_train.npy does not match train_idx of the split file")
            ds = FP16MemmapBatchDataset(mm_dir, "train", y_mean, y_std)
            sampler = RandomBatchSampler(len(ds), BATCH, seed=seed, drop_last=True)
            sampler.set_epoch(0)
            loader = DataLoader(ds, sampler=sampler, **loader_kwargs)
            provenance.update(source=str(mm_dir))
        it = iter(loader)

    t_wait, t_h2d, t_step, in_bytes, losses = [], [], [], [], []
    torch.cuda.reset_peak_memory_stats(device)
    t_meas0 = None
    for i in range(n_total):
        if i == args.warmup:
            torch.cuda.synchronize()
            t_meas0 = time.perf_counter()
        t0 = time.perf_counter()
        xb, yb = next(it)
        t1 = time.perf_counter()
        nbytes = 0 if xb.is_cuda else xb.element_size() * xb.nelement() + yb.element_size() * yb.nelement()

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(xb)
        loss = criterion(out.float(), yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        if i >= args.warmup:
            t_wait.append(t1 - t0)
            t_h2d.append(t3 - t2)
            t_step.append(t4 - t3)
            in_bytes.append(nbytes)
            losses.append(float(loss.detach()))
    wall = time.perf_counter() - t_meas0
    del it, loader

    n = args.n_batches
    total_wait = float(np.sum(t_wait))
    total_bytes = float(np.sum(in_bytes))
    res = {
        "tag": args.tag,
        "mode": args.mode,
        "cudnn": args.cudnn,
        "amp": bool(args.amp),
        "n_batches_measured": n,
        "warmup": args.warmup,
        "batch_size": BATCH,
        "num_workers": num_workers if args.mode != "compute" else None,
        "prefetch_factor": prefetch_factor if args.mode != "compute" and num_workers > 0 else None,
        "seed": seed,
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WD},
        "loss": "MSELoss",
        "config": str(args.config),
        "model_class": mcfg["class_name"],
        "model_kwargs": mcfg["kwargs"],
        "provenance": provenance,
        "t_wait": summary(t_wait),
        "t_h2d": summary(t_h2d),
        "t_step": summary(t_step),
        "wall_measured_s": wall,
        "wait_fraction": total_wait / wall,
        "epoch_extrapolated_s": {"from_mean_wall": wall / n * EPOCH_BATCHES,
                                 "from_medians": float(np.median(t_wait) + np.median(t_h2d)
                                                       + np.median(t_step)) * EPOCH_BATCHES,
                                 "epoch_batches": EPOCH_BATCHES},
        "input_MB_per_batch": total_bytes / n / 1e6 if total_bytes else None,
        "input_MB_per_s_wall": total_bytes / wall / 1e6 if total_bytes else None,
        "loss_first_last": [losses[0], losses[-1]],
        "max_memory_allocated_MB": torch.cuda.max_memory_allocated(device) / 2**20,
        "max_memory_reserved_MB": torch.cuda.max_memory_reserved(device) / 2**20,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "numpy": np.__version__, "h5py": h5py.__version__},
        "cudnn_flags": {"enabled": torch.backends.cudnn.enabled,
                        "deterministic": torch.backends.cudnn.deterministic,
                        "benchmark": torch.backends.cudnn.benchmark},
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print(f"[{args.tag}] mode={args.mode} cudnn={args.cudnn} amp={args.amp} workers={res['num_workers']}")
    for k in ("t_wait", "t_h2d", "t_step"):
        print(f"  {k:7s} median={res[k]['median_s'] * 1e3:8.2f} ms  total={res[k]['total_s']:8.2f} s")
    print(f"  wall={wall:.2f} s  wait_fraction={res['wait_fraction']:.3f}  "
          f"epoch~{res['epoch_extrapolated_s']['from_mean_wall'] / 60:.1f} min")
    if total_bytes:
        print(f"  input {res['input_MB_per_batch']:.1f} MB/batch, {res['input_MB_per_s_wall']:.1f} MB/s")
    print(f"  max GPU memory allocated {res['max_memory_allocated_MB']:.0f} MB  -> {out_path}")
    with open(out_path, "x") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
