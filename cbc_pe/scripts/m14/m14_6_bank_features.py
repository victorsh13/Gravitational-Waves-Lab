"""M14.6: matched-filter peak features of val G3T for the P512 bank and the learned bank L32 of M14.5b (VM, GPU).

Per val row, template k and detector d (docs/experiments/M14_6_bank_probes.md):
  P512 (templates_<UTC>.npz of m14_6_make_templates.py, whitened per val PSD block and detector, unit norm):
      rho_kd(q) = |c_I + i c_Q|, c_I(q) = sum_m x_d[m] h_kd[m - q + p] (template peak p aligned at window sample q),
      c_Q with the Hilbert transform of h_kd; linear correlation (FFT length 2N, zeros outside the window, no
      circular wrap), q = 0..N-1. |c_I + i c_Q| is the modulus of the analytic signal of c_I (one-sided spectrum).
  L32: the bank layer of the M14.5b checkpoint, reproduced (circular, length N, independent template_i / template_q,
      unit norm, out = sqrt(c_I^2 + c_Q^2 + 1e-12)), as TemplateBank.forward. Gate G3: on the first 64 val rows,
      max |reproduced - model.template_bank(x)| / max |model.template_bank(x)| <= 1e-4.
Summaries: peak over time per detector (3) and network peak sqrt(sum_d peak_d^2) (1) -> (rows, K, 4) fp32, columns
[H1, L1, V1, network] (fp16_memmap detector order).

Output: <data_root>/results/M13_3/M14_6/features_val_<UTC>.npz (never overwrites), rows in fp16_memmap order.
PyTorch 1.9.1 compatible (torch.fft); no new dependencies.

Cost (P100, estimate): ~40000 x 3 x 512 complex FFTs of length 32768 plus products; ~20-40 min. GPU memory:
template spectra of one block ~0.2 GB + 3 complex (B, 3, chunk, 2N) buffers (~4.8 GB with B = 16, chunk = 128).
Host RAM: ~2.8 GB for the templates array.

Usage (from cbc_pe, VM):
    python scripts/m14/m14_6_bank_features.py --templates <data_root>/results/M13_3/M14_6/templates_<UTC>.npz
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import resolve_data_root
from scripts.predict_cnn_hdf5 import import_model_class

CHECKPOINT_REL = "M13_3/M14_5b_4a_G3T_tbank_fullcov_s123_final.pt"
MEMMAP_REL = "processed/M13_G3_band1024_w16_500k/m13_13_g3t/fp16_memmap"
OUT_SUBDIR = "results/M13_3/M14_6"
G3_ROWS = 64
G3_TOL = 1e-4


def require(condition, message):
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--templates", type=Path, required=True, help="templates_<UTC>.npz")
    p.add_argument("--data-root", type=Path, default=None, help="Precedence: CLI, CBC_PE_DATA_ROOT, default")
    p.add_argument("--batch", type=int, default=16, help="val rows per GPU batch")
    p.add_argument("--chunk", type=int, default=128, help="templates per GPU chunk")
    return p.parse_args()


def summarize(peaks):
    """peaks (B, 3, K) -> (B, K, 4): [peak_H1, peak_L1, peak_V1, sqrt(sum_d peak_d^2)]."""
    net = torch.sqrt((peaks * peaks).sum(1, keepdim=True))
    return torch.cat([peaks, net], 1).permute(0, 2, 1)


def physical_peaks(x, H, n, chunk):
    """x (B, 3, n) float32; H (K, 3, n + 1) rfft (length 2n) of the templates laid out with their peak at index 0.

    Analytic correlation: one-sided spectrum Z = X conj(H) (x2 on 1..n-1 positive bins, DC and Nyquist x1, zeros
    for negative frequencies), ifft of length 2n; |.| = sqrt(c_I^2 + c_Q^2). Lags 0..n-1 only (peak inside window).
    """
    X = torch.fft.rfft(x, n=2 * n)                                   # (B, 3, n + 1)
    w = torch.ones(n + 1, device=x.device)
    w[1:n] = 2.0
    out = []
    for s in range(0, H.shape[0], chunk):
        Hc = H[s:s + chunk].permute(1, 0, 2)                         # (3, k, n + 1)
        Z = X.unsqueeze(2) * Hc.conj().unsqueeze(0) * w              # (B, 3, k, n + 1)
        Z = torch.cat([Z, torch.zeros_like(Z[..., 1:n])], -1)        # (B, 3, k, 2n)
        c = torch.fft.ifft(Z, n=2 * n)[..., :n]
        out.append(c.abs().amax(-1))                                 # (B, 3, k)
        del Z, c
    return torch.cat(out, 2)


def learned_bank_out(x, ti, tq, n):
    """Independent reproduction of TemplateBank.forward (L = n): (B, 3, K, n) |c|, circular."""
    def layout(h):
        h = h / (h.norm(dim=-1, keepdim=True) + 1e-12)
        return torch.roll(h, shifts=1 - h.shape[-1], dims=-1)
    X = torch.fft.rfft(x, n=n).unsqueeze(2)
    ci = torch.fft.irfft(X * torch.fft.rfft(layout(ti), n=n).conj(), n=n)
    cq = torch.fft.irfft(X * torch.fft.rfft(layout(tq), n=n).conj(), n=n)
    return torch.sqrt(ci * ci + cq * cq + 1e-12)


def load_model(path):
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False  # trusted own checkpoint
    ck = torch.load(path, **load_kwargs)
    mc = ck["model_config"]
    kw = dict(mc["model_kwargs"])
    kw.setdefault("n_detectors", 3)
    kw.setdefault("n_outputs", 3)
    model = import_model_class(mc["class_name"])(**kw)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    require(model.template_bank is not None, "checkpoint has no template bank")
    return model, ck


def main():
    args = parse_args()
    data_root = resolve_data_root(cli_data_root=args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| torch", torch.__version__, flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = data_root / OUT_SUBDIR / f"features_val_{stamp}.npz"
    require(not out_path.exists(), f"{out_path} exists")

    memmap_dir = data_root / MEMMAP_REL
    row_ids = np.load(memmap_dir / "row_ids_val.npy").astype(np.int64)
    X_val = np.load(memmap_dir / "X_val.npy", mmap_mode="r")
    n_rows, n_det, n = X_val.shape
    require(n_det == 3 and len(row_ids) == n_rows, f"X_val {X_val.shape}")

    t_load = time.perf_counter()
    with np.load(args.templates, allow_pickle=False) as z:
        tmeta = json.loads(str(z["metadata_json"]))
        require(np.array_equal(z["val_row_ids"], row_ids), "templates val_row_ids != row_ids_val of the memmap")
        block_index = np.asarray(z["val_block_index"], dtype=np.int64)
        p = int(z["peak_sample"])
        T = np.asarray(z["templates"], dtype=np.float32)              # (K, n_blocks, 3, n)
    K, n_blocks = T.shape[:2]
    require(T.shape[2:] == (3, n), f"templates {T.shape}")
    print(f"templates: K = {K}, {n_blocks} blocks, peak sample {p} ({time.perf_counter() - t_load:.0f} s)", flush=True)

    # ---- L32 and gate G3 ----
    ck_path = data_root / "models" / "checkpoints" / CHECKPOINT_REL
    model, ck = load_model(ck_path)
    bank = model.template_bank.to(device)
    ti, tq = bank.template_i.detach(), bank.template_q.detach()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(X_val[:G3_ROWS], dtype=np.float32)).to(device)
        num = den = 0.0
        for s in range(0, G3_ROWS, 8):
            ref = bank(x[s:s + 8])                                    # (b, 3 * K_L, n), channels (det, template)
            rep = learned_bank_out(x[s:s + 8], ti, tq, n).reshape(ref.shape)
            num = max(num, float((rep - ref).abs().max()))
            den = max(den, float(ref.abs().max()))
    g3 = dict(rows=G3_ROWS, max_abs_diff=num, max_abs_ref=den, rel_diff=num / den, tol=G3_TOL,
              passed=bool(num / den <= G3_TOL))
    print(f"G3: max|rep - layer| / max|layer| = {num / den:.3e} (tol {G3_TOL:g}) -> "
          f"{'PASS' if g3['passed'] else 'FAIL'}", flush=True)
    require(g3["passed"], "G3 FAILED: the reproduced bank layer does not match the checkpoint forward")

    K_L = ti.shape[0]
    feat_L = np.zeros((n_rows, K_L, 4), dtype=np.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, n_rows, args.batch):
            x = torch.from_numpy(np.asarray(X_val[s:s + args.batch], dtype=np.float32)).to(device)
            feat_L[s:s + len(x)] = summarize(learned_bank_out(x, ti, tq, n).amax(-1)).cpu().numpy()
    print(f"L32: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    del model, bank
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ---- P512, block by block ----
    feat_P = np.zeros((n_rows, K, 4), dtype=np.float32)
    t0 = time.perf_counter()
    done = 0
    with torch.no_grad():
        for b in range(n_blocks):
            pos = np.flatnonzero(block_index == b)
            # Template with its peak at index 0 of a 2n buffer (zeros elsewhere): output index q = peak at q.
            h = torch.zeros(K, 3, 2 * n)
            h[..., :n] = torch.from_numpy(T[:, b])
            h = torch.roll(h, shifts=-p, dims=-1).to(device)
            H = torch.fft.rfft(h, n=2 * n)
            del h
            for s in range(0, len(pos), args.batch):
                idx = pos[s:s + args.batch]
                x = torch.from_numpy(np.asarray(X_val[idx], dtype=np.float32)).to(device)
                feat_P[idx] = summarize(physical_peaks(x, H, n, args.chunk)).cpu().numpy()
            done += len(pos)
            del H
            el = time.perf_counter() - t0
            print(f"block {b + 1}/{n_blocks}: {done}/{n_rows} rows, {el / 60:.1f} min, "
                  f"ETA {el / done * (n_rows - done) / 60:.1f} min", flush=True)
    require(np.isfinite(feat_P).all() and np.isfinite(feat_L).all(), "non-finite features")
    if device.type == "cuda":
        print(f"max GPU memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    meta = dict(experiment="M14.6", created_utc=stamp, templates_file=str(args.templates.resolve()),
                templates_sha256=sha256_file(args.templates), templates_metadata=tmeta,
                checkpoint_file=str(ck_path), checkpoint_sha256=sha256_file(ck_path),
                checkpoint_tag=str(ck.get("checkpoint_tag")), memmap_dir=str(memmap_dir),
                columns=["peak_H1", "peak_L1", "peak_V1", "network"], gate_G3=g3,
                P_definition="analytic linear correlation, FFT length 2N, lags with the template peak in the window",
                L_definition="TemplateBank.forward reproduced (circular, length N)",
                batch=args.batch, chunk=args.chunk, torch_version=torch.__version__, device=str(device),
                elapsed_P_s=time.perf_counter() - t0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("xb") as f:
        np.savez(f, row_ids=row_ids, features_P=feat_P, features_L32=feat_L, metadata_json=np.array(json.dumps(meta)))
    print("Saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
