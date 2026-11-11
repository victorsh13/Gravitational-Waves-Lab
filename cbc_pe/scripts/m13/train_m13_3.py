"""M13.3a/b: retrain the M11 model (per-step warmup + cosine schedule, random batches, fp16 memmap).

Data: <data_root>/processed/<dataset_id>/fp16_memmap/ (train and val only; cal/test are never read).
row_ids_<split>.npy must equal sort(<split>_idx) of the config split file; manifest.json SHA256 is
recorded. If the source HDF5 exists, a fixed batch of 256 shuffled positions per split is compared
against it before training (|dX| <= 0.5 ulp_fp16(|x|) + 1e-7, y identical); any failure aborts.

Training: model class/kwargs from the config, AdamW(lr=peak_lr, wd), MSE on train-only standardized
labels [chirp_mass, total_mass, chi_eff], batch 256, drop_last, RandomBatchSampler(seed). Per-step
LambdaLR: linear warmup over warmup_epochs to peak_lr, cosine to 0 at the end of max_epochs. No early
stopping; the registered model is the last epoch (_final.pt). _bestval.pt is informative only.
Optional training keys (M13.9/M13.10; absent = the behaviour above): schedule.warmup_steps replaces
schedule.warmup_epochs; train_subset = {n, seed} trains on nested rows default_rng(seed).permutation(...)[:n];
total_steps stops the run at that step, possibly in the middle of the last epoch.
Optional heteroscedastic head (M14.1; model.heteroscedastic = true, requires training.loss = "beta_nll" and
training.beta_nll = beta): the model outputs [mu, log_sigma] per label (standardized units, log_sigma clamped
in the model) and is trained with beta-NLL (Seitzer et al. 2022): per label
0.5 * (log sigma^2 + (y - mu)^2 / sigma^2) * stop_grad(sigma^2)^beta, mean over labels and batch. Logged
val/train40k mse and slope are computed on mu; prediction npz files add sigma_<split> (standardized).
Optional full covariance (M14.5; model.covariance = "full" with model.heteroscedastic = true, requires
training.loss = "beta_nll_full" and training.beta_nll = beta): the model outputs [mu, log L_ii, L21, L31, L32]
(Sigma = L L^T, L lower triangular, standardized units); e = L^-1 (y - mu) by forward substitution and per
label (e_i^2 / 2 + log L_ii) * stop_grad(L_ii^2)^beta, mean over labels and batch (equals beta_nll for L
diagonal). Prediction npz files: pred_<split> = mu, sigma_<split> = sqrt(diag Sigma), cov_<split> = Sigma
(N, 3, 3) float32, standardized.
Optional template bank (M14.2; model.template_bank = {n_templates, template_length, n_samples, chunk_templates,
checkpoint, init}): network.TemplateBank amplitudes sqrt(c_I^2 + c_Q^2) per (detector, template) are concatenated
after the input before block1; templates initialized by init_template_bank (train rows, default_rng(142001)),
the rows used recorded in run_info.procedure.template_bank.
Optional dropout switch-off: from 0-based epoch ceil((1 - f) * max_epochs) every nn.Dropout gets p=0.

Optional noise augmentation (M13.6, only if training.noise_augmentation.enabled): train X is rebuilt
per batch from <processed>/<memmap_dir>/{X_clean_train, psd_key_train, bank_X, bank_psd_key}.npy
(merge_m13_6_clean.py; L(s), L(n) before z-score, rows in G3 row_ids_train order). On the GPU in
float32: x[b, d] = clean[b, d] + s_bd * R_bd(bank[key_b, k_bd, d]), k_bd ~ U{0..K-1}, s_bd = -1 w.p.
0.5 (sign_flip), R_bd = time reversal w.p. 0.5 (time_reverse), all independent per row and detector,
drawn in that order by default_rng(seed + 10000 * epoch + batch_index); then the per-sample
per-detector z-score (x - mean) / (std + 1e-6), population std, as
normalize_input_per_sample_per_detector_zscore. Val and the train40k monitor read the G3 memmap X.

Monitoring (eval mode, every epoch): per-label MSE (standardized units) and OLS slope of theta on f
(theta = truth, f = prediction) on full val and on a fixed 40000-row train subset
(np.random.default_rng(133400)). Epochs are 0-based in logs and history; checkpoint "epoch" is the
number of completed epochs.

Outputs (refused if any exists, except the resumable ones with --resume):
    <data_root>/models/checkpoints/M13_3/<tag>_{last,final,bestval}.pt
    <data_root>/results/M13_3/<tag>/{run_info.json, history.npz,
        val_predictions_embeddings.npz, train40k_predictions.npz}
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import dataset_processed_dir, resolve_data_root, resolve_processed_artifact  # noqa: E402
from src.models import network  # noqa: E402
from src.models.memmap_dataset import (  # noqa: E402
    FP16MemmapBatchDataset, RandomBatchSampler, SequentialBatchSampler)

BATCH = 256
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
TRAIN_MONITOR_N = 40000
TRAIN_MONITOR_SEED = 133400
CHECK_N = 256
CHECK_SEED = {"train": 133401, "val": 133402}
DEFAULT_LABELS = ["chirp_mass", "total_mass", "chi_eff"]
# Split name used in the train-subset npz keys (pred_<split>, idx_<split>, ...), as in the M11 schema.
TRAIN_MONITOR_SPLIT = "train"
AUG_BANK_K = 64
AUG_SEED_EPOCH_STRIDE = 10000  # > batches per epoch (400000 // 256 = 1562): no seed reused across epochs
AUG_ZSCORE_EPS = 1e-6  # M10_INPUT_ZSCORE_EPS used to build the G3 X
TB_INIT_SEED = 142001  # M14.2 template-bank row selection
TB_T_SUN = 4.925490947e-6  # G Msun / c^3 [s]
TB_F_REF = 30.0  # Hz, reference frequency of tau30


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="continue from <tag>_last.pt")
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_lr_lambda(warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


def dropout_off_epoch(fraction, max_epochs):
    if fraction <= 0.0:
        return None
    # round() guards against float noise, e.g. (1 - 0.2) * 80 must give 64, not 65.
    return int(math.ceil(round((1.0 - fraction) * max_epochs, 9)))


def schedule_steps(tcfg, steps_per_epoch, max_epochs):
    """Return (warmup_steps, total_steps); without the optional keys, epochs times steps_per_epoch."""
    sched = tcfg["schedule"]
    has_steps = "warmup_steps" in sched
    warmup_steps = int(sched["warmup_steps"]) if has_steps else steps_per_epoch * int(sched["warmup_epochs"])
    total_steps = int(tcfg.get("total_steps", steps_per_epoch * max_epochs))
    if not (max_epochs - 1) * steps_per_epoch < total_steps <= max_epochs * steps_per_epoch:
        raise SystemExit(f"total_steps={total_steps} does not end in epoch {max_epochs} "
                         f"({steps_per_epoch} steps/epoch)")
    if has_steps and not 0 < warmup_steps < total_steps:
        raise SystemExit(f"warmup_steps={warmup_steps} must be in (0, {total_steps})")
    return warmup_steps, total_steps


class SubsetBatchSampler(RandomBatchSampler):
    """RandomBatchSampler over a fixed set of train positions (same per-epoch permutation rule)."""

    def __init__(self, positions, batch_size, seed, drop_last=True):
        super().__init__(positions.size, batch_size, seed, drop_last)
        self.positions = positions

    def _order(self):
        return self.positions[super()._order()]


def make_train_sampler(n_train, tcfg, seed):
    """Return (sampler, positions); positions is None when the whole train split is used."""
    sub = tcfg.get("train_subset")
    if not sub:
        return RandomBatchSampler(n_train, BATCH, seed=seed, drop_last=True), None
    n = int(sub["n"])
    if not 0 < n <= n_train:
        raise SystemExit(f"train_subset.n={n} must be in (0, {n_train}]")
    # Same permutation cut at n, so smaller subsets are nested in larger ones with the same seed.
    positions = np.random.default_rng(int(sub["seed"])).permutation(n_train)[:n]
    return SubsetBatchSampler(positions, BATCH, seed, drop_last=True), positions


def disable_dropout(model):
    names = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
            names.append(name)
    return names


class NoiseAugTrainDataset(FP16MemmapBatchDataset):
    """Train split for noise augmentation: X_clean from the m13_6_aug memmap, y from the G3 memmap.

    The index is (batch_seed, positions). Returns (X_clean f16, noise f16, sign f32, reverse bool, y);
    noise[b, d] = bank[bank_start[pos_b] + k_bd, d] before sign/reversal (applied on the GPU).
    """

    def __init__(self, g3_dir, aug_dir, y_mean, y_std, bank_start, bank_k, sign_flip, time_reverse):
        super().__init__(g3_dir, "train", y_mean, y_std)
        self.x_path = Path(aug_dir) / "X_clean_train.npy"
        self.bank_path = Path(aug_dir) / "bank_X.npy"
        Xc = np.load(self.x_path, mmap_mode="r")
        bank = np.load(self.bank_path, mmap_mode="r")
        if Xc.dtype != np.float16 or tuple(Xc.shape) != self.x_shape:
            raise ValueError(f"X_clean_train {Xc.dtype} {Xc.shape} does not match G3 X_train {self.x_shape}")
        if bank.dtype != np.float16 or tuple(bank.shape[1:]) != self.x_shape[1:]:
            raise ValueError(f"bank_X {bank.dtype} {bank.shape} does not match {self.x_shape[1:]}")
        bank_start = np.asarray(bank_start, dtype=np.int64)
        if bank_start.shape != (self.x_shape[0],) or bank_start.min() < 0 or bank_start.max() + bank_k > bank.shape[0]:
            raise ValueError("bank_start out of range of bank_X")
        del Xc, bank
        self.bank_start = bank_start
        self.bank_k = int(bank_k)
        self.sign_flip = bool(sign_flip)
        self.time_reverse = bool(time_reverse)
        self._bank = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_bank"] = None
        return state

    def _open(self):
        super()._open()
        if self._bank is None:
            self._bank = np.load(self.bank_path, mmap_mode="r")

    def __getitem__(self, item):
        batch_seed, batch_positions = item
        xc, y = super().__getitem__(batch_positions)
        pos = np.asarray(batch_positions, dtype=np.int64)
        n, C, T = pos.size, self.x_shape[1], self.x_shape[2]
        # Fixed draw order (k, sign, reverse) whatever the flags, so k does not depend on them.
        rng = np.random.default_rng(int(batch_seed))
        k = rng.integers(0, self.bank_k, size=(n, C))
        sign = np.where(rng.random((n, C)) < 0.5, -1.0, 1.0).astype(np.float32)
        rev = rng.random((n, C)) < 0.5
        if not self.sign_flip:
            sign[:] = 1.0
        if not self.time_reverse:
            rev[:] = False
        idx = self.bank_start[pos][:, None] + k
        noise = np.empty((n, C, T), dtype=np.float16)
        for d in range(C):
            order = np.argsort(idx[:, d], kind="stable")  # monotone memmap reads
            noise[order, d] = self._bank[idx[order, d], d]
        return xc, torch.from_numpy(noise), torch.from_numpy(sign), torch.from_numpy(rev), y


class SeededBatchSampler(Sampler):
    """Wraps RandomBatchSampler; yields (seed + AUG_SEED_EPOCH_STRIDE * epoch + batch_index, positions)."""

    def __init__(self, base, seed):
        self.base = base
        self.seed = int(seed)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        for b, pos in enumerate(self.base):
            yield self.seed + AUG_SEED_EPOCH_STRIDE * self.base.epoch + b, pos


def augment_on_gpu(xc, noise, sign, rev, device):
    x = xc.to(device, non_blocking=True).float()
    nz = noise.to(device, non_blocking=True).float()
    rev = rev.to(device, non_blocking=True).unsqueeze(-1)
    nz = torch.where(rev, nz.flip(-1), nz)
    x = x + sign.to(device, non_blocking=True).unsqueeze(-1) * nz
    # As normalize_input_per_sample_per_detector_zscore: population std, eps added to std.
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / (std + AUG_ZSCORE_EPS)


def load_noise_aug(acfg, base_dir, g3_mm_dir, row_ids_train):
    """Validate the m13_6_aug memmap against the G3 train rows; return (aug_dir, bank_start, K, info)."""
    aug_dir = base_dir / acfg["memmap_dir"]
    aug_manifest = aug_dir / "manifest.json"
    if not aug_manifest.exists():
        raise SystemExit(f"Missing {aug_manifest}")
    man = json.loads(aug_manifest.read_text())
    K = int(acfg.get("bank_k", AUG_BANK_K))
    if int(man["bank_X"]["k_per_key"]) != K:
        raise SystemExit(f"m13_6_aug manifest k_per_key={man['bank_X']['k_per_key']} != {K}")
    # X_clean_train rows follow the G3 row_ids_train order (checked by the merge against this file).
    if man["g3_memmap"]["row_ids_train_sha256"] != sha256(g3_mm_dir / "row_ids_train.npy"):
        raise SystemExit("m13_6_aug was merged against a different G3 row_ids_train.npy")
    keys_train = np.load(aug_dir / "psd_key_train.npy")
    bank_keys = np.load(aug_dir / "bank_psd_key.npy")
    if keys_train.shape != row_ids_train.shape:
        raise SystemExit(f"psd_key_train {keys_train.shape} != row_ids_train {row_ids_train.shape}")
    if bank_keys.size % K:
        raise SystemExit(f"bank_psd_key size {bank_keys.size} is not a multiple of K={K}")
    kb = bank_keys.reshape(-1, K)
    uk = kb[:, 0]
    if not (kb == uk[:, None]).all() or not (uk[1:] > uk[:-1]).all():
        raise SystemExit("bank_psd_key is not K consecutive entries per sorted unique psd_key")
    j = np.searchsorted(uk, keys_train)
    if (j >= uk.size).any() or not np.array_equal(uk[np.minimum(j, uk.size - 1)], keys_train):
        raise SystemExit("Some train psd_key has no entries in the noise bank")
    info = {"config": acfg, "bank_k": K, "memmap_dir": str(aug_dir),
            "manifest": {"path": str(aug_manifest), "sha256": sha256(aug_manifest),
                         "outputs_sha256": man.get("outputs_sha256"), "script_sha256": man.get("script_sha256"),
                         "config_sha256": man.get("config_sha256")},
            "n_psd_keys_bank": int(uk.size), "n_psd_keys_train": int(np.unique(keys_train).size),
            "batch_rng": f"default_rng(seed + {AUG_SEED_EPOCH_STRIDE} * epoch + batch_index); "
                         "draws k, sign, reverse per (row, detector)",
            "zscore": {"eps": AUG_ZSCORE_EPS, "std": "population", "dtype": "float32 on GPU"}}
    return aug_dir, (j * K).astype(np.int64), K, info


def tau_newtonian(mc):
    """Leading-order time to coalescence from 30 Hz [s] (as scripts/m13/m13_1_diagnostics.py)."""
    return 5.0 / 256.0 * (TB_T_SUN * mc)**(-5.0 / 3.0) * (np.pi * TB_F_REF)**(-8.0 / 3.0)


def hilbert_quadrature(h):
    """Hilbert transform of a real 1D series by FFT (cos -> sin); DC and Nyquist set to 0."""
    H = np.fft.rfft(h) * -1j
    H[0] = 0.0
    if h.size % 2 == 0:
        H[-1] = 0.0
    return np.fft.irfft(H, n=h.size)


def init_template_bank(bank, icfg, base_dir, g3_mm_dir, row_ids_train):
    """M14.2: deterministic template-bank initialization from clean train signals (train rows only).

    K = n_tau * n_chi cells: n_tau train quantile bins of tau30 (Newtonian, true chirp mass), each split
    into n_chi train quantile bins of chi_eff within it; one row per cell by default_rng(seed).choice.
    Per row: detector d = argmax_d sum(X_clean_train[pos, d]**2) in float64 (L(s) whitened, before the
    z-score; proportional to the optimal SNR^2 in the window; M14.2 amendment 1, no HDF5 for G3T); detector
    names only if a manifest has a "detector_order" key, else only the index d is recorded. X_clean_train of
    that detector, z-scored as the input (population std, + eps); merger = argmax of the
    analytic-signal envelope; the L samples ending at the merger form template_i (merger at the end,
    zero-padded at the start if needed), template_q = Hilbert(template_i); both unit-norm.
    """
    seed = int(icfg.get("seed", TB_INIT_SEED))
    n_chi = int(icfg.get("n_chi_eff_bins", 4))
    K, L = bank.n_templates, bank.template_length
    if n_chi <= 0 or K % n_chi:
        raise SystemExit(f"n_templates={K} is not a multiple of n_chi_eff_bins={n_chi}")
    n_tau = K // n_chi
    mm_dir = base_dir / icfg.get("memmap_dir", "memmap")
    man = json.loads((mm_dir / "manifest.json").read_text())
    if man["g3_memmap"]["row_ids_train_sha256"] != sha256(g3_mm_dir / "row_ids_train.npy"):
        raise SystemExit(f"{mm_dir} X_clean_train was built against a different G3 row_ids_train.npy")
    y = np.load(g3_mm_dir / "y_train.npy", mmap_mode="r")  # unstandardized, G3 row_ids_train order
    if y.shape != (row_ids_train.size, 3):
        raise SystemExit(f"y_train {y.shape} does not match row_ids_train {row_ids_train.shape}")
    tau, chi = tau_newtonian(np.asarray(y[:, 0], dtype=np.float64)), np.asarray(y[:, 2], dtype=np.float64)
    rng = np.random.default_rng(seed)
    tau_edges = np.quantile(tau, np.linspace(0.0, 1.0, n_tau + 1))
    tau_bin = np.clip(np.searchsorted(tau_edges[1:-1], tau, side="right"), 0, n_tau - 1)
    pos, cells = [], []
    for i in range(n_tau):
        rows_i = np.flatnonzero(tau_bin == i)
        chi_edges = np.quantile(chi[rows_i], np.linspace(0.0, 1.0, n_chi + 1))
        chi_bin = np.clip(np.searchsorted(chi_edges[1:-1], chi[rows_i], side="right"), 0, n_chi - 1)
        for j in range(n_chi):
            cell = rows_i[chi_bin == j]
            if cell.size == 0:
                raise SystemExit(f"empty template-bank cell tau{i} chi{j}")
            pos.append(int(rng.choice(cell)))
            cells.append([i, j])
    pos = np.asarray(pos, dtype=np.int64)
    dets = None
    for mpath in (mm_dir / "manifest.json", g3_mm_dir / "manifest.json"):
        if mpath.exists() and "detector_order" in (m := json.loads(mpath.read_text())):
            dets = [str(v) for v in (m["detector_order"].split(",") if isinstance(m["detector_order"], str)
                                     else m["detector_order"])]
            break
    Xc = np.load(mm_dir / "X_clean_train.npy", mmap_mode="r")
    t_i, t_q, det, peak, energy = np.zeros((K, L)), np.zeros((K, L)), [], [], []
    for k in range(K):
        row = np.asarray(Xc[pos[k]], dtype=np.float64)
        e = (row * row).sum(axis=1)
        d = int(np.argmax(e))
        s = row[d]
        z = (s - s.mean()) / (s.std() + AUG_ZSCORE_EPS)
        p = int(np.argmax(np.abs(z + 1j * hilbert_quadrature(z))))
        seg = z[max(0, p - L + 1):p + 1]
        t_i[k, L - seg.size:] = seg
        t_q[k] = hilbert_quadrature(t_i[k])
        det.append(d)
        energy.append(e.tolist())
        peak.append(p)
    t_i /= np.linalg.norm(t_i, axis=1, keepdims=True)
    t_q /= np.linalg.norm(t_q, axis=1, keepdims=True)
    bank.set_templates(t_i, t_q)
    return {"seed": seed, "n_tau_bins": n_tau, "n_chi_eff_bins": n_chi,
            "rule": "train quantile bins of tau30, then of chi_eff within each tau30 bin; "
                    "rng.choice per cell in (tau, chi) order; detector = argmax sum X_clean^2 (whitened, pre-zscore)",
            "detector_rule_amendment": "M14.2 enmienda 1: snr_<det> del HDF5 sustituido por la energía blanqueada "
                                       "(no existe HDF5 de G3T)",
            "detector_names": dets,
            "memmap_dir": str(mm_dir), "tau30_edges_s": tau_edges.tolist(),
            "templates": [{"k": k, "train_position": int(pos[k]), "row_id": int(row_ids_train[pos[k]]),
                           "cell_tau_chi": cells[k], "detector_index": det[k],
                           **({"detector": dets[det[k]]} if dets is not None else {}), "merger_index": peak[k],
                           "chirp_mass": float(y[pos[k], 0]), "total_mass": float(y[pos[k], 1]),
                           "chi_eff": float(y[pos[k], 2]), "tau30_s": float(tau[pos[k]]),
                           "clean_energy": energy[k]} for k in range(K)]}


def beta_nll_loss(out, y, beta):
    """M14.1 beta-NLL on out = [mu, log_sigma] (B, 2K) and y (B, K), standardized units.

    Per label 0.5 * (log sigma^2 + (y - mu)^2 / sigma^2) * stop_grad(sigma^2)^beta; mean over labels and batch.
    """
    k = y.shape[1]
    mu, log_sigma = out[:, :k], out[:, k:]
    log_var = 2.0 * log_sigma
    nll = 0.5 * (log_var + (y - mu) ** 2 * torch.exp(-log_var))
    if beta != 0.0:
        nll = nll * torch.exp(beta * log_var).detach()
    return nll.mean()


def beta_nll_full_loss(out, y, beta):
    """M14.5 beta-NLL with full covariance on out = [mu, log L_ii, L21, L31, L32] (B, 9), y (B, 3).

    r = y - mu, e = L^-1 r by explicit forward substitution (no torch.linalg; PyTorch 1.9.1). Per label
    (e_i^2 / 2 + log L_ii) * stop_grad(L_ii^2)^beta; mean over labels and batch, as beta_nll_loss.
    Sum over labels with beta = 0: 0.5 r^T Sigma^-1 r + 0.5 log det Sigma, Sigma = L L^T.
    """
    if y.shape[1] != 3 or out.shape[1] != 9:
        raise ValueError(f"beta_nll_full_loss expects K = 3 (out 9, y 3), got out {out.shape}, y {y.shape}")
    r = y - out[:, :3]
    log_d = out[:, 3:6]
    inv_d = torch.exp(-log_d)
    l21, l31, l32 = out[:, 6], out[:, 7], out[:, 8]
    e1 = r[:, 0] * inv_d[:, 0]
    e2 = (r[:, 1] - l21 * e1) * inv_d[:, 1]
    e3 = (r[:, 2] - l31 * e1 - l32 * e2) * inv_d[:, 2]
    e = torch.stack([e1, e2, e3], dim=1)
    nll = 0.5 * e ** 2 + log_d
    if beta != 0.0:
        nll = nll * torch.exp(2.0 * beta * log_d).detach()
    return nll.mean()


def split_output(out, n_targets, heteroscedastic, full_cov=False):
    """Return (mu, sigma or None, cov or None), standardized units.

    diag: sigma = exp(log_sigma); full (M14.5): sigma = sqrt(diag Sigma), cov = Sigma (B, K, K).
    """
    if not heteroscedastic:
        return out, None, None
    if full_cov:
        mu, _, cov = network.full_covariance_from_output(out, n_targets)
        return mu, torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2)), cov
    return out[:, :n_targets], torch.exp(out[:, n_targets:]), None


def make_loader(ds, sampler):
    return DataLoader(ds, sampler=sampler, batch_size=None, num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=False, prefetch_factor=PREFETCH_FACTOR)


@torch.no_grad()
def predict(model, loader, device):
    # Returns (pred, emb, y) as before; heteroscedastic models return (mu, emb, y, sigma);
    # full-covariance models (M14.5) return (mu, emb, y, sigma, cov).
    model.eval()
    hetero = bool(getattr(model, "heteroscedastic", False))
    full_cov = hetero and getattr(model, "covariance", "diag") == "full"
    preds, embs, ys, sigmas, covs = [], [], [], [], []
    for xb, yb in loader:
        out, emb = model(xb.to(device, non_blocking=True).float(), return_embedding=True)
        pred, sigma, cov = split_output(out, yb.shape[1], hetero, full_cov)
        preds.append(pred.cpu().numpy())
        embs.append(emb.cpu().numpy())
        ys.append(yb.numpy())
        if hetero:
            sigmas.append(sigma.cpu().numpy())
        if full_cov:
            covs.append(cov.float().cpu().numpy())
    res = (np.concatenate(preds), np.concatenate(embs), np.concatenate(ys))
    if full_cov:
        return res + (np.concatenate(sigmas), np.concatenate(covs).astype(np.float32))
    return res + (np.concatenate(sigmas),) if hetero else res


def label_metrics(pred, y):
    f = pred.astype(np.float64)
    t = y.astype(np.float64)
    mse = np.mean((f - t) ** 2, axis=0)
    fc = f - f.mean(axis=0)
    tc = t - t.mean(axis=0)
    slope = (fc * tc).sum(axis=0) / (fc * fc).sum(axis=0)  # OLS slope of theta regressed on f
    return mse, slope


def check_against_hdf5(h5_path, ds, row_ids, split):
    import h5py
    pos = np.random.default_rng(CHECK_SEED[split]).choice(len(ds), size=min(CHECK_N, len(ds)), replace=False)
    X_mm, y_mm = ds[pos]
    X_mm = X_mm.numpy().astype(np.float64)
    y_mm = y_mm.numpy()
    rows = row_ids[pos]
    order = np.argsort(rows)
    with h5py.File(h5_path, "r") as f:
        X_sorted = f["X"][rows[order]]
        y_sorted = f["y"][rows[order]]
    X_h5 = np.empty_like(X_sorted)
    X_h5[order] = X_sorted
    y_h5 = np.empty_like(y_sorted)
    y_h5[order] = y_sorted
    x_ref = X_h5.astype(np.float64)
    tol = 0.5 * np.spacing(np.abs(x_ref).astype(np.float16)).astype(np.float64) + 1e-7
    excess = np.abs(X_mm - x_ref) - tol
    y_ref = ((y_h5.astype(np.float32) - ds.y_mean) / (ds.y_std + 1e-8)).astype(np.float32)
    ok_x = bool(np.all(np.isfinite(X_mm)) and np.all(excess <= 0.0))
    ok_y = bool(np.array_equal(y_mm, y_ref))
    res = {"split": split, "n": int(pos.size), "seed": CHECK_SEED[split], "x_ok": ok_x, "y_ok": ok_y,
           "max_abs_dx": float(np.max(np.abs(X_mm - x_ref))), "max_excess_over_tol": float(np.max(excess))}
    print(f"[hdf5 check] {res}", flush=True)
    if not (ok_x and ok_y):
        raise SystemExit(f"fp16 memmap does not match HDF5 for split {split}")
    return res


def save_predictions(path, split, pred, emb, y, idx, common, sigma=None, cov=None):
    # Same schema as scripts/predict_cnn_hdf5.py: standardized pred/y, original HDF5 row ids, ascending.
    # Heteroscedastic models (M14.1) add sigma_<split> (standardized); pred_<split> is mu.
    # Full-covariance models (M14.5) also add cov_<split> (N, K, K) float32, standardized.
    payload = dict(common)
    payload["available_splits"] = np.array([split])
    payload[f"pred_{split}"] = pred
    payload[f"emb_{split}"] = emb
    payload[f"y_{split}"] = y
    payload[f"idx_{split}"] = idx
    if sigma is not None:
        payload[f"sigma_{split}"] = sigma
    if cov is not None:
        payload[f"cov_{split}"] = cov
    with open(path, "xb") as fh:
        np.savez_compressed(fh, **payload)
    print(f"saved {path}", flush=True)


def atomic_save(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cfg = json.loads(args.config.read_text())
    dcfg, tcfg, mcfg = cfg["dataset"], cfg["training"], cfg["model"]
    dataset_id = dcfg["dataset_id"]
    tag = cfg["outputs"]["checkpoint_tag"]
    seed = int(tcfg["seed"])
    max_epochs = int(tcfg["max_epochs"])
    sched = tcfg["schedule"]
    peak_lr = float(sched["peak_lr"])
    if ("warmup_epochs" in sched) == ("warmup_steps" in sched):
        raise SystemExit("schedule needs exactly one of warmup_epochs, warmup_steps")
    warmup_epochs = int(sched["warmup_epochs"]) if "warmup_epochs" in sched else None
    weight_decay = float(tcfg["weight_decay"])
    off_fraction = float(tcfg.get("dropout_off_fraction", 0.0))
    tr_every = int(tcfg.get("monitor_train40k_every", 1))
    if sched.get("type") != "cosine":
        raise SystemExit(f"Only the cosine schedule is implemented, got {sched.get('type')}")
    hetero = bool(mcfg.get("heteroscedastic", False))
    covariance = mcfg.get("covariance", "diag")
    if covariance not in ("diag", "full") or (covariance == "full" and not hetero):
        raise SystemExit(f"model.covariance {covariance!r}: expected diag, or full with model.heteroscedastic = true")
    full_cov = covariance == "full"
    loss_name = tcfg.get("loss", "MSELoss")
    if int(tcfg["batch_size"]) != BATCH:
        raise SystemExit("Expected batch_size 256")
    expected_loss = "beta_nll_full" if full_cov else ("beta_nll" if hetero else "MSELoss")
    if loss_name != expected_loss:
        raise SystemExit(f"loss {loss_name!r}: expected {expected_loss} for heteroscedastic={hetero}, "
                         f"covariance={covariance}")
    beta = float(tcfg["beta_nll"]) if hetero else None
    if float(tcfg["learning_rate"]) != peak_lr:
        raise SystemExit("training.learning_rate must equal schedule.peak_lr")
    if tr_every < 1:
        raise SystemExit(f"monitor_train40k_every must be >= 1, got {tr_every}")
    if not 0.0 <= off_fraction < 1.0:
        raise SystemExit(f"dropout_off_fraction must be in [0, 1), got {off_fraction}")
    if cfg.get("input_normalization", {}).get("enabled", False):
        raise SystemExit("memmap export holds HDF5 values; config normalizes on the fly")
    off_epoch = dropout_off_epoch(off_fraction, max_epochs)
    acfg = tcfg.get("noise_augmentation") or {}
    use_aug = bool(acfg.get("enabled", False))

    data_root = resolve_data_root(cli_data_root=args.data_root, config_data_root=cfg.get("data_root"))
    ckpt_dir = data_root / "models" / "checkpoints" / "M13_3"
    res_dir = data_root / "results" / "M13_3" / tag
    p_last = ckpt_dir / f"{tag}_last.pt"
    p_final = ckpt_dir / f"{tag}_final.pt"
    p_best = ckpt_dir / f"{tag}_bestval.pt"
    p_info = res_dir / "run_info.json"
    p_hist = res_dir / "history.npz"
    p_val = res_dir / "val_predictions_embeddings.npz"
    p_tr = res_dir / "train40k_predictions.npz"
    resumable = {p_last, p_best, p_info}
    for p in (p_last, p_final, p_best, p_info, p_hist, p_val, p_tr):
        if p.exists() and not (args.resume and p in resumable):
            raise SystemExit(f"Output exists, refusing to overwrite: {p}")
    if args.resume and not p_last.exists():
        raise SystemExit(f"--resume given but {p_last} does not exist")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data: train/val membership from the split file; cal/test indices are never loaded.
    split_path = resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id, file_name=dcfg["split_file"])
    stats_path = resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id,
                                            file_name=dcfg["label_stats_file"])
    h5_path = resolve_processed_artifact(data_root=data_root, dataset_id=dataset_id, file_name=dcfg["dataset_file"])
    with np.load(split_path) as sp:
        split_idx = {s: sp[f"{s}_idx"].astype(np.int64) for s in ("train", "val")}
    stats = np.load(stats_path)
    y_mean = stats["y_mean"].astype(np.float32)
    y_std = stats["y_std"].astype(np.float32)
    label_names = [str(x) for x in stats["label_names"].tolist()] if "label_names" in stats.files else DEFAULT_LABELS
    if label_names != DEFAULT_LABELS:
        raise SystemExit(f"Unexpected label order {label_names}")
    mm_dir = dataset_processed_dir(data_root, dataset_id) / "fp16_memmap"
    manifest = mm_dir / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"Missing {manifest}")
    row_ids, ds = {}, {}
    for s in ("train", "val"):
        row_ids[s] = np.load(mm_dir / f"row_ids_{s}.npy")
        if not np.array_equal(row_ids[s], np.sort(split_idx[s])):
            raise SystemExit(f"row_ids_{s}.npy does not match sort({s}_idx) of the split file")
        ds[s] = FP16MemmapBatchDataset(mm_dir, s, y_mean, y_std)

    if h5_path.exists():
        hdf5_check = [check_against_hdf5(h5_path, ds[s], row_ids[s], s) for s in ("train", "val")]
    else:
        print(f"WARNING: HDF5 {h5_path} not found; skipping memmap-vs-HDF5 check", flush=True)
        hdf5_check = None

    train_sampler, subset_pos = make_train_sampler(len(ds["train"]), tcfg, seed)
    if use_aug:
        aug_dir, bank_start, bank_k, aug_info = load_noise_aug(acfg, dataset_processed_dir(data_root, dataset_id),
                                                               mm_dir, row_ids["train"])
        ds_aug = NoiseAugTrainDataset(mm_dir, aug_dir, y_mean, y_std, bank_start, bank_k,
                                      sign_flip=bool(acfg.get("sign_flip", False)),
                                      time_reverse=bool(acfg.get("time_reverse", False)))
        train_loader = make_loader(ds_aug, SeededBatchSampler(train_sampler, seed))
        print(f"noise augmentation: K={bank_k}, {aug_info['n_psd_keys_train']} train keys, {aug_dir}", flush=True)
    else:
        aug_info = None
        train_loader = make_loader(ds["train"], train_sampler)
    val_loader = make_loader(ds["val"], SequentialBatchSampler(len(ds["val"]), BATCH))
    tr_pos = np.sort(np.random.default_rng(TRAIN_MONITOR_SEED).choice(len(ds["train"]), TRAIN_MONITOR_N,
                                                                        replace=False))
    tr_loader = make_loader(ds["train"], [tr_pos[i:i + BATCH] for i in range(0, tr_pos.size, BATCH)])

    steps_per_epoch = len(train_sampler)
    warmup_steps, total_steps = schedule_steps(tcfg, steps_per_epoch, max_epochs)

    model_kwargs = dict(mcfg["kwargs"])
    if hetero:
        # Stored in model_kwargs so predict_cnn_hdf5.py rebuilds the heteroscedastic head from the checkpoint.
        model_kwargs["heteroscedastic"] = True
    if full_cov:
        # Only for M14.5, so diag checkpoints keep their model_kwargs unchanged.
        model_kwargs["covariance"] = "full"
    tb_cfg = mcfg.get("template_bank")
    if tb_cfg is not None:
        # Architecture keys go to model_kwargs (rebuilt by predict_cnn_hdf5.py); "init" only here.
        model_kwargs["template_bank"] = {k: v for k, v in tb_cfg.items() if k != "init"}
    model = getattr(network, mcfg["class_name"])(**model_kwargs).to(device)
    if hetero and not getattr(model, "heteroscedastic", False):
        raise SystemExit(f"{mcfg['class_name']} has no heteroscedastic option")
    if full_cov and getattr(model, "covariance", "diag") != "full":
        raise SystemExit(f"{mcfg['class_name']} has no full covariance option")
    tb_info = None
    if tb_cfg is not None:
        if getattr(model, "template_bank", None) is None:
            raise SystemExit(f"{mcfg['class_name']} has no template_bank option")
        # Deterministic; on --resume the checkpoint state_dict below overwrites these values.
        tb_info = init_template_bank(model.template_bank, tb_cfg.get("init", {}),
                                     dataset_processed_dir(data_root, dataset_id), mm_dir, row_ids["train"])
        tb_info["architecture"] = model_kwargs["template_bank"]
        tb_info["n_parameters_bank"] = sum(p.numel() for p in model.template_bank.parameters())
        print(f"template bank: K={model.template_bank.n_templates}, L={model.template_bank.template_length}, "
              f"rows {[t['row_id'] for t in tb_info['templates']]}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(warmup_steps, total_steps))
    if full_cov:
        def criterion(out, y):
            return beta_nll_full_loss(out, y, beta)
    elif hetero:
        def criterion(out, y):
            return beta_nll_loss(out, y, beta)
    else:
        criterion = nn.MSELoss()
    model_config = {"class_name": mcfg["class_name"], "architecture_name": mcfg.get("architecture_name"),
                    "model_kwargs": model_kwargs, "input_normalization": cfg.get("input_normalization")}
    dropout_original = {n: float(m.p) for n, m in model.named_modules() if isinstance(m, nn.Dropout)}

    start_epoch, best_val, history, dropout_off = 0, float("inf"), {}, False
    if args.resume:
        ck = torch.load(p_last, map_location="cpu")
        if ck["config"] != cfg:
            raise SystemExit("Config differs from the one stored in the last checkpoint")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = int(ck["epoch"])
        best_val = float(ck["best_val_loss"])
        history = ck["history"]
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        if ck["dropout"]["off"]:
            disable_dropout(model)
            dropout_off = True
        print(f"resumed from {p_last} after {start_epoch} completed epochs", flush=True)

    try:
        import h5py
        h5py_version = h5py.__version__
    except ImportError:
        h5py_version = None
    hashed = [Path(__file__).resolve(), ROOT / "src/models/network.py", ROOT / "src/models/memmap_dataset.py",
              ROOT / "src/paths.py", args.config.resolve()]
    run_info = {
        "tag": tag,
        "config_path": str(args.config.resolve()),
        "config": cfg,
        "resume": bool(args.resume),
        "start_epoch": start_epoch,
        "sha256": {str(p.relative_to(ROOT)) if ROOT in p.parents else str(p): sha256(p) for p in hashed},
        "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
        "data": {"memmap_dir": str(mm_dir), "hdf5_path": str(h5_path), "split_file": str(split_path),
                 "label_stats_file": str(stats_path), "n_train": len(ds["train"]), "n_val": len(ds["val"]),
                 "label_names": label_names, "hdf5_check": hdf5_check,
                 "train_monitor": {"n": TRAIN_MONITOR_N, "seed": TRAIN_MONITOR_SEED}},
        "procedure": {"optimizer": {"name": "AdamW", "lr": peak_lr, "weight_decay": weight_decay},
                      "loss": "MSELoss (standardized labels)", "batch_size": BATCH, "drop_last": True,
                      "sampler": {"name": type(train_sampler).__name__, "seed": seed},
                      "schedule": {"type": "linear warmup + cosine to 0, per step", "peak_lr": peak_lr,
                                   "warmup_epochs": warmup_epochs, "warmup_steps": warmup_steps,
                                   "steps_per_epoch": steps_per_epoch, "total_steps": total_steps},
                      "max_epochs": max_epochs, "early_stopping": False, "registered_model": "last epoch",
                      "dropout_off_fraction": off_fraction, "dropout_off_epoch_0based": off_epoch,
                      "dropout_original_p": dropout_original,
                      "loader": {"num_workers": NUM_WORKERS, "pin_memory": True, "prefetch_factor": PREFETCH_FACTOR},
                      "cudnn": {"benchmark": True, "deterministic": False}},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "h5py": h5py_version},
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if use_aug:
        run_info["data"]["noise_augmentation"] = aug_info
    if hetero:
        run_info["procedure"]["loss"] = (f"beta_nll (beta={beta}, standardized labels; model outputs [mu, log_sigma], "
                                         f"log_sigma clamped to [{type(model).LOG_SIGMA_MIN}, "
                                         f"{type(model).LOG_SIGMA_MAX}]; logged mse/slope on mu)")
        run_info["procedure"]["heteroscedastic"] = True
    if full_cov:
        run_info["procedure"]["loss"] = (f"beta_nll_full (beta={beta}, standardized labels; model outputs "
                                         f"[mu, log L_ii, L21, L31, L32], Sigma = L L^T, log L_ii clamped to "
                                         f"[{type(model).LOG_SIGMA_MIN}, {type(model).LOG_SIGMA_MAX}]; "
                                         f"logged mse/slope on mu)")
        run_info["procedure"]["covariance"] = "full"
    if tb_info is not None:
        run_info["procedure"]["template_bank"] = tb_info
    if subset_pos is not None:
        # Hash of the HDF5 row ids (sorted int64 bytes), so the chosen rows can be checked without the rng.
        rows = np.sort(row_ids["train"][subset_pos]).astype(np.int64)
        run_info["data"]["train_subset"] = {"n": int(subset_pos.size), "seed": int(tcfg["train_subset"]["seed"]),
                                            "row_ids_sha256": hashlib.sha256(rows.tobytes()).hexdigest()}
    info_path = p_info if not args.resume else res_dir / f"run_info_resume_e{start_epoch:03d}.json"
    with open(info_path, "x") as f:
        json.dump(run_info, f, indent=2)
    print(f"[{tag}] steps/epoch={steps_per_epoch} warmup_steps={warmup_steps} total_steps={total_steps} "
          f"dropout_off_epoch={off_epoch} -> {info_path}", flush=True)

    def base_ckpt(epochs_done):
        return {"epoch": epochs_done, "model_state_dict": model.state_dict(),
                "y_mean": torch.as_tensor(y_mean), "y_std": torch.as_tensor(y_std),
                "model_config": model_config, "config": cfg, "checkpoint_tag": tag,
                "dropout": {"off": dropout_off, "off_epoch_0based": off_epoch, "original_p": dropout_original}}

    def fmt(a):
        return "[" + " ".join(f"{v:.4f}" for v in a) + "]"

    last_eval = None
    t_run0 = time.time()
    for epoch in range(start_epoch, max_epochs):
        if off_epoch is not None and epoch >= off_epoch and not dropout_off:
            names = disable_dropout(model)
            dropout_off = True
            print(f"epoch {epoch:02d}: dropout p=0.0 in {len(names)} nn.Dropout modules: {names}", flush=True)

        train_sampler.set_epoch(epoch)
        model.train()
        wait, n_seen, lr_first, lr_last = 0.0, 0, None, None
        loss_sum = torch.zeros((), device=device)
        t0 = time.perf_counter()
        it = iter(train_loader)
        while True:
            tw = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            wait += time.perf_counter() - tw
            if use_aug:
                xc, noise, sign, rev, yb = batch
                xb = augment_on_gpu(xc, noise, sign, rev, device)
            else:
                xb, yb = batch
                xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True)
            lr_last = optimizer.param_groups[0]["lr"]
            lr_first = lr_last if lr_first is None else lr_first
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach() * xb.shape[0]
            n_seen += xb.shape[0]
            if scheduler.last_epoch >= total_steps:
                break
        del it
        train_loss = float(loss_sum) / n_seen
        t_train = time.perf_counter() - t0
        expected_step = min((epoch + 1) * steps_per_epoch, total_steps)
        if scheduler.last_epoch != expected_step:
            raise RuntimeError(f"scheduler step {scheduler.last_epoch} != {expected_step}")

        t0 = time.perf_counter()
        val_out = predict(model, val_loader, device)
        t_val = time.perf_counter() - t0
        # train40k is always evaluated at the last epoch (its predictions are saved).
        do_tr = (epoch + 1) % tr_every == 0 or epoch == max_epochs - 1
        val_mse, val_slope = label_metrics(val_out[0], val_out[2])
        if do_tr:
            t0 = time.perf_counter()
            tr_out = predict(model, tr_loader, device)
            t_tr = time.perf_counter() - t0
            tr_mse, tr_slope = label_metrics(tr_out[0], tr_out[2])
            last_eval = (val_out, tr_out)
        else:
            tr_mse = tr_slope = np.full_like(val_mse, np.nan)
            t_tr = float("nan")
        val_loss = float(val_mse.mean())

        rec = {"epoch": epoch, "lr_first": lr_first, "lr_last": lr_last, "train_loss": train_loss,
               "val_loss": val_loss, "val_mse": val_mse, "val_slope": val_slope, "train40k_mse": tr_mse,
               "train40k_slope": tr_slope, "train_time": t_train, "val_time": t_val, "train40k_time": t_tr,
               "data_wait": wait, "dropout_off": dropout_off}
        for k, v in rec.items():
            history.setdefault(k, []).append(v)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            ck = base_ckpt(epoch + 1)
            ck["best_val_loss"] = best_val
            atomic_save(ck, p_best)
        ck = base_ckpt(epoch + 1)
        ck.update(optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                  best_val_loss=best_val, history=history,
                  rng={"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                       "numpy": np.random.get_state(), "python": random.getstate()})
        atomic_save(ck, p_last)

        print(f"epoch {epoch:02d}/{max_epochs - 1} | lr {lr_first:.3e}->{lr_last:.3e} | train {train_loss:.5f} | "
              f"val {val_loss:.5f}{' *' if improved else ''} mse {fmt(val_mse)} slope {fmt(val_slope)} | "
              f"{f'tr40k mse {fmt(tr_mse)} slope {fmt(tr_slope)}' if do_tr else 'tr40k skipped'} | "
              f"dropout {'off' if dropout_off else 'on'} | "
              f"t_train {t_train:.0f}s wait {wait:.1f}s t_val {t_val:.0f}s t_tr40k {t_tr:.0f}s | "
              f"elapsed {(time.time() - t_run0) / 60:.1f} min", flush=True)

    # Registered model: last epoch.
    ck = base_ckpt(max_epochs)
    ck.update(best_val_loss=best_val, history=history)
    with open(p_final, "xb") as fh:
        torch.save(ck, fh)
    print(f"saved {p_final}", flush=True)
    np.savez(p_hist, **{k: np.asarray(v) for k, v in history.items()}, label_names=np.array(label_names))
    print(f"saved {p_hist}", flush=True)

    if last_eval is None:  # resumed after the last epoch had already completed
        last_eval = (predict(model, val_loader, device), predict(model, tr_loader, device))
    common = {"y_mean": y_mean, "y_std": y_std, "label_names": np.array(label_names),
              "checkpoint_file": str(p_final), "dataset_path": str(h5_path), "split_path": str(split_path),
              "label_stats_path": str(stats_path), "model_config": np.array(model_config, dtype=object),
              "input_normalization": np.array(cfg.get("input_normalization"), dtype=object),
              "memmap_dir": str(mm_dir)}
    (vp, ve, vy, *vs), (tp, te, ty, *ts) = last_eval
    save_predictions(p_val, "val", vp, ve, vy, row_ids["val"], common, sigma=vs[0] if vs else None,
                     cov=vs[1] if len(vs) > 1 else None)
    save_predictions(p_tr, TRAIN_MONITOR_SPLIT, tp, te, ty, row_ids["train"][tr_pos], common,
                     sigma=ts[0] if ts else None, cov=ts[1] if len(ts) > 1 else None)


if __name__ == "__main__":
    main()
