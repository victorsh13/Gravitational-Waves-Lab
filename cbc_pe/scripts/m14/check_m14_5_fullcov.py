"""M14.5 checks of the full-covariance head and loss (CPU, random data, no data files; seconds).

(a) with L21 = L31 = L32 = 0, beta_nll_full_loss == beta_nll_loss (beta 0.5 and 0; atol 1e-6);
(b) with beta = 0, 3 * beta_nll_full_loss == mean of the multivariate Gaussian NLL
    0.5 r^T Sigma^-1 r + 0.5 log det Sigma, computed in float64 with explicit inverse and determinant
    (atol 1e-5); network.full_covariance_from_output gives the same Sigma = L L^T;
(c) finite gradients of beta_nll_full_loss (float32);
(d) M14.5a config, B = 2: output (2, 9) finite, Sigma positive definite, state_dict keys equal to M14.1
    (shapes equal except head.*); for the same torch seed every non-head tensor and the head [mu, log L_ii]
    rows equal the M14.1 initial weights;
(e) M14.5b config (template bank), B = 2: output (2, 9) finite, Sigma positive definite, state_dict keys
    equal to M14.2 (shapes equal except head.*).

Usage (from the repository root): python scripts/m14/check_m14_5_fullcov.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models import network  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_m13_3", ROOT / "scripts/m13/train_m13_3.py")
trainer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainer)

CFG = ROOT / "configs/experiments/m14"
CFG_M14_1 = CFG / "train_m14_1_4aG3T_hetero_s123.json"
CFG_M14_2 = CFG / "train_m14_2_4aG3T_tbank_s123.json"
CFG_M14_5A = CFG / "train_m14_5a_4aG3T_fullcov_s123.json"
CFG_M14_5B = CFG / "train_m14_5b_4aG3T_tbank_fullcov_s123.json"

results = {}


def report(name, ok, detail=""):
    results[name] = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def random_out(rng, b, off_zero=False, dtype=torch.float64):
    mu = rng.normal(size=(b, 3))
    log_d = rng.uniform(-1.0, 1.0, size=(b, 3))
    off = np.zeros((b, 3)) if off_zero else rng.normal(scale=0.7, size=(b, 3))
    out = torch.tensor(np.concatenate([mu, log_d, off], axis=1), dtype=dtype)
    y = torch.tensor(mu + rng.normal(scale=1.5, size=(b, 3)), dtype=dtype)
    return out, y


def build_model(cfg_path):
    # Same model_kwargs construction as train_m13_3.py.
    mcfg = json.loads(cfg_path.read_text())["model"]
    kw = dict(mcfg["kwargs"])
    if mcfg.get("heteroscedastic", False):
        kw["heteroscedastic"] = True
    if mcfg.get("covariance", "diag") == "full":
        kw["covariance"] = "full"
    tb = mcfg.get("template_bank")
    if tb is not None:
        kw["template_bank"] = {k: v for k, v in tb.items() if k != "init"}
    n_samples = int(tb["n_samples"]) if tb is not None else 16384
    return getattr(network, mcfg["class_name"])(**kw), int(kw["n_detectors"]), n_samples


def check_forward(name, cfg_path, ref_path, check_init):
    torch.manual_seed(0)
    model, n_det, n_samples = build_model(cfg_path)
    torch.manual_seed(0)
    ref, _, _ = build_model(ref_path)
    model.eval()
    x = torch.randn(2, n_det, n_samples)
    with torch.no_grad():
        out = model(x)
        _, _, cov = network.full_covariance_from_output(out, 3)
    eig = np.linalg.eigvalsh(cov.double().numpy())
    sd, sd_ref = model.state_dict(), ref.state_dict()
    keys_ok = list(sd) == list(sd_ref)
    shape_diff = sorted(k for k in sd if k in sd_ref and sd[k].shape != sd_ref[k].shape)
    ok = (tuple(out.shape) == (2, 9) and bool(torch.isfinite(out).all()) and bool((eig > 0).all())
          and keys_ok and all(k.startswith("head.") for k in shape_diff))
    detail = f"out {tuple(out.shape)}, min eig {eig.min():.3e}, keys equal {keys_ok}, shape diff {shape_diff}"
    if check_init:
        same = all(torch.equal(sd[k], sd_ref[k]) for k in sd if not k.startswith("head."))
        same_head = (torch.equal(sd["head.weight"][:6], sd_ref["head.weight"])
                     and torch.equal(sd["head.bias"][:6], sd_ref["head.bias"]))
        ok = ok and same and same_head
        detail += f", init equal non-head {same}, head rows 0-5 {same_head}"
    report(name, ok, detail)


def main():
    rng = np.random.default_rng(145000)

    # (a) diagonal L: same loss as M14.1.
    out, y = random_out(rng, 64, off_zero=True)
    diffs = [abs(float(trainer.beta_nll_full_loss(out, y, b)) - float(trainer.beta_nll_loss(out[:, :6], y, b)))
             for b in (0.5, 0.0)]
    report("(a) diag L == beta_nll_loss", max(diffs) <= 1e-6, f"max |diff| {max(diffs):.2e}")

    # (b) beta = 0: multivariate Gaussian NLL (without the 1.5 log 2 pi constant), float64 reference.
    out, y = random_out(rng, 64)
    o, r = out.numpy(), (y - out[:, :3]).numpy()
    ref = []
    sig_ref = np.empty((len(o), 3, 3))
    for n in range(len(o)):
        L = np.diag(np.exp(o[n, 3:6]))
        L[1, 0], L[2, 0], L[2, 1] = o[n, 6], o[n, 7], o[n, 8]
        S = L @ L.T
        sig_ref[n] = S
        ref.append(0.5 * r[n] @ np.linalg.inv(S) @ r[n] + 0.5 * np.log(np.linalg.det(S)))
    d_loss = abs(3.0 * float(trainer.beta_nll_full_loss(out, y, 0.0)) - float(np.mean(ref)))
    _, _, sig = network.full_covariance_from_output(out, 3)
    d_sig = float(np.max(np.abs(sig.numpy() - sig_ref)))
    report("(b) beta=0 == MVN NLL", d_loss <= 1e-5 and d_sig <= 1e-10,
           f"|diff| loss {d_loss:.2e}, max |diff| Sigma {d_sig:.2e}")

    # (c) finite gradients (float32, as in training).
    out, y = random_out(rng, 64, dtype=torch.float32)
    out.requires_grad_(True)
    grads_ok = []
    for b in (0.5, 0.0):
        out.grad = None
        loss = trainer.beta_nll_full_loss(out, y, b)
        loss.backward()
        grads_ok.append(bool(torch.isfinite(loss)) and bool(torch.isfinite(out.grad).all())
                        and bool((out.grad[:, 6:] != 0).any()))
    report("(c) finite gradients", all(grads_ok), f"beta 0.5/0: {grads_ok}")

    check_forward("(d) M14.5a forward", CFG_M14_5A, CFG_M14_1, check_init=True)
    check_forward("(e) M14.5b forward", CFG_M14_5B, CFG_M14_2, check_init=False)

    print("ALL PASS" if all(results.values()) else f"FAILED: {[k for k, v in results.items() if not v]}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
