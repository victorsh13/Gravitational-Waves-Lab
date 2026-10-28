"""M13.3.0: does dropout explain the train-set slope > 1 of theta on f for M11? (HIP-DROPOUT)

Forward passes of the frozen M11 checkpoint on a train-only subset of G1, in eval mode and in
train mode (dropout active, K seeded realizations). All metrics in standardized label units,
labels ordered [chirp_mass, total_mass, chi_eff]. Val/cal/test rows are never read (X or y).
Output: $CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M13_3_0_dropout/analysis.json
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_data_root  # noqa: E402
from src.models.network import SimpleCNN_ResidualDilated  # noqa: E402
from scripts.m12.g1_npe_contract import (  # noqa: E402  (read-only reuse; file is hash-pinned)
    DATASET, LABELS, digest, load_contract, require, write_json)

CONFIG = ROOT / 'configs/experiments/train_m11_final_G1_500k.json'
CKPT_NAME = ('M11_final_G1_500k_SimpleCNN_ResidualDilated_M11_final_G1_inputzscore_resdilated_'
             'emb64_d124_train400k_MSELoss_seed123_checkpoint.pt')
OUT_NAME = 'M13_3_0_dropout'

CHUNK_ROWS = 64
CHUNK_SEED = 133001
DROP_SEED0 = 133100  # torch.manual_seed(DROP_SEED0 + 10*batch_index + k) before every train-mode pass
K = 4
BATCH = 256
EQUIV_TOL = 1e-6
# Expected dropout rates (M11 config): conv blocks and residual blocks 0.05, embedding 0.1.
P_CONV, P_DENSE = 0.05, 0.1
# Readout thresholds (protocol).
DROP_BAND = (0.985, 1.015)
EVAL_MIN = 1.04
REJECT_MIN = 1.04
NORM_MAX = 0.05
NEXT_SUSPECT = ('parada temprana / no convergencia de la cabeza (detener el descenso de gradiente '
                'equivale aproximadamente a ridge y contrae: pendiente > 1); se pone a prueba con R1 '
                '(planificador de presupuesto fijo, dropout de M11)')


def ols(x, y):
    """OLS of y on x (with intercept): (slope, intercept)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    xc = x - x.mean()
    slope = float(xc @ (y - y.mean()) / (xc @ xc))
    return slope, float(y.mean() - slope * x.mean())


def ls_head(e, theta):
    """Least-squares linear head with intercept; returns W of shape (emb+1, 3), bias last."""
    a = np.hstack([e, np.ones((e.shape[0], 1))])
    w, *_ = np.linalg.lstsq(a, theta, rcond=None)
    return w


def apply_head(w, e):
    return e @ w[:-1] + w[-1]


def load_checkpoint(path):
    try:
        obj = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch < 1.13 (VM: 1.9.1) has no weights_only
        obj = torch.load(str(path), map_location='cpu')
    require(isinstance(obj, dict) and all(k in obj for k in ('model_state_dict', 'y_mean', 'y_std')),
            'Checkpoint lacks model_state_dict / y_mean / y_std')
    return obj


def module_inventory(model, kwargs):
    """Abort on any module with running statistics; list norms and dropouts."""
    norms, drops = [], []
    for name, m in model.named_modules():
        running = [b for b, _ in m.named_buffers(recurse=False) if b.startswith('running_')]
        require(not isinstance(m, nn.modules.batchnorm._BatchNorm) and not running
                and not getattr(m, 'track_running_stats', False),
                f'Module with running statistics found: {name} ({type(m).__name__})')
        if isinstance(m, nn.GroupNorm):
            norms.append(dict(name=name, type='GroupNorm', num_groups=m.num_groups,
                              num_channels=m.num_channels))
        elif isinstance(m, (nn.LayerNorm, nn.modules.instancenorm._InstanceNorm)):
            norms.append(dict(name=name, type=type(m).__name__))
        if isinstance(m, nn.modules.dropout._DropoutNd):
            drops.append(dict(name=name, type=type(m).__name__, p=float(m.p)))
    for d in drops:
        expected = P_DENSE if d['name'].startswith('embedding_layer.') else P_CONV
        require(d['type'] == 'Dropout' and abs(d['p'] - expected) < 1e-12,
                f"Unexpected dropout {d['name']}: {d['type']} p={d['p']} (expected {expected})")
    require(abs(kwargs['dropout_conv'] - P_CONV) < 1e-12 and
            abs(kwargs['dropout_dense'] - P_DENSE) < 1e-12, 'Config dropout rates changed')
    return dict(normalizations=norms, dropouts=drops, n_dropouts=len(drops),
                running_stats_modules=[])


def set_mode(model, dropout_train, rest_train):
    model.train(rest_train)
    for m in model.modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.train(dropout_train)


def select_rows(is_train, n_total, n_rows):
    """Random 64-row chunks (seeded), then their train rows in ascending order, up to n_rows.

    Chunks are drawn in permutation order until n_rows train rows are reached; the last drawn
    chunk contributes only its first train rows needed to reach n_rows exactly.
    """
    n_chunks = -(-n_total // CHUNK_ROWS)
    perm = np.random.default_rng(CHUNK_SEED).permutation(n_chunks)
    chosen, rows, count = [], [], 0
    for c in perm:
        r = np.flatnonzero(is_train[c * CHUNK_ROWS:(c + 1) * CHUNK_ROWS]) + c * CHUNK_ROWS
        if r.size == 0:
            continue
        r = r[:n_rows - count]
        chosen.append(int(c))
        rows.append(r)
        count += r.size
        if count == n_rows:
            break
    rows = np.sort(np.concatenate(rows)).astype(np.int64)
    return rows, sorted(chosen), n_chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--n-rows', type=int, default=10000)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    data_root = resolve_data_root(cli_data_root=args.data_root, config_data_root=cfg.get('data_root'))
    out_dir = data_root / 'results' / DATASET / OUT_NAME
    require(not out_dir.exists(), f'Output directory already exists: {out_dir}')
    checks = []

    # Frozen G1 contract: config, pinned split/label-stats hashes, HDF5 header and membership.
    paths, splits, y_mean, y_std, identity = load_contract(data_root)
    checks.append('frozen G1 contract (g1_npe_contract.load_contract)')
    n_total = sum(v.size for v in splits.values())
    require(0 < args.n_rows <= splits['train'].size, f'--n-rows must be in [1, {splits["train"].size}]')

    ckpt_path = data_root / 'models' / 'checkpoints' / DATASET / CKPT_NAME
    require(ckpt_path.exists(), f'Checkpoint not found: {ckpt_path}')
    ckpt_sha = digest(ckpt_path)
    ckpt = load_checkpoint(ckpt_path)
    c_mean = ckpt['y_mean'].numpy().astype(np.float64)
    c_std = ckpt['y_std'].numpy().astype(np.float64)
    require(np.allclose(c_mean, y_mean, rtol=1e-6, atol=0.0) and
            np.allclose(c_std, y_std, rtol=1e-6, atol=1e-8),
            'Checkpoint y_mean/y_std differ from train-only label stats')
    checks.append('checkpoint y_mean/y_std equal train-only label stats (rtol 1e-6; std atol 1e-8)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    kwargs = cfg['model']['kwargs']
    model = SimpleCNN_ResidualDilated(**kwargs)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    inventory = module_inventory(model, kwargs)
    checks.append('no module with running statistics; dropout rates match M11 config')
    w_head = np.vstack([model.head.weight.detach().cpu().numpy().T,
                        model.head.bias.detach().cpu().numpy()[None]]).astype(np.float64)

    is_train = np.zeros(n_total, dtype=bool)
    is_train[splits['train']] = True
    rows, chunks, n_chunks = select_rows(is_train, n_total, args.n_rows)
    require(rows.size == args.n_rows and is_train[rows].all() and np.unique(rows).size == rows.size,
            'Row selection is not N unique train rows')
    checks.append('selected rows are N unique train rows')
    rows_sha = hashlib.sha256(np.ascontiguousarray(rows, dtype='<i8').tobytes()).hexdigest()

    n = rows.size
    theta = np.empty((n, 3), np.float64)
    f_eval = np.empty((n, 3), np.float64)
    f_drop = np.empty((K, n, 3), np.float64)
    e_eval, e_drop, equiv = None, None, None
    with h5py.File(paths['dataset_file'], 'r') as h, torch.no_grad():
        x_ds, y_ds = h['X'], h['y']
        hdf5_chunks = list(x_ds.chunks) if x_ds.chunks is not None else None
        for start in range(0, n, BATCH):
            sl = slice(start, min(start + BATCH, n))
            idx = rows[sl]  # ascending: chunks are visited in increasing order, train rows only
            # G1 X is stored already z-scored per sample/detector (input_normalization.enabled=False).
            x = torch.from_numpy(np.ascontiguousarray(x_ds[idx], dtype=np.float32)).to(device)
            theta[sl] = (y_ds[idx].astype(np.float64) - y_mean) / y_std

            set_mode(model, dropout_train=False, rest_train=False)
            f, e = model(x, return_embedding=True)
            if e_eval is None:
                e_eval = np.empty((n, e.shape[1]), np.float64)
                e_drop = np.empty((K, n, e.shape[1]), np.float64)
                # Train mode with every Dropout in eval must reproduce model.eval().
                set_mode(model, dropout_train=False, rest_train=True)
                equiv = float((model(x) - f).abs().max().item())
                require(equiv <= EQUIV_TOL, f'Train mode changes more than dropout: max|diff|={equiv}')
                checks.append(f'train mode minus dropout == eval on first batch (max|diff| <= {EQUIV_TOL})')
            f_eval[sl], e_eval[sl] = f.cpu().numpy(), e.cpu().numpy()

            set_mode(model, dropout_train=True, rest_train=True)
            for k in range(K):
                # Distinct seed per (batch, realization): masks independent across batches (K < 10).
                torch.manual_seed(DROP_SEED0 + 10 * (start // BATCH) + k)
                f, e = model(x, return_embedding=True)
                f_drop[k, sl], e_drop[k, sl] = f.cpu().numpy(), e.cpu().numpy()
            print(f'[{sl.stop}/{n}]', flush=True)
    model.eval()
    require(all(np.isfinite(a).all() for a in (theta, f_eval, f_drop, e_eval, e_drop)),
            'Non-finite outputs')
    require(np.abs(apply_head(w_head, e_eval) - f_eval).max() < 1e-4, 'head(e_eval) != f_eval')
    checks.append('outputs finite; head(e_eval) reproduces f_eval')

    theta_k = np.tile(theta, (K, 1))
    f_drop_s = f_drop.reshape(K * n, 3)
    e_drop_s = e_drop.reshape(K * n, -1)
    w_ls_drop = ls_head(e_drop_s, theta_k)
    w_ls_eval = ls_head(e_eval, theta)
    f_ls_drop = apply_head(w_ls_drop, e_eval)
    f_ls_eval = apply_head(w_ls_eval, e_eval)

    metrics = {}
    for j, lab in enumerate(LABELS):
        b_eval, a_eval = ols(f_eval[:, j], theta[:, j])
        b_drop, a_drop = ols(f_drop_s[:, j], theta_k[:, j])
        dev = f_drop_s[:, j] - np.tile(f_eval[:, j], K)
        metrics[lab] = dict(
            slope_eval=b_eval, intercept_eval=a_eval, slope_drop=b_drop, intercept_drop=a_drop,
            var_ratio_drop_noise=float(dev.var() / f_eval[:, j].var()),
            mse_eval=float(np.mean((f_eval[:, j] - theta[:, j])**2)),
            mse_drop=float(np.mean((f_drop_s[:, j] - theta_k[:, j])**2)),
            ls_head_on_e_drop=dict(
                slope_theta_on_pred_e_eval=ols(f_ls_drop[:, j], theta[:, j])[0],
                rel_norm_vs_m11_head=float(np.linalg.norm(w_ls_drop[:, j] - w_head[:, j]) /
                                           np.linalg.norm(w_head[:, j]))),
            ls_head_on_e_eval=dict(
                slope_theta_on_pred_e_eval=ols(f_ls_eval[:, j], theta[:, j])[0],
                rel_norm_vs_m11_head=float(np.linalg.norm(w_ls_eval[:, j] - w_head[:, j]) /
                                           np.linalg.norm(w_head[:, j]))),
            head_contraction_c=ols(f_ls_eval[:, j], f_eval[:, j])[0])
    global_norms = {name: float(np.linalg.norm(w - w_head) / np.linalg.norm(w_head))
                    for name, w in (('ls_head_on_e_drop', w_ls_drop), ('ls_head_on_e_eval', w_ls_eval))}

    m = metrics
    supported = all(DROP_BAND[0] <= m[l]['slope_drop'] <= DROP_BAND[1] and m[l]['slope_eval'] >= EVAL_MIN
                    and m[l]['ls_head_on_e_drop']['rel_norm_vs_m11_head'] <= NORM_MAX for l in LABELS)
    rejected = any(m[l]['slope_drop'] >= REJECT_MIN for l in LABELS)
    verdict = 'APOYADA' if supported else 'RECHAZADA' if rejected else 'PARCIAL'
    readout = dict(
        verdict=verdict,
        criteria=dict(
            APOYADA=f'all labels: slope_drop in {list(DROP_BAND)}, slope_eval >= {EVAL_MIN}, '
                    f'rel_norm (LS on e_drop, per label, bias included) <= {NORM_MAX}',
            RECHAZADA=f'slope_drop >= {REJECT_MIN} in any label', PARCIAL='otherwise'),
        fraction_explained=({l: (None if m[l]['slope_eval'] == 1.0 else
                                 (m[l]['slope_eval'] - m[l]['slope_drop']) / (m[l]['slope_eval'] - 1.0))
                             for l in LABELS} if verdict == 'PARCIAL' else None),
        next_suspect=(None if verdict == 'APOYADA' else dict(
            text=NEXT_SUSPECT, head_contraction_c={l: m[l]['head_contraction_c'] for l in LABELS})))

    inventory['train_mode_effect'] = (
        f'train mode only activates dropout (p={P_CONV} in ConvBlock/ResidualDilatedBlock, '
        f'p={P_DENSE} in embedding_layer); verified on the first batch: train mode with all '
        f'Dropout in eval vs model.eval(), max|diff|={equiv} <= {EQUIV_TOL}')
    result = dict(
        experiment='M13.3.0', hypothesis='HIP-DROPOUT', labels=LABELS, units='standardized (train-only scaler)',
        provenance=dict(
            data_root=str(data_root), dataset_file=str(paths['dataset_file']), config=str(CONFIG),
            checkpoint=str(ckpt_path), checkpoint_sha256=ckpt_sha, checkpoint_epoch=ckpt.get('epoch'),
            script=str(Path(__file__).resolve()), script_sha256=digest(Path(__file__).resolve()),
            g1_contract=identity),
        settings=dict(
            n_rows=n, K=K, batch_size=BATCH, chunk_rows=CHUNK_ROWS, chunk_seed=CHUNK_SEED,
            dropout_seed_rule=(f'torch.manual_seed({DROP_SEED0} + 10*batch_index + k) before every '
                               f'train-mode forward; batch_index = 0, 1, ... (batches of {BATCH} '
                               f'over ascending row ids), k = 0..{K - 1}'),
            device=str(device), cuda_device=(torch.cuda.get_device_name(0) if device.type == 'cuda' else None),
            cudnn_deterministic=True, cudnn_benchmark=False, dtype='float32 forward, float64 metrics',
            versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                          h5py=h5py.__version__)),
        rows=dict(split='train', n_chunks_total=n_chunks, n_chunks_selected=len(chunks), chunk_ids=chunks,
                  hdf5_X_chunks=hdf5_chunks, row_ids_sha256=rows_sha, row_ids_dtype='<i8',
                  row_ids=rows.tolist()),
        module_inventory=inventory,
        checks_passed=checks,
        metrics=metrics,
        global_rel_norm_vs_m11_head=global_norms,
        readout=readout)

    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / 'analysis.json', result)
    print(json.dumps(dict(verdict=verdict, out=str(out_dir / 'analysis.json')), indent=2))


if __name__ == '__main__':
    main()
