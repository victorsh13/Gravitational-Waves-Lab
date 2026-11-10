"""M14.2 amendment 1: CPU check of init_template_bank without HDF5 (synthetic data in tmp_path, no GPU).

Run: python -m pytest tests/test_m14_2_template_bank_init.py
The detector of each template must be argmax_d sum(X_clean[pos, d]**2); templates must be unit-norm.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "m13" / "train_m13_3.py"

spec = importlib.util.spec_from_file_location("train_m13_3", TRAINER)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

N_ROWS, N_SAMPLES, K = 64, 2048, 8


def make_fixture(tmp_path):
    g3, base = tmp_path / "fp16_memmap", tmp_path
    g3.mkdir()
    (base / "memmap").mkdir()
    rng = np.random.default_rng(0)
    row_ids = np.sort(rng.choice(1000, N_ROWS, replace=False)).astype(np.int64)
    np.save(g3 / "row_ids_train.npy", row_ids)
    mc = rng.uniform(10.0, 60.0, N_ROWS)
    y = np.stack([mc, 2.4 * mc, rng.uniform(-0.8, 0.8, N_ROWS)], axis=1).astype(np.float32)
    np.save(g3 / "y_train.npy", y)
    # Dominant detector known per row: d = row % 3 (amplitude 3 vs 1 and 0.5 in the other two).
    dom = np.arange(N_ROWS) % 3
    t = np.arange(N_SAMPLES)
    merger = rng.integers(N_SAMPLES // 2, N_SAMPLES - 10, N_ROWS)
    Xc = np.lib.format.open_memmap(base / "memmap" / "X_clean_train.npy", "w+", np.float16, (N_ROWS, 3, N_SAMPLES))
    for r in range(N_ROWS):
        chirp = np.exp(-np.clip(merger[r] - t, 0, None) / 400.0) * (t <= merger[r]) \
            * np.sin(2 * np.pi * (30 + 0.05 * t) * t / 4096.0)
        amp = np.roll(np.array([3.0, 1.0, 0.5]), dom[r])
        Xc[r] = amp[:, None] * chirp[None]
    Xc.flush()
    del Xc
    sha = hashlib.sha256((g3 / "row_ids_train.npy").read_bytes()).hexdigest()
    (base / "memmap" / "manifest.json").write_text(json.dumps({"g3_memmap": {"row_ids_train_sha256": sha}}))
    return base, g3, row_ids, dom


def test_init_without_hdf5(tmp_path):
    base, g3, row_ids, dom = make_fixture(tmp_path)
    assert not list(tmp_path.rglob("*.h5"))
    bank = tr.network.TemplateBank(n_templates=K, template_length=N_SAMPLES, n_samples=N_SAMPLES, checkpoint=False)
    info = tr.init_template_bank(bank, {"seed": 142001, "n_chi_eff_bins": 4, "memmap_dir": "memmap"},
                                 base, g3, row_ids)
    temps = info["templates"]
    assert len(temps) == K and len({t["row_id"] for t in temps}) == K
    for t in temps:
        assert t["detector_index"] == dom[t["train_position"]]
        assert int(np.argmax(t["clean_energy"])) == t["detector_index"]
        assert "detector" not in t and "snr" not in t  # no detector_order key: index only
    assert info["detector_names"] is None
    assert "argmax sum X_clean^2" in info["rule"] and "enmienda 1" in info["detector_rule_amendment"]
    ti = bank.template_i.detach().numpy().astype(np.float64)
    tq = bank.template_q.detach().numpy().astype(np.float64)
    assert np.allclose(np.linalg.norm(ti, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(tq, axis=1), 1.0, atol=1e-5)
    # Deterministic: same rows and detectors on a second call.
    bank2 = tr.network.TemplateBank(n_templates=K, template_length=N_SAMPLES, n_samples=N_SAMPLES, checkpoint=False)
    assert tr.init_template_bank(bank2, {"seed": 142001}, base, g3, row_ids) == info
