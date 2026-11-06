"""M13.9/M13.10: deterministic CPU check of the optional train_m13_3.py keys (no data, no GPU).

Run: python tests/test_m13_9_10_trainer_options.py
(c) drives a copy of the training loop's stop logic with the trainer's own sampler and LR schedule;
the source assertions tie that copy to the lines in main().
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "experiments" / "m13"
TRAINER = ROOT / "scripts" / "m13" / "train_m13_3.py"
N_TRAIN = 400000
SEED = 123
SUBSET_SEED = 131001
TOTAL = 124960

spec = importlib.util.spec_from_file_location("train_m13_3", TRAINER)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def cfg(name):
    return json.loads((CFG_DIR / f"train_m13_{name}.json").read_text())["training"]


def resolved(tcfg, n_train=N_TRAIN):
    sampler, pos = tr.make_train_sampler(n_train, tcfg, int(tcfg["seed"]))
    spe = len(sampler)
    return (sampler, pos, spe) + tr.schedule_steps(tcfg, spe, int(tcfg["max_epochs"]))


def first_batches(sampler, epoch, k=3):
    sampler.set_epoch(epoch)
    it = iter(sampler)
    return [next(it) for _ in range(k)]


def check_a_default_unchanged():
    tcfg = cfg("13_4aG3T_s123")
    new, pos, spe, warmup, total = resolved(tcfg)
    # Copy of the code before the change (commit 1119194).
    old = tr.RandomBatchSampler(N_TRAIN, tr.BATCH, seed=SEED, drop_last=True)
    old_spe = len(old)
    old_total = old_spe * int(tcfg["max_epochs"])
    old_warmup = old_spe * int(tcfg["schedule"]["warmup_epochs"])
    assert pos is None and type(new) is tr.RandomBatchSampler
    assert (spe, warmup, total) == (old_spe, old_warmup, old_total) == (1562, 3124, TOTAL)
    for epoch in (0, 1):
        for a, b in zip(first_batches(new, epoch), first_batches(old, epoch)):
            assert np.array_equal(a, b)
    print(f"(a) OK: steps_per_epoch={spe} warmup_steps={warmup} total_steps={total}; "
          "3 first batches of epochs 0 and 1 identical to the previous code")


def check_b_subsets():
    s200, p200, _, _, _ = resolved(cfg("10_4aG3T_n200k_s123"))
    s100, p100, _, _, _ = resolved(cfg("10_4aG3T_n100k_s123"))
    assert p100.size == 100000 and p200.size == 200000
    assert np.unique(p100).size == 100000 and np.unique(p200).size == 200000
    assert p100.min() >= 0 and p200.max() < N_TRAIN
    assert np.isin(p100, p200).all() and np.array_equal(p100, p200[:100000])
    for name, sampler, pos in (("100k", s100, p100), ("200k", s200, p200)):
        seen_orders = []
        for epoch in (0, 1, 320 if name == "100k" else 159):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            used = np.concatenate(batches)
            assert all(b.size == tr.BATCH for b in batches) and len(batches) == len(sampler)
            assert np.unique(used).size == used.size, "position repeated within an epoch"
            assert np.isin(used, pos).all(), "position outside the subset"
            seen_orders.append(used[:tr.BATCH])
        assert not np.array_equal(seen_orders[0], seen_orders[1])
        print(f"(b) OK: {name}: no repeated position within an epoch, all inside the subset "
              f"({used.size}/{pos.size} used per epoch, drop_last)")
    print("(b) OK: 100k nested in 200k")


def check_c_stop():
    src = TRAINER.read_text()
    assert "if scheduler.last_epoch >= total_steps:\n                break\n" in src
    assert "expected_step = min((epoch + 1) * steps_per_epoch, total_steps)" in src
    tcfg = cfg("10_4aG3T_n100k_s123")
    max_epochs = int(tcfg["max_epochs"])
    sampler, _, spe, warmup, total = resolved(tcfg)
    assert (spe, warmup, total, max_epochs) == (390, 3124, TOTAL, 321)
    wrapped = tr.SeededBatchSampler(sampler, SEED)
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=3e-4)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, tr.make_lr_lambda(warmup, total))
    last_epoch_steps = 0
    for epoch in range(max_epochs):
        wrapped.set_epoch(epoch)
        last_epoch_steps = 0
        for _batch_seed, _pos in wrapped:
            opt.step()
            sch.step()
            last_epoch_steps += 1
            if sch.last_epoch >= total:
                break
        expected = min((epoch + 1) * spe, total)
        assert sch.last_epoch == expected, (sch.last_epoch, expected)
    assert sch.last_epoch == TOTAL and epoch == 320 and last_epoch_steps == 160
    assert abs(opt.param_groups[0]["lr"]) < 1e-12
    print(f"(c) OK: stopped at step {sch.last_epoch} in 0-based epoch {epoch} "
          f"after {last_epoch_steps} of {spe} steps; final lr={opt.param_groups[0]['lr']:.1e}")


def check_configs():
    for name, want, every, off_step in (("9_4aG3T_short20_s123", (781, 31240), 1, 16 * 1562),
                                        ("9_4aG3T_short20_s124", (781, 31240), 1, 16 * 1562),
                                        ("10_4aG3T_n200k_s123", (3124, TOTAL), 2, 99968),
                                        ("10_4aG3T_n100k_s123", (3124, TOTAL), 4, 100230)):
        tcfg = cfg(name)
        _, _, spe, warmup, total = resolved(tcfg)
        off = tr.dropout_off_epoch(float(tcfg["dropout_off_fraction"]), int(tcfg["max_epochs"]))
        assert (warmup, total) == want and off * spe == off_step
        assert int(tcfg.get("monitor_train40k_every", 1)) == every
        print(f"configs OK: {name}: steps/epoch={spe} warmup={warmup} total={total} "
              f"dropout_off_step={off * spe}")


if __name__ == "__main__":
    check_a_default_unchanged()
    check_b_subsets()
    check_c_stop()
    check_configs()
    print("ALL OK")
