from pathlib import Path

import h5py
import numpy as np


ROOT = Path(
    "/data/vserrano/cbc_pe_data/processed/m11_6_domains"
)

REFERENCE_H5 = (
    ROOT / "m11_6_G0_26k.h5"
)

SPLITS_OUT = (
    ROOT / "m11_6_splits.npz"
)

STATS_OUT = (
    ROOT / "m11_6_label_stats_train_only.npz"
)


with h5py.File(
    REFERENCE_H5,
    "r",
) as h5:

    split = (
        h5["split"][:]
        .astype(str)
    )

    y = h5["y"][:].astype(
        np.float32
    )

    source_id = (
        h5["source_id"][:]
        .astype(str)
    )


train_idx = np.flatnonzero(
    split == "train"
).astype(np.int64)

val_idx = np.flatnonzero(
    split == "val"
).astype(np.int64)

cal_idx = np.flatnonzero(
    split == "cal"
).astype(np.int64)

test_idx = np.flatnonzero(
    split == "test"
).astype(np.int64)


print(
    "train:",
    len(train_idx),
)

print(
    "val:",
    len(val_idx),
)

print(
    "cal:",
    len(cal_idx),
)

print(
    "test:",
    len(test_idx),
)

np.savez_compressed(
    SPLITS_OUT,
    train_idx=train_idx,
    val_idx=val_idx,
    cal_idx=cal_idx,
    test_idx=test_idx,
    seed=np.int64(123),
)

#############################
##         SCALER          ##
#############################

y_train = y[
    train_idx
]

y_mean = y_train.mean(
    axis=0
).astype(np.float32)

y_std = y_train.std(
    axis=0
).astype(np.float32)

label_names = np.asarray(
    [
        "chirp_mass",
        "total_mass",
        "chi_eff",
    ]
)

np.savez_compressed(
    STATS_OUT,
    y_mean=y_mean,
    y_std=y_std,
    label_names=label_names,
)

print(
    "y_mean:",
    y_mean,
)

print(
    "y_std:",
    y_std,
)


#######################
#   PAIRING CONTROL   #
#######################

for domain in [
    "G1",
    "R",
]:

    path = (
        ROOT
        / f"m11_6_{domain}_26k.h5"
    )

    with h5py.File(
        path,
        "r",
    ) as h5:

        ids = (
            h5["source_id"][:]
            .astype(str)
        )

        y_domain = (
            h5["y"][:]
            .astype(np.float32)
        )

    assert np.array_equal(
        ids,
        source_id,
    )

    assert np.array_equal(
        y_domain,
        y,
    )

print(
    "Shared source/label contract: PASS"
)