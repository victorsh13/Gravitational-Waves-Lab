import h5py
import numpy as np
from pathlib import Path

src_path = Path(
    "/data/vserrano/cbc_pe_data/processed/M11_final_G1_500k/"
    "m11_final_G1_500k.h5"
)

dst_path = src_path.with_name(
    "m11_final_G1_500k_trainio.h5"
)

BLOCK = 256

with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:

    # Copy root attributes
    for key, value in src.attrs.items():
        dst.attrs[key] = value

    # Add explicit provenance
    dst.attrs["storage_layout_variant"] = "trainio"
    dst.attrs["storage_layout_source"] = src_path.name
    dst.attrs["storage_layout_note"] = (
        "Exact logical copy of M11-final dataset with training-optimized "
        "HDF5 chunking; scientific content unchanged."
    )

    # X
    X_src = src["X"]
    X_dst = dst.create_dataset(
        "X",
        shape=X_src.shape,
        dtype=X_src.dtype,
        chunks=(64, 3, 16384),
        compression=None,
    )

    # y
    y_src = src["y"]
    y_dst = dst.create_dataset(
        "y",
        shape=y_src.shape,
        dtype=y_src.dtype,
        chunks=(1024, 3),
        compression=None,
    )

    n = X_src.shape[0]

    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)

        X_dst[start:stop] = X_src[start:stop]
        y_dst[start:stop] = y_src[start:stop]

        if start % 10000 == 0:
            print(f"{start}/{n}")

    # Copy all other datasets/groups unchanged
    for key in src.keys():
        if key in {"X", "y"}:
            continue
        src.copy(key, dst)

print("Wrote:", dst_path)