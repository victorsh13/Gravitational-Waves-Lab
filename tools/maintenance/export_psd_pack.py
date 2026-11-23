#!/usr/bin/env python3
"""Exporta el paquete de PSDs de gwsim (psd_pack_o3b.npz) desde los datos de origen.

Solo para el mantenedor, en la máquina con los datos de origen. Solo lee esos datos.

Para cada bloque del banco de ruido del proyecto de origen (load_bank, con las sumas sha256 del banco
congelado) lee los 128 s de strain de la caché de GWOSC (resolve_cached_psd_files, read_psd_intervals) y
calcula la PSD de Welch exactamente como estimate_raw_offsource_psd antes de interpolar:
strain.time_slice(psd_start, psd_end).psd(8.0). gwsim interpola y recorta después
(gwsim/physics/steps.py:interpolate_raw_psd). Los nombres de origen están en docs/PROCEDENCIA.md.

  python tools/maintenance/export_psd_pack.py --data-root <raíz de datos> --output <fichero.npz> [--splits train val ...]

No escribe en el repositorio: --output es obligatorio y nunca se sobrescribe. Imprime tamaño y sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gwsim.noise import PACK_FORMAT_VERSION  # noqa: E402
from gwsim.physics.steps import DETECTORS, PSD_SEGMENT_DURATION_S, require  # noqa: E402

SPLITS = ("train", "val", "cal", "test")
# Sección "bank" de configs/generation/generate_m11_final_G1_500k.json @ 4b34677 (copiada).
BANK = {
    "dataset_id": "m11_6_manifests",
    "files": {
        "groups": "m11_6_file_groups.csv",
        "blocks": "m11_6_real_noise_blocks.csv",
        "crops": "m11_6_real_noise_crops.csv",
    },
    "sha256": {
        "groups": "60cb3073134a0e5294184cab6c62c3938ca6f70e3b673244d8fc5750815f87e1",
        "blocks": "4423deda5bd987cdec6a4525eb1bfbb28ab5e52c952b899951de2eaded3f787c",
        "crops": "452bcee1ff07f7bdef7703e640f367d907d16f62cf9d35773a18fa89df69b314",
    },
    "environment_count": 29,
    "block_count": 292,
    "crop_count": 4380,
}
DATA_ROOT_ENV_VAR = "CBC_PE_DATA_ROOT"


# Copiado de scripts/prepare_m11_final_manifest.py:load_bank @ 4b34677; cambios: recibe la sección "bank"
# (BANK) en vez de la configuración completa; resolve_processed_artifact(allow_legacy_flat=False) sustituido
# por su ruta canónica <data_root>/processed/<dataset_id>/<fichero>. Se conservan las sumas sha256.
def load_bank(bank, data_root):
    """Fail closed if the accepted bank changes; do not redownload or rebuild it."""
    tables = {}
    for key, name in bank["files"].items():
        path = Path(data_root) / "processed" / bank["dataset_id"] / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == bank["sha256"][key],
                f"Frozen bank checksum mismatch: {path}")
        tables[key] = pd.read_csv(path)
    groups, blocks, crops = (tables[k] for k in ("groups", "blocks", "crops"))
    for table, column, count in (
        (groups, "file_group_id", bank["environment_count"]),
        (blocks, "block_id", bank["block_count"]),
        (crops, "noise_crop_id", bank["crop_count"]),
    ):
        require(len(table) == count and table[column].is_unique, f"Invalid {column} bank")
        require(set(table["split"]) == set(SPLITS), "Missing bank split")
        require(table.groupby("file_group_id")["split"].nunique().eq(1).all(),
                "Environment shared across splits")
    require(blocks["passes_basic_qc"].eq(True).all(), "Unaccepted block")
    group_splits = groups.set_index("file_group_id")["split"]
    require(blocks["file_group_id"].map(group_splits).eq(blocks["split"]).all(),
            "Block/group split mismatch")
    block_lookup = blocks.set_index("block_id")
    for column in ("file_group_id", "split", "psd_start", "psd_end"):
        require(crops["block_id"].map(block_lookup[column]).eq(crops[column]).all(),
                f"Crop/block mismatch: {column}")
    require(np.allclose(crops.processing_end - crops.processing_start, 4.8125,
                        rtol=0, atol=1e-6), "Invalid processing interval")
    require(np.allclose(crops.psd_end - crops.psd_start, 128, rtol=0, atol=1e-6),
            "Invalid PSD interval")
    require((crops.processing_start >= crops.psd_end + 8).all(), "PSD guard violation")
    # Same temporal-support audit as the M11.6 preproduction split notebook.
    supports = list(blocks.itertuples(index=False))
    for i, a in enumerate(supports):
        for b in supports[i + 1:]:
            if a.split != b.split:
                require(min(a.block_end, b.block_end) <= max(a.block_start, b.block_start),
                        "Cross-split temporal overlap")
    return crops


# Copiado de scripts/generate_m11_final_G1.py:resolve_cached_psd_files @ 4b34677; cambios: no devuelve la
# procedencia JSON (tamaño y mtime de los ficheros); `manifest` es la tabla de recortes.
def resolve_cached_psd_files(root, manifest):
    """Read cached file headers only; fail on missing/ambiguous assigned intervals."""
    files = {ifo: [] for ifo in DETECTORS}
    for ifo in DETECTORS:
        for path in sorted((root / "gwosc_cache/m11").glob(f"*-{ifo}_*.hdf5")):
            with h5py.File(path, "r") as h:
                start = float(h["meta/GPSstart"][()])
                duration = float(h["meta/Duration"][()])
                rate = len(h["strain/Strain"]) / duration
            if rate == 4096:
                files[ifo].append((path, start, start + duration))
    selected = {}
    for row in manifest.drop_duplicates("block_id").itertuples():
        for ifo in DETECTORS:
            matches = [(p, start) for p, start, end in files[ifo]
                       if start <= row.psd_start and row.psd_end <= end]
            require(len(matches) == 1, f"Expected one cached {ifo} file for {row.block_id}; got {len(matches)}")
            path, start = matches[0]
            selected[row.block_id, ifo] = (path, start)
    return selected


# Copiado de scripts/generate_m11_final_G1.py:read_psd_intervals @ 4b34677; cambios: ninguno
def read_psd_intervals(row, selected):
    from pycbc.types import TimeSeries
    strains = {}
    for ifo in DETECTORS:
        path, start = selected[row.block_id, ifo]
        first = int(round((row.psd_start - start) * 4096))
        last = first + 128 * 4096
        require(abs(start + first / 4096 - row.psd_start) < 1e-7, "PSD sample misalignment")
        with h5py.File(path, "r") as h:
            values = h["strain/Strain"][first:last]
        require(len(values) == 128 * 4096, "Incomplete PSD reference")
        strains[ifo] = TimeSeries(values, delta_t=1 / 4096, epoch=float(row.psd_start))
    return strains


# Copiado de scripts/generate_m11_final_G1.py:estimate_raw_offsource_psd @ 4b34677; cambios: solo la parte
# anterior a la interpolación (la posterior está en gwsim/physics/steps.py:interpolate_raw_psd).
def welch_offsource_psd(
    strain,
    *,
    psd_start: float,
    psd_end: float,
    psd_segment_duration: float = PSD_SEGMENT_DURATION_S,
):
    psd_data = strain.time_slice(float(psd_start), float(psd_end))

    if len(psd_data) == 0:
        raise ValueError("PSD reference window is empty.")

    if not np.all(np.isfinite(psd_data.numpy())):
        raise ValueError("PSD reference contains non-finite values.")

    psd = psd_data.psd(float(psd_segment_duration))
    return psd


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, help=f"Raíz de datos de origen (si no, ${DATA_ROOT_ENV_VAR})")
    parser.add_argument("--output", type=Path, required=True, help="Fichero .npz nuevo (no se sobrescribe)")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    args = parser.parse_args()
    root = args.data_root or (Path(os.environ[DATA_ROOT_ENV_VAR]) if os.environ.get(DATA_ROOT_ENV_VAR) else None)
    require(root is not None, f"Usa --data-root o define {DATA_ROOT_ENV_VAR}")
    root = root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(output.suffix == ".npz", "--output debe terminar en .npz")
    require(not output.exists(), f"Ya existe, no se sobrescribe: {output}")
    repo = Path(__file__).resolve().parents[2]
    require(repo not in output.parents, f"--output no puede estar dentro del repositorio gwsim ({repo})")

    crops = load_bank(BANK, root)
    crops = crops[crops["split"].isin(args.splits)]
    blocks = crops.drop_duplicates("block_id").sort_values("block_id", kind="stable")
    print(f"{len(blocks)} bloques, {len(crops)} recortes, splits {args.splits}", flush=True)
    selected = resolve_cached_psd_files(root, blocks)

    psd_arrays, delta_f = [], []
    for k, row in enumerate(blocks.itertuples()):
        strains = read_psd_intervals(row, selected)
        per_ifo = []
        for ifo in DETECTORS:
            psd = welch_offsource_psd(strains[ifo], psd_start=float(row.psd_start), psd_end=float(row.psd_end))
            per_ifo.append((np.asarray(psd.numpy(), dtype=np.float64), float(psd.delta_f)))
        require(len({len(a) for a, _ in per_ifo}) == 1 and len({d for _, d in per_ifo}) == 1,
                f"Block {row.block_id}: PSDs of different length or delta_f")
        require(all(a.dtype == np.float64 for a, _ in per_ifo), "Welch PSD is not float64")
        psd_arrays.append(np.stack([a for a, _ in per_ifo]))
        delta_f.append(per_ifo[0][1])
        if (k + 1) % 20 == 0 or k + 1 == len(blocks):
            print(f"  {k + 1}/{len(blocks)} bloques", flush=True)

    payload = dict(
        format_version=np.int64(PACK_FORMAT_VERSION),
        detectors=np.array(DETECTORS),
        psd=np.stack(psd_arrays),
        delta_f=np.array(delta_f, dtype=np.float64),
        block_id=np.array([str(b) for b in blocks["block_id"]]),
        split=np.array([str(s) for s in blocks["split"]]),
        file_group_id=blocks["file_group_id"].to_numpy(dtype=np.int64),
        psd_start=blocks["psd_start"].to_numpy(dtype=np.float64),
        psd_end=blocks["psd_end"].to_numpy(dtype=np.float64),
        crop_noise_crop_id=np.array([str(c) for c in crops["noise_crop_id"]]),
        crop_block_id=np.array([str(b) for b in crops["block_id"]]),
        crop_center_gps=crops["center_gps"].to_numpy(dtype=np.float64),
        crop_split=np.array([str(s) for s in crops["split"]]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:  # creación exclusiva
        np.savez_compressed(handle, **payload)
    print(f"Escrito {output}", flush=True)
    print(f"  tamaño {output.stat().st_size} bytes ({output.stat().st_size / 2**20:.1f} MiB)", flush=True)
    print(f"  sha256 {sha256_file(output)}", flush=True)


if __name__ == "__main__":
    main()
