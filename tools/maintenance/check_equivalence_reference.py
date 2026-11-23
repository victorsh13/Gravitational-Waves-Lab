#!/usr/bin/env python3
"""Prueba de equivalencia de gwsim frente a la producción de referencia del proyecto de origen.

Solo para el mantenedor, en la máquina con los datos de origen. Un proceso; solo lee esos datos.

Toma --n filas equiespaciadas del split de validación de la referencia (row_ids_val.npy y X_val.npy de su
memmap float16); sus parámetros, semillas, SNR objetivo y ventanas salen del manifest de origen con
adjust_processing_windows; la PSD, de --psd-pack. Llama a gwsim.recipe.simulate_row con la física por
defecto, convierte X a float16 como to_fp16 y compara con X_val. Los nombres de los conjuntos de origen
(constantes de abajo) están explicados en docs/PROCEDENCIA.md.

Criterio (fijado antes de ejecutar): PASS si en TODAS las filas la diferencia máxima es <= 1 ulp de
float16 y las etiquetas [chirp_mass, total_mass, chi_eff] (float32) coinciden exactamente con y_val.

  python tools/maintenance/check_equivalence_reference.py --data-root <raíz> --psd-pack <pack.npz> \
         --output <informe.json> [--n 64]
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gwsim  # noqa: E402
from gwsim.noise import PsdPackNoise  # noqa: E402
from gwsim.physics.steps import LABELS, require, to_fp16  # noqa: E402
from gwsim.physics.parameters import CBCParameters  # noqa: E402
from gwsim.recipe import PARAMETER_NAMES, Physics, RowOptions, simulate_row  # noqa: E402

G1_DATASET = "M11_final_G1_500k"
G1_MANIFEST = "m11_final_G1_500k_manifest.csv"
G3_DATASET = "M13_G3_band1024_w16_500k"
G3T_MEMMAP = Path("m13_13_g3t") / "fp16_memmap"
G1_CONTEXT = (1664, 1664)   # simulation.processing_context_{start,end}_samples de G1
TIME_TOL_S = 1e-6
MAX_ULP = 1
DATA_ROOT_ENV_VAR = "CBC_PE_DATA_ROOT"


# Copiado de scripts/m13/generate_m13_G3_whitening.py:adjust_processing_windows @ 4b34677; cambios: los
# contextos de G1 y G3 y dt/duration se pasan como argumentos (en producción salen de spec_g1 y g1.CONFIG);
# sin el print final.
def adjust_processing_windows(rows, config, ctx1):
    dt = config.delta_t
    duration = config.duration
    ctx1_start, ctx1_end = ctx1
    ctx3_start = config.processing_context_start_samples
    ctx3_end = config.processing_context_end_samples
    g1_processing_duration = duration + (ctx1_start + ctx1_end) * dt

    start1 = rows["processing_start"].to_numpy(dtype=np.float64)
    end1 = rows["processing_end"].to_numpy(dtype=np.float64)
    reference = rows["reference_geocentric_time"].to_numpy(dtype=np.float64)
    require(np.isclose(end1 - start1, g1_processing_duration).all(), "G1 processing duration mismatch")
    start3 = start1 - (ctx3_start - ctx1_start) * dt
    end3 = end1 + (ctx3_end - ctx1_end) * dt
    require(np.isclose(end3 - start3, config.processing_duration).all(),
            "Adjusted processing duration differs from CONFIG.processing_duration")

    final_start1 = start1 + ctx1_start * dt
    final_start3 = start3 + ctx3_start * dt
    final_center1 = final_start1 + 0.5 * duration
    final_center3 = final_start3 + 0.5 * duration
    require(np.max(np.abs(final_start3 - final_start1)) < TIME_TOL_S, "final_start changed")
    require(np.max(np.abs(final_center3 - final_center1)) < TIME_TOL_S, "final_center changed")
    # Same tolerance as build_g1_row, before and after the adjustment.
    require(np.max(np.abs(final_center1 - reference)) < dt, "G1 final_center != reference GPS")
    require(np.max(np.abs(final_center3 - reference)) < dt, "G3 final_center != reference GPS")

    adjusted = rows.copy()
    adjusted["g1_processing_start"] = start1
    adjusted["g1_processing_end"] = end1
    adjusted["processing_start"] = start3
    adjusted["processing_end"] = end3
    return adjusted


def fp16_ulp_distance(a, b):
    """|a - b| en ulp de float16 (orden de los patrones de bits; +0 y -0 son el mismo punto)."""
    def ordered(x):
        i = np.ascontiguousarray(x, dtype=np.float16).view(np.int16).astype(np.int32)
        return np.where(i < 0, -(i & 0x7FFF), i)
    return np.abs(ordered(a) - ordered(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, help=f"Raíz de datos de origen (si no, ${DATA_ROOT_ENV_VAR})")
    parser.add_argument("--psd-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Informe JSON nuevo (no se sobrescribe)")
    parser.add_argument("--n", type=int, default=64)
    args = parser.parse_args()
    root = args.data_root or (Path(os.environ[DATA_ROOT_ENV_VAR]) if os.environ.get(DATA_ROOT_ENV_VAR) else None)
    require(root is not None, f"Usa --data-root o define {DATA_ROOT_ENV_VAR}")
    root = root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(not output.exists(), f"Ya existe, no se sobrescribe: {output}")
    require(args.n >= 1, "--n debe ser >= 1")

    memmap = root / "processed" / G3_DATASET / G3T_MEMMAP
    row_ids = np.load(memmap / "row_ids_val.npy", mmap_mode="r")
    X_val = np.load(memmap / "X_val.npy", mmap_mode="r")
    y_val = np.load(memmap / "y_val.npy", mmap_mode="r")
    require(X_val.dtype == np.float16 and X_val.shape[0] == row_ids.size == y_val.shape[0],
            "Unexpected reference val memmap")
    positions = np.unique(np.round(np.linspace(0, row_ids.size - 1, args.n)).astype(np.int64))

    manifest_path = root / "processed" / G1_DATASET / G1_MANIFEST
    df = pd.read_csv(manifest_path, float_precision="round_trip")
    require(np.array_equal(df["manifest_index"].to_numpy(), np.arange(len(df))), "Manifest order changed")
    ids = np.asarray(row_ids[positions], dtype=np.int64)
    rows = df.loc[ids]
    require(bool(rows["split"].eq("val").all()), "row_ids_val has rows of other splits")

    physics = Physics()  # física por defecto
    rows = adjust_processing_windows(rows, physics.config, G1_CONTEXT)
    noise = PsdPackNoise(args.psd_pack, ["val"])

    records, passed = [], True
    t0 = time.perf_counter()
    for k, (pos, (row_id, row)) in enumerate(zip(positions, rows.iterrows())):
        rec = dict(position=int(pos), row_id=int(row_id), block_id=str(row["block_id"]))
        try:
            opts = RowOptions(physics=physics, mode="snr", outputs=("X",), click_free=True,
                              processing_start=float(row["processing_start"]))
            outputs, meta = simulate_row(
                {name: row[name] for name in PARAMETER_NAMES}, float(row["reference_geocentric_time"]),
                noise.psds(str(row["block_id"]), physics.config), int(row["placement_seed"]),
                int(row["gaussian_seed"]), target_snr=float(row["target_network_snr"]), opts=opts)
            x16 = to_fp16(np.asarray(outputs["X"], dtype=np.float32), f"X row {row_id}")
            ref = np.asarray(X_val[pos])
            ulp = fp16_ulp_distance(x16, ref)
            p = CBCParameters(**{name: float(row[name]) for name in PARAMETER_NAMES}, distance=1000.0)
            y = np.array([p.chirp_mass, p.total_mass, p.chi_eff], dtype=np.float32)
            labels_ok = bool(np.array_equal(y, np.asarray(y_val[pos])))
            rec.update(n_differing=int((x16.view(np.int16) != ref.view(np.int16)).sum()),
                       max_ulp=int(ulp.max()), labels_match=labels_ok,
                       final_network_snr=float(meta["final_network_snr"]),
                       target_network_snr=float(row["target_network_snr"]),
                       final_distance=float(meta["final_distance"]),
                       total_mass=float(p.total_mass), band=[meta["band_f_lower"], meta["band_taper_end_hz"]],
                       align_residual_rad=meta["align_residual_rad"])
            rec["pass"] = bool(rec["max_ulp"] <= MAX_ULP and labels_ok)
        except Exception as exc:
            rec.update(error=f"{type(exc).__name__}: {exc}", **{"pass": False})
        passed &= rec["pass"]
        records.append(rec)
        print(f"[{k + 1}/{positions.size}] fila {rec['row_id']}: "
              + (f"max {rec['max_ulp']} ulp, {rec['n_differing']} muestras distintas, etiquetas "
                 f"{'OK' if rec['labels_match'] else 'DISTINTAS'}" if "error" not in rec else rec["error"]),
              flush=True)

    report = dict(
        verdict="PASS" if passed else "FAIL",
        criterion=f"every row: max |X_gwsim_fp16 - X_val| <= {MAX_ULP} float16 ulp and labels equal (float32)",
        n_rows=len(records), n_pass=sum(r["pass"] for r in records),
        max_ulp=max((r.get("max_ulp", -1) for r in records), default=-1),
        g3t_memmap=str(memmap), g1_manifest=str(manifest_path), psd_pack=str(args.psd_pack.resolve()),
        gwsim_version=gwsim.__version__,
        software_versions={name: version(name) for name in ("numpy", "scipy", "h5py", "pandas", "PyCBC", "lalsuite")},
        wall_time_s=time.perf_counter() - t0, rows=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(f"{report['verdict']}: {report['n_pass']}/{report['n_rows']} filas; máx {report['max_ulp']} ulp -> {output}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
