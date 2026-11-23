"""Escritura HDF5 reanudable (esquema en docs/SALIDAS.md).

Una fila pasa por status 0 (pendiente) -> 1 (hecha) o 2 (fallida). El status se escribe después de
los datos de la fila: si el proceso muere entre medias, la fila sigue en 0 y --resume la rehace
(la fila es determinista: mismas semillas, mismo resultado).
"""
from __future__ import annotations

from datetime import datetime, timezone

import h5py
import numpy as np

from . import __version__
from .physics.g1 import DETECTORS, LABELS

STATUS_PENDING, STATUS_DONE, STATUS_FAILED = 0, 1, 2
PLAN_FLOAT = ("mass_1", "mass_2", "spin_1z", "spin_2z", "inclination", "ra", "dec", "polarization_angle",
              "chirp_mass", "total_mass", "q", "chi_eff", "target_network_snr", "distance_input", "gps")
PLAN_INT = ("placement_seed", "gaussian_seed")
PLAN_STR = ("block_id", "noise_crop_id", "noise_split")
RESULT_FLOAT = ("final_distance", "reference_network_snr", "final_network_snr",
                *(f"snr_{ifo}" for ifo in DETECTORS), "actual_geocentric_time", "placement_offset_s",
                "full_network_duration", "required_final_duration", "band_f_lower", "band_taper_end_hz",
                "align_residual_rad")
RESULT_INT = tuple(f"edge_samples_{ifo}" for ifo in DETECTORS)
RESULT_SEED = tuple(f"gaussian_seed_{ifo}" for ifo in DETECTORS)
STR = h5py.string_dtype("utf-8")


def create_output(path, plan, outputs, length, attrs):
    """Crea el fichero (falla si ya existe) con el plan completo y status = 0."""
    n = len(plan)
    with h5py.File(path, "x") as h:
        for name in outputs:
            h.create_dataset(name, shape=(n, len(DETECTORS), length), dtype=np.float32,
                             chunks=(1, len(DETECTORS), length), compression="lzf")
        for name in PLAN_FLOAT:
            h.create_dataset(name, data=plan[name].to_numpy(dtype=np.float64))
        for name in PLAN_INT:
            h.create_dataset(name, data=plan[name].to_numpy(dtype=np.int64))
        for name in PLAN_STR:
            h.create_dataset(name, data=np.array([str(v) for v in plan[name]], dtype=object), dtype=STR)
        h.create_dataset("y", data=plan[list(LABELS)].to_numpy(dtype=np.float32))
        for name in RESULT_FLOAT:
            h.create_dataset(name, shape=(n,), dtype=np.float64, fillvalue=np.nan)
        for name in RESULT_INT:
            h.create_dataset(name, shape=(n,), dtype=np.int64, fillvalue=-1)
        for name in RESULT_SEED:
            h.create_dataset(name, shape=(n,), dtype=np.uint32)
        h.create_dataset("is_truncated", shape=(n,), dtype=bool)
        h.create_dataset("error", shape=(n,), dtype=STR)
        h.create_dataset("status", shape=(n,), dtype=np.uint8)
        h.attrs.update(attrs)
        h.attrs["gwsim_version"] = __version__
        h.attrs["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        h.attrs["detectors"] = ",".join(DETECTORS)
        h.attrs["labels"] = ",".join(LABELS)
        h.attrs["outputs"] = ",".join(outputs)
        h.attrs["n_rows"] = n
        h.attrs["initialization_complete"] = True


def verify_for_resume(h, config_json):
    if not bool(h.attrs.get("initialization_complete", False)):
        raise ValueError("El HDF5 no terminó de inicializarse; bórralo y genera de nuevo")
    if h.attrs["config_json"] != config_json:
        raise ValueError("--resume con otra configuración: la del fichero (attrs['config_json']) es distinta. "
                         "Usa la misma --config y los mismos overrides de la CLI")
    if h.attrs["gwsim_version"] != __version__:
        raise ValueError(f"El fichero es de gwsim {h.attrs['gwsim_version']}; esta versión es {__version__}")


def read_plan_rows(h, indices):
    """Lista de dicts (una por fila) con lo que necesita simulate_row."""
    indices = np.asarray(indices)
    cols = {name: h[name][:] for name in (*PLAN_FLOAT, *PLAN_INT)}
    cols["block_id"] = h["block_id"].asstr()[:]
    return [{name: values[i] for name, values in cols.items()} | {"row_index": int(i)} for i in indices]


def write_results(h, results):
    """results: [(índice, status, salidas | None, metadatos, error)]. Datos primero, status al final."""
    results = sorted(results, key=lambda r: r[0])
    idx = [r[0] for r in results]
    for i, status, outputs, _, _ in results:
        if status == STATUS_DONE:
            for name, array in outputs.items():
                h[name][i] = array
    for name in (*RESULT_FLOAT, *RESULT_INT, *RESULT_SEED, "is_truncated"):
        default = {"is_truncated": False}.get(name, -1 if name in RESULT_INT else
                                               0 if name in RESULT_SEED else np.nan)
        h[name][idx] = np.array([r[3].get(name, default) for r in results], dtype=h[name].dtype)
    h["error"][idx] = np.array([r[4] for r in results], dtype=object)
    h.flush()
    h["status"][idx] = np.array([r[1] for r in results], dtype=np.uint8)
    h.flush()
