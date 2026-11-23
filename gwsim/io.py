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
from .physics.steps import DETECTORS, LABELS

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


# Bytes aproximados de los metadatos de una fila (todos los datasets 1-D, y, y unos 64 bytes de texto).
METADATA_BYTES_PER_ROW = (8 * (len(PLAN_FLOAT) + len(PLAN_INT) + len(RESULT_FLOAT) + len(RESULT_INT))
                          + 4 * len(RESULT_SEED) + 4 * len(LABELS) + 2 + 64)


def load(path, outputs=("X",), only_ok=True):
    """Lee un HDF5 de gwsim: dict nombre -> array con las salidas pedidas y todos los metadatos por fila.

    Con only_ok=True solo devuelve las filas con status == 1 (y "row_index" con su índice en el fichero).
    Las salidas se cargan enteras en memoria: para ficheros grandes, lee por bloques con h5py.
    """
    with h5py.File(path, "r") as h:
        available = h.attrs["outputs"].split(",")
        missing = [o for o in outputs if o not in available]
        if missing:
            raise KeyError(f"{path} no contiene {missing}; contiene {available}")
        status = h["status"][:]
        rows = np.flatnonzero(status == STATUS_DONE) if only_ok else np.arange(len(status))
        data = {"row_index": rows}
        for name in outputs:
            data[name] = h[name][rows] if len(rows) else np.zeros((0,) + h[name].shape[1:], np.float32)
        for name, dataset in h.items():
            if name in available or dataset.ndim == 0:
                continue
            values = dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:]
            data[name] = values[rows]
    return data


def estimate_size(config, n):
    """Bytes previstos del HDF5 con n filas, sin comprimir y en float32 (orden de magnitud).

    config: ruta a un JSON de gwsim o un dict con la misma estructura (las claves ausentes toman el
    valor por defecto). No valida la configuración.
    """
    import json
    from pathlib import Path
    from .config import DEFAULT_CONFIG
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    outputs = config.get("outputs", DEFAULT_CONFIG["outputs"])
    sim = {**DEFAULT_CONFIG["simulation"], **config.get("simulation", {})}
    length = int(round(float(sim["duration"]) * float(sim["sampling_frequency"])))
    per_row = len(outputs) * len(DETECTORS) * length * np.dtype(np.float32).itemsize + METADATA_BYTES_PER_ROW
    return int(n) * int(per_row)
