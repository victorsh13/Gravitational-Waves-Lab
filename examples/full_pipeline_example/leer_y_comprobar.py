#!/usr/bin/env python3
"""Ejemplo completo, paso 2: lee salidas/ejemplo.h5, resume, comprueba y dibuja.

Uso (con el entorno de gwsim activado, desde cualquier carpeta):
    python examples/full_pipeline_example/leer_y_comprobar.py [ruta/al/fichero.h5]

Imprime un resumen y una lista de comprobaciones, cada una con OK o FALLO, y guarda dos figuras en
salidas/. Termina con código 0 si todas las comprobaciones dan OK y con 1 si alguna falla.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))           # para importar gwsim sin instalarlo

import h5py                              # noqa: E402
import numpy as np                       # noqa: E402
import matplotlib                        # noqa: E402
matplotlib.use("Agg")                    # sin ventana: las figuras van a ficheros
import matplotlib.pyplot as plt          # noqa: E402

import gwsim                             # noqa: E402

COLORS = {"H1": "#d55e00", "L1": "#0072b2", "V1": "#009e73"}   # paleta Okabe-Ito
IFOS = ("H1", "L1", "V1")
TOL_ZSCORE = 1e-3        # |media| y |desviación − 1| por canal
TOL_SNR = 0.05           # |SNR final − objetivo| / objetivo, la misma tolerancia que gwsim
TOL_PRIOR = 1e-9         # holgura numérica en los límites del prior


def prior_limits(prior):
    """Límites (mín, máx) que impone el prior a cada columna; solo las distribuciones de esta config."""
    limits = {}
    masses = prior["masses"]
    if masses["distribution"] == "components_uniform":
        limits["mass_1"] = limits["mass_2"] = tuple(masses["range"])
    else:
        limits["chirp_mass"] = tuple(masses["chirp_mass_range"])
        limits["q"] = tuple(masses["q_range"])
    for name in ("spin_1z", "spin_2z"):
        spec = prior[name]
        limits[name] = (spec["fixed"], spec["fixed"]) if "fixed" in spec else tuple(spec["range"])
    natural = {"inclination": (0.0, math.pi), "dec": (-math.pi / 2, math.pi / 2),
               "ra": (0.0, 2 * math.pi), "polarization_angle": (0.0, 2 * math.pi)}
    for name, bounds in natural.items():
        spec = prior[name]
        limits[name] = (spec["fixed"], spec["fixed"]) if "fixed" in spec else bounds
    spec = prior["target_network_snr"]
    limits["target_network_snr"] = (spec["fixed"], spec["fixed"]) if "fixed" in spec else tuple(spec["range"])
    return limits


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "salidas" / "ejemplo.h5"
    if not path.is_file():
        sys.exit(f"No existe {path}. Genera primero: bash {(HERE / 'generar.sh').relative_to(ROOT)}")
    out_dir = path.parent

    # --- 1. Leer ---------------------------------------------------------------------------------
    with h5py.File(path, "r") as h:
        cfg = json.loads(h.attrs["config_json"])          # la configuración exacta de este fichero
    everything = gwsim.load(path, outputs=(), only_ok=False)   # solo metadatos, todas las filas
    d = gwsim.load(path, outputs=("X", "X_noise"))             # salidas + metadatos, filas hechas
    status, errors = everything["status"], everything["error"]

    # --- 2. Resumen --------------------------------------------------------------------------------
    n_ok, n_fail, n_pending = int((status == 1).sum()), int((status == 2).sum()), int((status == 0).sum())
    print(f"Fichero: {path}")
    print(f"  ruido: {cfg['noise']['source']}, modo: {cfg['mode']}, semilla: {cfg['seed']}")
    print(f"  filas: {len(status)} -> hechas {n_ok}, fallidas {n_fail}, pendientes {n_pending}")
    for i in np.flatnonzero(status == 2):
        print(f"    fila {i} fallida: {errors[i]}")
    print(f"  X: {d['X'].shape} {d['X'].dtype}   X_noise: {d['X_noise'].shape}   y: {d['y'].shape}")
    if n_ok:
        for name, label in (("total_mass", "M_tot [M☉]"), ("chi_eff", "χ_eff"), ("final_network_snr", "SNR final")):
            v = d[name]
            print(f"  {label:<11} mín {v.min():8.3f}   máx {v.max():8.3f}")

    # --- 3. Comprobaciones -------------------------------------------------------------------------
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(f"  [{'OK' if ok else 'FALLO':>5}] {name}" + (f": {detail}" if detail else ""))

    print("\nComprobaciones:")
    check("hay filas hechas", n_ok > 0, f"{n_ok} de {len(status)}")
    if n_ok:
        X, Xn = d["X"].astype(np.float64), d["X_noise"].astype(np.float64)
        check("forma (3, 16384) por señal", X.shape[1:] == (3, 16384) and Xn.shape[1:] == (3, 16384),
              f"X {X.shape}, X_noise {Xn.shape}")
        check("valores finitos", np.isfinite(X).all() and np.isfinite(Xn).all())
        mean_err = float(np.abs(X.mean(axis=2)).max())
        std_err = float(np.abs(X.std(axis=2) - 1.0).max())
        check("media ≈ 0 por canal en X", mean_err < TOL_ZSCORE, f"máx |media| = {mean_err:.2e}")
        check("desviación ≈ 1 por canal en X", std_err < TOL_ZSCORE, f"máx |desv − 1| = {std_err:.2e}")
        if cfg["mode"] == "snr":
            rel = np.abs(d["final_network_snr"] - d["target_network_snr"]) / d["target_network_snr"]
            check(f"|SNR final − objetivo| / objetivo ≤ {TOL_SNR}", float(rel.max()) <= TOL_SNR,
                  f"máx {float(rel.max()):.2e}")
        if not cfg["parameters_file"]:
            bad = []
            for name, (lo, hi) in prior_limits(cfg["prior"]).items():
                v = d[name]
                if not ((v >= lo - TOL_PRIOR) & (v <= hi + TOL_PRIOR)).all():
                    bad.append(f"{name} fuera de [{lo:g}, {hi:g}]")
            check("parámetros dentro del prior", not bad, "; ".join(bad))

    # --- 4. Figuras --------------------------------------------------------------------------------
    if n_ok:
        fs = cfg["simulation"]["sampling_frequency"]
        t = np.arange(d["X"].shape[-1]) / fs
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for ax, (j, ifo) in zip(axes, enumerate(IFOS)):
            ax.plot(t, d["X_noise"][0, j], color="0.65", lw=0.5, label=f"{ifo}: X_noise (solo ruido)")
            ax.plot(t, d["X"][0, j], color=COLORS[ifo], lw=0.5, alpha=0.85, label=f"{ifo}: X (señal + ruido)")
            ax.set_ylabel("X (adimensional)")
            ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
        axes[-1].set_xlabel("tiempo en la ventana [s]")
        axes[0].set_title(f"Fila {d['row_index'][0]}: M_tot = {d['total_mass'][0]:.1f} M☉, "
                          f"SNR de red = {d['final_network_snr'][0]:.1f}")
        fig.tight_layout()
        fig.savefig(out_dir / "senal_vs_ruido.png", dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, (k, xlabel) in zip(axes, enumerate(("masa de chirp 𝓜 [M☉]", "masa total M_tot [M☉]", "espín efectivo χ_eff"))):
            ax.hist(d["y"][:, k], bins=10, color="#0072b2", alpha=0.8, label=f"{n_ok} señales")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("número de señales")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "etiquetas.png", dpi=120)
        plt.close(fig)
        print(f"\nFiguras: {out_dir / 'senal_vs_ruido.png'}, {out_dir / 'etiquetas.png'}")

    passed = all(checks)
    print("\nRESULTADO:", "todas las comprobaciones OK" if passed else "hay comprobaciones con FALLO")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
