"""Plan de la generación: parámetros, SNR o distancia, asignación de ruido y semillas de cada fila.

Todo el plan se sortea en el proceso padre antes de generar y se guarda en el HDF5, así que el
resultado no depende de --workers ni de --resume.

Semillas: `seed` global -> un flujo independiente por propósito,
    np.random.default_rng(np.random.SeedSequence([seed, k]))
con k = 1 parámetros, 2 SNR/distancia, 3 asignación de ruido, 4 colocación, 5 ruido.
Los flujos 4 y 5 dan, por fila, los enteros placement_seed y gaussian_seed que usa simulate_row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .physics.g1 import require
from .recipe import PARAMETER_NAMES

STREAM_PARAMETERS = 1
STREAM_AMPLITUDE = 2   # SNR objetivo (modo snr) o distancia (modo distance)
STREAM_NOISE_ASSIGNMENT = 3
STREAM_PLACEMENT = 4
STREAM_NOISE = 5
SEED_HIGH = 2**63 - 1


def stream(seed, k):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(k)]))


def _uniform_or_fixed(rng, spec, n, default_low, default_high):
    # Se sortea también cuando el valor está fijado, para no desplazar el flujo de los demás parámetros.
    lo, hi = spec.get("range", (default_low, default_high))
    values = rng.uniform(lo, hi, size=n)
    if "fixed" in spec:
        values = np.full(n, float(spec["fixed"]))
    return values


def sample_parameters(prior, n, rng):
    """Mismas fórmulas que src/sampling.py:ParameterSampler.sample_one, vectorizadas.

    Cada parámetro se sortea siempre (aunque esté fijado) en un orden fijo, de modo que fijar uno no
    cambia los valores de los demás.
    """
    masses = prior["masses"]
    if masses["distribution"] == "components_uniform":
        lo, hi = masses["range"]
        m1 = rng.uniform(lo, hi, size=n)
        m2 = rng.uniform(lo, hi, size=n)
    else:  # chirp_mass_q_uniform, q = m2/m1 <= 1
        mc = rng.uniform(*masses["chirp_mass_range"], size=n)
        q = rng.uniform(*masses["q_range"], size=n)
        m1 = mc * (1.0 + q) ** (1.0 / 5.0) / q ** (3.0 / 5.0)
        m2 = q * m1
    s1 = _uniform_or_fixed(rng, prior["spin_1z"], n, -1.0, 1.0)
    s2 = _uniform_or_fixed(rng, prior["spin_2z"], n, -1.0, 1.0)
    # Isótropo: cos(inclination) ~ U(-1, 1), sin(dec) ~ U(-1, 1); ra, psi ~ U(0, 2 pi).
    inclination = np.arccos(rng.uniform(-1.0, 1.0, size=n))
    ra = rng.uniform(0.0, 2.0 * np.pi, size=n)
    dec = np.arcsin(rng.uniform(-1.0, 1.0, size=n))
    psi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    values = {"mass_1": m1, "mass_2": m2, "spin_1z": s1, "spin_2z": s2, "inclination": inclination,
              "ra": ra, "dec": dec, "polarization_angle": psi}
    for name in ("inclination", "ra", "dec", "polarization_angle"):
        if "fixed" in prior[name]:
            values[name] = np.full(n, float(prior[name]["fixed"]))
    return pd.DataFrame(values)


def order_masses(df):
    """m1 >= m2, intercambiando también los espines (como CBCParameters.__post_init__)."""
    swap = df["mass_1"].to_numpy() < df["mass_2"].to_numpy()
    if swap.any():
        for a, b in (("mass_1", "mass_2"), ("spin_1z", "spin_2z")):
            va, vb = df[a].to_numpy().copy(), df[b].to_numpy().copy()
            df.loc[swap, a], df.loc[swap, b] = vb[swap], va[swap]
    return df


def sample_amplitude(spec, n, rng, kind):
    dist = spec.get("distribution", "fixed" if "fixed" in spec else None)
    if dist == "fixed":
        return np.full(n, float(spec["fixed"]))
    lo, hi = (float(x) for x in spec["range"])
    if dist == "uniform":
        return rng.uniform(lo, hi, size=n)
    if dist == "loguniform":
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
    if dist == "uniform_volume":  # p(d) ∝ d^2 (volumen euclídeo; aproximación sin cosmología)
        return (rng.uniform(0.0, 1.0, size=n) * (hi ** 3 - lo ** 3) + lo ** 3) ** (1.0 / 3.0)
    raise ValueError(f"Unknown {kind} distribution {dist!r}")


def add_derived(df):
    m1, m2 = df["mass_1"].to_numpy(), df["mass_2"].to_numpy()
    s1, s2 = df["spin_1z"].to_numpy(), df["spin_2z"].to_numpy()
    total = m1 + m2
    # Mismas expresiones que CBCParameters.chirp_mass / chi_eff.
    df["chirp_mass"] = (m1 * m2) ** (3.0 / 5.0) / (total ** (1.0 / 5.0))
    df["total_mass"] = total
    df["q"] = m2 / m1
    df["chi_eff"] = (m1 * s1 + m2 * s2) / total
    return df


def read_parameters_file(path, mode):
    df = pd.read_csv(path, float_precision="round_trip")
    amplitude = "target_network_snr" if mode == "snr" else "distance"
    missing = [c for c in (*PARAMETER_NAMES, amplitude) if c not in df.columns]
    require(not missing, f"parameters_file {path}: missing columns {missing}")
    require(len(df) > 0, f"parameters_file {path} is empty")
    numeric = [*PARAMETER_NAMES, amplitude]
    require(np.isfinite(df[numeric].to_numpy(dtype=np.float64)).all(),
            f"parameters_file {path}: non-finite values in {numeric}")
    return df


def build_plan(cfg, noise):
    """DataFrame de n filas con todo lo que simulate_row necesita (salvo las PSD)."""
    mode, seed = cfg["mode"], cfg["seed"]
    file_df = None
    if cfg["parameters_file"]:
        file_df = read_parameters_file(cfg["parameters_file"], mode)
        n = len(file_df)
    else:
        n = int(cfg["n_samples"])

    if mode == "noise_only":
        plan = pd.DataFrame({name: np.full(n, np.nan) for name in PARAMETER_NAMES})
        for name in ("chirp_mass", "total_mass", "q", "chi_eff"):
            plan[name] = np.nan
    elif file_df is not None:
        plan = order_masses(file_df[list(PARAMETER_NAMES)].astype(np.float64).reset_index(drop=True))
        plan = add_derived(plan)
    else:
        plan = add_derived(order_masses(sample_parameters(cfg["prior"], n, stream(seed, STREAM_PARAMETERS))))

    plan["target_network_snr"] = np.nan
    plan["distance_input"] = np.nan
    if mode == "snr":
        plan["target_network_snr"] = (file_df["target_network_snr"].to_numpy(dtype=np.float64)
                                      if file_df is not None else
                                      sample_amplitude(cfg["prior"]["target_network_snr"], n,
                                                       stream(seed, STREAM_AMPLITUDE), "SNR"))
    elif mode == "distance":
        plan["distance_input"] = (file_df["distance"].to_numpy(dtype=np.float64)
                                  if file_df is not None else
                                  sample_amplitude(cfg["prior"]["distance"], n,
                                                   stream(seed, STREAM_AMPLITUDE), "distance"))

    if file_df is not None and noise.source == "psd_pack" and "noise_crop_id" in file_df.columns:
        assignment = noise.crops_by_id(file_df["noise_crop_id"].astype(str).tolist())
    else:
        assignment = noise.assign(n, stream(seed, STREAM_NOISE_ASSIGNMENT))
    if file_df is not None and noise.source == "analytic" and "gps" in file_df.columns:
        assignment["gps"] = file_df["gps"].to_numpy(dtype=np.float64)
    for key, values in assignment.items():
        plan[key] = values

    plan["placement_seed"] = stream(seed, STREAM_PLACEMENT).integers(0, SEED_HIGH, size=n, dtype=np.int64)
    plan["gaussian_seed"] = stream(seed, STREAM_NOISE).integers(0, SEED_HIGH, size=n, dtype=np.int64)
    for key in ("placement_seed", "gaussian_seed"):
        if file_df is not None and key in file_df.columns:
            plan[key] = file_df[key].to_numpy(dtype=np.int64)
    if mode == "noise_only":
        plan["placement_seed"] = -1
    require((plan["gaussian_seed"] >= 0).all(), "gaussian_seed must be non-negative")
    return plan
