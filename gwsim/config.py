"""Carga, valores por defecto y validación de la configuración de gwsim (ver docs/CONFIGURACION.md).

load_config devuelve (cfg, avisos). Un error de configuración lanza ConfigError con un mensaje que
dice qué clave falla y por qué; los avisos no paran nada, solo se imprimen y se guardan en el HDF5.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .recipe import (DEFAULT_SIGNAL_PROCESSOR, DEFAULT_SIMULATION, M_SWITCH, MODES, OUTPUT_NAMES, SIGNAL_OUTPUTS)

SPLITS = ("train", "val", "cal", "test")
# O3b: 1 nov 2019 15:00 UTC - 27 mar 2020 17:00 UTC (solo para el ruido analítico).
O3B_GPS_RANGE = [1256655618.0, 1269363618.0]
# Aproximantes con precesión: gwsim solo tiene espines alineados (spin_1z, spin_2z).
PRECESSING_APPROXIMANTS = {
    "IMRPhenomPv2", "IMRPhenomPv2_NRTidal", "IMRPhenomPv3", "IMRPhenomPv3HM", "IMRPhenomXP",
    "IMRPhenomXPHM", "IMRPhenomTP", "IMRPhenomTPHM", "SEOBNRv3", "SEOBNRv3_opt", "SEOBNRv4P",
    "SEOBNRv4PHM", "SEOBNRv5PHM", "NRSur7dq2", "NRSur7dq4", "SpinTaylorT1", "SpinTaylorT4", "SpinTaylorT5",
}
VALIDATED_APPROXIMANT = "SEOBNRv4_opt"
M_TOT_UNVALIDATED = 250.0
# Copiado de scripts/m15/m15_build_g4.py (constante GEN_M_CHECK) @ 4b34677; cambios: nombre.
M_TOT_MAY_NOT_GENERATE = 16.0

DEFAULT_CONFIG = {
    "description": "",
    "n_samples": 10,
    "seed": 0,
    "mode": "snr",
    "outputs": ["X"],
    "click_free": True,
    "noise": {
        "source": "psd_pack",
        "psd_pack": None,
        "splits": list(SPLITS),
        "gps_range": list(O3B_GPS_RANGE),
    },
    # Prior por defecto: masas U[5, 90], espines U[-1, 1], ángulos isótropos y SNR U[10, 25).
    "prior": {
        "masses": {"distribution": "components_uniform", "range": [5.0, 90.0]},
        "spin_1z": {"distribution": "uniform", "range": [-1.0, 1.0]},
        "spin_2z": {"distribution": "uniform", "range": [-1.0, 1.0]},
        "inclination": {"distribution": "isotropic"},
        "ra": {"distribution": "uniform"},
        "dec": {"distribution": "isotropic"},
        "polarization_angle": {"distribution": "uniform"},
        "target_network_snr": {"distribution": "uniform", "range": [10.0, 25.0]},
        "distance": {"distribution": "uniform_volume", "range": [100.0, 3000.0]},
    },
    "parameters_file": None,
    "simulation": dict(DEFAULT_SIMULATION),
    "signal_processor": dict(DEFAULT_SIGNAL_PROCESSOR),
}
SECTIONS = ("noise", "prior", "simulation", "signal_processor")
ANGLE_LIMITS = {"inclination": (0.0, math.pi, "isotropic"), "dec": (-math.pi / 2, math.pi / 2, "isotropic"),
                "ra": (0.0, 2 * math.pi, "uniform"), "polarization_angle": (0.0, 2 * math.pi, "uniform")}


class ConfigError(ValueError):
    pass


def _fail(message):
    raise ConfigError(message)


def _check(condition, message):
    if not condition:
        _fail(message)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _range(spec, key, name, lo_bound=None, hi_bound=None, positive=False):
    value = spec.get(key)
    _check(isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_number(v) for v in value),
           f"{name}.{key} debe ser una lista [mín, máx] de números; recibido {value!r}")
    lo, hi = float(value[0]), float(value[1])
    _check(lo <= hi, f"{name}.{key} debe estar ordenado [mín, máx]; recibido {value!r}")
    _check(not positive or lo > 0, f"{name}.{key} debe ser positivo; recibido {value!r}")
    _check(lo_bound is None or lo >= lo_bound, f"{name}.{key}: el mínimo debe ser >= {lo_bound}")
    _check(hi_bound is None or hi <= hi_bound, f"{name}.{key}: el máximo debe ser <= {hi_bound}")
    return [lo, hi]


def _merge(base, user, where):
    unknown = set(user) - set(base)
    _check(not unknown, f"Claves desconocidas en {where}: {sorted(unknown)}. Permitidas: {sorted(base)}")
    out = copy.deepcopy(base)
    for key, value in user.items():
        out[key] = value
    return out


def _validate_prior(prior, mode):
    masses = prior["masses"]
    dist = masses.get("distribution")
    if dist == "components_uniform":
        _check(set(masses) <= {"distribution", "range"}, f"prior.masses: claves {sorted(masses)} no válidas")
        masses["range"] = _range(masses, "range", "prior.masses", positive=True)
    elif dist == "chirp_mass_q_uniform":
        _check(set(masses) <= {"distribution", "chirp_mass_range", "q_range"},
               f"prior.masses: claves {sorted(masses)} no válidas")
        masses["chirp_mass_range"] = _range(masses, "chirp_mass_range", "prior.masses", positive=True)
        masses["q_range"] = _range(masses, "q_range", "prior.masses", positive=True, hi_bound=1.0)
    else:
        _fail("prior.masses.distribution debe ser 'components_uniform' o 'chirp_mass_q_uniform'; "
              f"recibido {dist!r}")
    for name in ("spin_1z", "spin_2z"):
        spec = prior[name]
        if "fixed" in spec:
            _check(_is_number(spec["fixed"]) and -1.0 <= spec["fixed"] <= 1.0,
                   f"prior.{name}.fixed debe estar en [-1, 1]")
        else:
            _check(spec.get("distribution") == "uniform",
                   f"prior.{name}: usa {{'distribution': 'uniform', 'range': [a, b]}} o {{'fixed': v}}")
            spec["range"] = _range(spec, "range", f"prior.{name}", lo_bound=-1.0, hi_bound=1.0)
    for name, (lo, hi, natural) in ANGLE_LIMITS.items():
        spec = prior[name]
        if "fixed" in spec:
            _check(_is_number(spec["fixed"]) and lo <= spec["fixed"] <= hi,
                   f"prior.{name}.fixed debe estar en [{lo:.6g}, {hi:.6g}] rad")
        else:
            _check(spec == {"distribution": natural},
                   f"prior.{name}: usa {{'distribution': '{natural}'}} o {{'fixed': valor_en_rad}}")
    if mode == "snr":
        _amplitude(prior["target_network_snr"], "prior.target_network_snr", ("uniform", "loguniform"))
    if mode == "distance":
        _amplitude(prior["distance"], "prior.distance", ("uniform_volume", "uniform"))


def _amplitude(spec, name, allowed):
    if "fixed" in spec:
        _check(_is_number(spec["fixed"]) and spec["fixed"] > 0, f"{name}.fixed debe ser positivo")
        return
    _check(spec.get("distribution") in allowed,
           f"{name}.distribution debe ser una de {list(allowed)} (o usa {{'fixed': v}}); "
           f"recibido {spec.get('distribution')!r}")
    spec["range"] = _range(spec, "range", name, positive=True)


def _mtot_bounds(cfg):
    """(mín, máx) de M_tot = m1 + m2 que puede dar el prior o el fichero."""
    if cfg["parameters_file"]:
        import pandas as pd
        df = pd.read_csv(cfg["parameters_file"], usecols=["mass_1", "mass_2"])
        total = df["mass_1"] + df["mass_2"]
        return float(total.min()), float(total.max())
    masses = cfg["prior"]["masses"]
    if masses["distribution"] == "components_uniform":
        lo, hi = masses["range"]
        return 2 * lo, 2 * hi
    # M = Mc q^(-3/5) (1 + q)^(6/5): creciente en Mc y decreciente en q.
    (mc_lo, mc_hi), (q_lo, q_hi) = masses["chirp_mass_range"], masses["q_range"]
    f = lambda q: q ** (-3.0 / 5.0) * (1.0 + q) ** (6.0 / 5.0)
    return mc_lo * f(q_hi), mc_hi * f(q_lo)


def _resolve(path, base):
    p = Path(path).expanduser()
    return str((p if p.is_absolute() else base / p).resolve())


def load_config(path, overrides=None):
    """Lee el JSON, aplica valores por defecto y overrides de la CLI y valida. Devuelve (cfg, avisos)."""
    path = Path(path)
    _check(path.is_file(), f"No existe el fichero de configuración: {path}")
    try:
        user = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        _fail(f"{path} no es JSON válido: {exc}")
    _check(isinstance(user, dict), f"{path}: la configuración debe ser un objeto JSON")
    cfg = _merge(DEFAULT_CONFIG, {k: v for k, v in user.items() if k not in SECTIONS}, "la raíz")
    for section in SECTIONS:
        if section in user:
            _check(isinstance(user[section], dict), f"'{section}' debe ser un objeto JSON")
            cfg[section] = _merge(DEFAULT_CONFIG[section], user[section], f"'{section}'")
    base = path.resolve().parent
    if cfg["noise"]["psd_pack"]:
        cfg["noise"]["psd_pack"] = _resolve(cfg["noise"]["psd_pack"], base)
    if cfg["parameters_file"]:
        cfg["parameters_file"] = _resolve(cfg["parameters_file"], base)
    # Overrides de la CLI (rutas relativas al directorio actual).
    overrides = overrides or {}
    if overrides.get("n_samples") is not None:
        cfg["n_samples"] = int(overrides["n_samples"])
    if overrides.get("psd_pack") is not None:
        cfg["noise"]["psd_pack"] = str(Path(overrides["psd_pack"]).expanduser().resolve())
    warnings = validate(cfg)
    return cfg, warnings


def validate(cfg):
    warnings = []
    mode = cfg["mode"]
    _check(mode in MODES, f"mode debe ser uno de {list(MODES)}; recibido {mode!r}")
    _check(isinstance(cfg["n_samples"], int) and not isinstance(cfg["n_samples"], bool) and cfg["n_samples"] > 0,
           "n_samples debe ser un entero positivo")
    _check(isinstance(cfg["seed"], int) and not isinstance(cfg["seed"], bool) and cfg["seed"] >= 0,
           "seed debe ser un entero >= 0")
    _check(isinstance(cfg["click_free"], bool), "click_free debe ser true o false")
    outputs = cfg["outputs"]
    _check(isinstance(outputs, list) and outputs and len(set(outputs)) == len(outputs),
           "outputs debe ser una lista no vacía y sin repetidos")
    bad = [o for o in outputs if o not in OUTPUT_NAMES]
    _check(not bad, f"outputs desconocidas {bad}; permitidas: {list(OUTPUT_NAMES)}")
    if mode == "noise_only":
        _check(not set(outputs) & set(SIGNAL_OUTPUTS),
               f"En mode 'noise_only' no hay señal: quita {sorted(set(outputs) & set(SIGNAL_OUTPUTS))} de outputs")
        _check(not cfg["parameters_file"], "parameters_file no se usa en mode 'noise_only'")

    noise = cfg["noise"]
    _check(noise["source"] in ("psd_pack", "analytic"), "noise.source debe ser 'psd_pack' o 'analytic'")
    if noise["source"] == "psd_pack":
        _check(noise["psd_pack"], "noise.source = 'psd_pack' necesita noise.psd_pack (o --psd-pack en la CLI)")
        _check(Path(noise["psd_pack"]).is_file(), f"No existe noise.psd_pack: {noise['psd_pack']}")
        _check(isinstance(noise["splits"], list) and noise["splits"] and set(noise["splits"]) <= set(SPLITS),
               f"noise.splits debe ser una lista no vacía con elementos de {list(SPLITS)}")
        _check(float(cfg["simulation"].get("sampling_frequency", 0)) <= 4096,
               "Con noise.source = 'psd_pack', simulation.sampling_frequency debe ser <= 4096 Hz: "
               "el paquete de PSDs se estimó con datos a 4096 Hz y llega a 2048 Hz")
    else:
        noise["gps_range"] = _range(noise, "gps_range", "noise", positive=True)

    if cfg["parameters_file"]:
        _check(Path(cfg["parameters_file"]).is_file(), f"No existe parameters_file: {cfg['parameters_file']}")
        from .priors import read_parameters_file
        try:
            read_parameters_file(cfg["parameters_file"], mode)
        except ValueError as exc:
            _fail(str(exc))
        warnings.append("parameters_file: se ignoran n_samples y el prior; una fila por señal")
    if mode != "noise_only" and not cfg["parameters_file"]:
        _validate_prior(cfg["prior"], mode)

    # Física: las mismas comprobaciones que SimulationConfig / SignalProcessor / check_physics.
    from .recipe import Physics
    try:
        Physics(cfg["simulation"], cfg["signal_processor"])
    except (ValueError, TypeError) as exc:
        _fail(f"simulation / signal_processor: {exc}")
    for section, reference in (("simulation", DEFAULT_SIMULATION), ("signal_processor", DEFAULT_SIGNAL_PROCESSOR)):
        changed = sorted(k for k in reference if cfg[section].get(k) != reference[k])
        if changed:
            warnings.append(f"{section}: {changed} distintos de la configuración por defecto; esa física no está validada")

    approximant = cfg["simulation"]["waveform_approximant"]
    _check(approximant not in PRECESSING_APPROXIMANTS,
           f"{approximant} tiene precesión; gwsim solo admite espines alineados")
    if mode != "noise_only":
        if approximant != VALIDATED_APPROXIMANT and cfg["click_free"]:
            warnings.append(f"La señal sin clic (bandas de arranque baja y alta) solo se ha validado con {VALIDATED_APPROXIMANT}; con "
                            f"{approximant} revisa el resultado o usa click_free: false")
        if not cfg["click_free"]:
            warnings.append("click_free: false inyecta A, con el clic de 30 Hz, en vez de la señal sin clic")
        m_lo, m_hi = _mtot_bounds(cfg)
        if m_hi > M_TOT_UNVALIDATED:
            warnings.append(f"M_tot puede llegar a {m_hi:.1f} > {M_TOT_UNVALIDATED:g}: región sin validar")
        if m_hi > M_SWITCH:
            warnings.append(f"M_tot puede superar {M_SWITCH:g}: se usa la banda de arranque alta (10→14 Hz), "
                            "validada a nivel de forma de onda hasta M_tot = 250; equivalencia con la producción "
                            "no comprobada en esta región")
        if m_lo < M_TOT_MAY_NOT_GENERATE:
            warnings.append(f"M_tot puede bajar a {m_lo:.1f} < {M_TOT_MAY_NOT_GENERATE:g}: algunas formas de onda "
                            "pueden no generarse (esas filas quedarán con status = 2)")
    return warnings


def config_json(cfg):
    return json.dumps(cfg, sort_keys=True)
