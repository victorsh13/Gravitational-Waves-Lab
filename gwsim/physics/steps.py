"""Pasos de la cadena por fila: PSD, señal en la ventana final, ruido gaussiano, apilado y z-score.

Copiados de los generadores del proyecto de origen (ver docs/PROCEDENCIA.md). Allí leen objetos
globales de módulo; aquí reciben un objeto `phys` (gwsim.recipe.Physics) con los mismos atributos:
phys.config, phys.waveform_generator, phys.projector, phys.window_selector, phys.processor. El orden
de operaciones, los tipos y las llamadas a pycbc no cambian.
"""
from __future__ import annotations

import numpy as np
from pycbc.noise import gaussian
from pycbc.psd import interpolate

# Copiado de scripts/generate_m11_final_G1.py (constantes DETECTORS, LABELS, PSD_SEGMENT_DURATION_S, M10_INPUT_ZSCORE_EPS) @ 4b34677; cambios: M10_INPUT_ZSCORE_EPS -> INPUT_ZSCORE_EPS (mismo valor)
DETECTORS = ("H1", "L1", "V1")
LABELS = ("chirp_mass", "total_mass", "chi_eff")
PSD_SEGMENT_DURATION_S = 8.0
INPUT_ZSCORE_EPS = 1e-6
# Copiado de scripts/m13/generate_m13_6_clean.py (constante FP16_ABS_LIMIT) @ 4b34677.
FP16_ABS_LIMIT = 6e4  # float16 max is 65504


# Copiado de scripts/generate_m11_final_G1.py:require @ 4b34677; cambios: ninguno
def require(condition, message):
    if not condition:
        raise ValueError(message)


# Copiado de src/models/dataset.py:normalize_input_per_sample_per_detector_zscore @ 4b34677; cambios: ninguno
def normalize_input_per_sample_per_detector_zscore(
    X: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize one input sample per detector/channel.

    Expected shape:
        X: (C, T)

    Output:
        Each detector channel has approximately mean 0 and std 1.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X shape (C, T), got {X.shape}")

    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)

    return ((X - mean) / (std + eps)).astype(np.float32)


# Copiado de scripts/generate_m11_final_G1.py:estimate_raw_offsource_psd @ 4b34677; cambios: solo la
# parte posterior a `psd = psd_data.psd(8.0)` (la PSD de Welch llega ya calculada desde el paquete de
# PSDs, que la calcula con la parte anterior en tools/maintenance/export_psd_pack.py); se recibe `psd` en vez de
# `strain, psd_start, psd_end`.
def interpolate_raw_psd(
    psd,
    *,
    delta_f: float,
    target_flength: int,
):
    """
    Raw empirical PSD:
        strain -> Welch -> interpolation

    No inverse-spectrum truncation here.
    SignalProcessor applies whitening conditioning exactly once later.
    """
    psd = interpolate(psd, float(delta_f))

    target_flength = int(target_flength)

    if len(psd) > target_flength:
        psd = psd[:target_flength]
    elif len(psd) < target_flength:
        raise ValueError(
            f"Interpolated PSD shorter than target_flength: "
            f"{len(psd)} < {target_flength}"
        )

    if not np.all(np.isfinite(psd.numpy())):
        raise ValueError("Raw empirical PSD contains non-finite values.")

    return psd


# Copiado de scripts/generate_m11_final_G1.py:estimate_block_psds_for_detector @ 4b34677; cambios: recibe
# `config` y la PSD de Welch (`welch_psd`) en vez del strain y el intervalo; llama a interpolate_raw_psd.
def estimate_block_psds_for_detector(
    config,
    *,
    welch_psd,
):
    return {
        "snr": interpolate_raw_psd(
            welch_psd,
            delta_f=config.delta_f,
            target_flength=config.flength,
        ),
        "proc": interpolate_raw_psd(
            welch_psd,
            delta_f=config.processing_delta_f,
            target_flength=config.processing_flength,
        ),
    }


# Copiado de scripts/generate_m11_final_G1.py:build_signal_network_for_params @ 4b34677; cambios: globales
# de módulo -> atributos de `phys`.
def build_signal_network_for_params(
    phys,
    params,
    *,
    geocentric_time: float,
    final_start: float,
    injector,
):
    waveform = phys.waveform_generator.generate(params)

    projection = phys.projector.project(
        h_plus=waveform.h_plus,
        h_cross=waveform.h_cross,
        parameters=params,
        geocentric_coalescence_time=geocentric_time,
    )

    windowed = phys.window_selector.select(
        projected_strains=projection.strains,
        max_duration=phys.config.duration,
    )

    zeros = {
        ifo: injector.build_zero_strain(
            start_time=final_start,
            length=phys.config.length,
        )
        for ifo in DETECTORS
    }

    injected = injector.inject_network(
        noises=zeros,
        signals=windowed.strains,
    )

    segments = {
        ifo: injected[ifo].strain
        for ifo in DETECTORS
    }

    return {
        "waveform": waveform,
        "projection": projection,
        "windowed": windowed,
        "segments": segments,
    }


# Copiado de scripts/generate_m11_final_G1.py:build_gaussian_processing_noise @ 4b34677; cambios: CONFIG
# global -> phys.config.
def build_gaussian_processing_noise(
    phys,
    *,
    psds_proc: dict,
    processing_start: float,
    detector_seeds: dict,
    injector,
):
    noises = {}

    for ifo in DETECTORS:
        noise = gaussian.noise_from_psd(
            psd=psds_proc[ifo],
            length=phys.config.processing_length,
            delta_t=phys.config.delta_t,
            seed=int(detector_seeds[ifo]),
        )

        noises[ifo] = injector.set_strain_start_time(
            strain=noise,
            start_time=processing_start,
            expected_length=phys.config.processing_length,
        )

    return noises


# Copiado de scripts/generate_m11_final_G1.py:stack_and_normalize_network @ 4b34677; cambios: CONFIG global -> phys.config; M10_INPUT_ZSCORE_EPS -> INPUT_ZSCORE_EPS (mismo valor, 1e-6).
def stack_and_normalize_network(phys, network: dict):
    X_raw = np.stack(
        [np.asarray(network[ifo]) for ifo in DETECTORS],
        axis=0,
    )

    expected_shape = (len(DETECTORS), phys.config.length)

    if X_raw.shape != expected_shape:
        raise ValueError(
            f"Unexpected processed network shape: "
            f"{X_raw.shape}, expected={expected_shape}"
        )

    if not np.all(np.isfinite(X_raw)):
        raise ValueError("Processed network contains non-finite values.")

    X = normalize_input_per_sample_per_detector_zscore(
        X_raw,
        eps=INPUT_ZSCORE_EPS,
    )

    return X_raw, X


# Copiado de scripts/m13/generate_m13_6_clean.py:stack_raw @ 4b34677; cambios: g1.CONFIG -> phys.config.
def stack_raw(phys, network):
    X = np.stack([np.asarray(network[ifo], dtype=np.float64) for ifo in DETECTORS], axis=0)
    require(X.shape == (len(DETECTORS), phys.config.length), f"Unexpected network shape {X.shape}")
    return X


# Copiado de scripts/m13/generate_m13_6_clean.py:to_fp16 @ 4b34677; cambios: ninguno
def to_fp16(x, what):
    require(np.isfinite(x).all(), f"Non-finite values in {what}")
    peak = float(np.abs(x).max())
    require(peak <= FP16_ABS_LIMIT, f"|x| = {peak:.4g} > {FP16_ABS_LIMIT:g} in {what}")
    return x.astype(np.float16)
