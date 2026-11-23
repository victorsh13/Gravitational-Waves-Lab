"""Receta por fila de gwsim (la física por defecto reproduce la del proyecto de origen; docs/PROCEDENCIA.md).

    simulate_row(params, gps, psds, placement_seed, gaussian_seed, target_snr=... | distance=..., opts=...)

Pasos (configuración por defecto):
  1. A = SEOBNRv4_opt desde 30 Hz a 1000 Mpc; colocación random_contained (placement_seed); reproyección
     en el GPS real; SNR de red con la PSD "snr" (30-1024 Hz); distancia reescalada a la SNR objetivo;
     A regenerada a esa distancia y comprobación de la SNR (tolerancia relativa 0,05).
  2. Ruido gaussiano de longitud de procesado con la PSD "proc"; semillas por detector derivadas de
     gaussian_seed: default_rng(gaussian_seed).integers(0, 2**32 - 1) para H1, L1 y V1.
  3. Señal S sin clic: banda de arranque baja (15 Hz, taper hasta 25 Hz) si M_tot <= 180, alta (10 Hz,
     taper hasta 14 Hz) si no; alineada con A, con taper de borde.
  4. s + n -> process_network (PSD "proc") -> z-score por muestra y detector -> float32.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics.config import SimulationConfig
from .physics.detectors import DetectorProjector
from .physics.injection import SignalInjector
from .physics.parameters import CBCParameters
from .physics.processing import SignalProcessor
from .physics.snr import (compute_network_optimal_snr, rescale_distance_for_target_network_snr,
                          validate_snr_rescaling)
from .physics.waveform import WaveformGenerator
from .physics.windowing import ProjectedNetworkWindowSelector
from .physics.steps import (DETECTORS, build_gaussian_processing_noise, build_signal_network_for_params,
                         require, stack_and_normalize_network, stack_raw)
from .physics.clickfree import AlignmentError, align_abort, clickfree_strains

# Copiado de configs/generation/generate_m11_final_G1_500k.json ("simulation", "signal_processor") con los overrides de configs/generation/generate_m13_G3_band1024_w16.json @ 4b34677; cambios: nombres de las constantes.
DEFAULT_SIMULATION = {
    "simulation_regime": "BBH",
    "waveform_family": "IMR",
    "waveform_approximant": "SEOBNRv4_opt",
    "sampling_frequency": 4096,
    "duration": 4.0,
    "processing_context_start_samples": 33408,
    "processing_context_end_samples": 33408,
    "low_frequency_cutoff": 30.0,
    "snr_high_frequency_cutoff": 1024.0,
    "snr_on_truncated_signal": True,
    "snr_relative_tolerance": 0.05,
    "truncation_policy": "keep_last_segment",
    "required_final_duration": 1.0,
    "safe_margin_start": 0.0,
    "safe_margin_end": 0.0,
}
DEFAULT_SIGNAL_PROCESSOR = {
    "whitening_method": "psd",
    "apply_highpass": True,
    "apply_lowpass": True,
    "apply_standardization": False,
    "output_mode": "crop_to_config",
    "whitening_low_frequency_cutoff": 30.0,
    "whitening_max_filter_duration": 16.0,
    "whitening_trunc_method": "hann",
    "highpass_frequency": 30.0,
    "lowpass_frequency": 1024.0,
    "fir_order": 256,
    "fir_beta": 5.0,
    "remove_corrupted": True,
}

# Copiado de scripts/m15/m15_build_g4.py (constantes M_SWITCH, BAND_LOW, BAND_HIGH; enmienda 2) @ 4b34677.
M_SWITCH = 180.0
BAND_LOW = (15.0, 25.0)    # banda de arranque baja: 15 Hz, taper hasta 25 Hz
BAND_HIGH = (10.0, 14.0)   # banda de arranque alta (masas grandes): 10 Hz, taper hasta 14 Hz

# Distancia de referencia a la que se mide la SNR antes de reescalar.
REFERENCE_DISTANCE_MPC = 1000.0

MODES = ("snr", "distance", "noise_only")
OUTPUT_NAMES = ("X", "X_noise", "X_unnormalized", "signal_whitened", "signal_projected", "strain_noisy_raw")
SIGNAL_OUTPUTS = ("signal_whitened", "signal_projected")
PARAMETER_NAMES = ("mass_1", "mass_2", "spin_1z", "spin_2z", "inclination", "ra", "dec", "polarization_angle")


# Copiado de scripts/m15/m15_build_g4.py:band_for @ 4b34677; cambios: ninguno
def band_for(total_mass):
    return BAND_LOW if float(total_mass) <= M_SWITCH else BAND_HIGH


class Physics:
    """Objetos de la física, construidos una vez: config (SimulationConfig), waveform_generator,
    projector, window_selector y processor. Sin argumentos, la física por defecto.
    """

    # Copiado de scripts/generate_m11_final_G1.py:initialize_physics y scripts/m13/generate_m13_G3_whitening.py:initialize_g3_physics @ 4b34677; cambios: objeto en vez de globales de módulo.
    def __init__(self, simulation: dict | None = None, signal_processor: dict | None = None):
        self.simulation = {**DEFAULT_SIMULATION, **(simulation or {})}
        self.signal_processor = {**DEFAULT_SIGNAL_PROCESSOR, **(signal_processor or {})}
        self.config = SimulationConfig(**self.simulation)
        self.waveform_generator = WaveformGenerator(self.config)
        self.projector = DetectorProjector(list(DETECTORS))
        self.window_selector = ProjectedNetworkWindowSelector(self.config)
        self.processor = SignalProcessor(config=self.config, **self.signal_processor)
        # Copiado de scripts/m13/generate_m13_G3_whitening.py:check_physics (última comprobación).
        margin = self.processor.corrupted_margin_samples_per_side()
        ctx_start = self.config.processing_context_start_samples
        ctx_end = self.config.processing_context_end_samples
        require(ctx_start >= margin and ctx_end >= margin,
                f"Context per side ({ctx_start}, {ctx_end}) < corrupted margin {margin} samples")
        require(self.processor.output_mode == "crop_to_config",
                "gwsim assumes signal_processor.output_mode='crop_to_config'")


def td_waveform_kwargs(config, parameters):
    """Argumentos con los que WaveformGenerator.generate llama a get_td_waveform (physics/waveform.py).

    Mismas claves y mismos valores que esa llamada; son los argumentos de A con los que se genera S.
    """
    return dict(
        mass1=parameters.mass_1,
        mass2=parameters.mass_2,
        distance=parameters.distance,
        inclination=parameters.inclination,
        spin1z=parameters.spin_1z,
        spin2z=parameters.spin_2z,
        f_lower=config.low_frequency_cutoff,
        delta_t=config.delta_t,
        approximant=config.waveform_approximant,
    )


# Copiado de scripts/m13/m13_11_prepare_rows.py:crop_to_final @ 4b34677; cambios: g1.CONFIG -> phys.config.
def crop_to_final(phys, series, final_start, injector):
    """Projected signal cropped to the final window, with the generator's injection rounding."""
    zero = injector.build_zero_strain(start_time=final_start, length=phys.config.length)
    return injector.inject(zero, series).strain


# Copiado de scripts/m13/m13_11_prepare_rows.py:process_clean @ 4b34677; cambios: g1.CONFIG / g1.PROCESSOR -> phys.config / phys.processor.
def process_clean(phys, strains, *, processing_start, psds, injector):
    """L(s): noiseless projection in zero processing-length strain, processor, no z-score."""
    zeros = {ifo: injector.build_zero_strain(start_time=processing_start, length=phys.config.processing_length)
             for ifo in DETECTORS}
    injected = {ifo: injector.inject(zeros[ifo], strains[ifo]).strain for ifo in DETECTORS}
    return stack_raw(phys, phys.processor.process_network(strains=injected, psds=psds))


@dataclass
class RowOptions:
    physics: Physics
    mode: str = "snr"                   # "snr" | "distance" | "noise_only"
    outputs: tuple = ("X",)             # subconjunto de OUTPUT_NAMES
    click_free: bool = True             # True: señal S sin clic (por defecto); False: A, con el clic de 30 Hz
    # Inicio de la ventana de procesado. None: gps - duration/2 - contexto (la ventana final queda centrada
    # en gps). La prueba de equivalencia pasa el valor de la referencia para no depender del redondeo de
    # esa resta.
    processing_start: float | None = None


def _nan_metadata():
    meta = {name: float("nan") for name in (
        "reference_network_snr", "final_network_snr", "final_distance", "actual_geocentric_time",
        "placement_offset_s", "full_network_duration", "required_final_duration",
        "band_f_lower", "band_taper_end_hz", "align_residual_rad")}
    meta["is_truncated"] = False
    for ifo in DETECTORS:
        meta[f"snr_{ifo}"] = float("nan")
        meta[f"edge_samples_{ifo}"] = -1
    return meta


# Copiado de scripts/generate_m11_final_G1.py:build_g1_row y scripts/m13/m13_13_build_g3t.py:noisy_row @ 4b34677; cambios: globales -> `phys`; ventana derivada de gps u opts.processing_start; modos distance y noise_only; sin las comprobaciones de reconstrucción ni el procesado de A; salidas opcionales (detalle en docs/PROCEDENCIA.md).
def simulate_row(params, gps, psds, placement_seed, gaussian_seed, *, target_snr=None, distance=None,
                 opts: RowOptions):
    """Una fila. Devuelve (salidas, metadatos).

    params: mapeo con PARAMETER_NAMES (la distancia se ignora: la de referencia es 1000 Mpc).
    gps: centro de la ventana final.
    psds: {"snr": {ifo: FrequencySeries}, "proc": {ifo: FrequencySeries}}.
    salidas: dict nombre -> array (3, length) float32, solo las pedidas en opts.outputs.
    """
    phys = opts.physics
    CONFIG = phys.config
    mode = opts.mode
    require(mode in MODES, f"Unknown mode {mode!r}; expected one of {MODES}")
    unknown = set(opts.outputs) - set(OUTPUT_NAMES)
    require(not unknown, f"Unknown outputs {sorted(unknown)}; expected a subset of {OUTPUT_NAMES}")
    if mode == "snr":
        require(target_snr is not None and np.isfinite(float(target_snr)) and float(target_snr) > 0,
                "mode 'snr' needs a positive target_snr")
    if mode == "distance":
        require(distance is not None and np.isfinite(float(distance)) and float(distance) > 0,
                "mode 'distance' needs a positive distance")
    if mode == "noise_only":
        require(not set(opts.outputs) & set(SIGNAL_OUTPUTS), "noise_only has no signal outputs")

    gps = float(gps)
    if opts.processing_start is None:
        processing_start = gps - 0.5 * CONFIG.duration - CONFIG.processing_context_start_seconds
    else:
        processing_start = float(opts.processing_start)

    final_start = (
        processing_start
        + CONFIG.processing_context_start_seconds
    )
    final_end = final_start + CONFIG.duration
    final_center = 0.5 * (final_start + final_end)

    require(abs(final_center - gps) < CONFIG.delta_t,
            "Reference GPS does not match assigned final segment")

    gaussian_rng = np.random.default_rng(
        int(gaussian_seed)
    )

    gaussian_seeds = {
        ifo: int(gaussian_rng.integers(0, 2**32 - 1))
        for ifo in DETECTORS
    }

    require(len(set(gaussian_seeds.values())) == 3, "Detector Gaussian seed collision")

    psds_emp_snr = psds["snr"]
    psds_emp_proc = psds["proc"]
    meta = _nan_metadata()

    if mode == "noise_only":
        # inject/build_zero_strain/set_strain_start_time never draw from rng.
        injector = SignalInjector(config=CONFIG, rng=np.random.default_rng(0))
        noise = build_gaussian_processing_noise(
            phys, psds_proc=psds_emp_proc, processing_start=processing_start,
            detector_seeds=gaussian_seeds, injector=injector,
        )
        signal_strains = None
        injected_strains = {ifo: noise[ifo] for ifo in DETECTORS}
    else:
        placement_rng = np.random.default_rng(
            int(placement_seed)
        )

        injector = SignalInjector(
            config=CONFIG,
            rng=placement_rng,
        )

        params_ref = CBCParameters(
            mass_1=float(params["mass_1"]),
            mass_2=float(params["mass_2"]),
            distance=float(REFERENCE_DISTANCE_MPC),
            inclination=float(params["inclination"]),
            ra=float(params["ra"]),
            dec=float(params["dec"]),
            spin_1z=float(params["spin_1z"]),
            spin_2z=float(params["spin_2z"]),
            polarization_angle=float(params["polarization_angle"]),
        )

        # Provisional placement geometry at reference distance.
        waveform_ref = phys.waveform_generator.generate(params_ref)

        projection_provisional = phys.projector.project(
            h_plus=waveform_ref.h_plus,
            h_cross=waveform_ref.h_cross,
            parameters=params_ref,
            geocentric_coalescence_time=final_center,
        )

        windowed_provisional = phys.window_selector.select(
            projected_strains=projection_provisional.strains,
            max_duration=CONFIG.duration,
        )

        abstract_placement = (
            injector.choose_segment_placement_containing_network(
                signals=windowed_provisional.strains,
                placement_policy="random_contained",
                safe_margin_start=float(CONFIG.safe_margin_start),
                safe_margin_end=float(CONFIG.safe_margin_end),
                enforce_safe_margins=True,
            )
        )

        placement_shift = (
            final_start
            - float(abstract_placement.segment_start_time)
        )
        geocentric_time = final_center + placement_shift
        placement_offset_s = geocentric_time - final_center

        # Final reference projection at actual placement time.
        projection_ref = phys.projector.project(
            h_plus=waveform_ref.h_plus,
            h_cross=waveform_ref.h_cross,
            parameters=params_ref,
            geocentric_coalescence_time=geocentric_time,
        )

        windowed_ref = phys.window_selector.select(
            projected_strains=projection_ref.strains,
            max_duration=CONFIG.duration,
        )

        tol = 2.0 * CONFIG.delta_t

        if (
            windowed_ref.metadata.used_window_start_time
            < final_start - tol
        ):
            raise ValueError(
                "signal starts outside final segment."
            )

        if (
            windowed_ref.metadata.used_window_end_time
            > final_end + tol
        ):
            raise ValueError(
                "signal ends outside final segment."
            )

        zero_final = {
            ifo: injector.build_zero_strain(
                start_time=final_start,
                length=CONFIG.length,
            )
            for ifo in DETECTORS
        }

        signal_only_ref_results = injector.inject_network(
            noises=zero_final,
            signals=windowed_ref.strains,
        )

        signal_segments_ref = {
            ifo: signal_only_ref_results[ifo].strain
            for ifo in DETECTORS
        }

        _, snr_ref = compute_network_optimal_snr(signal_segments_ref, psds_emp_snr, CONFIG)
        if mode == "snr":
            target_snr = float(target_snr)
            distance = rescale_distance_for_target_network_snr(
                params_ref.distance, snr_ref, target_snr,
            )
        else:
            # Modo distance: distancia física dada, sin reescalado (cambio respecto a build_g1_row).
            distance = float(distance)
        params_final = params_ref.with_distance(distance)
        signal = build_signal_network_for_params(
            phys, params_final, geocentric_time=geocentric_time,
            final_start=final_start, injector=injector,
        )
        detector_snrs, final_snr = compute_network_optimal_snr(
            signal["segments"], psds_emp_snr, CONFIG,
        )
        if mode == "snr":
            validate_snr_rescaling(final_snr, target_snr, CONFIG.snr_relative_tolerance)
        noise = build_gaussian_processing_noise(
            phys, psds_proc=psds_emp_proc, processing_start=processing_start,
            detector_seeds=gaussian_seeds, injector=injector,
        )

        meta.update({
            "reference_network_snr": snr_ref,
            "final_network_snr": final_snr,
            "final_distance": distance,
            "actual_geocentric_time": geocentric_time,
            "placement_offset_s": placement_offset_s,
            "full_network_duration": windowed_ref.metadata.full_network_duration,
            "required_final_duration": windowed_ref.metadata.required_available_final_duration,
            "is_truncated": windowed_ref.metadata.is_truncated,
        })
        for ifo in DETECTORS:
            meta[f"snr_{ifo}"] = detector_snrs[ifo]

        if opts.click_free:
            # Misma fuente, t_c, ruido y PSD; solo cambia la señal (S en vez de A).
            # La distancia y la SNR guardadas son las de A.
            opt_args = td_waveform_kwargs(CONFIG, params_final)
            band = band_for(float(opt_args["mass1"]) + float(opt_args["mass2"]))
            signal_strains, n_before, align = clickfree_strains(
                phys, opt_args, params_final, float(geocentric_time), processing_start, band)
            # Alineación fallida (|residuo| > ALIGN_ABORT o NaN): la fila falla (status = 2) y se sigue.
            if align_abort(align["residual_rad"]):
                raise AlignmentError(f"alineación: residuo {float(align['residual_rad']):.6g} rad")
            meta["band_f_lower"], meta["band_taper_end_hz"] = float(band[0]), float(band[1])
            meta["align_residual_rad"] = float(align["residual_rad"])
            for ifo in DETECTORS:
                meta[f"edge_samples_{ifo}"] = int(n_before[ifo])
        else:
            # Señal A (signal["projection"].strains), con el clic de 30 Hz.
            signal_strains = signal["projection"].strains

        injected = injector.inject_network(noises=noise, signals=signal_strains)
        injected_strains = {ifo: injected[ifo].strain for ifo in DETECTORS}

    processed = phys.processor.process_network(strains=injected_strains, psds=psds_emp_proc)
    X_raw, X = stack_and_normalize_network(phys, processed)
    require(np.isfinite(X).all(), "Non-finite normalized input")

    for ifo in DETECTORS:
        meta[f"gaussian_seed_{ifo}"] = gaussian_seeds[ifo]

    outputs = {}
    if "X" in opts.outputs:
        outputs["X"] = np.asarray(X, dtype=np.float32)
    if "X_unnormalized" in opts.outputs:
        outputs["X_unnormalized"] = np.asarray(X_raw, dtype=np.float32)
    if "X_noise" in opts.outputs:
        # Misma realización de ruido, mismo procesado y z-score propio.
        _, X_noise = stack_and_normalize_network(
            phys, phys.processor.process_network(strains=noise, psds=psds_emp_proc))
        outputs["X_noise"] = np.asarray(X_noise, dtype=np.float32)
    if "signal_whitened" in opts.outputs:
        outputs["signal_whitened"] = np.asarray(process_clean(
            phys, signal_strains, processing_start=processing_start, psds=psds_emp_proc,
            injector=injector), dtype=np.float32)
    if "signal_projected" in opts.outputs:
        outputs["signal_projected"] = np.stack(
            [np.asarray(crop_to_final(phys, signal_strains[ifo], final_start, injector)) for ifo in DETECTORS],
            axis=0).astype(np.float32)
    if "strain_noisy_raw" in opts.outputs:
        # La ventana final ocupa las muestras [contexto, contexto + length) del segmento de procesado.
        c0 = CONFIG.processing_context_start_samples
        outputs["strain_noisy_raw"] = np.stack(
            [np.asarray(injected_strains[ifo])[c0:c0 + CONFIG.length] for ifo in DETECTORS],
            axis=0).astype(np.float32)
    return outputs, meta
