# Procedencia del código

## Origen

Todo el código de física se ha copiado del proyecto de origen:

- repositorio `victorsh13/Gravitational-Waves-Lab`;
- rama `m12-estimator-regression-bias`;
- carpeta `cbc_pe/`;
- commit **`4b34677`**.

gwsim no importa nada de ese repositorio. **La configuración por defecto reproduce la producción G3T
del proyecto de origen**; la equivalencia bit a bit está verificada (ver VALIDEZ.md y MANTENIMIENTO.md).

Cada función o clase copiada lleva una cabecera `# Copiado de <fichero>:<función> @ 4b34677; cambios: …`,
en la que el fichero de origen sirve de referencia. En todo lo que afecta a los números (orden de
operaciones, tipos y llamadas a PyCBC/LAL) la copia es literal. Solo se han quitado las comprobaciones
de procedencia (hashes, manifest, rutas de `/data`), se han cambiado los imports y, donde el original
leía objetos globales de módulo, se pasan como argumento. En la fase 2 solo se cambiaron nombres,
imports, mensajes y textos.

## Nombres neutros → origen

En los documentos de usuario, los mensajes, la CLI y el HDF5, gwsim no usa los nombres internos del
proyecto de origen. Esta es la correspondencia:

| Nombre en gwsim | Nombre en el proyecto de origen |
|---|---|
| configuración por defecto / física por defecto / `configs/default.json` | producción G3T (G3 con la señal sin clic): `configs/generation/generate_m11_final_G1_500k.json` + overrides de `generate_m13_G3_band1024_w16.json`, señal de `scripts/m13/m13_13_build_g3t.py` |
| `DEFAULT_SIMULATION`, `DEFAULT_SIGNAL_PROCESSOR` (`gwsim/recipe.py`) | secciones `simulation` y `signal_processor` de G3 (antes `G3T_SIMULATION`, `G3T_SIGNAL_PROCESSOR` en gwsim 0.1 fase 1) |
| prior por defecto (masas U[5, 90], espines U[−1, 1], SNR U[10, 25)) | prior de G1 (`parameter_sampler` + `target_network_snr_range`) |
| señal A | forma de onda de G1/G3 (SEOBNRv4_opt desde 30 Hz) |
| señal S sin clic | S de M13.11/M13.12/M13.13 |
| banda de arranque baja (15 Hz, taper hasta 25 Hz) | S15 (`S15_F_LOWER`, `S15_TAPER_END_HZ`; `BAND_LOW` de M15) |
| banda de arranque alta (10 Hz, taper hasta 14 Hz) | S10 (`BAND_HIGH` de M15, enmienda 2; G-TAPER-G4b superada hasta M_tot = 250, confirmado el 23-11-2026) |
| referencia de la señal sin clic (10 Hz → 14 Hz) | Sref (`SREF_F_LOWER`, `SREF_TAPER_END_HZ`) |
| validación de la forma de onda (desajuste ≤ 8·10⁻⁴) | puerta G-TAPER |
| alineación de S con A; error `"alineación: residuo <valor> rad"` | puerta G-ALIGN (`ALIGN_TOL = 0.04`, `ALIGN_ABORT = 0.5`, enmienda 3) |
| taper de borde | puerta G-EDGE |
| comprobaciones de reconstrucción del origen, no copiadas | G-REC, G-NOISE |
| aviso de M_tot < 16 | G-GEN (`GEN_M_CHECK`, M15) |
| `clickfree_strains` (`physics/clickfree.py`) | `s15_strains` (M13.13) / `s15_strains_g4` (M15) |
| `physics/steps.py` | (en la fase 1, `physics/g1.py`) funciones de `generate_m11_final_G1.py` y `generate_m13_6_clean.py` |
| paquete `psd_pack_o3b.npz` | PSD de Welch de los 292 bloques del banco de ruido de G1/M11.6 (`m11_6_manifests`), caché `gwosc_cache/m11` |
| splits `train`/`val`/`cal`/`test` del paquete | splits por entorno del banco M11.6, reutilizados en G1 |
| `tools/maintenance/export_psd_pack.py` | (fase 1: `tools/export_psd_pack.py`) |
| `tools/maintenance/check_equivalence_reference.py` | (fase 1: `tools/check_equivalence_g3t.py`); compara con `M13_G3_band1024_w16_500k/m13_13_g3t/fp16_memmap` |
| referencia de la prueba de equivalencia | val de G3T (`X_val.npy`, `row_ids_val.npy`) y manifest de G1 (`M11_final_G1_500k`) |

El ε del z-score, `INPUT_ZSCORE_EPS = 1e-6` (`physics/steps.py`), se llama `M10_INPUT_ZSCORE_EPS` en el
origen; se renombró también dentro de la función copiada `stack_and_normalize_network`, con el mismo valor.

## Módulos copiados enteros (`gwsim/physics/`)

En todos ellos, la única diferencia con el original son las cabeceras añadidas. Los imports relativos
no cambian.

| Fichero gwsim | Origen @ 4b34677 | Contenido |
|---|---|---|
| `physics/config.py` | `src/config.py` | `SimulationConfig` |
| `physics/parameters.py` | `src/parameters.py` | `CBCParameters` |
| `physics/waveform.py` | `src/waveform.py` | `WaveformMetadata`, `GeneratedWaveform`, `WaveformGenerator` |
| `physics/detectors.py` | `src/detectors.py` | `ProjectionMetadata`, `ProjectedStrains`, `DetectorProjector` |
| `physics/windowing.py` | `src/windowing.py` | `NetworkWindowMetadata`, `WindowedProjectedNetwork`, `ProjectedNetworkWindowSelector` |
| `physics/injection.py` | `src/injection.py` | `InjectionResult`, `SegmentPlacement`, `SignalInjector` |
| `physics/snr.py` | `src/snr.py` | `compute_*_snr`, `rescale_distance_for_target_network_snr`, `validate_snr_rescaling`, … |
| `physics/processing.py` | `src/processing.py` | `SignalProcessor` |
| `physics/noise.py` | `src/noise.py` | `NoiseModel` (PSD analíticas) |

## Funciones copiadas de los scripts

| Función gwsim | Origen @ 4b34677 | Cambios |
|---|---|---|
| `physics/steps.py:require` | `scripts/generate_m11_final_G1.py:require` | ninguno |
| `physics/steps.py:normalize_input_per_sample_per_detector_zscore` | `src/models/dataset.py` | ninguno |
| `physics/steps.py:interpolate_raw_psd` | `scripts/generate_m11_final_G1.py:estimate_raw_offsource_psd` | solo la parte posterior a `psd(8.0)`; recibe la PSD de Welch |
| `physics/steps.py:estimate_block_psds_for_detector` | `scripts/generate_m11_final_G1.py` | recibe `config` y la PSD de Welch |
| `physics/steps.py:build_signal_network_for_params` | `scripts/generate_m11_final_G1.py` | globales → `phys` |
| `physics/steps.py:build_gaussian_processing_noise` | `scripts/generate_m11_final_G1.py` | `CONFIG` → `phys.config` |
| `physics/steps.py:stack_and_normalize_network` | `scripts/generate_m11_final_G1.py` | `CONFIG` → `phys.config` |
| `physics/steps.py:stack_raw` | `scripts/m13/generate_m13_6_clean.py` | `g1.CONFIG` → `phys.config` |
| `physics/steps.py:to_fp16` (+ `FP16_ABS_LIMIT`) | `scripts/m13/generate_m13_6_clean.py` | ninguno |
| `physics/clickfree.py:_opt_click_free` | `scripts/m13/m13_11_prepare_rows.py` | `CLICK_F_LOWER` y `CLICK_TAPER_END_HZ` → argumentos |
| `physics/clickfree.py:_advance`, `_align_to_A` | `scripts/m13/m13_11_prepare_rows.py` | ninguno |
| `physics/clickfree.py:opt_click_free_aligned` | `scripts/m13/m13_11_prepare_rows.py:_opt_click_free_aligned` | recibe la banda |
| `physics/clickfree.py:edge_taper` (+ `EDGE_TAPER_S`, `EDGE_BETA`) | `scripts/m13/m13_13_build_g3t.py` | ninguno (solo el docstring) |
| `physics/clickfree.py:align_abort` (+ `ALIGN_ABORT`) | `scripts/m13/m13_13_build_g3t.py` | ninguno |
| `physics/clickfree.py:clickfree_strains` | `scripts/m13/m13_13_build_g3t.py:s15_strains` y `scripts/m15/m15_build_g4.py:s15_strains_g4` | nombre; proyector → `phys.projector`; `cap.params` y `cap.geocentric_time` → argumentos; la banda → argumento |
| `recipe.py:band_for` (+ `M_SWITCH`, `BAND_LOW`, `BAND_HIGH`) | `scripts/m15/m15_build_g4.py` | ninguno |
| `recipe.py:crop_to_final`, `process_clean` | `scripts/m13/m13_11_prepare_rows.py` | globales → `phys` |
| `recipe.py:Physics` | `generate_m11_final_G1.py:initialize_physics`, `generate_m13_G3_whitening.py:initialize_g3_physics` y `check_physics` | objeto en vez de globales; la configuración se construye directamente con el cutoff de 1024 Hz (en el origen, con `dataclasses.replace`, que da el mismo `SimulationConfig`) |
| `recipe.py:td_waveform_kwargs` | `m13_11_prepare_rows.py:capture_row` (`opt_args`) y `src/waveform.py:WaveformGenerator.generate` | construye los mismos argumentos de `get_td_waveform` en vez de interceptar la llamada |
| `recipe.py:simulate_row` | `scripts/generate_m11_final_G1.py:build_g1_row` y `scripts/m13/m13_13_build_g3t.py:noisy_row` | globales → `phys`; ventana derivada de `gps` (o de `opts.processing_start`); modos `distance` y `noise_only`; sin G-REC ni G-NOISE ni el procesado de A; alineación fallida → `status = 2` en vez de parar; salidas opcionales |
| `noise.py:PsdPackNoise.sample_assignments` | `scripts/prepare_m11_final_manifest.py:sample_assignments` | un único sorteo sobre la unión de los splits elegidos; arrays del paquete. Mismo orden de llamadas al generador por fila, pero no reproduce los sorteos de G1 |
| `tools/maintenance/export_psd_pack.py:load_bank` | `scripts/prepare_m11_final_manifest.py:load_bank` | sección `bank` copiada; ruta canónica; conserva las sumas sha256 |
| `tools/maintenance/export_psd_pack.py:resolve_cached_psd_files`, `read_psd_intervals` | `scripts/generate_m11_final_G1.py` | sin la procedencia JSON / ninguno |
| `tools/maintenance/export_psd_pack.py:welch_offsource_psd` | `scripts/generate_m11_final_G1.py:estimate_raw_offsource_psd` | solo la parte anterior a la interpolación |
| `tools/maintenance/check_equivalence_reference.py:adjust_processing_windows` | `scripts/m13/generate_m13_G3_whitening.py` | contextos y `dt`/`duration` como argumentos; sin `print` |

## Código nuevo (no copiado)

- `priors.py`: muestreo configurable. Usa las mismas fórmulas isótropas que
  `src/sampling.py:ParameterSampler`, vectorizadas y con otro orden de sorteo.
- `config.py`: carga y validación.
- `io.py`: HDF5 reanudable, `load` y `estimate_size`.
- `__main__.py`: CLI, procesos y `selftest`.
- `noise.py`: lectura del paquete y PSD analíticas (salvo `sample_assignments`).
- El resto de `tools/maintenance/` (lo no indicado en la tabla anterior).
