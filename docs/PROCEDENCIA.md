# Procedencia del código

Todo el código de física se ha copiado del repositorio CBC-PE (`cbc_pe/`, rama
`m12-estimator-regression-bias`, commit **4b34677**). gwsim no importa nada de ese repositorio. Cada
función o clase copiada lleva una cabecera `# Copiado de <fichero>:<función> @ 4b34677; cambios: …`.

En todo lo que afecta a los números (orden de operaciones, tipos y llamadas a PyCBC/LAL) la copia es
literal. Solo se han quitado las comprobaciones de procedencia (hashes, manifest, rutas de `/data`),
se han cambiado los imports y, donde el original leía objetos globales de módulo, se pasan como
argumento.

## Módulos copiados enteros (`gwsim/physics/`)

En todos ellos, la única diferencia con el original son las cabeceras añadidas. Los imports
relativos no cambian.

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
| `physics/g1.py:require` | `scripts/generate_m11_final_G1.py:require` | ninguno |
| `physics/g1.py:normalize_input_per_sample_per_detector_zscore` | `src/models/dataset.py` | ninguno |
| `physics/g1.py:interpolate_raw_psd` | `scripts/generate_m11_final_G1.py:estimate_raw_offsource_psd` | solo la parte posterior a `psd(8.0)`; recibe la PSD de Welch |
| `physics/g1.py:estimate_block_psds_for_detector` | `scripts/generate_m11_final_G1.py:estimate_block_psds_for_detector` | recibe `config` y la PSD de Welch |
| `physics/g1.py:build_signal_network_for_params` | `scripts/generate_m11_final_G1.py` | globales → `phys` |
| `physics/g1.py:build_gaussian_processing_noise` | `scripts/generate_m11_final_G1.py` | `CONFIG` → `phys.config` |
| `physics/g1.py:stack_and_normalize_network` | `scripts/generate_m11_final_G1.py` | `CONFIG` → `phys.config` |
| `physics/g1.py:stack_raw` | `scripts/m13/generate_m13_6_clean.py:stack_raw` | `g1.CONFIG` → `phys.config` |
| `physics/g1.py:to_fp16` (+ `FP16_ABS_LIMIT`) | `scripts/m13/generate_m13_6_clean.py:to_fp16` | ninguno |
| `physics/clickfree.py:_opt_click_free` | `scripts/m13/m13_11_prepare_rows.py:_opt_click_free` | `CLICK_F_LOWER` y `CLICK_TAPER_END_HZ` → argumentos |
| `physics/clickfree.py:_advance` | `scripts/m13/m13_11_prepare_rows.py:_advance` | ninguno |
| `physics/clickfree.py:_align_to_A` | `scripts/m13/m13_11_prepare_rows.py:_align_to_A` | ninguno |
| `physics/clickfree.py:opt_click_free_aligned` | `scripts/m13/m13_11_prepare_rows.py:_opt_click_free_aligned` | recibe la banda |
| constantes `CLICK_BETA`, `CLICK_F_REF`, `CLICK_ALIGN_*` | `scripts/m13/m13_11_prepare_rows.py` | ninguno |
| `physics/clickfree.py:edge_taper` (+ `EDGE_TAPER_S`, `EDGE_BETA`) | `scripts/m13/m13_13_build_g3t.py:edge_taper` | ninguno |
| `physics/clickfree.py:s15_strains` | `scripts/m13/m13_13_build_g3t.py:s15_strains` y `scripts/m15/m15_build_g4.py:s15_strains_g4` | proyector → `phys.projector`; `cap.params` y `cap.geocentric_time` → argumentos; la banda (de `taper_band(*band_for(...))`) → argumento |
| `recipe.py:band_for` (+ `M_SWITCH`, `BAND_LOW`, `BAND_HIGH`) | `scripts/m15/m15_build_g4.py:band_for` | ninguno |
| `recipe.py:crop_to_final` | `scripts/m13/m13_11_prepare_rows.py:crop_to_final` | `g1.CONFIG` → `phys.config` |
| `recipe.py:process_clean` | `scripts/m13/m13_11_prepare_rows.py:process_clean` | `g1.CONFIG` y `g1.PROCESSOR` → `phys` |
| `recipe.py:Physics` | `generate_m11_final_G1.py:initialize_physics`, `generate_m13_G3_whitening.py:initialize_g3_physics` y `check_physics` | objeto en vez de globales; la configuración G3 se construye directamente con el cutoff de 1024 Hz (en G3 se hace con `dataclasses.replace`, que da el mismo `SimulationConfig`) |
| `recipe.py:td_waveform_kwargs` | `m13_11_prepare_rows.py:capture_row` (`opt_args`, de `_TdArgsTap`) y `src/waveform.py:WaveformGenerator.generate` | construye los mismos argumentos de `get_td_waveform` en vez de interceptar la llamada |
| `recipe.py:simulate_row` | `scripts/generate_m11_final_G1.py:build_g1_row` y `scripts/m13/m13_13_build_g3t.py:noisy_row` | globales → `phys`; la ventana se deriva de `gps` (o de `opts.processing_start`) en vez de leerse del manifest; modos `distance` y `noise_only`; sin G-REC ni G-NOISE ni el procesado de A, que X_T no usa; salidas opcionales |
| `recipe.py:G3T_SIMULATION`, `G3T_SIGNAL_PROCESSOR` | `configs/generation/generate_m11_final_G1_500k.json` + overrides de `generate_m13_G3_band1024_w16.json` | valores copiados |
| `noise.py:PsdPackNoise.sample_assignments` | `scripts/prepare_m11_final_manifest.py:sample_assignments` | un único sorteo sobre la unión de los splits elegidos; arrays del paquete en vez del DataFrame. Mismo orden de llamadas al generador por fila, pero no reproduce los sorteos de G1 (otra semilla y otra población) |
| `tools/export_psd_pack.py:load_bank` | `scripts/prepare_m11_final_manifest.py:load_bank` | recibe la sección `bank` copiada; ruta canónica en vez de `resolve_processed_artifact`; **conserva** las sumas sha256 del banco |
| `tools/export_psd_pack.py:resolve_cached_psd_files` | `scripts/generate_m11_final_G1.py` | no devuelve la procedencia JSON |
| `tools/export_psd_pack.py:read_psd_intervals` | `scripts/generate_m11_final_G1.py` | ninguno |
| `tools/export_psd_pack.py:welch_offsource_psd` | `scripts/generate_m11_final_G1.py:estimate_raw_offsource_psd` | solo la parte anterior a la interpolación |
| `tools/check_equivalence_g3t.py:adjust_processing_windows` | `scripts/m13/generate_m13_G3_whitening.py` | contextos y `dt`/`duration` como argumentos; sin `print` |

## Código nuevo (no copiado)

- `priors.py`: muestreo configurable. Usa las mismas fórmulas isótropas que
  `src/sampling.py:ParameterSampler`, pero vectorizadas y con otro orden de sorteo.
- `config.py`: carga y validación.
- `io.py`: HDF5 reanudable.
- `__main__.py`: CLI y procesos.
- `noise.py`: lectura del paquete y PSD analíticas (salvo `sample_assignments`).
- `tools/*` (salvo lo indicado en la tabla).
