# Salidas: esquema del HDF5

Un fichero contiene n filas, una por muestra. Todos los datasets tienen n como primera dimensión.

## Arrays de entrada (solo los pedidos en `outputs`)

Todos son float32 con forma `(n, 3, 16384)` (detectores H1, L1, V1 × 4 s a 4096 Hz). Se guardan
comprimidos con lzf, en bloques de una fila. Salen de la misma realización de ruido de la fila.

| Nombre | Contenido |
|---|---|
| `X` | z-score de process(s + n): la entrada de G3T. |
| `X_noise` | z-score de process(n), con su propia media y desviación. Es la pareja «solo ruido» de `X`. |
| `X_unnormalized` | process(s + n) sin z-score. |
| `signal_whitened` | process(s), sin ruido ni z-score (`process_clean`). Como el procesado es lineal, `X_unnormalized` = process(n) + `signal_whitened`. |
| `signal_projected` | s cruda (strain) en la ventana final, sin procesar. |
| `strain_noisy_raw` | s + n cruda (strain) en la ventana final, sin procesar. |

En las filas con `status != 1`, los arrays valen 0.

## Por fila

| Nombre | Tipo | Contenido |
|---|---|---|
| `status` | uint8 | 0 pendiente, 1 hecha, 2 fallida. |
| `error` | texto | Mensaje de la excepción si `status = 2`. |
| `mass_1`, `mass_2` | f64 | M☉ en el marco del detector, con m₁ ≥ m₂. |
| `spin_1z`, `spin_2z` | f64 | Espines alineados. |
| `inclination`, `ra`, `dec`, `polarization_angle` | f64 | En radianes. |
| `chirp_mass`, `total_mass`, `q`, `chi_eff` | f64 | Derivados. |
| `y` | f32 (n, 3) | `[chirp_mass, total_mass, chi_eff]`, en el orden de las etiquetas del proyecto. |
| `target_network_snr` | f64 | SNR objetivo (modo `snr`; NaN en los demás). |
| `distance_input` | f64 | Distancia dada (modo `distance`; NaN en los demás). |
| `final_distance` | f64 | Distancia de la señal inyectada, en Mpc. |
| `reference_network_snr` | f64 | SNR de red de A a 1000 Mpc. |
| `final_network_snr`, `snr_H1`, `snr_L1`, `snr_V1` | f64 | SNR óptima final (de A; 30–1024 Hz, PSD "snr"). |
| `gps` | f64 | Centro de la ventana final. |
| `actual_geocentric_time` | f64 | t_c geocéntrico tras la colocación. |
| `placement_offset_s` | f64 | t_c − `gps`. |
| `full_network_duration`, `required_final_duration` | f64 | Duración de la señal proyectada y tramo final exigido. |
| `is_truncated` | bool | La señal A duraba más de 4 s y se recortó su parte inicial para colocarla. |
| `band_f_lower`, `band_taper_end_hz` | f64 | Banda de S: (15, 25) para S15 y (10, 14) para S10; NaN si no se usó S. |
| `align_residual_rad` | f64 | Fase residual de la alineación de S con A. Si \|residuo\| > 0,5 rad (o NaN), la fila tiene `status = 2` y `error = "G-ALIGN: residuo <valor> rad"`. |
| `edge_samples_H1/L1/V1` | i64 | Muestras de S anteriores a la ventana de procesado (> 0: se aplicó el taper de borde); −1 sin S. |
| `block_id`, `noise_crop_id`, `noise_split` | texto | Bloque, recorte y split del paquete de PSDs (`analytic` / vacío con ruido analítico). |
| `placement_seed`, `gaussian_seed` | i64 | Semillas de la fila (`placement_seed = −1` en `noise_only`). |
| `gaussian_seed_H1/L1/V1` | u32 | Semillas por detector, derivadas de `gaussian_seed`. |

## Atributos

| Atributo | Contenido |
|---|---|
| `config_json` | Configuración completa, con los valores por defecto aplicados. |
| `gwsim_version`, `created_utc` | Versión y fecha de creación. |
| `warnings` | Avisos de la validación (lista JSON). |
| `outputs`, `detectors`, `labels`, `n_rows` | Descripción del contenido. |

## Leerlo con h5py

```python
import json
import h5py
import numpy as np

with h5py.File("senales.h5", "r") as h:
    cfg = json.loads(h.attrs["config_json"])
    ok = np.flatnonzero(h["status"][:] == 1)
    X = h["X"][ok]                    # (n_ok, 3, 16384) float32; h5py exige índices crecientes
    X_noise = h["X_noise"][ok] if "X_noise" in h else None
    y = h["y"][:][ok]                 # [chirp_mass, total_mass, chi_eff]
    snr = h["final_network_snr"][:][ok]
    block = h["block_id"].asstr()[:][ok]
    fallidas = h["error"].asstr()[:][h["status"][:] == 2]
```

Para conjuntos grandes, lee por bloques (`h["X"][i:i + 1024]`) en lugar de cargarlo todo en memoria:
cada fila de `X` ocupa 192 KiB.
