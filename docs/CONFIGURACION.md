# Configuración

Una configuración es un fichero JSON. Las claves que no pongas toman el valor por defecto, que es la
física de G3T con el prior de G1. Si escribes una clave desconocida, gwsim da un error que lista las
claves permitidas. Los **avisos** (física distinta de G3T, masas fuera de la región validada…) se
imprimen y se guardan en el HDF5, pero no paran la generación.

La configuración completa, con los valores por defecto ya aplicados y las rutas absolutas, se guarda
en `attrs["config_json"]` del HDF5.

## Raíz

| Clave | Por defecto | Qué es |
|---|---|---|
| `description` | `""` | Texto libre. |
| `n_samples` | `10` | Número de filas. Se ignora si hay `parameters_file`. En la CLI, `--n-samples` lo sustituye. |
| `seed` | `0` | Semilla global, un entero ≥ 0 (ver «Semillas»). |
| `mode` | `"snr"` | `"snr"`, `"distance"` o `"noise_only"`. |
| `outputs` | `["X"]` | Subconjunto de `X`, `X_noise`, `X_unnormalized`, `signal_whitened`, `signal_projected` y `strain_noisy_raw` (ver SALIDAS.md). |
| `click_free` | `true` | `true`: señal S15/S10, como G3T. `false`: señal A, con el clic de 30 Hz, como G3. |
| `parameters_file` | `null` | CSV con una fila por señal; sustituye al prior. La ruta es relativa al JSON. |

## `noise`

| Clave | Por defecto | Qué es |
|---|---|---|
| `source` | `"psd_pack"` | `"psd_pack"` (PSD empíricas de O3b) o `"analytic"` (aLIGOZeroDetHighPower para H1 y L1, AdvVirgo para V1). |
| `psd_pack` | `null` | Ruta al `.npz`, relativa al JSON. En la CLI, `--psd-pack` lo sustituye (ruta relativa al directorio actual). |
| `splits` | los cuatro | Splits del paquete que se usan: `train`, `val`, `cal`, `test`. Son entornos de ruido (grupos de ficheros de O3b) disjuntos. Si usas configuraciones con splits distintos para tu entrenamiento y tu evaluación, sus ruidos no comparten entornos. |
| `gps_range` | O3b | Solo con `analytic`: el GPS se sortea uniforme en `[mín, máx]`. |

Con `psd_pack`, cada fila se asigna a un recorte de ruido sorteando, de forma uniforme y en este
orden, un entorno, un bloque dentro del entorno y un recorte dentro del bloque, como en G1. El GPS
de la fila es el centro del recorte (`center_gps`).

```json
"noise": {"source": "psd_pack", "psd_pack": "/datos/psd_pack_g1.npz", "splits": ["train"]}
```

## `prior`

Por defecto, el prior es el de G1: masas U[5, 90] M☉, espines U[−1, 1], ángulos isótropos y
SNR U[10, 25).

| Clave | Formas admitidas |
|---|---|
| `masses` | `{"distribution": "components_uniform", "range": [5, 90]}`: m₁ y m₂ independientes y uniformes, y después ordenadas (m₁ ≥ m₂, intercambiando también los espines). `{"distribution": "chirp_mass_q_uniform", "chirp_mass_range": [10, 40], "q_range": [0.25, 1]}`: 𝓜 y q = m₂/m₁ uniformes. |
| `spin_1z`, `spin_2z` | `{"distribution": "uniform", "range": [-1, 1]}` o `{"fixed": 0.0}`. |
| `inclination` | `{"distribution": "isotropic"}` (cos ι uniforme) o `{"fixed": rad}`. |
| `dec` | `{"distribution": "isotropic"}` (sen δ uniforme) o `{"fixed": rad}`. |
| `ra`, `polarization_angle` | `{"distribution": "uniform"}` (U[0, 2π)) o `{"fixed": rad}`. |
| `target_network_snr` | Modo `snr`: `{"distribution": "uniform" \| "loguniform", "range": [10, 25]}` o `{"fixed": 15}`. |
| `distance` | Modo `distance`, en Mpc: `{"distribution": "uniform_volume", "range": [100, 3000]}` (p(d) ∝ d², volumen euclídeo, sin cosmología), `"uniform"` o `{"fixed": 800}`. |

Si cambias una sola entrada del prior, se sustituye entera, pero las demás conservan su valor por
defecto. Cada parámetro se sortea siempre, aunque lo fijes, de modo que fijar uno no cambia los
valores sorteados de los demás.

```json
"prior": {"masses": {"distribution": "components_uniform", "range": [20, 60]},
          "spin_1z": {"fixed": 0.0}, "spin_2z": {"fixed": 0.0},
          "target_network_snr": {"distribution": "loguniform", "range": [8, 30]}}
```

## `parameters_file`

Es un CSV con una fila por señal. Tiene que traer las columnas `mass_1`, `mass_2`, `spin_1z`,
`spin_2z`, `inclination`, `ra`, `dec` y `polarization_angle` (ángulos en radianes y masas en M☉,
marco del detector), más `target_network_snr` (modo `snr`) o `distance` (modo `distance`, en Mpc).

Columnas opcionales:

- `placement_seed` y `gaussian_seed`: si vienen, sustituyen a las semillas derivadas de `seed`;
- `noise_crop_id`: solo con `psd_pack`; fija el recorte de ruido;
- `gps`: solo con `analytic`.

Ver [examples/params_example.csv](../examples/params_example.csv). Con una fila del manifest de G1
(sus parámetros, `target_network_snr`, `placement_seed`, `gaussian_seed` y `noise_crop_id`) se
obtiene la fila de G3T, salvo por la ventana de procesado: G3 la recalcula a partir de la de G1 y puede diferir
de la que deriva gwsim en el redondeo de coma flotante (< 1 µs). Para la equivalencia exacta se usa
`RowOptions.processing_start` (ver VALIDEZ.md).

## `simulation` y `signal_processor`

Son las mismas claves que las secciones homónimas de las configuraciones de producción
(`SimulationConfig` y `SignalProcessor`), con los valores de G3T por defecto: 4096 Hz, 4 s, un
contexto de 33408 muestras por lado, una SNR en 30–1024 Hz con tolerancia 0,05, SEOBNRv4_opt desde
30 Hz, un blanqueado `psd` con filtro de 16 s y truncado Hann, FIR paso alto de 30 Hz y paso bajo de
1024 Hz (orden 256, β = 5) y salida `crop_to_config`.

Se pueden cambiar, pero cualquier diferencia con G3T da un aviso: esa física no está validada.
Algunas restricciones:

- con `psd_pack`, `sampling_frequency` debe ser ≤ 4096 Hz;
- el contexto por lado debe cubrir el borde que estropea el procesado (el error lo indica);
- los aproximantes con precesión están prohibidos (gwsim solo tiene espines alineados);
- con otro aproximante y `click_free: true` aparece un aviso, porque S15/S10 solo se ha validado
  con SEOBNRv4_opt.

## Semillas

A partir de `seed` se crea un flujo independiente por propósito,
`default_rng(SeedSequence([seed, k]))`, con k igual a:

1. parámetros;
2. SNR o distancia;
3. asignación de ruido;
4. colocación;
5. ruido.

Cada fila guarda sus enteros `placement_seed` y `gaussian_seed`. Todo el plan se sortea antes de
generar, así que el resultado no depende de `--workers` ni de `--resume`.

## Ejemplos

- [configs/default_g3t.json](../configs/default_g3t.json): G3T exacto, escrito de forma explícita.
- [configs/example_noise_only.json](../configs/example_noise_only.json): solo ruido, de los entornos de `train`.
- [configs/example_custom_prior.json](../configs/example_custom_prior.json): 𝓜 y q uniformes,
  espines nulos, SNR log-uniforme, ruido analítico y pares `X` / `X_noise`.
- [configs/example_from_file.json](../configs/example_from_file.json): parámetros desde un CSV.
