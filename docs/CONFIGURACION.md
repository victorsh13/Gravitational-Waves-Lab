# Configuración

Una configuración es un fichero JSON. Las claves que no pongas toman el valor por defecto: la física
validada (ver VALIDEZ.md) con el prior por defecto. Si escribes una clave desconocida, gwsim da un
error que lista las claves permitidas. Los **avisos** (física distinta de la de por defecto, masas
fuera de la región validada…) se imprimen y se guardan en el HDF5, pero no paran la generación.

La configuración completa, con los valores por defecto ya aplicados y las rutas absolutas, se guarda
en `attrs["config_json"]` del HDF5.

Para validar una configuración sin generar nada, ver «Validar sin generar» al final.

## Raíz

| Clave | Por defecto | Qué es |
|---|---|---|
| `description` | `""` | Texto libre. |
| `n_samples` | `10` | Número de filas. Se ignora si hay `parameters_file`. En la CLI, `--n-samples` lo sustituye. |
| `seed` | `0` | Semilla global, un entero ≥ 0 (ver «Semillas»). |
| `mode` | `"snr"` | `"snr"`, `"distance"` o `"noise_only"`. |
| `outputs` | `["X"]` | Subconjunto de `X`, `X_noise`, `X_unnormalized`, `signal_whitened`, `signal_projected` y `strain_noisy_raw` (ver SALIDAS.md). |
| `click_free` | `true` | `true`: señal S sin clic (bandas de arranque baja y alta; ver FISICA.md §6). `false`: señal A, que empieza de golpe a 30 Hz (con clic). |
| `parameters_file` | `null` | CSV con una fila por señal; sustituye al prior. La ruta es relativa al JSON. |

## `noise`

| Clave | Por defecto | Qué es |
|---|---|---|
| `source` | `"psd_pack"` | `"psd_pack"` (PSD empíricas de O3b) o `"analytic"` (aLIGOZeroDetHighPower para H1 y L1, AdvVirgo para V1). |
| `psd_pack` | `null` en el código; `"../psd_pack_o3b.npz"` en las configuraciones de `configs/` | Ruta al `.npz`, relativa al JSON. Con las configuraciones de `configs/`, el paquete se busca en la raíz del repositorio. En la CLI, `--psd-pack` lo sustituye (ruta relativa al directorio actual). |
| `splits` | los cuatro | Splits del paquete que se usan: `train`, `val`, `cal`, `test`. Son entornos de ruido (grupos de ficheros de O3b) disjuntos. Si usas configuraciones con splits distintos para tu entrenamiento y tu evaluación, sus ruidos no comparten entornos. |
| `gps_range` | O3b | Solo con `analytic`: el GPS se sortea uniforme en `[mín, máx]`. |

Con `psd_pack`, cada fila se asigna a un recorte de ruido sorteando, de forma uniforme y en este
orden, un entorno, un bloque dentro del entorno y un recorte dentro del bloque. El GPS de la fila es
el centro del recorte (`center_gps`). Sortear primero el entorno hace que todos los entornos pesen lo
mismo, aunque unos tengan más bloques que otros.

```json
"noise": {"source": "psd_pack", "psd_pack": "../psd_pack_o3b.npz", "splits": ["train"]}
```

## `prior`

Por defecto: masas U[5, 90] M☉, espines U[−1, 1], ángulos isótropos y SNR U[10, 25).

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

Por qué «isótropo» significa cos ι uniforme y no ι uniforme: las orientaciones del momento angular
están repartidas uniformemente sobre la esfera, y el elemento de área de la esfera es
sen ι dι = −d(cos ι). Con δ ocurre lo mismo, pero con sen δ, porque la declinación se mide desde el
ecuador.

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

Ver [examples/params_example.csv](../examples/params_example.csv).

## `simulation` y `signal_processor`

Son los parámetros numéricos de la cadena (clases `SimulationConfig` y `SignalProcessor` de
`gwsim/physics/`). Valores por defecto:

- 4096 Hz y 4 s, con un contexto de 33408 muestras por lado;
- SNR en 30–1024 Hz, con tolerancia 0,05;
- SEOBNRv4_opt desde 30 Hz;
- blanqueado `psd` con filtro de 16 s y truncado Hann;
- FIR paso alto de 30 Hz y paso bajo de 1024 Hz (orden 256, β = 5);
- salida `crop_to_config`.

Se pueden cambiar, pero cualquier diferencia da un aviso: esa física no está validada. Algunas
restricciones:

- con `psd_pack`, `sampling_frequency` debe ser ≤ 4096 Hz;
- el contexto por lado debe cubrir el borde que estropea el procesado (el error lo indica);
- los aproximantes con precesión están prohibidos (gwsim solo tiene espines alineados);
- con otro aproximante y `click_free: true` aparece un aviso, porque la señal sin clic solo se ha
  validado con SEOBNRv4_opt.

## Semillas

A partir de `seed` se crea un flujo independiente por propósito,
`default_rng(SeedSequence([seed, k]))`, con k igual a:

1. parámetros;
2. SNR o distancia;
3. asignación de ruido;
4. colocación;
5. ruido.

Cada fila guarda sus enteros `placement_seed` y `gaussian_seed`. Todo el plan se sortea antes de
generar, así que el resultado no depende de `--workers` ni de `--resume`. Dos ficheros con el mismo
`seed` y el mismo prior tienen **las mismas fuentes y el mismo ruido**: para conjuntos
independientes, usa semillas distintas.

## Validar sin generar

La CLI no tiene un modo de solo validar; desde Python, `load_config` hace la misma validación que
`generate`, sin escribir nada:

```bash
python -c "from gwsim.config import load_config; cfg, avisos = load_config('configs/mi_config.json'); print('Configuración válida'); [print('AVISO:', a) for a in avisos]"
```

Si la configuración tiene un error, se lanza `ConfigError` con un mensaje que dice qué clave falla.

## Ejemplos

- [configs/default.json](../configs/default.json): la configuración por defecto, escrita de forma explícita.
- [configs/example_noise_only.json](../configs/example_noise_only.json): solo ruido, de los entornos de `train`.
- [configs/example_custom_prior.json](../configs/example_custom_prior.json): 𝓜 y q uniformes,
  espines nulos, SNR log-uniforme, ruido analítico y pares `X` / `X_noise`.
- [configs/example_from_file.json](../configs/example_from_file.json): parámetros desde un CSV.
