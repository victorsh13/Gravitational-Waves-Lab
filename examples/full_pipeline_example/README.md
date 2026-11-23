# Ejemplo completo: de la configuración a las comprobaciones

Este ejemplo recorre todo el flujo de gwsim con 20 señales:

```
config.json  →  generar.sh (python -m gwsim generate)  →  salidas/ejemplo.h5  →  leer_y_comprobar.py
```

Está pensado para que lo **copies y cambies solo los parámetros**. La misma secuencia, explicada paso a
paso, está en el notebook [`tutorials/00_full_pipeline_example.ipynb`](../../tutorials/00_full_pipeline_example.ipynb).

## Ejecutarlo

Con el entorno de gwsim activado (guía, paso 3) y, si puede ser, el paquete de PSD en la raíz del
repositorio (guía, paso 5):

```bash
bash examples/full_pipeline_example/generar.sh
python examples/full_pipeline_example/leer_y_comprobar.py
```

Los dos comandos funcionan desde cualquier carpeta.

## Qué debe salir

**`generar.sh`** (de 1 a 3 minutos):

1. Valida la configuración: `Configuración válida: 20 señales, modo snr, salidas ['X', 'X_noise']`.
2. Genera, con líneas de progreso `k/20 (hechas …, fallidas …)`.
3. Termina con el resumen de `gwsim info`.

Si no encuentra `psd_pack_o3b.npz`, avisa y usa `config_analitico.json` (ruido con curvas de diseño, no
con las PSD de O3b). Si `salidas/ejemplo.h5` ya existe, se niega a sobrescribirlo y te dice cómo
borrarlo.

**`leer_y_comprobar.py`** imprime un resumen:

- filas hechas y fallidas (con el error de cada fallida);
- las formas de `X`, `X_noise` e `y`;
- los rangos de M_tot, χ_eff y SNR.

Después hace estas comprobaciones, cada una con `OK` o `FALLO`:

```
  [   OK] hay filas hechas: 20 de 20
  [   OK] forma (3, 16384) por señal: X (20, 3, 16384), X_noise (20, 3, 16384)
  [   OK] valores finitos
  [   OK] media ≈ 0 por canal en X: máx |media| = …
  [   OK] desviación ≈ 1 por canal en X: máx |desv − 1| = …
  [   OK] |SNR final − objetivo| / objetivo ≤ 0.05: máx …
  [   OK] parámetros dentro del prior
```

Por último guarda dos figuras en `salidas/`:

- `senal_vs_ruido.png`: X frente a X_noise de la primera señal, un panel por detector;
- `etiquetas.png`: histogramas de 𝓜, M_tot y χ_eff.

Termina con código 0 si todo da OK y con 1 si algo falla. Si una fila sale con `status = 2`, su error
aparece en el resumen; la guía, en el paso 11, explica las causas habituales.

## La configuración (`config.json`), clave a clave

| Clave | Valor | Qué significa |
|---|---|---|
| `description` | texto | Para recordar qué es; se guarda en el HDF5. |
| `n_samples` | `20` | Número de señales. |
| `seed` | `20261123` | Semilla global. Con la misma semilla y la misma configuración sale exactamente el mismo fichero. |
| `mode` | `"snr"` | La distancia se ajusta para que cada señal tenga su SNR de red objetivo. |
| `outputs` | `["X", "X_noise"]` | `X` = señal + ruido procesados; `X_noise` = **la misma** realización de ruido, sin señal. |
| `click_free` | `true` | Señal sin el clic artificial del arranque a 30 Hz (FISICA.md §6). Déjalo así. |
| `noise.source` | `"psd_pack"` | Ruido gaussiano con las PSD reales de O3b. |
| `noise.psd_pack` | `"../../psd_pack_o3b.npz"` | Ruta al paquete, relativa a esta carpeta (es decir, la raíz del repositorio). |
| `noise.splits` | `["train"]` | Solo los entornos de ruido del split `train`. |
| `prior.masses` | `components_uniform`, `[20, 60]` | m₁ y m₂ uniformes en 20–60 M☉ (ordenadas, m₁ ≥ m₂). |
| `prior.spin_1z`, `spin_2z` | `uniform`, `[-0.5, 0.5]` | Espines alineados uniformes en −0,5…0,5. |
| `prior.inclination`, `dec` | `isotropic` | Orientación y posición en el cielo isótropas (cos ι y sen δ uniformes). |
| `prior.ra`, `polarization_angle` | `uniform` | Uniformes en [0, 2π). |
| `prior.target_network_snr` | `uniform`, `[10, 25]` | SNR de red objetivo uniforme en [10, 25). |

`config_analitico.json` es idéntica, salvo en `noise`: `{"source": "analytic", "gps_range": […]}`.
Todas las claves posibles están en [docs/CONFIGURACION.md](../../docs/CONFIGURACION.md).

## Qué cambiar para tu estudio

Copia la carpeta **dentro de `examples/`**, para que las rutas relativas sigan valiendo
(`cp -r examples/full_pipeline_example examples/mi_estudio`), edita `config.json` y vuelve a ejecutar los
dos comandos sobre `examples/mi_estudio/`.

| Clave | Efecto | Ejemplo |
|---|---|---|
| `prior.masses` | Rango o forma del prior de masas | `{"distribution": "components_uniform", "range": [10, 80]}` · `{"distribution": "chirp_mass_q_uniform", "chirp_mass_range": [10, 40], "q_range": [0.25, 1]}` |
| `prior.spin_1z`, `spin_2z` | Espines alineados | `{"distribution": "uniform", "range": [-1, 1]}` · `{"fixed": 0.0}` |
| `prior.target_network_snr` | Distribución de la SNR (modo `snr`) | `{"distribution": "loguniform", "range": [8, 30]}` · `{"fixed": 12}` |
| `mode` + `prior.distance` | Distancia física en lugar de SNR; la SNR resulta | `"mode": "distance"` y `"distance": {"distribution": "uniform_volume", "range": [200, 2000]}` |
| `mode: "noise_only"` | Solo ruido, con el mismo procesado (la distribución nula) | `"mode": "noise_only"`, `"outputs": ["X"]` (no admite `signal_*`) |
| `outputs` | Qué arrays se guardan (cada uno ocupa 192 KiB por señal) | `["X"]` · `["X", "X_noise", "signal_whitened"]` (lista en SALIDAS.md) |
| `noise.splits` + `seed` | Conjuntos sin fuga de ruido: un split del paquete y una semilla por conjunto | train: `["train"]`, `seed: 101` · val: `["val"]`, `seed: 202` · test: `["test"]`, `seed: 303` |
| `n_samples` | Cuántas señales | `100000` (calcula antes el tamaño con `gwsim.estimate_size`) |
| `--workers` (CLI) | Procesos en paralelo | `WORKERS=4 bash examples/mi_estudio/generar.sh` |

Al cambiar el prior, revisa los avisos de la validación: masas totales < 16 M☉, > 180 M☉ o > 250 M☉
(guía, paso 11, y VALIDEZ.md).
