# Uso desde Python

La CLI (`python -m gwsim generate`) es la forma normal de generar conjuntos. Desde Python puedes:

- generar una sola fila y mirar sus pasos intermedios (`simulate_row`);
- construir PSD, priors y configuraciones a mano;
- leer y dimensionar ficheros (`load`, `estimate_size`).

Todos los ejemplos suponen que Python se ejecuta en la raíz del repositorio. Si no, añade antes la raíz
al `sys.path`: `import sys; sys.path.insert(0, "/ruta/a/gwsim")`.

## `Physics`: la física, construida una vez

```python
from gwsim.recipe import Physics

phys = Physics()                          # física por defecto
cfg = phys.config                         # SimulationConfig
cfg.sampling_frequency, cfg.duration, cfg.length, cfg.processing_length
# (4096, 4.0, 16384, 83200)
cfg.delta_f, cfg.processing_delta_f       # Δf de las PSD "snr" y "proc"
```

`Physics(simulation={...}, signal_processor={...})` acepta las mismas claves que las secciones
homónimas de la configuración (CONFIGURACION.md). Lo que no pases toma el valor por defecto. Los
objetos que contiene son:

- `phys.waveform_generator`: genera h₊ y h× con `get_td_waveform`;
- `phys.projector`: proyecta sobre H1, L1 y V1;
- `phys.window_selector`;
- `phys.processor`: blanqueado, banda y recorte.

Construir `Physics` cuesta poco, pero conviene hacerlo una sola vez y reutilizarlo.

## PSD (`gwsim.noise`)

```python
from gwsim.noise import PsdPackNoise, AnalyticNoise
import numpy as np

noise = PsdPackNoise("psd_pack_o3b.npz", splits=["train"])   # PSD empíricas de O3b
# noise = AnalyticNoise([1256655618.0, 1269363618.0])        # alternativa: curvas de diseño

a = noise.assign(3, np.random.default_rng(0))   # sortea 3 recortes: entorno -> bloque -> recorte
a["block_id"], a["noise_crop_id"], a["gps"]
psds = noise.psds(a["block_id"][0], phys.config)
psds["snr"]["H1"].delta_f, psds["proc"]["H1"].delta_f       # 0.25 y ≈ 0.0492 Hz
```

`psds` es `{"snr": {ifo: FrequencySeries}, "proc": {ifo: FrequencySeries}}`: la misma PSD de Welch,
interpolada a las dos rejillas de frecuencia (FISICA.md §5). Otros atributos del paquete:

- `noise.block_id`, `noise.block_split`: los bloques y sus splits;
- `noise.psd`: array `(bloques, 3, frecuencias)` con la PSD de Welch sin interpolar;
- `noise.delta_f`: su resolución, Δf = 1/8 Hz;
- `noise.crops_by_id([...])`: los recortes por identificador.

## `simulate_row`: una fila

```python
from gwsim.recipe import Physics, RowOptions, simulate_row

phys = Physics()
params = dict(mass_1=36.0, mass_2=29.0, spin_1z=0.0, spin_2z=0.0,
              inclination=0.4, ra=1.2, dec=-0.3, polarization_angle=0.5)
opts = RowOptions(physics=phys, mode="snr",
                  outputs=("X", "X_noise", "signal_whitened", "signal_projected"))
out, meta = simulate_row(params, a["gps"][0], psds, placement_seed=1, gaussian_seed=2,
                         target_snr=15.0, opts=opts)
out["X"].shape                     # (3, 16384), float32
meta["final_network_snr"], meta["final_distance"], meta["snr_H1"]
```

Argumentos:

- `params`: dict o mapeo con `mass_1`, `mass_2`, `spin_1z`, `spin_2z`, `inclination`, `ra`, `dec` y
  `polarization_angle`. La distancia no se pasa: la de referencia es 1000 Mpc y después se reescala.
- `gps`: el centro de la ventana final de 4 s.
- `psds`: las de `noise.psds(...)`.
- `placement_seed`, `gaussian_seed`: enteros ≥ 0. Con los mismos valores sale exactamente la misma
  fila. Con el mismo `gaussian_seed` y la misma PSD sale la misma realización de ruido.
- `target_snr=` (modo `snr`) o `distance=` en Mpc (modo `distance`). En `noise_only` no se pasa
  ninguno, y `placement_seed` se ignora (puede ser `None`).

Campos de `RowOptions`:

- `physics`;
- `mode`: `"snr"`, `"distance"` o `"noise_only"`;
- `outputs`: los nombres de SALIDAS.md;
- `click_free`: `True` por defecto;
- `processing_start`: `None` por defecto, de modo que la ventana se centra en `gps`.

Devuelve `(out, meta)`:

- `out` es un dict nombre → array `(3, 16384)` float32, con las salidas pedidas;
- `meta` tiene los metadatos de SALIDAS.md (SNR, distancia, t_c, banda, alineación, semillas por
  detector…).

Si algo falla, lanza la excepción; la CLI la captura y marca la fila con `status = 2`.

Modo `distance` y modo `noise_only`:

```python
out_d, meta_d = simulate_row(params, a["gps"][0], psds, 1, 2, distance=800.0,
                             opts=RowOptions(physics=phys, mode="distance"))
meta_d["final_network_snr"]        # la SNR que resulta a 800 Mpc

out_n, _ = simulate_row(None, a["gps"][0], psds, None, 2,
                        opts=RowOptions(physics=phys, mode="noise_only", outputs=("X",)))
```

Con el mismo `gaussian_seed` (2), la misma PSD y el mismo `gps`, `out_n["X"]` coincide con el
`X_noise` de la fila con señal: es la misma realización de ruido, con el mismo procesado.

## Priors (`gwsim.priors`)

```python
import numpy as np
from gwsim.config import DEFAULT_CONFIG
from gwsim.priors import sample_parameters, order_masses, add_derived, sample_amplitude

prior = DEFAULT_CONFIG["prior"]
rng = np.random.default_rng(0)
df = add_derived(order_masses(sample_parameters(prior, 1000, rng)))
df[["mass_1", "mass_2", "chirp_mass", "total_mass", "q", "chi_eff"]].describe()
snr = sample_amplitude({"distribution": "loguniform", "range": [8, 30]}, 1000, rng, "SNR")
```

- `sample_parameters(prior, n, rng)` devuelve un DataFrame con los 8 parámetros.
- `order_masses` impone m₁ ≥ m₂ (intercambiando también los espines).
- `add_derived` añade 𝓜, M_tot, q y χ_eff.
- `sample_amplitude(spec, n, rng, tipo)` sortea la SNR o la distancia.

Para el plan completo de un fichero (parámetros, ruido y semillas), `build_plan(cfg, noise)`, con un
`cfg` de `load_config`.

## Configuraciones (`gwsim.config`)

```python
from gwsim.config import load_config, ConfigError

try:
    cfg, avisos = load_config("configs/default.json")
except ConfigError as exc:
    print("Configuración no válida:", exc)
```

`load_config` aplica los valores por defecto, resuelve las rutas y valida. Devuelve el dict completo y
la lista de avisos. Es la misma validación que hace `generate`, sin escribir nada.

## `gwsim.load`: leer un fichero

```python
import gwsim

d = gwsim.load("senales.h5", outputs=("X", "X_noise"), only_ok=True)
d["X"].shape, d["X_noise"].shape          # (n_ok, 3, 16384)
d["y"]                                     # (n_ok, 3): [chirp_mass, total_mass, chi_eff]
d["row_index"]                             # índice de cada fila en el fichero
d["block_id"][:3], d["final_network_snr"][:3]
```

Devuelve un dict con:

- las salidas pedidas;
- **todos** los metadatos por fila (textos ya decodificados);
- `row_index`.

Con `only_ok=True` (por defecto), solo las filas con `status == 1`; con `outputs=()`, solo los
metadatos, sin cargar los arrays grandes. Lo carga todo en memoria: para ficheros grandes, lee por
bloques con h5py (SALIDAS.md y el tutorial 06).

## `gwsim.estimate_size`: cuánto ocupará

```python
import gwsim
gwsim.estimate_size("configs/default.json", 100_000) / 2**30     # GiB, ≈ 18.3 con outputs = ["X"]
gwsim.estimate_size({"outputs": ["X", "X_noise"]}, 100_000) / 2**30   # ≈ 36.7
```

Son los bytes previstos del HDF5, sin comprimir y en float32, según las salidas pedidas, más unos cientos
de bytes de metadatos por fila. `config` puede ser una ruta o un dict; las claves ausentes toman el
valor por defecto, y no se valida.
