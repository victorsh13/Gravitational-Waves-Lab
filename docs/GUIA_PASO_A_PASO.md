# Guía paso a paso

Esta guía te lleva desde un ordenador sin nada instalado hasta generar tus propios conjuntos de
datos. En cada paso tienes:

- el bloque de comandos exacto, para copiar y pegar en una terminal;
- la salida que deberías ver;
- el porqué de lo que haces.

Si algo no sale como se describe, ve al paso 11 («Problemas frecuentes»).

## El flujo de trabajo, de un vistazo

```
configuración JSON  →  python -m gwsim generate  →  fichero HDF5  →  gwsim.load / h5py  →  tu análisis
```

1. **Edita una configuración** (`configs/*.json`): qué fuentes (prior), cuántas, qué ruido y qué
   salidas quieres (paso 8).
2. **Lanza la generación** con `python -m gwsim generate --config … --output …` (pasos 6 y 9).
3. **Se crea un fichero HDF5**, el `--output` que tú elijas: un array por salida pedida (`X`, `X_noise`…),
   los metadatos de cada señal y la configuración completa (paso 7).
4. **Léelo** con `gwsim.load("fichero.h5")` o, si es grande, por bloques con h5py (pasos 6 y 7).

Si prefieres empezar por un ejemplo completo que funciona y en el que solo tienes que cambiar
parámetros, ve a [`examples/full_pipeline_example/`](../examples/full_pipeline_example/README.md)
(o a su versión notebook, `tutorials/00_full_pipeline_example.ipynb`) después del paso 5.

Convenciones:

- `$` no se escribe: indica que lo que sigue es un comando de terminal.
- Lo que empieza por `>>>` se escribe dentro de Python (`python` en la terminal); también lo puedes
  pegar en un notebook, sin los `>>>`.
- Todas las órdenes `python -m gwsim …` se ejecutan **desde la carpeta raíz del repositorio** (la que
  contiene `README.md`).

---

## 1. Requisitos

- **Sistema operativo:** Linux o macOS. En Windows, solo dentro de **WSL2** (Windows Subsystem for
  Linux), porque LALSuite y PyCBC no se distribuyen para Windows nativo. Para instalar WSL2, abre
  PowerShell como administrador, ejecuta `wsl --install` y reinicia; después trabaja siempre en la
  terminal de Ubuntu.
- **conda o miniforge** (ruta A del paso 3, la recomendada) **o Python 3.10** (ruta B, con `venv` y
  `pip`). Recomendamos miniforge, que usa conda-forge por defecto. Si no tienes ninguno:
  ```bash
  $ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  $ bash Miniforge3-$(uname)-$(uname -m).sh       # acepta la licencia y deja que inicialice la shell
  ```
  Cierra y vuelve a abrir la terminal. `conda --version` debe responder con un número de versión.
- **git** (`git --version`) y, para descargar el paquete de PSD de forma cómoda, la CLI de GitHub
  `gh` (opcional; también se puede descargar desde la web).
- **Disco:**
  - unos **3 GB** para el entorno de conda;
  - **unos 105 MB** para el paquete de PSD;
  - lo que ocupen tus datos: con la salida `X`, cada señal ocupa **192 KiB** (3 × 16384 × 4 bytes),
    es decir, unos 20 GB por cada 100 000 señales. El paso 9 explica cómo calcularlo.
- **Memoria:** 2 GB libres bastan para un proceso; cada proceso extra (`--workers`) suma memoria
  (paso 9).

## 2. Descargar el código

Clona solo la rama `gwsim` del repositorio (el resto del repositorio no hace falta):

```bash
$ git clone --single-branch --branch gwsim https://github.com/victorsh13/Gravitational-Waves-Lab.git gwsim
$ cd gwsim
$ ls
```

Salida esperada (el orden puede variar):

```
README.md  configs  docs  environment.yml  examples  gwsim  requirements-notebooks.txt  requirements.txt  tools  tutorials
```

`--single-branch --branch gwsim` descarga solo esa rama, y el último `gwsim` es el nombre de la carpeta
que se crea. A partir de aquí, **todos los comandos se ejecutan dentro de esta carpeta**.

## 3. Crear el entorno

Un *entorno* es una instalación de Python aislada, con versiones fijas de cada paquete. Así gwsim usa
exactamente las versiones con las que se verificó (ver VALIDEZ.md) y no rompe otros proyectos tuyos.

Hay dos rutas con **las mismas versiones**. Elige **una**:

- **Ruta A, conda (recomendada):** funciona en cualquier Linux o macOS e instala también Python 3.10.
- **Ruta B, venv + pip:** si ya tienes Python 3.10 y prefieres no usar conda.

### Ruta A: conda (recomendada)

```bash
$ conda env create -f environment.yml
$ conda activate gwsim
$ python -c "import pycbc, lal, numpy, h5py; print('pycbc', pycbc.__version__, '| lalsuite', lal.__version__, '| numpy', numpy.__version__, '| h5py', h5py.__version__)"
```

- **El primer comando** tarda varios minutos: descarga Python 3.10, numpy, scipy, h5py, pandas,
  astropy, lalsuite, PyCBC y matplotlib. Al terminar escribe algo como
  `To activate this environment, use $ conda activate gwsim`.
- **Después de `conda activate gwsim`**, el indicador de la terminal empieza por `(gwsim)`. **Tienes
  que activar el entorno cada vez que abras una terminal nueva.**
- **El tercer comando** comprueba las versiones. Salida esperada:

  ```
  pycbc 2.9.0 | lalsuite 7.26.11 | numpy 2.2.6 | h5py 3.16.0
  ```

### Ruta B: venv + pip (opcional)

Solo si ya tienes **Python 3.10** (exactamente esa versión menor). Primero compruébalo:

```bash
$ python3.10 --version
```

Debe responder `Python 3.10.x`. Si dice `command not found`, usa la ruta A.

```bash
$ python3.10 -m venv .venv
$ source .venv/bin/activate
$ python -m pip install --upgrade pip
$ pip install -r requirements.txt
$ python -c "import pycbc, lal, numpy, h5py; print('pycbc', pycbc.__version__, '| lalsuite', lal.__version__, '| numpy', numpy.__version__, '| h5py', h5py.__version__)"
```

- **`python3.10 -m venv .venv`** crea el entorno en la carpeta `.venv/` dentro del repositorio
  (`.gitignore` ya la excluye).
- **`source .venv/bin/activate`** lo activa: el indicador empieza por `(.venv)`. Como en la ruta A, hay
  que activarlo en cada terminal nueva.
- **`requirements.txt`** tiene exactamente las mismas versiones que `environment.yml`.
- **La última línea** debe dar la misma salida que en la ruta A.

lalsuite y PyCBC se instalan como *wheels* (paquetes ya compilados) solo en **Linux x86_64 y macOS**. En
otra plataforma (Linux ARM, como una Raspberry Pi o algunos servidores, o Windows nativo), pip intentará
compilarlos y fallará. En ese caso usa la ruta A, o WSL2 en Windows.

### En las dos rutas: el aviso de `pkg_resources`

Puede aparecer un aviso que menciona `pkg_resources` (`UserWarning: pkg_resources is deprecated …`).
**Es inofensivo**: PyCBC usa un módulo antiguo de setuptools para algo que gwsim no necesita. Puedes
ignorarlo; aparecerá también en otros comandos.

### Opcional: Jupyter para los tutoriales

Los notebooks de `tutorials/` necesitan Jupyter y un *kernel* (el proceso de Python que ejecuta las
celdas) que use este entorno. Con el entorno activado, en cualquiera de las dos rutas:

```bash
$ pip install -r requirements-notebooks.txt
$ python -m ipykernel install --user --name gwsim --display-name "Python (gwsim)"
$ jupyter lab
```

- **El segundo comando** registra el entorno como kernel con el nombre «Python (gwsim)».
- **En JupyterLab**, abre un notebook y elige ese kernel en *Kernel → Change Kernel…* (o arriba a la
  derecha).
- **En VS Code**, abre el notebook, pulsa *Select Kernel* (arriba a la derecha) → *Jupyter Kernel…* →
  «Python (gwsim)». Si no aparece, elige *Python Environments…* y la carpeta del entorno: `.venv`, o el
  entorno `gwsim` de conda.

## 4. Autoprueba

```bash
$ python -m gwsim selftest
```

`selftest` genera 2 señales con ruido analítico (curvas de diseño, sin necesidad del paquete de PSD) en
un directorio temporal y comprueba:

- que las dos filas se han hecho (`status = 1`);
- que `X` tiene forma `(3, 16384)` y valores finitos;
- que la SNR final está a menos del 5 % de la objetivo.

Tarda del orden de segundos a un minuto. Salida esperada (los números de segundos varían):

```
Creado /tmp/gwsim_selftest_…/selftest.h5: 2 filas
2 filas pendientes de 2; 1 proceso(s)
2/2 (hechas 2, fallidas 0); 7 s
Terminado: /tmp/gwsim_selftest_…/selftest.h5 (hechas 2, fallidas 0)
OK
```

Lo importante es la última línea: **`OK`**. Si sale `FALLO:` seguido de una lista, cada línea dice qué
fila y qué comprobación ha fallado (ver el paso 11).

## 5. El paquete de PSD

### Qué es y por qué hace falta

El ruido de gwsim es gaussiano y está *coloreado* con la densidad espectral de potencia (PSD) real de
los detectores durante O3b. El paquete `psd_pack_o3b.npz` contiene esas PSD para 292 bloques de datos
de O3b (H1, L1 y V1), estimadas con el método de Welch sobre 128 s de datos públicos de GWOSC.
FISICA.md §5 explica qué es la PSD de Welch.

Ocupa 108 361 400 bytes (≈ 103 MiB). Como supera el límite de 100 MB por fichero de git, no está en el
repositorio: se distribuye como recurso de una *release* de GitHub llamada **`gwsim-psd-o3b-v1`**.

### Descargarlo

Con la CLI de GitHub (`gh`; la primera vez te pedirá `gh auth login`):

```bash
$ gh release download gwsim-psd-o3b-v1 --repo victorsh13/Gravitational-Waves-Lab
```

O desde la web: https://github.com/victorsh13/Gravitational-Waves-Lab/releases, release
`gwsim-psd-o3b-v1`, fichero `psd_pack_o3b.npz`.

### Dónde guardarlo

**En la raíz del repositorio**, junto a `README.md`, con el nombre `psd_pack_o3b.npz`. Las
configuraciones de `configs/` lo buscan ahí (`"psd_pack": "../psd_pack_o3b.npz"`, ruta relativa a la
carpeta `configs/`), y los tutoriales también. `.gitignore` ya excluye los `.npz`, así que no se subirá
a git por error. Si lo guardas en otro sitio, pásalo con `--psd-pack /ruta/psd_pack_o3b.npz` en cada
`generate`.

### Comprobar que está bien descargado

```bash
$ sha256sum psd_pack_o3b.npz          # en macOS: shasum -a 256 psd_pack_o3b.npz
```

Salida esperada, exactamente:

```
1dc4ff2d730e42b3fc68e8b26a7c9bb2fd6ad38372aaa09f75bf555be0f500fa  psd_pack_o3b.npz
```

El sha256 es una huella del fichero: si cambia un solo byte, cambia por completo. Si no coincide, la
descarga está corrupta o incompleta; bórrala y descárgala de nuevo.

## 6. Primer conjunto: 10 señales

```bash
$ python -m gwsim generate --config configs/default.json --output senales.h5 --n-samples 10
```

`configs/default.json` es la configuración por defecto, escrita entera:

- masas m₁, m₂ uniformes en [5, 90] M☉;
- espines alineados uniformes en [−1, 1];
- orientación y posición en el cielo isótropas;
- SNR de red objetivo uniforme en [10, 25);
- ruido con las PSD del paquete.

`--n-samples 10` sustituye al número de señales de la configuración.

Salida esperada (los tiempos varían):

```
AVISO: M_tot puede bajar a 10.0 < 16: algunas formas de onda pueden no generarse (esas filas quedarán con status = 2)
Creado senales.h5: 10 filas
10 filas pendientes de 10; 1 proceso(s)
1/10 (hechas 1, fallidas 0); 3 s
2/10 (hechas 2, fallidas 0); 7 s
…
10/10 (hechas 10, fallidas 0); 35 s
Terminado: senales.h5 (hechas 10, fallidas 0)
```

El `AVISO` es normal con este prior: dice que, con masas tan pequeñas, alguna forma de onda podría no
generarse. Esas filas quedarían marcadas como fallidas y la generación seguiría. Para ver un resumen
del fichero:

```bash
$ python -m gwsim info senales.h5
```

Para leerlo desde Python (ejecuta `python` en la raíz del repositorio):

```python
>>> import gwsim
>>> d = gwsim.load("senales.h5")                 # solo las filas hechas (status == 1)
>>> d["X"].shape                                  # (señales, detectores H1/L1/V1, muestras)
(10, 3, 16384)
>>> d["y"][0]                                     # [masa de chirp, masa total, chi_eff]
>>> d["final_network_snr"]                        # SNR de red de cada señal
>>> sorted(d)                                     # todos los metadatos disponibles
```

SALIDAS.md describe cada campo. `X` está normalizado por detector (media 0 y desviación 1 en cada uno
de los tres canales); FISICA.md §8 explica por qué.

## 7. Qué hay dentro del fichero

**Dónde está.** El fichero es exactamente el `--output` que eliges: si escribes
`--output salidas/train.h5`, se crea `salidas/train.h5`, y la carpeta también, si no existe. gwsim
**nunca sobrescribe**: si el fichero ya existe, se niega a empezar (o continúa con `--resume`, paso 9).

**Qué contiene.** Un HDF5 es un fichero con «carpetas» y arrays con nombre (*datasets*) y atributos.
El de gwsim tiene esta forma (con `outputs: ["X", "X_noise"]` y n señales):

```
senales.h5
├── X                      (n, 3, 16384) float32   una entrada por salida pedida en "outputs"
├── X_noise                (n, 3, 16384) float32
├── status                 (n,) uint8              0 pendiente · 1 hecha · 2 fallida
├── error                  (n,) texto              por qué falló la fila (vacío si no)
├── y                      (n, 3) float32          [chirp_mass, total_mass, chi_eff]
├── mass_1, mass_2, spin_1z, …, chi_eff            (n,)  parámetros y derivados de cada señal
├── target_network_snr, final_network_snr, snr_H1, … (n,)  SNR objetivo y obtenidas
├── final_distance, gps, actual_geocentric_time, … (n,)  distancia y tiempos
├── block_id, noise_crop_id, noise_split           (n,)  de dónde sale el ruido
└── placement_seed, gaussian_seed, …               (n,)  semillas (reproducibilidad)
atributos: config_json (la configuración completa), gwsim_version, created_utc, outputs, warnings, …
```

SALIDAS.md tiene la lista completa de campos.

**Verlo desde la terminal.** Si tienes las herramientas de HDF5 (`h5ls`):

```bash
$ h5ls -r senales.h5
```

Si no, estas 5 líneas de Python recorren el árbol e imprimen nombre, forma y tipo de cada dataset:

```bash
$ python - <<'EOF'
import h5py
with h5py.File("senales.h5", "r") as h:
    h.visititems(lambda name, obj: print(f"{name:<28} {str(obj.shape):<18} {obj.dtype}") if isinstance(obj, h5py.Dataset) else None)
    print({k: v for k, v in h.attrs.items() if k != "config_json"})
EOF
```

**Recuperar la configuración exacta.** La configuración completa, con los valores por defecto
aplicados, se guarda dentro del fichero. Para sacarla como JSON y poder repetir o variar la
generación:

```bash
$ python -c "import h5py, json; cfg = json.loads(h5py.File('senales.h5', 'r').attrs['config_json']); open('config_recuperada.json', 'w').write(json.dumps(cfg, indent=2))"
```

`config_recuperada.json` se puede pasar tal cual a `generate --config`.

**Tamaño y compresión.** Cada salida ocupa 192 KiB por señal sin comprimir (3 × 16384 × 4 bytes).
gwsim comprime con lzf, que es rápido y sin pérdidas; el ahorro es modesto, porque el ruido apenas se
comprime. Para estimar el tamaño antes de generar, usa `gwsim.estimate_size` (paso 9).

**Un fichero por split.** Conviene un fichero para train, otro para val y otro para test (paso 10):

- cada uno lleva su propia configuración (con su split de ruido y su semilla);
- se pueden regenerar o compartir por separado;
- es imposible mezclarlos por error al leer.

## 8. Tu propia configuración

Copia la configuración por defecto y edítala:

```bash
$ cp configs/default.json configs/mi_config.json
```

Solo necesitas escribir las claves que cambian: las demás toman el valor por defecto. Por ejemplo,
`configs/mi_config.json` podría quedar así:

```json
{
  "description": "Masas 20-60, espines nulos, SNR log-uniforme 8-30, pares señal / solo ruido",
  "n_samples": 1000,
  "seed": 101,
  "mode": "snr",
  "outputs": ["X", "X_noise"],
  "noise": {"source": "psd_pack", "psd_pack": "../psd_pack_o3b.npz", "splits": ["train"]},
  "prior": {
    "masses": {"distribution": "components_uniform", "range": [20.0, 60.0]},
    "spin_1z": {"fixed": 0.0},
    "spin_2z": {"fixed": 0.0},
    "target_network_snr": {"distribution": "loguniform", "range": [8.0, 30.0]}
  }
}
```

JSON no admite comentarios, así que la explicación va aparte, línea a línea:

| Línea | Qué hace |
|---|---|
| `description` | Texto libre para recordar qué es; se guarda en el HDF5. |
| `n_samples` | Cuántas señales. |
| `seed` | Semilla global. **Con la misma semilla y el mismo prior salen las mismas fuentes y el mismo ruido**; usa semillas distintas para conjuntos independientes. |
| `mode` | `snr`: la distancia se ajusta para lograr la SNR objetivo. Alternativas: `distance` y `noise_only`. |
| `outputs` | `X` (señal + ruido) y `X_noise` (la misma realización de ruido, sin señal). |
| `noise` | PSD del paquete, solo de los entornos del split `train`. |
| `prior.masses` | m₁ y m₂ uniformes en [20, 60] M☉ (y ordenadas, m₁ ≥ m₂). |
| `prior.spin_1z`, `spin_2z` | Espines fijos a 0. |
| `prior.target_network_snr` | log-uniforme: tantas señales entre 8 y 16 como entre 15 y 30. |

Todas las claves, con sus valores por defecto, están en CONFIGURACION.md.

Antes de generar, **valida la configuración** (la CLI no tiene un modo de solo validar; esto hace la
misma validación que `generate`, sin escribir nada):

```bash
$ python -c "from gwsim.config import load_config; cfg, avisos = load_config('configs/mi_config.json'); print('Configuración válida'); [print('AVISO:', a) for a in avisos]"
```

Salida esperada para el ejemplo:

```
Configuración válida
```

Si hay un error (una clave mal escrita, un rango al revés…), sale un `ConfigError` que dice qué clave
falla y por qué. Por ejemplo, con `"range": [60.0, 20.0]`:
`ConfigError: prior.masses.range debe estar ordenado [mín, máx]; recibido [60.0, 20.0]`.

## 9. Generar un conjunto grande

### Cuánto va a ocupar

```bash
$ python -c "import gwsim; b = gwsim.estimate_size('configs/mi_config.json', 100000); print(f'{b / 2**30:.1f} GiB')"
```

Para 100 000 señales con `X` y `X_noise`, la salida esperada es `36.7 GiB`: 2 salidas × 3 detectores ×
16384 muestras × 4 bytes por señal, más unos cientos de bytes de metadatos. Es una estimación **sin
comprimir**; el fichero real suele ocupar algo menos, porque se guarda con compresión lzf. Comprueba
que tienes ese espacio libre (`df -h .`).

### Varios procesos: `--workers`

```bash
$ python -m gwsim generate --config configs/mi_config.json --output train.h5 --workers 4
```

`--workers 4` reparte las señales entre 4 procesos; solo el proceso principal escribe en el fichero.

- **Cuántos usar:** como mucho, el número de núcleos físicos menos uno (`nproc` te dice cuántos
  núcleos lógicos hay).
- **Memoria por proceso:** cada proceso tiene su propia copia de la física y de las PSD, y genera
  formas de onda y FFT de segmentos de unos 20 s, del orden de unos cientos de MB. Las señales de masa
  pequeña, que empiezan a baja frecuencia y son más largas, consumen más. Antes de un conjunto grande,
  mide la memoria con un conjunto pequeño:
  ```bash
  $ /usr/bin/time -v python -m gwsim generate --config configs/mi_config.json --output prueba.h5 --n-samples 20 2>&1 | grep "Maximum resident"
  ```
  El resultado está en KB. Multiplícalo por el número de procesos y deja margen.
- **Tiempo:** mide los segundos por señal con esas 20 (la última línea de progreso) y multiplica.

### Si se interrumpe: `--resume`

Puede interrumpirse por un corte, por cerrar la sesión o por pulsar Ctrl-C. Cada señal tiene un
`status` en el fichero: 0 pendiente, 1 hecha, 2 fallida. Se marca como hecha **después** de escribir
sus datos, así que una señal a medias queda en 0. Para continuar:

```bash
$ python -m gwsim generate --config configs/mi_config.json --output train.h5 --workers 4 --resume
```

`--resume` rehace solo las filas con `status = 0`. Como todo el plan (parámetros, semillas, ruido) se
sorteó y guardó al crear el fichero, el resultado es idéntico al de una ejecución sin interrupción.
Para sesiones largas por SSH, lanza la generación dentro de `tmux` o `screen`, o con `nohup … &`, para
que no muera al cerrar la conexión.

## 10. Splits sin fuga de ruido

Si entrenas o calibras con unos datos y evalúas con otros, el ruido de la evaluación no debe venir de
los mismos tramos de datos de O3b. Si no, el estadístico podría «aprenderse» las particularidades de
esos tramos y el resultado de la evaluación saldría optimista.

El paquete de PSD está dividido en **entornos de ruido disjuntos** con cuatro etiquetas: `train`, `val`,
`cal` y `test`. Un entorno es un grupo de ficheros de O3b contiguos en el tiempo. Usa un split del
paquete para cada conjunto tuyo **y una semilla distinta**:

```bash
$ cp configs/mi_config.json configs/mi_train.json
$ cp configs/mi_config.json configs/mi_val.json
$ cp configs/mi_config.json configs/mi_test.json
```

Edita en cada una `seed` y `noise.splits`:

| Fichero | `seed` | `noise.splits` | `n_samples` |
|---|---|---|---|
| `mi_train.json` | `101` | `["train"]` | `80000` |
| `mi_val.json` | `202` | `["val"]` | `10000` |
| `mi_test.json` | `303` | `["test"]` | `10000` |

```bash
$ python -m gwsim generate --config configs/mi_train.json --output train.h5 --workers 4
$ python -m gwsim generate --config configs/mi_val.json   --output val.h5   --workers 4
$ python -m gwsim generate --config configs/mi_test.json  --output test.h5  --workers 4
```

Comprueba que no comparten bloques de ruido:

```python
>>> import gwsim
>>> tr = set(gwsim.load("train.h5", outputs=())["block_id"])
>>> te = set(gwsim.load("test.h5", outputs=())["block_id"])
>>> tr & te
set()
```

`set()` (vacío) significa que no hay ningún bloque común. La semilla distinta hace que las fuentes y
las realizaciones de ruido también sean distintas. El split `cal` queda libre por si necesitas un
cuarto conjunto (por ejemplo, para calibrar umbrales).

## 11. Problemas frecuentes

**Algunas filas tienen `status = 2`.** La generación no se para por una fila que falla: la marca y
sigue. Para ver por qué ha fallado:

```python
>>> d = gwsim.load("senales.h5", outputs=(), only_ok=False)
>>> fallidas = d["status"] == 2
>>> d["error"][fallidas], d["total_mass"][fallidas]
```

(`python -m gwsim info senales.h5` también muestra los 10 primeros errores). Las causas habituales
son:

- un error de la forma de onda con masas muy pequeñas (M_tot < 16 M☉);
- `SNR rescaling validation failed`: la SNR final no cayó dentro del 5 % de la objetivo;
- `alineación: residuo …`: la señal sin clic no se alineó con la de referencia.

Si son pocas, simplemente descártalas: `gwsim.load` ya lo hace por defecto. Si son muchas, revisa el
prior.

**Avisos de M_tot.** Al empezar, gwsim avisa si tu prior puede producir:

- masas totales < 16 M☉: la forma de onda puede no generarse;
- masas totales > 180 M☉: se usa la banda de arranque alta (10→14 Hz), validada a nivel de forma de onda hasta 250 M☉, pero sin comprobar la equivalencia con la producción en esa región;
- masas totales > 250 M☉: región sin validar.

No son errores, pero esas regiones no están validadas (VALIDEZ.md). Si tu estudio no las necesita,
ajusta el prior para no entrar en ellas.

**Falta de memoria** (`MemoryError`, `Killed` o el ordenador muy lento). Baja `--workers`. Al leer, no
cargues un fichero enorme de golpe con `gwsim.load`: lee por bloques con h5py (SALIDAS.md y el
tutorial 06).

**Un HDF5 a medias** (se cortó la generación). Continúa con `--resume` (paso 9). Si el corte ocurrió
justo al crear el fichero, `--resume` dice `El HDF5 no terminó de inicializarse`: bórralo y vuelve a
generar sin `--resume`.

**Cambié la configuración y usé `--resume`.** gwsim se niega con el mensaje
`--resume con otra configuración: …`. Hace bien: el fichero mezclaría señales de dos configuraciones
distintas. O vuelves a la configuración original (y a los mismos `--n-samples` y `--psd-pack`), o
generas un fichero nuevo con otro nombre.

**`… ya existe; usa --resume …`.** gwsim nunca sobrescribe un fichero. Usa otro `--output` o borra el
antiguo si de verdad no lo quieres.

**`No existe noise.psd_pack: …`.** El paquete no está en la raíz del repositorio con el nombre
`psd_pack_o3b.npz`. Muévelo ahí o usa `--psd-pack /ruta/psd_pack_o3b.npz` (paso 5).

**Avisos de `pkg_resources`.** Son inofensivos (paso 3).

---

Siguiente paso: los tutoriales de `tutorials/`, en orden, empezando por `00_full_pipeline_example.ipynb`. Para
abrirlos, necesitas Jupyter con el kernel «Python (gwsim)» (paso 3, «Opcional: Jupyter para los tutoriales»).
