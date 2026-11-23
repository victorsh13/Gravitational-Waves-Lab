# gwsim

gwsim simula señales de coalescencias binarias (SEOBNRv4_opt, espines alineados) en la red H1, L1 y
V1. El ruido es gaussiano coloreado con PSD empíricas de O3b, y el procesado es el blanqueado con
banda de 30–1024 Hz y z-score. Cada muestra es un array `(3, 16384)` (4 s a 4096 Hz); las fuentes,
los priors y las salidas son configurables.

## El flujo de trabajo

```
configuración JSON  →  python -m gwsim generate  →  fichero HDF5  →  gwsim.load / h5py  →  tu análisis
```

1. **Edita una configuración** (`configs/*.json`): qué fuentes, cuántas, qué ruido y qué salidas.
2. **Lanza la generación:** `python -m gwsim generate --config mi_config.json --output datos.h5`.
3. **Se crea `datos.h5`**, con un array por salida pedida (`X`, `X_noise`…), los metadatos de cada
   señal y la configuración completa con la que se generó. gwsim nunca sobrescribe.
4. **Léelo:** `d = gwsim.load("datos.h5")` y luego `d["X"]`, `d["y"]`, `d["final_network_snr"]`…

## Instalación

Dos rutas con las mismas versiones; elige una. Los detalles y la salida esperada de cada comando están
en la [guía paso a paso](docs/GUIA_PASO_A_PASO.md), pasos 2 a 5.

**Ruta A, conda (recomendada):**

```bash
git clone --single-branch --branch gwsim https://github.com/victorsh13/Gravitational-Waves-Lab.git gwsim
cd gwsim
conda env create -f environment.yml
conda activate gwsim
python -m gwsim selftest
gh release download gwsim-psd-o3b-v1 --repo victorsh13/Gravitational-Waves-Lab   # psd_pack_o3b.npz
```

**Ruta B, venv + pip** (solo con Python 3.10, en Linux x86_64 o macOS):

```bash
git clone --single-branch --branch gwsim https://github.com/victorsh13/Gravitational-Waves-Lab.git gwsim
cd gwsim
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip && pip install -r requirements.txt
python -m gwsim selftest
gh release download gwsim-psd-o3b-v1 --repo victorsh13/Gravitational-Waves-Lab   # psd_pack_o3b.npz
```

`selftest` debe terminar en `OK`. El paquete de PSD también se puede descargar desde la web, en la
página de *releases* del repositorio; guárdalo en la raíz, junto a este README. Para los notebooks:
`pip install -r requirements-notebooks.txt` (guía, paso 3).

## Lo primero que debes hacer: el ejemplo completo

```bash
bash examples/full_pipeline_example/generar.sh
python examples/full_pipeline_example/leer_y_comprobar.py
```

El ejemplo genera 20 señales, muestra qué hay en el fichero, lo lee, hace comprobaciones (cada una con
OK o FALLO) y dibuja dos figuras. Su [README](examples/full_pipeline_example/README.md) explica cada
clave de la configuración y tiene una tabla **«qué cambiar para tu estudio»**. La misma secuencia, con
explicaciones, está en el notebook
[`tutorials/00_full_pipeline_example.ipynb`](tutorials/00_full_pipeline_example.ipynb).

## Documentación

1. **[docs/GUIA_PASO_A_PASO.md](docs/GUIA_PASO_A_PASO.md)**: instalación, primer conjunto, qué hay en el
   fichero, configuración propia, conjuntos grandes, splits y problemas frecuentes.
2. [docs/FISICA.md](docs/FISICA.md): la cadena física, paso a paso y con el porqué.
3. [docs/CONFIGURACION.md](docs/CONFIGURACION.md): todas las claves de la configuración.
4. [docs/SALIDAS.md](docs/SALIDAS.md): dónde y cómo se guardan los datos; el esquema del HDF5.
5. [docs/API.md](docs/API.md): uso desde Python.
6. [docs/VALIDEZ.md](docs/VALIDEZ.md): qué está validado y qué no. **Léelo antes de usar los datos.**
7. [docs/ENTORNO.md](docs/ENTORNO.md): versiones, las dos rutas de instalación, `LAL_DATA_PATH` y Jupyter.
8. [docs/PROCEDENCIA.md](docs/PROCEDENCIA.md): de dónde viene el código.
9. [docs/MANTENIMIENTO.md](docs/MANTENIMIENTO.md): solo para el mantenedor.

## Tutoriales (`tutorials/`)

| Notebook | Contenido |
|---|---|
| [00_full_pipeline_example](tutorials/00_full_pipeline_example.ipynb) | El flujo completo: configuración, validación, generación, fichero, lectura, comprobaciones y figuras. |
| [01_primeros_pasos](tutorials/01_primeros_pasos.ipynb) | Generar y leer con la CLI y desde Python. |
| [02_anatomia_de_una_senal](tutorials/02_anatomia_de_una_senal.ipynb) | Una fila paso a paso: forma de onda, proyección, SNR, clic, blanqueado y z-score. |
| [03_ruido_y_psd](tutorials/03_ruido_y_psd.ipynb) | El paquete de PSD, el ruido y su blanqueado. |
| [04_priors_y_configuracion](tutorials/04_priors_y_configuracion.ipynb) | Priors, CSV de parámetros, modos y avisos. |
| [05_conjuntos_pareados_y_rangos](tutorials/05_conjuntos_pareados_y_rangos.ipynb) | Pares señal/ruido, splits disjuntos y un ejemplo de estadística de rangos. |
| [06_generacion_masiva](tutorials/06_generacion_masiva.ipynb) | Tamaño, procesos, reanudación y lectura por bloques. |

## Cita de los datos

Las PSD proceden de datos públicos de O3b de GWOSC (licencia CC BY 4.0). Si publicas resultados con
ellas, cita *Open Data from the Third Observing Run of LIGO, Virgo, KAGRA and GEO*, ApJS 267, 29 (2023),
e incluye el reconocimiento de https://gwosc.org/acknowledgement/.
