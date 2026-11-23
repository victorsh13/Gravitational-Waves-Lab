# gwsim

gwsim simula señales de coalescencias binarias (SEOBNRv4_opt, espines alineados) en la red H1, L1 y
V1. El ruido es gaussiano coloreado con PSD empíricas de O3b, y el procesado es el blanqueado con
banda de 30–1024 Hz y z-score. Cada muestra es un array `(3, 16384)` (4 s a 4096 Hz); las fuentes,
los priors y las salidas son configurables.

## Inicio rápido

```bash
git clone --single-branch --branch gwsim https://github.com/victorsh13/Gravitational-Waves-Lab.git gwsim
cd gwsim
conda env create -f environment.yml && conda activate gwsim
python -m gwsim selftest
gh release download gwsim-psd-o3b-v1 --repo victorsh13/Gravitational-Waves-Lab   # psd_pack_o3b.npz
python -m gwsim generate --config configs/default.json --output senales.h5 --n-samples 10
```

Para leer el resultado: `import gwsim; d = gwsim.load("senales.h5")` y `d["X"]`.

## Documentación

1. **[docs/GUIA_PASO_A_PASO.md](docs/GUIA_PASO_A_PASO.md)**: empieza aquí.
2. [docs/FISICA.md](docs/FISICA.md): la cadena física, paso a paso y con el porqué.
3. [docs/CONFIGURACION.md](docs/CONFIGURACION.md): todas las claves de la configuración.
4. [docs/SALIDAS.md](docs/SALIDAS.md): el esquema del HDF5.
5. [docs/API.md](docs/API.md): uso desde Python.
6. [docs/VALIDEZ.md](docs/VALIDEZ.md): qué está validado y qué no. **Léelo antes de usar los datos.**
7. [docs/ENTORNO.md](docs/ENTORNO.md): versiones, `LAL_DATA_PATH` y Jupyter.
8. [docs/PROCEDENCIA.md](docs/PROCEDENCIA.md): de dónde viene el código.
9. [docs/MANTENIMIENTO.md](docs/MANTENIMIENTO.md): solo para el mantenedor.

## Tutoriales (`tutorials/`)

| Notebook | Contenido |
|---|---|
| [00_primeros_pasos](tutorials/00_primeros_pasos.ipynb) | Generar y leer un primer conjunto. |
| [01_anatomia_de_una_senal](tutorials/01_anatomia_de_una_senal.ipynb) | Una fila paso a paso: forma de onda, proyección, SNR, clic, blanqueado y z-score. |
| [02_ruido_y_psd](tutorials/02_ruido_y_psd.ipynb) | El paquete de PSD, el ruido y su blanqueado. |
| [03_priors_y_configuracion](tutorials/03_priors_y_configuracion.ipynb) | Priors, CSV de parámetros, modos y avisos. |
| [04_conjuntos_pareados_y_rangos](tutorials/04_conjuntos_pareados_y_rangos.ipynb) | Pares señal/ruido, splits disjuntos y un ejemplo de estadística de rangos. |
| [05_generacion_masiva](tutorials/05_generacion_masiva.ipynb) | Tamaño, procesos, reanudación y lectura por bloques. |

## Cita de los datos

Las PSD proceden de datos públicos de O3b de GWOSC (licencia CC BY 4.0). Si publicas resultados con
ellas, cita *Open Data from the Third Observing Run of LIGO, Virgo, KAGRA and GEO*, ApJS 267, 29 (2023),
e incluye el reconocimiento de https://gwosc.org/acknowledgement/.
