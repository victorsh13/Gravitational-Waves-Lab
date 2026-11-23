# Entorno

## Crear el entorno

```bash
conda env create -f environment.yml
conda activate gwsim
```

`environment.yml` fija las versiones de gw-env, el entorno con el que se generó G3T: Python 3.10,
numpy 2.2.6, scipy 1.15.3, h5py 3.16.0, pandas 2.3.3, astropy 6.1.7, lalsuite 7.26.11 y
PyCBC 2.9.0. Para la equivalencia bit a bit con G3T hacen falta estas versiones; con otras, los
números pueden cambiar en los últimos bits.

gwsim no se instala como paquete: se ejecuta desde la raíz del repositorio (`python -m gwsim …`). Si
quieres importarlo desde otro sitio, añade esa raíz a `PYTHONPATH`.

## ¿Hace falta `LAL_DATA_PATH`?

**No, con la receta por defecto.** SEOBNRv4_opt es un modelo EOB en el dominio del tiempo que
integra sus ecuaciones sin ficheros de datos externos. Solo los modelos ROM/surrogate (por ejemplo,
SEOBNRv4_ROM o NRSur7dq4) leen datos de `lalsuite-extra` y necesitan `LAL_DATA_PATH`. Si cambias
`waveform_approximant` a uno de ellos y falta la variable, las filas fallarán con un error de LAL
que menciona el fichero de datos.

## El paquete de PSDs

`noise.source = "psd_pack"` (el valor por defecto) necesita un `.npz` con las PSD de Welch de los
bloques de ruido de G1. Lo genera `tools/export_psd_pack.py` en CIEMAT, a partir del banco de G1 y
de la caché de GWOSC:

```bash
python tools/export_psd_pack.py --data-root /ruta/cbc_pe_data --output /ruta/psd_pack_g1.npz
```

La herramienta imprime el tamaño y el sha256 del fichero: anótalos para saber qué paquete usaste.
Sin paquete, puedes usar `noise.source = "analytic"`.

## Comprobar la instalación

```bash
python -c "import numpy, scipy, h5py, pandas, pycbc, lal; print(pycbc.__version__, lal.__version__)"
python -m py_compile gwsim/*.py gwsim/physics/*.py tools/*.py
python -m gwsim generate --config configs/example_custom_prior.json --output /tmp/prueba_gwsim.h5 --n-samples 2
python -m gwsim info /tmp/prueba_gwsim.h5
```

Lo esperado: 2 filas hechas y la SNR final dentro del 5 % de la SNR objetivo. Una fila tarda del
orden de segundos (la generación de SEOBNRv4_opt y la alineación de S dominan); usa `--workers N`
para repartir.

En CIEMAT, la comprobación completa es la prueba de equivalencia con G3T (ver VALIDEZ.md):

```bash
python tools/check_equivalence_g3t.py --data-root /ruta/cbc_pe_data --psd-pack /ruta/psd_pack_g1.npz \
       --output /ruta/equivalencia_g3t.json --n 64
```
