# Entorno

La instalación paso a paso, con la salida esperada de cada comando, está en
[GUIA_PASO_A_PASO.md](GUIA_PASO_A_PASO.md). Este documento recoge los detalles.

## Versiones

`environment.yml` fija las versiones con las que se verificó la equivalencia bit a bit (ver
VALIDEZ.md): Python 3.10, numpy 2.2.6, scipy 1.15.3, h5py 3.16.0, pandas 2.3.3, astropy 6.1.7,
lalsuite 7.26.11, PyCBC 2.9.0 y, solo para los tutoriales, matplotlib 3.10.9. Con otras versiones,
gwsim funciona igual, pero los números pueden cambiar en los últimos bits.

gwsim no se instala como paquete: se ejecuta desde la raíz del repositorio (`python -m gwsim …`). Para
importarlo desde otro directorio, añade esa raíz a `PYTHONPATH`; los tutoriales lo hacen solos.

## ¿Hace falta `LAL_DATA_PATH`?

**No, con la receta por defecto.** SEOBNRv4_opt es un modelo EOB en el dominio del tiempo que integra
sus ecuaciones sin ficheros de datos externos. Solo los modelos ROM/surrogate (por ejemplo,
SEOBNRv4_ROM o NRSur7dq4) leen datos de `lalsuite-extra` y necesitan `LAL_DATA_PATH`. Si cambias
`waveform_approximant` a uno de ellos y falta la variable, las filas fallarán con un error de LAL que
menciona el fichero de datos.

## Jupyter para los tutoriales

Los notebooks de `tutorials/` necesitan un Jupyter que use el entorno `gwsim`. Si tu Jupyter está en
otro entorno, registra `gwsim` como kernel. Esto no forma parte de `environment.yml`; la versión es la
del entorno de verificación:

```bash
conda activate gwsim
pip install ipykernel==7.2.0
python -m ipykernel install --user --name gwsim --display-name "Python (gwsim)"
```

## Comprobar la instalación

```bash
python -c "import pycbc, lal, numpy, h5py; print('pycbc', pycbc.__version__, '| lalsuite', lal.__version__, '| numpy', numpy.__version__, '| h5py', h5py.__version__)"
python -m gwsim selftest
```

`selftest` genera 2 señales con ruido analítico en un directorio temporal. Comprueba que el status es 1,
que la forma es `(3, 16384)`, que los valores son finitos y que la SNR final está a menos del 5 % de la
objetivo. Escribe `OK` o el fallo concreto.
