# Entorno

La instalación paso a paso, con la salida esperada de cada comando, está en
[GUIA_PASO_A_PASO.md](GUIA_PASO_A_PASO.md), paso 3. Este documento recoge los detalles.

## Dos rutas, las mismas versiones

| Ruta | Ficheros | Cuándo |
|---|---|---|
| A, conda (recomendada) | `environment.yml` | Siempre que puedas; funciona en cualquier Linux o macOS e instala también Python 3.10. |
| B, venv + pip | `requirements.txt` | Si ya tienes Python 3.10 y estás en Linux x86_64 o macOS. |
| Notebooks (opcional, en las dos) | `requirements-notebooks.txt` | Para abrir los tutoriales de `tutorials/`. |

`environment.yml` y `requirements.txt` fijan **exactamente las mismas versiones**, las del entorno con
el que se verificó la equivalencia bit a bit (ver VALIDEZ.md): Python 3.10, numpy 2.2.6, scipy 1.15.3,
h5py 3.16.0, pandas 2.3.3, astropy 6.1.7, lalsuite 7.26.11, PyCBC 2.9.0 y, solo para los tutoriales,
matplotlib 3.10.9. Con otras versiones gwsim funciona igual, pero los números pueden cambiar en los
últimos bits. Si cambias una versión, cámbiala en los dos ficheros.

En la ruta B, lalsuite y PyCBC se instalan como *wheels* precompilados, que solo existen para Linux
x86_64 y macOS. En Linux ARM o en Windows nativo, pip intentaría compilarlos y fallaría: usa la ruta A,
o WSL2 en Windows.

gwsim no se instala como paquete: se ejecuta desde la raíz del repositorio (`python -m gwsim …`). Para
importarlo desde otro directorio, añade esa raíz a `PYTHONPATH`; los tutoriales y el ejemplo lo hacen
solos.

## ¿Hace falta `LAL_DATA_PATH`?

**No, con la receta por defecto.** SEOBNRv4_opt es un modelo EOB en el dominio del tiempo que integra
sus ecuaciones sin ficheros de datos externos. Solo los modelos ROM/surrogate (por ejemplo,
SEOBNRv4_ROM o NRSur7dq4) leen datos de `lalsuite-extra` y necesitan `LAL_DATA_PATH`. Si cambias
`waveform_approximant` a uno de ellos y falta la variable, las filas fallarán con un error de LAL que
menciona el fichero de datos.

## Jupyter

`requirements-notebooks.txt` añade `ipykernel` (versión fijada, la del entorno de verificación) y
`jupyterlab` (sin fijar: no formaba parte de ese entorno). La instalación, el registro del kernel
«Python (gwsim)» y cómo elegirlo en JupyterLab y en VS Code están en la guía, paso 3, «Opcional:
Jupyter para los tutoriales».

## Comprobar la instalación

```bash
python -c "import pycbc, lal, numpy, h5py; print('pycbc', pycbc.__version__, '| lalsuite', lal.__version__, '| numpy', numpy.__version__, '| h5py', h5py.__version__)"
python -m gwsim selftest
```

`selftest` genera 2 señales con ruido analítico en un directorio temporal. Comprueba que el status es 1,
que la forma es `(3, 16384)`, que los valores son finitos y que la SNR final está a menos del 5 % de la
objetivo. Escribe `OK` o el fallo concreto.
