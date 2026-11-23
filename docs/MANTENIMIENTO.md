# Mantenimiento (solo para el mantenedor)

Este documento es para quien mantiene gwsim en la máquina que tiene los datos del proyecto de
origen. Un usuario de gwsim no lo necesita. Las correspondencias de nombres están en PROCEDENCIA.md.

Las dos herramientas de `tools/maintenance/` solo leen los datos de origen y se niegan a sobrescribir
su salida:

- `export_psd_pack.py`;
- `check_equivalence_reference.py`.

Las dos resuelven la raíz de datos con `--data-root` o, si falta, con la variable de entorno
`CBC_PE_DATA_ROOT`.

## 1. Generar el paquete de PSD

```bash
cd /ruta/a/gwsim
conda activate gwsim          # o el entorno de producción, con las mismas versiones
python tools/maintenance/export_psd_pack.py \
       --data-root /data/vserrano/cbc_pe_data \
       --output /data/vserrano/gwsim/psd_pack_o3b.npz
```

Qué hace la herramienta:

1. Lee el banco de ruido del origen (`m11_6_manifests`) y comprueba sus sumas sha256 congeladas.
2. Para cada bloque, toma los 128 s de strain de la caché de GWOSC (`gwosc_cache/m11`) y calcula
   `psd(8.0)` exactamente como la producción antes de interpolar.
3. Escribe un `.npz` con las PSD (float64), los identificadores de bloque, split y entorno, y la tabla
   de recortes. Con `--splits` se puede restringir a algunos splits; por defecto van los cuatro.

**Resultado del 23-11-2026:**

| | |
|---|---|
| Bloques | 292 |
| Recortes | 4380 |
| Tamaño | 108 361 400 bytes |
| sha256 | `1dc4ff2d730e42b3fc68e8b26a7c9bb2fd6ad38372aaa09f75bf555be0f500fa` |

Si al repetirlo el sha256 cambia, el paquete es distinto. Puede deberse a otra versión de numpy, que
cambia la compresión del `.npz`, o a otros datos. En ese caso, publica una versión nueva de la release
(`…-v2`) y actualiza el sha256 en GUIA_PASO_A_PASO.md. No sustituyas la v1.

## 2. Repetir la prueba de equivalencia

```bash
python tools/maintenance/check_equivalence_reference.py \
       --data-root /data/vserrano/cbc_pe_data \
       --psd-pack /data/vserrano/gwsim/psd_pack_o3b.npz \
       --output /data/vserrano/gwsim/equivalencia_$(date +%Y%m%d).json \
       --n 64
```

La prueba funciona así:

- toma 64 filas equiespaciadas del split de validación de la referencia;
- reconstruye cada fila con `simulate_row` y la física por defecto, con los parámetros, las semillas
  y la ventana de procesado del manifest de origen;
- convierte X a float16 y lo compara con el X de la referencia.

**Criterio, fijado antes de ejecutar:** PASS si en todas las filas la diferencia máxima es ≤ 1 ulp de
float16 y las etiquetas coinciden. El proceso termina con código 0 si pasa y con 1 si no. El informe
JSON incluye, por fila, las muestras distintas, la diferencia máxima en ulp, la SNR final y la
distancia.

**Resultado del 23-11-2026:** PASS, 64/64 filas, 0 ulp (X idéntico bit a bit) y etiquetas iguales.

Repite la prueba después de cualquier cambio en `gwsim/physics/`, `gwsim/recipe.py`, `gwsim/noise.py`
o el entorno. Si cambia algo que solo afecta a nombres o textos, basta con `py_compile` y `selftest`.

## 3. Publicar el paquete como release de GitHub

GitHub no admite en git ficheros de más de 100 MB, así que el paquete (≈ 103 MiB) se distribuye como
recurso de una *release*, fuera del historial:

```bash
cd /ruta/a/gwsim            # el clon con la rama gwsim
sha256sum /data/vserrano/gwsim/psd_pack_o3b.npz     # debe coincidir con la tabla de arriba
gh release create gwsim-psd-o3b-v1 /data/vserrano/gwsim/psd_pack_o3b.npz \
   --repo victorsh13/Gravitational-Waves-Lab --target gwsim \
   --title "gwsim: paquete de PSD de O3b (v1)" \
   --notes "PSD de Welch (8 s, 128 s por bloque) de 292 bloques de O3b (GWOSC, CC BY 4.0), 4380 recortes. sha256 1dc4ff2d730e42b3fc68e8b26a7c9bb2fd6ad38372aaa09f75bf555be0f500fa"
```

El recurso tiene que llamarse `psd_pack_o3b.npz`: las configuraciones de `configs/` lo buscan con ese
nombre en la raíz del repositorio. Después, comprueba la descarga desde otra máquina, como en
GUIA_PASO_A_PASO.md, paso 5.
