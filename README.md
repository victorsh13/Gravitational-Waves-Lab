# gwsim

Simulador de señales de coalescencias binarias (CBC) en ruido gaussiano coloreado. La física es la
misma que la de la producción **G3T** del proyecto CBC-PE. Las fuentes, los priors y las salidas se
pueden configurar.

- La señal es SEOBNRv4_opt con espines alineados, proyectada sobre H1, L1 y V1, con una SNR de red
  objetivo.
- El ruido es gaussiano, con PSD empíricas de O3b (o PSD analíticas de diseño).
- El procesado es un blanqueado con un filtro de 16 s, una banda de 30–1024 Hz y un z-score por
  detector.
- Cada muestra tiene la forma `(3, 16384)`: 3 detectores × 4 s a 4096 Hz.

## Inicio rápido

```bash
conda env create -f environment.yml
conda activate gwsim
cd gwsim            # la raíz de este repositorio

# 10 señales con la física de G3T (necesitas el paquete de PSDs, ver docs/ENTORNO.md)
python -m gwsim generate --config configs/default_g3t.json --psd-pack /ruta/al/psd_pack_g1.npz \
       --output senales.h5 --workers 2

python -m gwsim info senales.h5
```

Sin paquete de PSDs puedes probar con ruido analítico (`configs/example_custom_prior.json`):

```bash
python -m gwsim generate --config configs/example_custom_prior.json --output prueba.h5 --n-samples 10
```

Para leer el fichero generado:

```python
import h5py, numpy as np
with h5py.File("senales.h5", "r") as h:
    ok = h["status"][:] == 1            # 1 = fila hecha, 2 = fallida
    X = h["X"][np.flatnonzero(ok)]      # (n, 3, 16384) float32, z-score por detector
    y = h["y"][:][ok]                   # [chirp_mass, total_mass, chi_eff]
    snr = h["final_network_snr"][:][ok]
```

## Modos

| `mode` | Qué hace |
|---|---|
| `snr` (por defecto) | Reescala la distancia para alcanzar la SNR de red objetivo. |
| `distance` | Usa la distancia física dada y guarda la SNR que resulta. |
| `noise_only` | Genera solo ruido, con el mismo procesado. |

Con `outputs` puedes pedir varias salidas en la misma fila y con la misma realización de ruido:
`X`, `X_noise`, `X_unnormalized`, `signal_whitened`, `signal_projected` y `strain_noisy_raw`.

## Documentación

- [docs/FISICA.md](docs/FISICA.md): la cadena física paso a paso.
- [docs/CONFIGURACION.md](docs/CONFIGURACION.md): todas las claves de la configuración.
- [docs/SALIDAS.md](docs/SALIDAS.md): el esquema del HDF5.
- [docs/VALIDEZ.md](docs/VALIDEZ.md): qué está validado y qué no. **Léelo antes de usar los datos.**
- [docs/PROCEDENCIA.md](docs/PROCEDENCIA.md): de dónde viene cada función.
- [docs/ENTORNO.md](docs/ENTORNO.md): instalación y comprobación.

## Datos y cita

Las PSD empíricas proceden de datos públicos de GWOSC (O3b), con licencia CC BY 4.0. Si publicas
resultados obtenidos con ellas, cita GWOSC y el artículo de la publicación de datos de O3 (ver
docs/VALIDEZ.md).
