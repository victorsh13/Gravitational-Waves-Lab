# M13.0c — Duración del filtro de blanqueado para G3

**Estado:** diseño y código preparados (26-10-2026). Barrido pendiente de ejecución por el usuario en la VM. Este documento se escribe **antes** de ver resultados.

## 1. Motivación

El blanqueado de G1 y G2 usa `inverse_spectrum_truncation` con `whitening_max_filter_duration = 0,5 s`. Eso equivale a una resolución espectral de ~2 Hz, así que las líneas estrechas de la PSD no se eliminan y sobreviven en la entrada X. En el piloto de G2 (ver [M13_0_G2_band1024.md](M13_0_G2_band1024.md), sección 6) aparecieron líneas de H1 a ~998–1002 Hz, de 7 a 95 veces la potencia del resto de la banda. También salió un grupo de V1 en ~884–907 Hz y quedaron líneas en 30–512 Hz que ya están en G1. G2 lo resolvió bajando el paso bajo a 850 Hz. M13.0c estudia la alternativa de un filtro de blanqueado más largo, con suficiente contexto, para un futuro G3.

## 2. Diseño

- **Generador** [scripts/m13/generate_m13_G3_whitening.py](../../scripts/m13/generate_m13_G3_whitening.py): es una variante del de G2, que queda intacto. Importa la física de `scripts/generate_m11_final_G1.py` y la orquestación de `scripts/m13/generate_m13_G2_band.py`. Solo admite estos overrides: `signal_processor.lowpass_frequency`, `signal_processor.whitening_max_filter_duration`, `simulation.processing_context_start_samples` y `simulation.processing_context_end_samples`. Cualquier otro aborta. El corte del SNR sigue en 512 Hz, así que las distancias finales siguen la misma regla que en G1.
- **Ventana de procesado.** `build_g1_row` exige `processing_end − processing_start = processing_duration`. Por eso, en cada fila: `processing_start' = processing_start − (ctx_start' − ctx_start_G1)·dt` y `processing_end' = processing_end + (ctx_end' − ctx_end_G1)·dt`, con dt = 1/4096. Se comprueba que `final_start` y `final_center` coinciden con G1 (< 1 µs) y con `reference_geocentric_time` (< dt) antes y después del ajuste. Los bordes originales se guardan como `g1_processing_start` y `g1_processing_end`. La ventana final de 4 s no cambia.
- **Contexto por lado** = `SignalProcessor.corrupted_margin_samples_per_side() + 128` muestras. Es la misma holgura que G1: a 0,5 s salen 1536 + 128 = 1664 muestras, y el barrido lo comprueba con un `assert`. El generador aborta si el contexto por lado es menor que el margen corrupto.
- **Piloto** (`--pilot N`, las N primeras filas del manifiesto): se escribe en `processed/<dataset_id>/pilot_N.h5`. Compara con G1 `final_distance`, `final_network_snr`, `actual_geocentric_time`, `placement_offset_s`, `snr_*` y `gaussian_seed_*`. Si difieren, se **informa sin abortar**. También informa del tiempo medio por fila.
- **Barrido** [scripts/m13/m13_0c_whitening_sweep.py](../../scripts/m13/m13_0c_whitening_sweep.py):

  | Filtro de blanqueado | Contexto por lado | Segmento procesado |
  |---|---|---|
  | 0,5 s | 1664 muestras | 4,81 s |
  | 2 s | 4736 muestras | 6,31 s |
  | 4 s | 8832 muestras | 8,31 s |
  | 8 s | 17 024 muestras | 12,31 s |

  Todos con paso bajo a 1024 Hz y 64 filas, lanzados en secuencia (`dataset_id = M13_G3_wsweep_<dur>s`). Sobre X se calcula una PSD de Welch (nperseg 4096, resolución 1 Hz) por fila y detector, con dos métricas:
  - `floor_ratio` = mediana de la PSD en [520, 1000) Hz / mediana en [40, 500) Hz.
  - `peak_ratio` = máximo / mediana de la PSD en [40, 1000) Hz.

  Se registran también la fracción de filas con `peak_ratio > 10` y el tiempo por fila. Las salidas van a `$CBC_PE_DATA_ROOT/results/M13_0c_whitening/` (`sweep.csv`, `summary.json`, `median_psd.png` y `configs/`); el script aborta si ese directorio ya existe.
- **Límite práctico.** La PSD de referencia se estima con Welch de segmentos de 8 s, es decir, con resolución de 0,125 Hz. Un filtro de más de 8 s no puede resolver líneas más finas que eso, así que 8 s es el máximo útil del barrido.

## 3. Cambio en `src/processing.py` (autorizado)

`SignalProcessor._validate_config` rechazaba `whitening_max_filter_duration ≥ config.duration`. Comparaba con la ventana final de 4 s, no con el segmento que realmente se filtra, y eso bloqueaba los puntos de 4 s y 8 s.

- **Antes:** error si `whitening_max_filter_duration >= config.duration`.
- **Después:** el límite es `config.processing_duration` si `output_mode == "crop_to_config"`, y `config.duration` en otro caso. El mensaje de error indica qué límite se usó.
- Efecto colateral: el atributo `source_module_sha256` de los HDF5 generados a partir de este cambio difiere del de G1, aunque el procesado numérico de cualquier configuración válida antes del cambio es idéntico. Una comparación de reproducibilidad de G1 debe hacerse por contenido (X, metadatos), no por ese hash.
- Ejecución: CPU en local, no en la VM (no usa PyTorch).

Es una relajación pura: no añade comprobaciones ni cambia ningún cálculo. G1, M10, M11 y G2 usan 0,5 s, que es menor que ambos límites, así que siguen siendo válidos y sus resultados numéricos no cambian. Lo único que cambia es la huella de `source_module_sha256` registrada en los atributos de los HDF5 nuevos.

## 4. Criterio (fijado antes de los resultados)

La duración elegida es la **más corta** que, en H1, L1 y V1 a la vez, cumpla dos condiciones:

- `floor_ratio` ∈ [0,8, 1,25] en al menos el 95 % de las filas;
- `peak_ratio` ≤ 10 en al menos el 95 % de las filas.

Si ninguna lo cumple, el resultado es **NONE**. Se escribe en `summary.json` y se imprime al terminar.

## 5. Qué no concluye

- No dice si el modelo mejora. El criterio mide solo la planitud y los picos residuales del espectro de X. Saber si un G3 mejora la estimación requiere entrenar y evaluar en validación aparte (otro experimento con un solo factor cambiado).
- Con 64 filas y estas fracciones no hay afirmaciones estadísticas; los umbrales son reglas de decisión, no niveles de confianza.
- **El ruido no queda emparejado con G1.** `noise_from_psd` depende de la longitud del segmento, así que el ruido gaussiano es otra realización aunque las semillas sean las mismas. Una comparación G1 frente a G3 tiene que ser estadística, no fila a fila sobre X. Los metadatos que no dependen del ruido (distancia, SNR, colocación y semillas) sí deberían coincidir; el piloto lo informa.
- **Datos reales.** Si se adopta un G3, el preprocesado de eventos reales tendrá que usar el mismo filtro de blanqueado, el mismo paso bajo y el mismo contexto por lado. Esto implica leer más strain alrededor del evento (hasta ~4,2 s por lado con 8 s). Este estudio no cambia ese protocolo.

## 6. Coste

- Cada piloto genera 64 filas. G2 tardó 0,49 s por fila con 0,5 s (pesimista, porque casi cada fila estima una PSD nueva). Las FFT crecen con el segmento procesado (de 4,8 a 12,3 s), así que se espera más tiempo por fila con los filtros largos; el barrido registra el valor medido.
- Disco: cuatro pilotos de 64 × 3 × 16384 float32 (~12,6 MB sin comprimir cada uno).
- Un G3 completo de 500 000 filas costaría aproximadamente lo que mida `seconds_per_row` × 500 000, repartido en fragmentos. Aún no hay modo de fragmentos: el generador G3 solo hace pilotos.

## 7. Comando (VM, entorno `cbc_torch`)

```bash
cd ~/cbc_pe
export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
mkdir -p logs/m13
OMP_NUM_THREADS=1 python scripts/m13/m13_0c_whitening_sweep.py \
    2>&1 | tee logs/m13/m13_0c_whitening_sweep.log
```

Si el barrido se interrumpe, los pilotos ya existentes se reanudan con `--resume`. Antes de relanzar hay que borrar a mano `results/M13_0c_whitening/`.

## 8. Resultados

- 16 s (último punto; 64 filas, mismos 47 intervalos de PSD de O3b): fracción de varianza en líneas, 40–1000 Hz, mediana / q95 / filas ≤ 2 %: H1 0,55 % / 2,3 % / 90,6 %; L1 0,22 % / 1,1 % / 98,4 %; V1 0,74 % / 4,5 % / 67 %. No cumple el criterio. 0,69 s por fila.
- Residuo: modos de violín (H1 ~505–514 y ~998 Hz; L1 ~510 Hz; V1 ~447–455 y ~899 Hz). El resto de líneas está a nivel de ruido.
- Decisión (salida prevista por la regla de parada): se acepta el filtro de 16 s con el residuo documentado. En el 95 % de las filas, el z-score atenúa la componente útil ≤ ~2 % (V1); en G1, V1 se atenuaba a ~0,11. No se aplican notches a los modos de violín; mejora acotada si en el futuro un análisis por detector lo justifica.
