# M13.0d — Generación de G3 (banda 30–1024 Hz, filtro de blanqueado de 16 s, 500k)

**Estado:** código y configuración preparados (26-10-2026). Piloto y generación pendientes de ejecución por el usuario en la VM.

## 1. Motivación

M13.0c ([M13_0c_whitening.md](M13_0c_whitening.md)) mostró que, con el filtro de blanqueado de 0,5 s de G1/G2, las líneas estrechas de la PSD sobreviven en X. En G1, V1 concentra el 98,7 % de la varianza en líneas. Con un filtro de 16 s y su contexto, la fracción de varianza en líneas cae a niveles de ruido salvo los modos de violín. G3 lleva esta configuración a los 500k de G1 y abre la banda a 30–1024 Hz.

## 2. Configuración

Config: [configs/generation/generate_m13_G3_band1024_w16.json](../../configs/generation/generate_m13_G3_band1024_w16.json), `dataset_id = M13_G3_band1024_w16_500k`, base `M11_final_G1_500k/m11_final_G1_500k.h5`.

| Parámetro | G1 | G3 |
|---|---|---|
| `signal_processor.lowpass_frequency` | 512 Hz | 1024 Hz |
| `signal_processor.whitening_max_filter_duration` | 0,5 s | 16 s |
| `simulation.processing_context_start/end_samples` | G1 | 33408 / 33408 |
| `simulation.snr_high_frequency_cutoff` | 512 Hz | 1024 Hz |
| Banda del SNR | 30–512 Hz | 30–1024 Hz |
| Ventana final | 4 s | 4 s (igual) |
| Almacenamiento X de los fragmentos | — | chunks (1,3,16384), lzf |

Contexto por lado = ceil(16·4096/2) + 2·256 (FIR) + 128 (holgura de G1) = 33408 muestras. El generador comprueba que cubre el margen corrupto del procesador.

`initialize_physics` de G1 exige un corte de 512 Hz y no se modifica. `generate_m13_G3_whitening.py` lo llama con el corte de G1 y después reconstruye `CONFIG` con `dataclasses.replace`, que solo cambia `snr_high_frequency_cutoff` y vuelve a validar. También reconstruye los tres objetos que guardan `CONFIG` al construirse (`WAVEFORM_GENERATOR`, `WINDOW_SELECTOR`, `PROCESSOR`). El resto de la física de G1 lee `CONFIG` en el momento de la llamada, así que se reutiliza sin copiarla.

## 3. Residuo aceptado

Quedan los modos de violín: H1 ~505–514 y ~998 Hz, L1 ~510 Hz, V1 ~447–455 y ~899 Hz. Con el filtro de 16 s, el q95 de la fracción de varianza en líneas (40–1000 Hz) es H1 2,3 %, L1 1,1 % y V1 4,5 %. No cumple el criterio previo de M13.0c, pero se acepta como residuo conocido de G3. El resto de las líneas queda a nivel de ruido.

## 4. Qué cambia respecto a G1

Se mantienen idénticos a G1: manifiesto, `source_id`, pertenencia a train/val/cal/test, semillas, colocación (`actual_geocentric_time`, `placement_offset_s`), ventana final, `reference_geocentric_time`, SNR objetivo y etiquetas `[chirp_mass, total_mass, chi_eff]`.

Cambia:

- **Ruido.** La longitud del segmento de procesado cambia, y `noise_from_psd` depende de ella, así que el ruido gaussiano es otra realización. G3 **no** está emparejado en ruido con G1 (`noise_paired_with_g1 = False`).
- **Distancia.** `final_distance` se recalcula para alcanzar el mismo SNR objetivo en la banda 30–1024 Hz. Una banda más ancha añade SNR a distancia fija, así que el cociente G3/G1 debe ser ≥ 1 (se espera ≤ ~1,02). Cambian también `final_network_snr` y `snr_{H1,L1,V1}` dentro de la tolerancia de reescalado, y `reference_network_snr`.
- **Procesado.** Paso bajo, filtro de blanqueado y contexto, según la tabla anterior. `processing_start/end` se amplían con el contexto; los bordes de G1 se guardan como `g1_processing_start/end`.

Guardas del piloto: abortar si `final_distance_G3/final_distance_G1 < 1 − 1e-9` o `> 1,05`. Si el máximo supera 1,02, solo se emite un aviso. Los demás metadatos se comparan de forma informativa.

Atributos del HDF5: `processing_configuration` (JSON con paso bajo, filtro, contexto, banda del SNR y `signal_processor` completo), `lowpass_frequency_hz`, `whitening_max_filter_duration_s`, `processing_context_*_samples`, `snr_low/high_frequency_cutoff_hz`, `distance_recomputed_for_snr_band`, `m13_reference` (M13.0c), `storage_X`, `source_module_sha256` (por fichero de `src/`) y `source_tree_sha256` (resumen), además de la procedencia de G1 heredada como en G2.

## 5. Coste estimado

0,69 s por fila en el piloto de 16 s de M13.0c, es decir, ~96 h de CPU para 500k. Con 4 fragmentos en paralelo (`OMP_NUM_THREADS=1`), son **~24 h**. Cada proceso imprime al terminar su pico de RSS (`resource.getrusage`).

## 6. Consecuencia para datos reales

El preprocesado de eventos reales que se evalúen con modelos entrenados en G3 deberá usar **el mismo contexto (33408/33408 muestras), el mismo filtro de blanqueado (16 s), el mismo paso bajo (1024 Hz) y el mismo corte del SNR (30–1024 Hz)**. Eso exige al menos ~8,2 s de datos limpios a cada lado de la ventana final de 4 s.

## 7. Comandos (VM, entorno `cbc_torch`)

```bash
cd ~/cbc_pe   # raíz del repositorio en la VM
export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
mkdir -p logs/m13
CFG=configs/generation/generate_m13_G3_band1024_w16.json

# Piloto (64 filas)
OMP_NUM_THREADS=1 python scripts/m13/generate_m13_G3_whitening.py --config $CFG --pilot 64 \
    2>&1 | tee logs/m13/g3_w16_pilot_64.log

# Fragmentos (N=4; el bloque k va al fragmento k % N)
N=4
for i in $(seq 0 $((N-1))); do
  OMP_NUM_THREADS=1 nohup python scripts/m13/generate_m13_G3_whitening.py --config $CFG \
      --shard $i --num-shards $N \
      > logs/m13/g3_w16_shard$(printf %02d $i)of$(printf %02d $N).log 2>&1 &
done

# Reanudar un fragmento interrumpido
OMP_NUM_THREADS=1 nohup python scripts/m13/generate_m13_G3_whitening.py --config $CFG \
    --shard 2 --num-shards 4 --resume >> logs/m13/g3_w16_shard02of04.log 2>&1 &

# Merge (tras completar todos los fragmentos)
nohup python scripts/m13/merge_m13_G3_shards.py --config $CFG --num-shards 4 \
    > logs/m13/g3_w16_merge.log 2>&1 &
```

Salidas:

- `processed/M13_G3_band1024_w16_500k/pilot_64.h5`
- `processed/M13_G3_band1024_w16_500k/shards/g3_shard{i:02d}of{n:02d}.h5` (`row_index` global, X lzf chunks (1,3,16384), `status`, checkpoint cada 32 filas)
- `processed/M13_G3_band1024_w16_500k/m13_G3_band1024_w16_500k_trainio.h5` (layout de `scripts/trainio.py`; `y` y `file_group_id` verificados contra el trainio de G1)
