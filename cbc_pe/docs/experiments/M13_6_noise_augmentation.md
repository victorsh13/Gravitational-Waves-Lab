# M13.6 — Aumentación de ruido: señal limpia + banco de ruido por PSD

**Estado:** cerrado (31-10-2026): aumentación adoptada como base (§8). Segunda semilla y M13.4b-aug en curso. Criterios de evaluación fijados el 28-10-2026, antes de entrenar (§7).

## 1. Motivación

M13.5 ([M13_5_renoise.md](M13_5_renoise.md)) verificó HIP-RUIDO-FIJO: la brecha train–val de M13.3b/M13.4a es memorización de la realización fija de ruido de cada fila. M13.6 da a cada fila de train una realización de ruido nueva en cada lote, sin regenerar el dataset.

## 2. Descomposición lineal

G3 construye X = zscore(L(s + n)), con L = `SignalProcessor.process_network` configurado con:
- `whitening_method="psd"`: PSD fija por bloque (Welch sobre la referencia off-source de 128 s del bloque, cacheada por `block_id`);
- FIR paso alto y paso bajo, `crop_to_config`, `apply_standardization=False`;
- la señal se suma al ruido (`SignalInjector.inject`, `+=`) antes de L.

L es lineal, así que **X = zscore(L(s) + L(n))**. El z-score por muestra y detector (`stack_and_normalize_network`) es el único paso no lineal. [generate_m13_6_clean.py](../../scripts/m13/generate_m13_6_clean.py) vuelve a comprobar estas condiciones.

Las filas se construyen con `build_g1_row` y la física de G3 sin cambios. Durante cada fila, `RowCapture` sustituye dos funciones auxiliares del módulo de G1: `build_gaussian_processing_noise` devuelve ruido cero y `stack_and_normalize_network` registra la red antes del z-score. Fuente, reescalado de distancia, colocación, PSD y ventana son los de G3. `psd_key = str(block_id)` (una PSD por bloque).

## 3. Paso 0: verificación de la descomposición

Modo `control`: 64 filas de train (`default_rng(136102)`); se compara zscore(L(s) + L(n_original)) con X del trainio de G3.

| Ruta | max\|ΔX\| | Umbral |
|---|---|---|
| float32 | 1,19e-7 | esperado ≤ 1e-5; aborto > 1e-3 |
| L(s), L(n) redondeados a float16 | 3,2e-3 | informativo (almacenamiento) |

Veredicto: **PASS** (`<data_root>/processed/M13_G3_band1024_w16_500k/m13_6_aug/control/control_report.json`). La descomposición reproduce X de G3 a precisión numérica.

## 4. Señal limpia y banco de ruido

Salidas en `<data_root>/processed/M13_G3_band1024_w16_500k/m13_6_aug/` (exclusivas):
- `signal`: filas de train, ruido 0, L(s) antes del z-score (float16, (n, 3, 16384)), con `psd_key`. 4 fragmentos.
- `noise_bank`: por cada `psd_key` de train, K = 64 realizaciones de L(n) solo ruido, con el procedimiento de ruido y la longitud de procesado de G3; semillas `default_rng(SeedSequence([key_hash, 136101, k]))`. 4 fragmentos.
- Merge ([merge_m13_6_clean.py](../../scripts/m13/merge_m13_6_clean.py)) a `m13_6_aug/memmap/`: `X_clean_train.npy` (orden de `row_ids_train` del memmap de G3, comprobado con `y_train`), `psd_key_train.npy`, `bank_X.npy` (ordenado por (psd_key, k)), `bank_psd_key.npy`, `manifest.json`. Cada `psd_key` de train debe tener exactamente k = 0..K−1.
- Orquestación: [run_m13_6_pipeline.sh](../../scripts/m13/run_m13_6_pipeline.sh) (espera de fragmentos, banco, merge, sincronización sin `--delete` y lanzamiento de los entrenamientos en la VM).

Solo se usan PSD de train: val/cal/test no entran en el banco.

## 5. Aumentación en entrenamiento

[train_m13_3.py](../../scripts/m13/train_m13_3.py) con `training.noise_augmentation.enabled`. Por lote, en GPU y en float32:

x[b, d] = clean[b, d] + s_bd · R_bd(bank[key_b, k_bd, d]),  k_bd ~ U{0..K−1}

- s_bd = −1 con probabilidad 0,5 (`sign_flip`); R_bd = inversión temporal con probabilidad 0,5 (`time_reverse`); independientes por fila y detector; sorteados con `default_rng(seed + 10000·epoch + batch_index)` (paso por época mayor que los 1562 lotes por época: ninguna semilla se repite entre épocas).
- Después, z-score por muestra y detector (x − media)/(std + 1e-6), como `normalize_input_per_sample_per_detector_zscore`.
- Val y el monitor train40k leen X del memmap de G3 (sin aumentar), así que siguen siendo comparables con M13.3b.

Con K = 64 y los dos volteos hay 256 variantes de ruido por detector y bloque PSD, frente a una sola en G3.

## 6. Plan

| Experimento | Config | Comparación |
|---|---|---|
| M13.6a | [train_m13_6a_G3aug_s123.json](../../configs/experiments/m13/train_m13_6a_G3aug_s123.json), [train_m13_6a_G3aug_s124.json](../../configs/experiments/m13/train_m13_6a_G3aug_s124.json) | M13.3b s123/s124 (mismo procedimiento; único cambio: aumentación) |
| M13.4a-aug | [train_m13_4a_G3aug_s123.json](../../configs/experiments/m13/train_m13_4a_G3aug_s123.json) | M13.6a (reevaluación de HIP-CONTEXTO en el régimen aumentado) |

Procedimiento idéntico a M13.3b (80 épocas, coseno, dropout apagado el último 20 %). M13.4b, pendiente de rediseño, se evaluaría en este mismo régimen.

## 7. Criterios fijados (28-10-2026, antes de entrenar)

**M13.6a frente a M13.3b.** M13.6a (s123, s124) frente a M13.3b (s123, s124) en val G3. Una diferencia de RMSE está **resuelta** si |Δ| > 2·max(ruido de semilla de 3b, ruido de semilla de 6a).

- **P-AUG1:** MSE_train40k / MSE_val ≥ 0,95. El train40k se evalúa con su ruido original de G3, que con aumentación nunca se ve en entrenamiento.
- **P-AUG2:** val final ≤ 0,5 % peor que la mejor época.
- **P-AUG3:** mejora resuelta en al menos una etiqueta y salvaguarda (ΔRMSE ≤ +2 %) en todas.

Si no se cumple P-AUG3, la aumentación no se adopta como base.

**M13.4a-aug frente a M13.6a.** Misma regla de resolución. HIP-CONTEXTO queda apoyada si la diferencia de ΔRMSE de χ_eff entre τ largos (D8–D10) y cortos (D1–D3) tiene un IC que excluye 0.

## 8. Resultados (31-10-2026)

Evaluación en [M13_6_evaluation.ipynb](../../notebooks/m13/M13_6_evaluation.ipynb). 1 semilla en las ejecuciones con aumentación.

**Verificado (val G3, 40 000 filas emparejadas; ruido de semilla de M13.3b: 1,4 / 1,1 / 0,6 %)**
1. **Aumentación adoptada como base.** M13.6a frente a M13.3b: ΔRMSE −6,0 % [−6,6; −5,4] / −5,8 % [−6,3; −5,3] / −5,7 % [−6,1; −5,2] (resueltos). P-AUG1 (train40k/val 0,99–1,03), P-AUG2 (mejor época = final) y P-AUG3: PASA. Pendiente θ sobre f en val 1,000 ± 0,004: la contracción desaparece también en val.
2. **C3 ≈ 0** (−0,02 a −0,05 %) en 6a y 4a-aug: el residuo de C3 de G1/G3 sin aumentación era efecto de la memorización, no de cambio de dominio (hipótesis previa descartada).
3. **M13.4a-aug (transformer)** frente a M13.6a: −2,4 % (no resuelto) / −3,1 % / −3,9 % (resueltos); frente a M13.3b: −8,2 / −8,8 / −9,4 %; frente a M11 en G1: −17,9 / −21,2 / −15,1 %. Mejor modelo del proyecto.
4. **HIP-CONTEXTO apoyada:** χ_eff mejora en todos los deciles de τ (~−2,5 %) y −9,1 % en el decil más largo; (D8–D10) − (D1–D3) = −2,1 % [−3,2; −0,9]. En sistemas ligeros (f220 > 1024 Hz), −51 % en M_c y M.
5. Colas: Q10 de M_c +0,63 → +0,47 M☉; Q90 de χ_eff −0,107 → −0,095. Mejora en todos los bins de SNR.
6. Orden temprano (G3, 5 modelos, descriptivo): Spearman con el final −0,4 a +0,3 en las épocas 0–10; 0,8 en la 20. Las primeras épocas no sirven para elegir arquitectura.

**Hipótesis / pendiente**
- Con aumentación, la mejor época es la final y la val sigue bajando: más épocas podrían mejorar (no probado).
- Confirmación con la segunda semilla y M13.4b-aug (en curso).

**Ejecuciones en curso (31-10-2026)**

| Experimento | Config | Notas |
|---|---|---|
| M13.4b-aug s123 | [train_m13_4b_G3aug_s123.json](../../configs/experiments/m13/train_m13_4b_G3aug_s123.json) | `MRDenseNet` con checkpointing de gradiente por etapa; monitor train40k cada 10 épocas |
| M13.6a s124 | [train_m13_6a_G3aug_s124_m10.json](../../configs/experiments/m13/train_m13_6a_G3aug_s124_m10.json) | `monitor_train40k_every = 10` |
| M13.4a-aug s124 | [train_m13_4a_G3aug_s124.json](../../configs/experiments/m13/train_m13_4a_G3aug_s124.json) | `monitor_train40k_every = 10` |
