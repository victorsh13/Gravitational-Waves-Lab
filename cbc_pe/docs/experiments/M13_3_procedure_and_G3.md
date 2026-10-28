# M13.3 — Procedimiento de entrenamiento corregido, G3 y M13.4a

**Estado:** cerrado (28-10-2026). M13.3.0, M13.3-IO, M13.3a, M13.3b y M13.4a cerrados. Todas las comparaciones son en **validación**; no hay evaluación en test ni conformal calibrado para los modelos nuevos (los cuantiles de M11 no se reutilizan).

Fuentes de las cifras (solo lectura):
- Conclusiones de [M13_3a_evaluation.ipynb](../../notebooks/m13/M13_3a_evaluation.ipynb) y [M13_3b_4a_evaluation.ipynb](../../notebooks/m13/M13_3b_4a_evaluation.ipynb).
- `<data_root>/results/M13_3/M13_3a_evaluation_summary.json` y `<data_root>/results/M13_3/M13_3b_4a_evaluation_summary.json`.
- M13.3.0 y M13.3-IO: resultado registrado en [M13_INDICE.md](M13_INDICE.md); detalle en `<data_root>/results/M11_final_G1_500k/M13_3_0_dropout/analysis.json` y `<data_root>/results/M13_3_io/*.json`.

Convenciones: etiquetas [chirp_mass, total_mass, chi_eff]; residuo r = f − θ (predicción menos verdad); "pendiente θ sobre f" = OLS de la verdad sobre la predicción (≈ 1 si calibrado); épocas 0-based. Escalador de etiquetas train-only de M11 y split del contrato congelado G1 (`M12.5.2-NPE-v1`), los mismos en G1 y G3.

## 1. M13.3.0 — Dilución por dropout en M11 (HIP-DROPOUT)

**Pregunta.** ¿Explica el dropout la pendiente θ sobre f > 1 de M11 en train?

**Diseño.** [m13_3_0_dropout_attenuation.py](../../scripts/m13/m13_3_0_dropout_attenuation.py): pasadas del checkpoint congelado de M11 sobre un subconjunto de train de G1 en modo eval y en modo train (dropout activo, K realizaciones con semilla). Métricas en unidades estandarizadas. No se leen val/cal/test.

**Resultado.** PARCIAL: fracción explicada 0,89 / 0,92 / 0,59 en [M_c, M, χ_eff]. La cabeza se ajusta a embeddings con dropout, que no son los de inferencia; en χ_eff queda una parte sin explicar.

## 2. M13.3-IO — Perfil de E/S y cómputo

**Diseño.** [m13_3_io_profile.py](../../scripts/m13/m13_3_io_profile.py): modelo M11 inicializado al azar, AdamW, MSE, lote 256, modo train; por paso se miden espera del iterador, copia a GPU y paso de cómputo (sincronizado). Rutas comparadas: solo cómputo, HDF5 y memmap fp16; variante AMP. No se guardan pesos.

**Resultado.** Limitado por cómputo (241 ms/lote). Se adopta el memmap fp16 ([build_fp16_memmap.py](../../scripts/m13/build_fp16_memmap.py), [memmap_dataset.py](../../src/models/memmap_dataset.py)); AMP descartado.

## 3. Procedimiento de entrenamiento (común a M13.3a/b y M13.4a)

[train_m13_3.py](../../scripts/m13/train_m13_3.py), configs en [configs/experiments/m13/](../../configs/experiments/m13/):
- Datos: memmap fp16 de train y val (`<data_root>/processed/<dataset_id>/fp16_memmap/`); `row_ids` iguales a los índices del split; comprobación de un lote fijo contra el HDF5 antes de entrenar. Cal/test no se leen.
- AdamW (lr 3e-4, wd 3e-4), MSE sobre etiquetas estandarizadas train-only, lote 256, `drop_last`, lotes aleatorios con semilla.
- Planificador por paso: calentamiento lineal 2 épocas y coseno hasta 0 en 80 épocas. Sin early stopping: el modelo registrado es la última época (`_final.pt`).
- Apagado del dropout (opcional): desde la época ceil((1 − 0,2)·80) = 64, todo `nn.Dropout` con p = 0.
- Monitor por época (modo eval): MSE y pendiente θ sobre f en val completa y en un subconjunto fijo de 40 000 filas de train (`train40k`).

## 4. M13.3a — M11 reentrenado en G1

**Hipótesis.** La contracción (pendiente > 1, regresión a la media) de M11 se debe en parte al procedimiento de entrenamiento (dropout activo en la cabeza, planificador, orden de lotes), no al dataset.

**Diseño.** Tres ejecuciones en G1 con la arquitectura de M11: R1 (seed 123, dropout de M11 todo el entrenamiento), R2 (seed 123) y R3 (seed 124), ambas con dropout apagado el último 20 %. Comparación emparejada por `idx` con M11 en val (40 000 filas). Ruido de semilla = |Δ%_R2 − Δ%_R3| (1 g.l.).

**Criterios fijados.**
- Salvaguarda: ΔRMSE% ≤ +2 % frente a M11 por etiqueta; "resuelto" si |Δ%| > 2·ruido de semilla.
- P1: pendiente en train40k del modelo final ∈ [0,99; 1,01] en las tres etiquetas, en R2 y R3.
- P2: pendiente en val ≤ 1,02 en las tres etiquetas.
- P3: C3 (ridge sobre el embedding, protocolo M12.5.3, LOEO por `file_group_id`): riesgo evitable < 1 % por etiqueta. Exploratorio, ajustado en val.
- P4: degeneración M–χ_eff por decil de τ: Δsesgo_χ y k·Δsesgo_M con el mismo signo en ≥ 8/10 deciles.

**Resultados (val G1).**

| Prueba | Resultado |
|---|---|
| Salvaguarda | PASA en R1, R2, R3 y las tres etiquetas; ninguna diferencia resuelta (ruido de semilla 1,3 / 1,7 / 0,2 %) |
| P1 | PASA (R2, R3: pendiente en train40k 0,999); R1 FALLA (1,02–1,03) |
| P2 | PASA: pendiente en val 1,04–1,06 (M11) → 0,97–0,99 |
| P3 | FALLA: riesgo evitable 1,5–3,0 % (masas), 1,0–1,5 % (χ_eff); R2 1,5/1,2/1,0 %, R3 2,6/3,0/1,5 % |
| P4 | FALLA (4/7/5 de 10 deciles en R1/R2/R3); prueba mal planteada |

Sesgo con signo en Q10/Q90 (M11 → R2/R3): M_c +2,9/−5,3 → +1,2/−3,8 M☉; M +7,9/−9,7 → +3,8/−6,8 M☉; χ_eff +0,196/−0,171 → +0,170/−0,125. RMSE por SNR: sin cambios en SNR < 16; en SNR ≥ 19, total_mass mejora un 3–7 % (descriptivo, no resuelto).

**Conclusiones.**
1. El procedimiento corregido no cambia el RMSE global de forma resoluble.
2. Elimina la contracción inducida por el entrenamiento. Mecanismo: al apagar el dropout (épocas 64–79) la pendiente en train pasa de 1,01–1,06 a 0,999; con dropout activo (R1) se queda en 1,02–1,03. Confirma M13.3.0. El planificador solo (R1) reduce la contracción en train de ~0,07 a 0,02–0,035.
3. Menor regresión a la media en las colas, consistente en las dos semillas. Parte del sesgo condicionado a la verdad es irreducible (contracción del estimador bayesiano); la reducción no indica cuánto falta hasta el óptimo.
4. P3: *hipótesis sin verificar* de que el riesgo evitable restante es sobre todo cambio de dominio train → val (pendiente en train 0,999 frente a val 0,97–0,98; MSE de train40k ~28 % menor que el de val). Pasa a M13.5.
5. P4: los deciles de τ dependen de M_c, muy correlacionada con M, y no separan la contracción de χ_eff de la degeneración M–χ_eff. Descriptivo: Spearman(r_M, r_χ) 0,64 (M11) → 0,68 (R2/R3).

**Decisión.** El procedimiento de R2/R3 es la base de M13.3b y M13.4.

## 5. M13.3b — Mismo procedimiento en G3

**Hipótesis.** G3 (banda 30–1024 Hz, filtro de blanqueado de 16 s, SNR en 30–1024 Hz, mismas fuentes que G1 y otro ruido; [M13_0d_G3_generation.md](M13_0d_G3_generation.md)) mejora el estimador frente a G1 con el mismo procedimiento.

**Preparación de G3.** [merge_m13_G3_shards.py](../../scripts/m13/merge_m13_G3_shards.py) y [finalize_m13_G3_merge.py](../../scripts/m13/finalize_m13_G3_merge.py) (cierre de las comprobaciones del merge sin tocar X); [prepare_g3_split_artifacts.py](../../scripts/m13/prepare_g3_split_artifacts.py) copia el split y las estadísticas de etiquetas train-only de G1 tras comprobar `y`, `file_group_id` y pertenencia fila a fila.

**Diseño.** `M13_3b_G3_s123` y `M13_3b_G3_s124` (procedimiento de R2). Comparación con la media de R2/R3 en val (40 000 filas emparejadas por fuente). ΔRMSE% = 100·(media G3 / media G1 − 1); ruido de semilla por dataset con dos semillas; "resuelto" si |Δ| > 2·max(ruidos). IC 95 %: bootstrap emparejado por filas (2000, semilla 136001), mide muestreo de filas, no variación entre semillas.

**Criterio de lectura por banda (fijado, descriptivo).** Clases de f₂₂₀ (<256, 256–512, 512–1024, >1024 Hz; `>512` agrupada). BANDA si en M o en χ_eff el ΔRMSE% de `>512` es más negativo que el de `<256` sin solape de IC; UNIFORME si se solapan en ambas; otro caso MIXTO. f₂₂₀ correlaciona con la masa: no separa banda de masa.

**Resultados (val).**

| Etiqueta | ΔRMSE% G3 vs G1 [IC 95 %] |
|---|---|
| chirp_mass | −10,2 [−11,9; −8,7] |
| total_mass | −12,6 [−14,2; −11,1] |
| chi_eff | −6,5 [−7,5; −5,4] |

- Entre 4 y 11 veces el ruido de semilla (≤ 1,7 %). Mejora en todos los bins de SNR (−10 a −13 %).
- Colas: Q10 de M_c +1,2 → +0,6 M☉; Q10 de M +3,7 → +2,2 M☉.
- Banda: lectura BANDA. M: `>512` −50,1 % [−55,0; −43,9] frente a `<256` −9,9 %. χ_eff: `>512` −3,8 % [−8,3; +0,8] frente a `<256` −6,9 %.
- Sistemas ligeros (f₂₂₀ > 512 Hz o M < 30 M☉, ~4 % de val): M_c −61 a −79 %, M −38 a −72 %. Coherente con recuperar fusión y ringdown cortados a 512 Hz en G1.
- Sistemas pesados (f₂₂₀ < 256 Hz, M ≥ 80 M☉): −7 a −12 % en las tres etiquetas; no atribuible a la banda; compatible con HIP-LÍNEAS. No demostrado.
- C3: riesgo evitable 0,7–1,3 % (sin resolver), frente a 1,5–3 % en G1.
- Sobreajuste: mejor val en las épocas 62–64; la final es un 1,1–1,5 % peor. Brecha MSE_train40k/MSE_val ≈ 0,71 en G1 y en G3.

**Conclusiones.** G3 mejora de forma resuelta. El efecto es un paquete (banda, filtro, realización de ruido, banda del SNR) y no se atribuye a una sola parte. χ_eff mejora de forma casi uniforme y no más en los ligeros, coherente con que χ_eff se mide sobre todo en el inspiral. El estimador está prácticamente en el óptimo de su embedding (C3).

**Decisión.** G3 es el dataset de producción y M13.3b (G3, procedimiento de R2) la base.

## 6. M13.4a — Agregación por transformer (HIP-CONTEXTO)

**Hipótesis (HIP-CONTEXTO).** El promedio global de M11 pierde contexto temporal largo; una agregación con atención debería mejorar χ_eff sobre todo en señales largas (τ alto).

**Diseño.** `SimpleCNN_ResidualDilatedTransformerAgg` ([network.py](../../src/models/network.py)): tronco de M11, AvgPool1d(8) → 256 tokens → transformer de 2 capas, 4 cabezas, con CLS, en lugar del promedio global. Coste: [count_model_cost.py](../../scripts/m13/count_model_cost.py). Una semilla (`M13_4a_G3_s123`), procedimiento de M13.3b, G3. Comparación con la media de M13.3b.

**Criterio fijado.** Por decil de τ30 (newtoniano, M_c verdadera, deciles de M13.1): ΔRMSE% de χ_eff frente a la media de 3b, IC por bootstrap dentro del decil (semilla 136003). Contraste media(D8–D10) − media(D1–D3): APOYA si es negativo y su IC excluye 0; si no, NO APOYA.

**Resultados (val).**
- Global, sin resolver: +2,3 / +1,7 / −0,05 % (ruido de semilla 1,4 / 1,1 / 0,6 %).
- HIP-CONTEXTO: APOYA. χ_eff mejora en D8–D10 y empeora en D1–D3; diferencia −3,0 % [−4,7; −1,3]. Una sola semilla: el IC no incluye variación entre semillas.
- Sobreajuste mayor: la final es un 3,3 % peor que la mejor val; brecha MSE_train40k/MSE_val 0,55–0,57.

**Hipótesis sin verificar.** La ganancia de contexto queda anulada por el sobreajuste (en la época 64 su val es igual a la de 3b). La brecha train/val que no cambia con el blanqueado apunta a memorización del ruido fijo (HIP-RUIDO-FIJO): se contrasta en M13.5 ([M13_5_renoise.md](M13_5_renoise.md)). Una fase sin dropout de 2–4 épocas podría corregir la pendiente sin el coste en val; pospuesto.

**Decisión.** M13.4a no se adopta. HIP-CONTEXTO queda apoyada como patrón y se reevaluará en el régimen con aumentación de ruido (M13.4a-aug, [M13_6_noise_augmentation.md](M13_6_noise_augmentation.md)). M13.4b (`MRDenseNet`) queda pendiente de rediseño (checkpointing) en ese mismo régimen.
