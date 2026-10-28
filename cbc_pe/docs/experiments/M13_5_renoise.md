# M13.5 — Brecha train–val: ruido fijo frente a entorno

**Estado:** cerrado (28-10-2026). Resultado: RUIDO_FIJO en los 3 modelos y las 3 etiquetas; control válido. HIP-RUIDO-FIJO verificada.

Fuente de las cifras: `<data_root>/processed/M13_G3_band1024_w16_500k/m13_5_renoise/m13_5_summary.json` (solo lectura).

## 1. Pregunta

En M13.3b/M13.4a la brecha MSE_train40k/MSE_val es ≈ 0,71 (0,55–0,57 en 4a) y no cambia de G1 a G3. Dos explicaciones:
- **HIP-RUIDO-FIJO:** la red memoriza la realización concreta de ruido de cada fila de train (cada fila se ve 80 veces con el mismo ruido).
- **ENTORNO:** train y val tienen entornos PSD disjuntos (`file_group_id`); la brecha es cambio de dominio.

Se distinguen regenerando filas de train **cambiando solo la realización del ruido gaussiano**: si la brecha es memorización, el MSE de esas filas sube hasta el de val; si es entorno, apenas cambia.

## 2. Diseño

- **Filas** ([m13_5_select_rows.py](../../scripts/m13/m13_5_select_rows.py)): 10 000 de train40k (`default_rng(135002)`, el mismo conjunto para los tres modelos) y 2000 de val (`default_rng(135003)`); 16 filas de control. IDs originales del HDF5.
- **Regeneración** ([generate_m13_5_renoise.py](../../scripts/m13/generate_m13_5_renoise.py)): física, metadatos, PSD, procesado, ventana y colocación de G3 sin cambios. Solo se sustituye `gaussian_seed`, que siembra únicamente los tres `noise_from_psd` por detector: `noise_seed_new = SeedSequence([gaussian_seed, 135001]).generate_state(1, uint64)[0] >> 1`, distinta de toda semilla del manifiesto. 4 fragmentos, 12 000 filas en total.
- **Modelos** ([m13_5_check_and_predict.py](../../scripts/m13/m13_5_check_and_predict.py)): `_final.pt` de `M13_3b_G3_s123`, `M13_3b_G3_s124` y `M13_4a_G3_s123`; inferencia en CPU, modo eval, X por float32 → float16 → float32 como en el memmap de entrenamiento.
- **Estadístico** (unidades estandarizadas train-only, por etiqueta):
  f = (MSE_train_new − MSE_train_orig) / (MSE_val40k − MSE_train_orig),
  fracción de la brecha train–val que recupera el cambio de ruido.
- **IC 95 %:** bootstrap percentil, 2000 réplicas, `default_rng(135004)`; train10k y val2k emparejados orig/new, val40k independiente; mismos remuestreos para todos los modelos.

## 3. Criterios fijados

- Sobre la estimación puntual de f: **RUIDO_FIJO** si f ≥ 0,7; **ENTORNO** si f ≤ 0,3; **MIXTO** en otro caso; **SIN_BRECHA** si MSE_val40k ≤ MSE_train_orig.
- Control en val: MSE_val2k_new / MSE_val2k_orig con IC que contiene 1 → **VALIDO** (cambiar el ruido en filas no vistas no debe cambiar el error).

## 4. Controles

| Control | Resultado |
|---|---|
| Reproducción (`--reproduce-original`, 16 filas) contra el trainio de G3 | max\|ΔX\| = 0,0; `y`, `final_distance` y `file_group_id` idénticos |
| Filas regeneradas (12 000) | `y`, `final_distance`, SNR, `file_group_id` y split idénticos a G3 (comprobación con aborto) |
| Correlación por fila X nuevo vs X de G3 | mediana 0,013 (q05 −0,004, q95 0,031): ruido distinto |
| Predicciones con el ruido de G3 vs npz guardados | max\|Δpred\| ≤ 5,7e-6 (estandarizado), `y` idéntica en los 3 modelos |

## 5. Resultado

| Modelo | Etiqueta | MSE train orig | MSE train new | MSE val40k | f [IC 95 %] | val new/orig [IC 95 %] | Lectura |
|---|---|---|---|---|---|---|---|
| 3b s123 | chirp_mass | 0,0728 | 0,0999 | 0,1010 | 0,96 [0,75; 1,20] | 0,89 [0,78; 1,02] | RUIDO_FIJO |
| 3b s123 | total_mass | 0,0513 | 0,0686 | 0,0704 | 0,90 [0,71; 1,13] | 0,96 [0,83; 1,10] | RUIDO_FIJO |
| 3b s123 | chi_eff | 0,1241 | 0,1714 | 0,1721 | 0,99 [0,82; 1,16] | 1,08 [0,94; 1,24] | RUIDO_FIJO |
| 3b s124 | chirp_mass | 0,0719 | 0,0991 | 0,0983 | 1,03 [0,81; 1,29] | 0,97 [0,85; 1,11] | RUIDO_FIJO |
| 3b s124 | total_mass | 0,0502 | 0,0681 | 0,0689 | 0,96 [0,78; 1,16] | 0,99 [0,87; 1,12] | RUIDO_FIJO |
| 3b s124 | chi_eff | 0,1219 | 0,1722 | 0,1701 | 1,04 [0,88; 1,22] | 1,00 [0,89; 1,13] | RUIDO_FIJO |
| 4a s123 | chirp_mass | 0,0575 | 0,1078 | 0,1042 | 1,08 [0,93; 1,23] | 0,89 [0,76; 1,04] | RUIDO_FIJO |
| 4a s123 | total_mass | 0,0412 | 0,0729 | 0,0719 | 1,03 [0,89; 1,17] | 0,95 [0,83; 1,10] | RUIDO_FIJO |
| 4a s123 | chi_eff | 0,1011 | 0,1752 | 0,1709 | 1,06 [0,94; 1,18] | 1,01 [0,90; 1,13] | RUIDO_FIJO |

Control de val VALIDO en las 9 combinaciones. Todas las cotas inferiores del IC de f son ≥ 0,71.

## 6. Conclusiones

1. **HIP-RUIDO-FIJO verificada.** Cambiar solo la realización del ruido en filas de train lleva su MSE al nivel de val (f ≈ 0,9–1,1); la brecha train–val es memorización del ruido fijo, no cambio de entorno PSD. El mismo resultado en las dos semillas de 3b y en 4a.
2. El mayor sobreajuste de 4a (brecha 0,55–0,57) es también memorización de ruido: su MSE en train con ruido nuevo queda al nivel de val.
3. Consecuencia para P3 de M13.3a: la hipótesis de cambio de dominio no se sostiene como explicación de la brecha.
4. Siguiente paso: aumentación de ruido en train (M13.6, [M13_6_noise_augmentation.md](M13_6_noise_augmentation.md)). Que la aumentación mejore val no se deduce de este resultado; se mide en M13.6a.
