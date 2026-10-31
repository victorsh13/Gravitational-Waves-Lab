# Índice de la fase M13

Este índice recoge los datasets, experimentos y términos de la fase M13, con su estado y su documentación.

## 1. Datasets

| Dataset | Descripción | Doc |
|---|---|---|
| G1 | Dataset actual, banda 30–512 Hz. Congelado. | — |
| G2 | G1 emparejado con banda 30–1024 Hz y blanqueado de G1. Descartado por líneas residuales (solo existe el piloto de 64 filas) | M13_0_G2_band1024.md |
| G3 | banda 30–1024 Hz, blanqueado largo (filtro de 16 s, M13.0c), SNR en 30–1024 Hz, mismas fuentes que G1, otro ruido. **Dataset de producción** desde M13.3b (28-10) | M13_0d_G3_generation.md, M13_3_procedure_and_G3.md |

## 2. Experimentos

| id | qué hace | dónde corre | estado | doc |
|---|---|---|---|---|
| M13.0a | comprobación de SNR por banda | CPU | cerrado (26-10): USER_DECISION, 1,47 %; se decide generar G2 | M13_0_G2_band1024.md |
| M13.0b | generación de G2: piloto, fragmentos, merge | CPU | piloto hecho; G2 descartado por líneas (26-10) | M13_0_G2_band1024.md |
| M13.0c | estudio del blanqueado: barrido de duración del filtro (0,5/2/4/8 s) sobre 64 filas | CPU | cerrado (26-10): filtro 16 s | M13_0c_whitening.md |
| M13.0d | generación de G3 | CPU local | cerrado: G3 completo; usado en M13.3b/M13.4a/M13.5 | M13_0d_G3_generation.md |
| M13.1a | Error y sesgo de M11 por τ, f220, q y SNR | CPU | cerrado (26-10) | M13_1_diagnostics.md |
| M13.1b | Correlación de residuos χ_eff–η | CPU | cerrado (26-10) | M13_1_diagnostics.md |
| M13.1c | Respuesta en frecuencia de los kernels de block1 de M11 | CPU | cerrado (26-10) | M13_1_diagnostics.md |
| M13.2a | M11 con el inspiral temprano enmascarado | inferencia GPU | propuesto | — |
| M13.2b | M11 con la entrada filtrada por debajo de 256 Hz | inferencia GPU | propuesto | — |
| M13.2c | M11 sin fusión ni ringdown | inferencia GPU | propuesto | — |
| M13.3.0 | dilución por dropout en M11 | inferencia GPU | cerrado (26-10): PARCIAL (fracción explicada 0,89 / 0,92 / 0,59) | M13_3_procedure_and_G3.md |
| M13.3-IO | perfil de E/S y cómputo | GPU | cerrado (26-10): limitado por cómputo (241 ms/lote); memmap fp16 adoptado; AMP descartado | M13_3_procedure_and_G3.md |
| M13.3a | M11 reentrenado (planificador coseno 80 épocas, lotes aleatorios, memmap fp16; R1 dropout M11, R2/R3 dropout apagado último 20 %) en G1 | entrenamiento GPU | cerrado (27-10): RMSE sin cambio resuelto; pendiente en val 1,04–1,06 → 0,97–0,99 y menor sesgo en colas; P3 y P4 fallan | M13_3_procedure_and_G3.md |
| M13.3b | M11 reentrenado como M13.3a (dropout apagado último 20 %) en G3 | entrenamiento GPU | cerrado (28-10): G3 frente a G1 −10,2 / −12,6 / −6,5 % RMSE (resuelto); pasa a ser la base | M13_3_procedure_and_G3.md |
| M13.4a | M11 + agregación por transformer: AvgPool1d(8) → 256 tokens → transformer 2 capas con CLS, en lugar del promedio global (`SimpleCNN_ResidualDilatedTransformerAgg`) | entrenamiento GPU | cerrado (28-10): global sin resolver; HIP-CONTEXTO apoyada (χ_eff τ largo − corto −3,0 % [−4,7; −1,3], 1 semilla); no se adopta | M13_3_procedure_and_G3.md |
| M13.4b | Multirresolución densa (7 etapas, 16384 → 256 pasos, ramas k = 3/9/27) con lectura por escala (atención temporal) y transformer sobre la última etapa (`MRDenseNet`) | entrenamiento GPU | con aumentación y checkpointing por etapa: en curso | M13_6_noise_augmentation.md |
| M13.4.x | Arquitecturas candidatas | entrenamiento GPU | por definir | — |
| M13.5 | diagnóstico de la brecha train–val: ruido fijo frente a entorno | CPU local | cerrado (28-10): RUIDO_FIJO en 3 modelos × 3 etiquetas (f 0,90–1,08), control válido | M13_5_renoise.md |
| M13.6 | aumentación de ruido: señal limpia + banco de ruido por PSD; M13.6a (s123, s124) y M13.4a-aug | CPU local + GPU | cerrado (31-10): aumentación adoptada como base; P-AUG1–3 PASA; 6a −6 % frente a 3b | M13_6_noise_augmentation.md |
| M13.4a-aug | M13.4a + aumentación de ruido (M13.6) | entrenamiento GPU | 4a + aumentación: mejor modelo (−8 a −9 % frente a 3b; HIP-CONTEXTO apoyada); 1 semilla, s124 en curso | M13_6_noise_augmentation.md |

## 3. Términos heredados de M12

| Término | Significado | Origen |
|---|---|---|
| C1 | Corrección afín | M12.5.3 |
| C2 | Corrección por bins | M12.5.3 |
| C3 | Ridge sobre embedding | M12.5.3 |
| C4 | kNN | M12.5.3 |
| HR | Reajuste de la cabeza en train | M12.5.4 |
| Tarjeta de evaluación | Métricas fijas para comparar modelos | M12 |

---

*Nota:* los nombres de las hipótesis se fijarán al diseñar cada experimento.

---

## 4. Hipótesis

- **HIP-LÍNEAS:** el blanqueado con filtro de 0,5 s deja líneas que dominan la varianza de X y, vía z-score, encogen la componente útil. Verificado en G1 (64 filas): V1 98,7 % de la varianza en líneas (mediana), L1 5,9 %, H1 0,9 %. Efecto sobre M11: no medido. Corrección: blanqueado largo en G3 (M13.0c).
- **HIP-CONTEXTO:** el contexto temporal largo (agregación por transformer) mejora χ_eff en señales de τ largo. **Apoyada por M13.4a-aug** (régimen aumentado): (D8–D10) − (D1–D3) = −2,1 % [−3,2; −0,9] en ΔRMSE de χ_eff frente a M13.6a (1 semilla).
- **HIP-RUIDO-FIJO:** la brecha train–val (MSE_train40k/MSE_val ≈ 0,71 en G1 y G3) se debe a que la red memoriza la realización fija de ruido de cada fila de train, no a un cambio de entorno PSD. **Verificada por M13.5:** cambiar solo el ruido gaussiano de 10 000 filas de train lleva su MSE al nivel de val (f = 0,90–1,08, ≥ 0,7 en los 3 modelos y las 3 etiquetas; control de val válido). Consecuencia: aumentación de ruido (M13.6).

## 5. Líneas futuras

- **Procesado aprendido en lugar de blanqueado (muy a futuro):** red que separe la componente de onda gravitacional del ruido (eliminación de ruido condicionada a la forma de onda) en lugar de blanquear. Separar componentes de ruido entre sí (sísmico, cuántico) no es identificable desde el strain solo: requeriría canales auxiliares (tipo DeepClean). Para ruido gaussiano el blanqueado no pierde información; el valor esperado está en lo no gaussiano (líneas, glitches, no estacionariedad). Conecta con la línea de glitches.