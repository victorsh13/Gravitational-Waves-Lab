# Índice de la fase M13

Este índice recoge los datasets, experimentos y términos de la fase M13, con su estado y su documentación.

## 1. Datasets

| Dataset | Descripción | Doc |
|---|---|---|
| G1 | Dataset actual, banda 30–512 Hz. Congelado. | — |
| G2 | G1 emparejado con banda 30–1024 Hz y blanqueado de G1. Descartado por líneas residuales (solo existe el piloto de 64 filas) | M13_0_G2_band1024.md |
| G3 | banda 30–1024 Hz, blanqueado largo (duración por M13.0c), SNR en 30–1024 Hz, mismas fuentes que G1, otro ruido. Candidato a dataset de producción | M13_0c_whitening.md |

## 2. Experimentos

| id | qué hace | dónde corre | estado | doc |
|---|---|---|---|---|
| M13.0a | comprobación de SNR por banda | CPU | cerrado (26-10): USER_DECISION, 1,47 %; se decide generar G2 | M13_0_G2_band1024.md |
| M13.0b | generación de G2: piloto, fragmentos, merge | CPU | piloto hecho; G2 descartado por líneas (26-10) | M13_0_G2_band1024.md |
| M13.0c | estudio del blanqueado: barrido de duración del filtro (0,5/2/4/8 s) sobre 64 filas | CPU | en curso | M13_0c_whitening.md |
| M13.1a | Error y sesgo de M11 por τ, f220, q y SNR | CPU | propuesto | — |
| M13.1b | Correlación de residuos χ_eff–η | CPU | propuesto | — |
| M13.1c | Respuesta en frecuencia de los kernels de block1 de M11 | CPU | propuesto | — |
| M13.2a | M11 con el inspiral temprano enmascarado | inferencia GPU | propuesto | — |
| M13.2b | M11 con la entrada filtrada por debajo de 256 Hz | inferencia GPU | propuesto | — |
| M13.2c | M11 sin fusión ni ringdown | inferencia GPU | propuesto | — |
| M13.3a/b | M11 reentrenado con procedimiento corregido en G1 / G2 | entrenamiento GPU | propuesto | — |
| M13.4.x | Arquitecturas candidatas | entrenamiento GPU | por definir | — |

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

## 5. Líneas futuras

- **Procesado aprendido en lugar de blanqueado (muy a futuro):** red que separe la componente de onda gravitacional del ruido (eliminación de ruido condicionada a la forma de onda) en lugar de blanquear. Separar componentes de ruido entre sí (sísmico, cuántico) no es identificable desde el strain solo: requeriría canales auxiliares (tipo DeepClean). Para ruido gaussiano el blanqueado no pierde información; el valor esperado está en lo no gaussiano (líneas, glitches, no estacionariedad). Conecta con la línea de glitches.