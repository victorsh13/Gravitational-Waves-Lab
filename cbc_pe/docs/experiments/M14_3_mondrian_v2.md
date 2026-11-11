# M14.3 — Mondrian v2: score normalizado por σ (M14.1)
**Estado:** pre-registrado el 11-11-2026.
**Origen (post hoc, declarado):** M14.1 cumplió C1 y falló C2 (una celda de 21, chirp D1–D3, grupo definido por la verdad). El análisis descriptivo sugiere que σ está calibrada en particiones que se calculan desde X. M14.3 responde directamente si σ sirve para el Mondrian; no modifica el veredicto de M14.1.
**Pregunta:** con el mismo modelo (M14.1), ¿dividir el score por σ da intervalos válidos y más estrechos que el Mondrian v1?

## Diseño (un factor)
- Modelo: M14.1 (μ, σ) sobre cal, val y test de G3T.
- **v1:** Mondrian de M13.13 (enmienda 6) sin cambios (método, taxonomías, número de bins, política conservative, nivel 0,90), aplicado a μ de M14.1.
- **v2:** idéntico a v1 (mismos bins y misma asignación de filas a bins), con el residuo r sustituido por r/σ en el score; el intervalo se reescala con σ por fila: μ ± q·σ en las etiquetas simétricas, [μ + q_lo·σ, μ + q_hi·σ] en chirp (asimétrico).
- Calibración en cal (30 000 filas). **Decisión en val** (40 000 filas). Test (30 000 filas): confirmación única de la política elegida.
- `src/conformal` no se modifica.

## Gates
- La inferencia de M14.1 en val reproduce el npz de val del entrenamiento (máx. |Δμ| y |Δσ| ≤ 1e-5, unidades estandarizadas).
- RMSE de μ en cal y en test dentro de ±2 % del de val.

## Criterios (val)
1. **Validez:** en v2, ninguna celda (marginal ni por bin) por debajo de su banda inferior (Beta-Binomial con el ruido de cal, procedimiento de la enmienda 6). La sobrecobertura se reporta junto a v1 y no invalida si v1 la muestra en la misma celda.
2. **Eficiencia (primario):** anchura media del intervalo, 100·(v2/v1 − 1), por etiqueta, con IC 95 % por bootstrap emparejado de filas de val (B = 2000, default_rng(143001); no incluye el ruido de cal). Mejora resuelta si IC sup ≤ −5 %; empeora si IC inf ≥ +5 %.
3. **Decisión por etiqueta:** v2 se adopta en las etiquetas con validez y mejora resuelta; en el resto se mantiene v1.

## Secundario (no decide)
- Cobertura condicional por tercil de SNR (SNR nominal de G3, como en M14.1): max_t |cobertura_t − 0,90| en v1 y en v2. Se espera menor en v2.

## Descriptivo pre-registrado (hipótesis de M14.1)
- En val: std(z) y mean(z) por grupos de τ30 **predicha** = τ_newtonian(μ_chirp), con cortes en los deciles de la τ30 predicha en val (D1–D3, D4–D7, D8–D10).
- Hipótesis: std(z) ∈ [0,9; 1,1] y |mean z| ≤ 0,1 en las 9 celdas. Si se cumple, apoya que el fallo de C2 venía de agrupar por la verdad. No cambia el veredicto de M14.1.

## Otros descriptivos
- Anchura mediana.
- Anchura y cobertura por tercil de SNR y por τ30 verdadera.
- Comparación con el Mondrian de la enmienda 6 (4a-G3T) en val y test.

## No se puede concluir
- Comportamiento en datos reales.
- Variación entre semillas.
- Si r/σ es el mejor score posible (paso futuro: revisión de scores).
- El IC no incluye la aleatoriedad de cal.

## Regla de parada
Una inferencia (cal, test y val como gate) y un notebook. Sin ajustar bins, nivel ni scores.

## Enmienda 1 (11-11-2026, antes de ejecutar la inferencia)
- Motivo: la inferencia de cal, val y test usa el memmap fp16. Si el entrenador predijo val con entradas float32, μ y σ difieren por el redondeo de la entrada (orden 1e-4–1e-3 estandarizado) y el gate de ≤ 1e-5 fallaría sin error de ruta.
- Gate de val: si el entrenador usó el memmap fp16, se mantiene ≤ 1e-5. Si usó float32: máx. |Δμ| ≤ 1e-2 (estandarizado), máx. |Δ log σ| ≤ 1e-2 y |ΔRMSE| ≤ 0,1 % por etiqueta. Se reportan también la mediana y el P99 de |Δμ| y |Δ log σ|.
- Justificación: el gate verifica la ruta de inferencia (checkpoint, separación μ/σ, exp(log σ), alineación de filas); un error de ruta da diferencias de orden 1.
- Predicciones para la decisión: las de la inferencia fp16 de val, la misma ruta de entrada que cal y test.
- Elección de la rama: la determina el dtype de X de val en el fichero que usó el entrenador (`training.data_loading_mode = hdf5_batch_slices`; ruta del bloque `dataset` del config de entrenamiento de M14.1): float16 → ≤ 1e-5; float32 → tolerancias de redondeo fp16. El notebook lo lee y lo registra en el gate.
