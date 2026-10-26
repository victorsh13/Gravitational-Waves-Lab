# M12.5.3 — Comprobación post-hoc de una condición necesaria de optimalidad de M11

Estado: protocolo **pre-registrado**, escrito antes de cualquier resultado.
Script: `scripts/m12/m12_5_3_posthoc_check.py`. Datos: solo el split **val** de M11 final G1.

## Hipótesis

Si el estimador puntual de M11, f(X), fuera la media posterior E[θ|X], entonces para cualquier
función g(X) se cumpliría E[θ − f(X) | g(X)] = 0. En consecuencia, ninguna función de
(f(X), emb(X)) podría reducir el MSE de M11. Si un corrector entrenado solo con la propia salida
y el embedding de M11 reduce el MSE fuera de fold, la condición necesaria se incumple.

## Datos y contrato

- `$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/m11_final_G1_500k_val_predictions_embeddings.npz`
  (`pred_val`, `y_val`, `emb_val`, `idx_val`, `y_mean`, `y_std`, `label_names`, `split_path`), cargado con
  `allow_pickle=False`. `model_config` e `input_normalization` no se leen.
- Orden de etiquetas: [chirp_mass, total_mass, chi_eff].
- El entorno PSD de cada fila es `file_group_id`, y la SNR objetivo es `target_network_snr`, ambos del
  HDF5 G1 e indexados por `idx_val`. X no se lee.
- Se reutiliza en solo lectura `g1_npe_contract.load_contract`, que comprueba los hashes fijados del split y del
  escalador, las cabeceras del HDF5 y la ausencia de fuga de entornos entre splits.
- Abortan la ejecución: `idx_val` con duplicados o distinto de la pertenencia a val; `y_val` fuera de unidades físicas
  (|media − y_mean| ≥ 0,1·y_std en alguna etiqueta), sin conversión automática; valores no finitos; menos de 3
  entornos en val. También abortan, como salvaguardas adicionales: el mismo criterio de unidades aplicado a `pred_val`,
  `y_val` ≠ `y[idx_val]` del HDF5, y `y_mean`/`y_std` del npz distintos de las estadísticas train-only fijadas.
- cal y test no se tocan.

## Diseño (fijo)

Validación cruzada dejando fuera un entorno de val cada vez (LOEO). Cada corrector se ajusta en los
demás entornos y predice el entorno retenido, en unidades físicas.

| Corrector | Definición | Hiperparámetro |
|---|---|---|
| C1 afín | θ ≈ W·f + b, W de 3×3, mínimos cuadrados | — |
| C2 binned | Para cada etiqueta, media de θ_j en 50 bins equi-cuantiles de f_j. Interpolación lineal entre centros (media de f_j en el bin) y valor del bin extremo fuera de rango. No se impone monotonía a las medias. | — |
| C3 ridge | Ridge sobre θ − f con entradas [emb, pred] estandarizadas en los entornos de entrenamiento. Intercepto no penalizado. | α ∈ {1e-2, …, 1e4} por etiqueta, LOEO interno |
| C4 kNN | Media de θ − f entre los k vecinos en emb estandarizado (distancia euclídea, por bloques). | k ∈ {16, 64, 256} por etiqueta, LOEO interno |

## Métricas

Todas son fuera de fold, por corrector, etiqueta y región.

- MSE de M11 y del corrector; d_i = e_M11² − e_corr² (un valor positivo favorece al corrector). Las diferencias negativas se
  conservan sin recortar.
- Bootstrap pareado: se remuestrean filas dentro de cada entorno conservando el número de filas por entorno, con 2000 réplicas
  y semilla 125301. Intervalo percentil marginal al 99,5833 % (Bonferroni sobre 4 correctores × 3 etiquetas).
- Margen práctico: 1 % del MSE de M11 en esa etiqueta y región.
- Pendientes, globales y por entorno, para M11 y para cada corrector: pendiente e intercepto OLS de la predicción sobre la verdad,
  R² = 1 − SSE/SST, y pendiente e intercepto de la verdad sobre la predicción.
  **Identidad de calibración:** si p = E[θ|·], entonces cov(p, θ) = var(p). Por tanto, pendiente(p sobre θ) = var(p)/var(θ) ≈ R²
  y pendiente(θ sobre p) ≈ 1. Se registra `slope_minus_r2`.
- Regiones (**solo descriptivas**, sin control de multiplicidad entre regiones): cada entorno; deciles de `pred_val`
  por etiqueta, definidos a partir de X y no de la verdad; SNR objetivo en [10,15), [15,20) y [20,25]. Los grupos con n < 200 se marcan
  como `sparse` y no reciben etiqueta de decisión.

## Criterio de decisión (solo comparaciones globales)

Sea [lo, hi] el intervalo al 99,5833 % de la media de d y m el margen.

- `AVOIDABLE_DEMONSTRATED`: lo > m (todo el intervalo por encima del margen).
- `NO_EVIDENCE_WITHIN_MARGIN`: −m ≤ lo y hi ≤ m (todo el intervalo dentro de ±m).
- `UNRESOLVED`: cualquier otro caso.

## Qué permite concluir

`AVOIDABLE_DEMONSTRATED` da una **cota inferior** del riesgo evitable de M11 en esa etiqueta. La cota se obtiene
usando solo la representación del propio M11 (f, emb).

## Qué no permite concluir

No encontrar mejora **no** demuestra que M11 sea óptimo respecto a X. Solo indica calibración relativa a la
representación de M11, dentro de la clase de correctores probada. Este experimento no valida ni selecciona ningún modelo.

## Limitaciones

- El bootstrap mantiene fijas las predicciones fuera de fold, así que no incluye la variabilidad del reajuste de los correctores.
- Hay pocos entornos en val: el LOEO y las variaciones entre entornos son poco precisos.
- Los correctores se ajustan en val. El estudio es exploratorio y no sirve para seleccionar el modelo final ni para fijar cuantiles conformales.
- cal y test no se tocan.

## Salida

El directorio exclusivo es `$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_3_posthoc/`. La ejecución aborta si ya existe.

- `risk.csv`: comparación global y decisión.
- `slopes.csv`: diagnóstico de pendientes.
- `regions.csv`: resultados descriptivos por región.
- `folds.csv`: hiperparámetros por fold, con el MSE del LOEO interno de cada candidato y el elegido.
- `analysis.json`: SHA256 del npz y del script, identidad del contrato, número de entornos y filas por entorno, comprobaciones
  superadas, origen de la SNR y decisiones.

## Regla de parada

Se hace **una sola ejecución** con estos 4 correctores y estos ajustes. No se añaden correctores, rejillas ni regiones a la vista de los
resultados. Cualquier seguimiento requiere un protocolo nuevo.

## Coste estimado

Minutos de CPU en local. El término dominante es el kNN con LOEO anidado, cuyo coste es de orden E·n² distancias en 64 dimensiones (n = 40 000). Si el número de
entornos E es grande, puede llegar a decenas de minutos. Memoria: unos cientos de MB.
