# M12.5.4 — Origen de la contracción excesiva de M11: entrenamiento frente a cambio de entorno (reajuste de la cabeza en train)

Estado: protocolo **pre-registrado**, escrito antes de cualquier resultado.
Script: `scripts/m12/m12_5_4_head_refit.py`. Notebook: `notebooks/m12/M12_5_4_head_refit_analysis.ipynb`.
Datos: splits **train** y **val** de M11 final G1. cal y test no se tocan; X no se lee.

## Contexto

En M12.5.3, C3 (ridge sobre el residuo con [h, f], ajustado en val con LOEO por entorno) redujo el MSE de M11 en val.
La cabeza de SimpleCNN_ResidualDilated es `Linear(64, 3)` aplicada directamente sobre h, de modo que f = A·h + a
(aquí se comprueba numéricamente). C3 equivale, por tanto, a reajustar la cabeza con datos de val. La pregunta es si
el mismo reajuste hecho **en train** (HR) recupera esa ganancia.

## Hipótesis

- **H1 (inducida por el entrenamiento).** La cabeza que se aprende no es la de MCO sobre h en el modo de evaluación,
  y la causa está en el propio entrenamiento: el dropout está antes de h, así que la media y la varianza de h cambian
  entre el modo de entrenamiento y el de evaluación; AdamW usa weight_decay 3e-4, que contrae los pesos de la cabeza;
  y la parada temprana sobre val detiene el ajuste antes de llegar al óptimo. *Predicción:* M11 también está
  contraído en train, s_tr > 1 (θ sobre f con pendiente mayor que 1). HR corrige esa contracción con datos de train y
  la corrección se transfiere a val: R ≈ 1.
- **H2 (cambio de entorno).** La cabeza es la óptima en train (s_tr ≈ 1). La descalibración de val se debe a que los
  entornos PSD de val difieren de los de train. *Predicción:* s_tr ≈ 1 y HR apenas cambia la cabeza, con lo que R ≈ 0.
  C3 solo gana porque se ajusta en los entornos de val.

## Diseño (fijo)

1. Config de inferencia `configs/prediction/predict_m11_final_G1_500k_train.json`, que es una copia exacta de la de
   val con `splits: ["train"]` y otro fichero de salida. El usuario ejecuta en GPU un pase solo hacia delante con
   `predict_cnn_hdf5.py`. El checkpoint de M11 no se modifica.
2. **Comprobaciones que abortan:** las de M12.5.3 aplicadas a cada npz; SHA256 fijado del npz de val; n_train = 400 000;
   h de dimensión 64; f lineal en h en cada partición (residuo máximo de lstsq(f ~ [h,1]) < 1e-4 en unidades
   estandarizadas); misma cabeza en las dos particiones (|Δcoef| ≤ 1e-3); reproducción de los MSE de M11 y de C3 de
   `M12_5_3_posthoc/risk.csv` con tolerancia relativa de 1e-6.
3. **A. Calibración de M11** en train y en val, global y por entorno: MSE, pendiente y ordenada de θ sobre f (MCO),
   pendiente de f sobre θ y R².
4. **B. Reajuste de la cabeza (HR):** para cada etiqueta, MCO de θ_j ~ [h, 1] con todas las filas de train, en unidades
   físicas y sin regularización. Los coeficientes (65×3) se guardan aparte en `head_refit_coefs.npz` y se aplican a val.
5. **C. Comparación en val:** d_HR = e²_M11 − e²_HR y d_C3 = e²_M11 − e²_C3, con C3 recalculado fuera de pliegue con los
   ajustes de M12.5.3. Bootstrap pareado dentro de cada entorno, con 2000 réplicas y semilla 125401; las mismas filas
   remuestreadas se usan para todas las cantidades. Intervalos percentiles al 98,333 % (Bonferroni sobre 3 etiquetas) para
   ΔMSE_HR/MSE_M11, ΔMSE_C3/MSE_M11, R = mean(d_HR)/mean(d_C3) y mean(d_C3 − d_HR). Los cocientes se calculan dentro de
   cada réplica.
6. **E. Descriptivo** (sin control de multiplicidad): en val, por decil con signo de la predicción de M11 (la misma
   definición que en M12.5.3), por intervalo de SNR objetivo y por entorno, para M11, HR y C3. Se dan n, sesgo
   (media de θ − pred), MSE, ganancia relativa frente a M11 y fracción de la ganancia global. También las pendientes de HR
   en val. En train, solo para M11: sesgo y MSE por decil con signo.

Todo el análisis se hace en unidades físicas y en float64.

## Por qué R es la métrica decisional

C3 da una cota inferior del riesgo de M11 que se puede evitar reajustando linealmente sobre (h, f). Como f es lineal en h,
esa cota es en esencia la de reajustar la cabeza. R mide qué fracción de esa ganancia se obtiene sin ver val, solo con
train. Las dos hipótesis predicen extremos opuestos de R (H1 ≈ 1, H2 ≈ 0). Además, R está normalizado por la ganancia
disponible, así que no depende de la escala del MSE de cada etiqueta. s_tr, por sí sola, no basta: describe la cabeza
en train, pero no dice si su corrección se transfiere a val.

## Factor de confusión: sobreajuste

Si M11 sobreajusta en train, sus predicciones en train están más cerca de θ que las de val. Eso empuja s_tr hacia
valores ≤ 1, aunque exista la contracción de H1, y hace que HR, ajustado en un train que la red ya "recuerda", se
transfiera peor a val, lo que reduce R. El sobreajuste, por tanto, sesga el resultado hacia ENV_SHIFT o MIXED. Se marca
`OVERFIT_CONFOUND` si MSE_M11,train < 0,9·MSE_M11,val. Es un indicador aparte que no modifica la decisión, pero
condiciona su lectura.

## Criterios de decisión (por etiqueta, estimaciones puntuales)

Sea s_tr la pendiente global de θ sobre f de M11 en train.

- `TRAINING_INDUCED`: s_tr ≥ 1,03 y R ≥ 0,75.
- `ENV_SHIFT`: |s_tr − 1| < 0,01 y R ≤ 0,25.
- `MIXED`: cualquier otro caso.

R solo tiene sentido como fracción transferida si mean(d_C3) > 0. Si no se cumple, la decisión es `MIXED` y se registra
`c3_gain_positive = False`. Los intervalos bootstrap se publican junto a la decisión, pero no entran en ella.

## Qué no permite concluir

- No separa los mecanismos de H1 (dropout entre los modos de entrenamiento y evaluación, weight decay, parada temprana).
  Solo distingue «entrenamiento» de «entorno».
- No dice nada sobre la regresión a la media de fondo, la que tendría incluso E[θ|X]. Solo trata la contracción que
  sobra respecto a la cabeza MCO sobre h.
- No valida un M11-HR desplegable. Para eso harían falta la evaluación en test y recalibrar el Mondrian para el predictor
  modificado, y ambas cosas quedan fuera del alcance. Los cuantiles conformales de M11 no se reutilizan.

## Coste

- GPU: un pase de inferencia solo hacia delante sobre los 400 000 ejemplos de train, que el usuario ya ha lanzado con
  `predict_cnn_hdf5.py` y la config del punto 1.
- CPU: unos minutos en local. Los términos dominantes son la lstsq de 400 000 × 65 y el C3 con LOEO anidado sobre val.
  Memoria: del orden de 1–2 GB (h de train en float64).

## Regla de parada

Se hace **una sola ejecución** con estos ajustes y sin variantes: ni otras regularizaciones de HR, ni otros
subconjuntos de train, ni umbrales nuevos. Cualquier seguimiento requiere un protocolo nuevo.

## M11 no se modifica

El checkpoint, las configs, las predicciones y los resultados de M11 y de M12.5.3 quedan intactos. Los coeficientes de
HR se guardan solo en el directorio de salida de este experimento.

## Salida

El directorio exclusivo es `$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_4_head_refit/`. La ejecución aborta si ya existe.

- `summary.csv`: métricas de C y D, decisión y `overfit_confound` por etiqueta.
- `calib.csv`: A, para M11 en train y val, más las pendientes de HR en val.
- `regions.csv`: E.
- `head_refit_coefs.npz`: coeficientes de HR (65×3, unidades físicas; la fila 64 es el intercepto).
- `analysis.json`: SHA256 de los dos npz y del script, comprobaciones superadas, ajustes, α elegidos por C3 y decisiones.
