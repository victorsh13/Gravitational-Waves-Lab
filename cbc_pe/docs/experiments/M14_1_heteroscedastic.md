# M14.1 — Cabeza heterocedástica (4a-G3T)
**Estado:** pre-registrado el 10-11-2026.
**Pregunta:** ¿puede la red predecir, además del valor, una incertidumbre σ por fila bien calibrada y adaptada (sobre todo al SNR), sin empeorar la media? Objetivo: score |r|/σ para el Mondrian v2.
**Diseño:** 4a-aug s123 en G3T, 80 épocas, procedimiento idéntico; solo cambian la cabeza (media y log σ) y la pérdida (β-NLL, β = 0,5). Referencia: 4a-G3T s123.
**Criterios (val G3T):**
1. Salvaguarda: ΔRMSE de μ ≤ +4 % por etiqueta frente a 4a-G3T s123 (IC por bootstrap emparejado de filas, B = 2000, default_rng(141001)).
2. Calibración: std(z), con z = (y − μ)/σ, en [0,9; 1,1] global, por tercil de SNR y por grupo de τ30 (D1–D3, D4–D7, D8–D10), por etiqueta.
3. Descriptivo: σ_media/RMSE por tercil de SNR (dentro de ±20 % = adaptado); Spearman(σ, |r|).
**Lectura:** se adopta para el Mondrian v2 si se cumplen 1 y 2.
**No se puede concluir:** el beneficio en anchura de los intervalos del Mondrian (requiere cal/test; siguiente paso). Una semilla.
**Regla de parada:** un entrenamiento.

## Implementación

- Config: `configs/experiments/m14/train_m14_1_4aG3T_hetero_s123.json` (copia de `configs/experiments/m13/train_m13_13_4aG3T_s123.json` con `model.heteroscedastic = true`, `training.loss = "beta_nll"`, `training.beta_nll = 0.5`, tag `M14_1_4a_G3T_hetero_s123`).
- Modelo: `SimpleCNN_ResidualDilatedTransformerAgg(heteroscedastic=True)` en `src/models/network.py`; la capa final da [μ, log σ] (3 + 3, unidades estandarizadas), log σ con clamp [−7, 3].
- Entrenador: `scripts/m13/train_m13_3.py` (`beta_nll_loss`); mse y pendiente registradas sobre μ.
- Predicciones: `pred_<split>` = μ; se añade `sigma_<split>` (estandarizada) en `val_predictions_embeddings.npz`, `train40k_predictions.npz` y en las salidas de `scripts/predict_cnn_hdf5.py`.

## Resultado (11-11-2026)
- **C1 (salvaguarda) PASA:** ΔRMSE de μ +0,37 % [−0,03; +0,82] / +0,44 % [−0,04; +0,94] / +0,06 % [−0,35; +0,48].
- **C2 (calibración) FALLA, 20/21 celdas:** chirp D1–D3 std(z) = 0,832. Global y por terciles de SNR, std(z) 0,977–1,011.
- **Veredicto pre-registrado:** σ no se adopta para el Mondrian v2.
- **Lectura post hoc (hipótesis):** los grupos de τ30 se definen con los parámetros verdaderos; ahí aparece la regresión a la media de μ (mean z = +0,40 en chirp D1–D3), que σ(X) no puede representar. En particiones que se calculan desde X (bins de σ, terciles de SNR), σ está bien calibrada: RMS(σ)/RMSE 0,99–1,01. Error de diseño del criterio 2: faltó un grupo equivalente definido por la predicción.
- **Consecuencia:** M14.3, experimento nuevo con su propio pre-registro.
- Notebook: `notebooks/m14/M14_1_evaluation.ipynb`; resumen: `results/M13_3/M14_1_4a_G3T_hetero_s123/M14_1_evaluation_summary_20261111T114355Z.json`.
