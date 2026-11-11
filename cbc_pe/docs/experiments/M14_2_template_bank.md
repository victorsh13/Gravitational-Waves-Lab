# M14.2 — Banco de filtros adaptados aprendibles (4a-G3T)
**Estado:** pre-registrado el 10-11-2026.
**Pregunta:** ¿una capa de filtrado adaptado (plantillas físicas aprendibles en cuadratura, correlación coherente en toda la ventana) mejora la estimación en señales largas, donde M13.8 encontró un suelo de χ_eff compatible con un estimador que no acumula bien la fase (hipótesis B)?
**Diseño:** 4a-aug s123 en G3T, 80 épocas, procedimiento idéntico; único cambio: banco de K plantillas en cuadratura antes del backbone, concatenado con la entrada. Referencia: 4a-G3T s123.
**Primario (val G3T):** ΔRMSE de chi_eff y chirp_mass en D8–D10 de τ30, con IC por bootstrap emparejado (B = 2000, default_rng(142002)). Mejora resuelta si −ΔRMSE > 2 × ruido de semilla de referencia en ese grupo (G3: chi_eff D10 4,7 %, global 1,6–2,0 %).
**Salvaguarda:** ΔRMSE ≤ +4 % en el resto de etiquetas y grupos.
**Lectura:** mejora resuelta → el estimador desaprovechaba la fase (B), dirección de M14. Sin mejora → no se descarta B (diseño concreto), pero apunta a límite de información; lo decide la referencia bayesiana (M13.11).
**No se puede concluir:** el diseño óptimo del banco; mejoras menores que el ruido de semilla (una semilla).
**Regla de parada:** un entrenamiento.

## Implementación

- Config: `configs/experiments/m14/train_m14_2_4aG3T_tbank_s123.json` (copia de `configs/experiments/m13/train_m13_13_4aG3T_s123.json` con solo `model.template_bank` y tag `M14_2_4a_G3T_tbank_s123`).
- Modelo: `TemplateBank` y `SimpleCNN_ResidualDilatedTransformerAgg(template_bank={...})` en `src/models/network.py`. Sin la opción el modelo es 4a sin cambios (mismas claves de `state_dict`).
  - K = 32 plantillas de longitud L = 16384 (ventana completa), `template_i` y `template_q` aprendibles e independientes, normalizadas a norma unidad en cada forward.
  - Correlación circular por FFT (`torch.fft.rfft/irfft`) de cada detector con cada plantilla (las mismas para los 3 detectores); la muestra final de la plantilla (merger) se coloca en el índice 0, de modo que la salida en t es el ajuste de los datos que terminan en t: una señal con merger en t da el pico en t, y cada detector conserva su propio retardo.
  - Salida |c| = sqrt(c_I² + c_Q² + 1e-12), (B, 3·K, T) con canales en orden (detector, plantilla); se concatena tras la entrada sin cambios y `block1` recibe 3 + 3K = 99 canales. Sin pooling ni proyección: la salida ya está a la resolución de la entrada y `block1` (stride 2) hace la primera reducción, como en 4a.
  - Memoria: los intermedios (B, 3, k, F) se calculan en bloques de 8 plantillas con gradient checkpointing (se recalculan en backward).
  - El resto de capas se construyen antes que el banco y el nuevo `block1`, así que reciben la misma inicialización que 4a con la misma semilla de torch.
- Inicialización (`init_template_bank` en `scripts/m13/train_m13_3.py`, solo filas de train): 8 bins por cuantiles de τ30 de train × 4 bins por cuantiles de χ_eff dentro de cada bin; una fila por celda con `default_rng(142001).choice`, en orden (τ30, χ_eff). De cada fila: detector con mayor `snr_<det>` (HDF5 trainio), `X_clean_train` (L(s) antes del z-score) de ese detector, z-score como la entrada, merger = argmax de la envolvente analítica; los L valores que terminan en el merger forman `template_i` (merger al final; ceros al principio si hace falta) y `template_q` = Hilbert(`template_i`). Las filas (row_id, posición, celda, detector, índice del merger, etiquetas, SNR) quedan en `run_info.json → procedure.template_bank`.
- Parámetros añadidos: 1 073 152 (banco 2 × 32 × 16384 = 1 048 576; `block1` +24 576). Total 1 574 675 frente a 501 523 en 4a.
- Coste estimado en P100 (batch 256): tensores guardados para backward 8,7 GiB frente a 6,2 GiB en 4a (medido en CPU con B = 2 y escalado), pico transitorio del banco ≈ 2–3 GiB; cabe en 16 GB. Tiempo estimado ≈ 0,55–0,7 s/paso frente a 0,345 s (≈ 15–19 min/época, ≈ 19–25 h para 80 épocas).
- Salida del banco: |c| (no |c|²) para que su escala sea comparable a la de la entrada (decidido antes de lanzar). Con K = 32 plantillas, el banco es mucho menor que uno de búsqueda; un resultado nulo es evidencia débil contra la hipótesis B.

## Enmienda 1 (antes de entrenar)

- **Motivo:** la inicialización elegía el detector de cada plantilla por `snr_<det>` del HDF5 trainio, pero para el dataset G3T (`m13_13_g3t`) solo existen memmaps; no hay HDF5 de G3T y el entrenamiento fallaba al arrancar.
- **Criterio nuevo:** detector de la plantilla k = argmax_d Σ_t X_clean_train[pos_k, d, t]² en float64 (L(s) blanqueada, antes del z-score). En `run_info.json → procedure.template_bank` se registra `clean_energy` (energía por detector) en lugar de `snr`, el índice `detector_index` y, solo si algún manifest trae la clave `detector_order`, el nombre del detector (los manifests actuales no la traen: solo se registra el índice).
- **Justificación:** el ruido blanqueado tiene varianza unidad por muestra, así que Σ x² de la señal limpia blanqueada es proporcional a ρ² óptimo en la ventana. Además mide la señal G3T que ve la red en la ventana, no la de G3 con clic que describía `snr_<det>`.
- Sin otros cambios: cuadrícula, semilla, selección de filas y extracción de la plantilla son las del pre-registro.

## Lanzamiento (GPU 2)

```
mkdir -p logs/m14 && CUDA_VISIBLE_DEVICES=2 nohup python scripts/m13/train_m13_3.py \
  --config configs/experiments/m14/train_m14_2_4aG3T_tbank_s123.json \
  > logs/m14/M14_2_4a_G3T_tbank_s123.log 2>&1 &
```

## Enmienda 2 (11-11-2026, antes de ver las predicciones de M14.2)
- **Motivo:** el criterio primario usa "2 × ruido de semilla de la referencia en ese grupo", pero no fija su valor en D8–D10.
- **Ruido de semilla en D8–D10:** por etiqueta, con el par 4a-aug G3 s123/s124 en val G3 y la definición de la sección 3b de M13_6_evaluation (aproximación: no hay dos semillas de 4a en G3T). Se calcula en el notebook antes de cargar M14.2 y se registra en el JSON.
- **Primario:** ΔRMSE% de chirp y χ_eff en D8–D10 frente a 4a-G3T s123 (bootstrap emparejado, B = 2000, default_rng(142002)). Mejora resuelta en una etiqueta si −ΔRMSE% > 2 × ruido de semilla de esa etiqueta en D8–D10.
- **Salvaguarda:** en cada celda no primaria (3 etiquetas × {global, D1–D3, D4–D7, D8–D10}), PASA si IC sup ≤ +4 %, FALLA si el punto > +4 %, INDETERMINADO en otro caso.
- **Veredicto:** mejora resuelta si se cumple en al menos una etiqueta primaria y ninguna celda de la salvaguarda FALLA.
- **Descriptivos:** ΔRMSE por decil D1–D10 y etiqueta; RMSE de χ_eff en D10 frente al suelo de 4a-G3T (0,207); solape normalizado entre plantillas iniciales y finales (si torch está disponible en local).
