# M13.9 — Protocolo corto (20 épocas) en G3T
**Estado:** pre-registrado el 6-11-2026.
**Pregunta:** ¿un entrenamiento de 20 épocas con su propio programa comprimido (warmup, coseno y dropout apagado el último 20 %) da un RMSE cercano al de 80 épocas y suficientemente estable para comparar variantes en M14?
**Diseño:** 4a-aug corto, semillas 123 y 124, G3T; referencia: 4a-G3T s123 (80 épocas). Evaluación en val G3T.
**Criterio:** se adopta si la penalización RMSE_corto (media de semillas)/RMSE_largo − 1 ≤ 5 % en las tres etiquetas y el ruido de semilla del corto (|Δ| entre semillas/media) ≤ 3 %.
**No se puede concluir:** si el corto conserva el orden entre arquitecturas distintas (una sola arquitectura).
**Regla de parada:** dos ejecuciones; sin repeticiones.
