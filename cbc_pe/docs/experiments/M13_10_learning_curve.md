# M13.10 — Curva de aprendizaje en número de fuentes (G3T)
**Estado:** pre-registrado el 6-11-2026.
**Pregunta:** ¿el error está limitado por el número de fuentes de entrenamiento?
**Diseño:** 4a-aug, semilla 123, G3T; N = 100k y 200k (subconjuntos anidados, seed 131001) y 400k (= 4a-G3T s123). Mismo número total de pasos (124 960) en todos, para que el único factor sea la variedad de fuentes (la aumentación da ruido nuevo en cada paso). Evaluación en val G3T completa.
**Medida primaria:** G = RMSE(200k)/RMSE(400k) − 1 por etiqueta, global y en D8–D10 de τ30, con IC por bootstrap emparejado de filas (B = 2000, default_rng(131002)).
**Criterio** (ruido de semilla de referencia de 4a ≈ 2 %, M13.6): limitado por datos si G ≥ 5 %; no limitado si G < 2 %; no concluyente en el resto. El punto de 100k es descriptivo (forma de la curva).
Combinación: cada uno de los 6 valores de G (3 etiquetas × global, D8–D10) se clasifica con los umbrales anteriores. Veredicto global: limitado por datos si al menos un G global o el G de χ_eff en D8–D10 es ≥ 5 %; no limitado si los 6 son < 2 %; no concluyente en el resto. Se informa siempre de la tabla completa. Notas: con 100k el apagado del dropout cae 262 pasos más tarde (0,21 %); con subconjuntos, P-AUG1 se separa a posteriori en filas entrenadas y no entrenadas del monitor.
**No se puede concluir:** si el límite restante es del modelo o de la información (→ referencia bayesiana); variación entre semillas por punto (una semilla).
**Regla de parada:** dos ejecuciones nuevas; sin semillas adicionales.
