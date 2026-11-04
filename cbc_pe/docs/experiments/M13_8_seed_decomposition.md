# M13.8 — Descomposición del error de χ_eff entre semillas

**Estado:** pre-registrado el 4-11-2026, antes de calcular. Solo val G3 (40 000 filas). Sin GPU.

## Pregunta y límites
Con r_s = ŷ_s − y = c(x) + v_s(x), donde c es la parte común a las semillas y v_s la propia de cada semilla (media cero):
- MSE medio = E[m²] + E[d²], con m = (r_a + r_b)/2 y d = (r_a − r_b)/2;
- varianza entre semillas: 2·E[d²];
- MSE de un ensemble infinito ≈ E[r_a·r_b].

**Se puede concluir:** qué fracción del error de χ_eff en señales largas se elimina promediando semillas del procedimiento de 4a-aug; si la ventaja de 4b sobre 4a en D10 se alcanza promediando 4a; si el residuo es común a varias arquitecturas; cómo escala con el SNR el error común.

**No se puede concluir:** que el error común sea el límite de información. Las semillas comparten datos de entrenamiento, procedimiento y prior; además, la contracción hacia el prior forma parte del estimador de Bayes. E[r_a·r_b] es una cota superior del error de Bayes.

Otros límites:
- Con 2 semillas, el bootstrap por filas no incluye la variabilidad por elección de semillas.
- El sesgo propio de cada semilla por decil es una sola realización.
- Una correlación positiva entre v_a y v_b haría que φ infraestimara la varianza para ensembles más diversos.

## Criterio primario (χ_eff, D10, par 4a-aug s123/s124)
φ = 2·E[d²] / E[(r_a² + r_b²)/2]; G∞ = 1 − √(1 − φ). IC 95 % por bootstrap emparejado sobre filas (B = 2000, semilla 13808).
- **Varianza relevante:** límite inferior del IC de G∞ ≥ 5 %. Prioridad: estabilizar los transformers (semillas, EMA o SWA) antes de M14.
- **Varianza secundaria:** límite superior del IC de G∞ < 5 %. El error es común; seguir con M13.10 y M14.
- **No concluyente:** resto. No se entrenan semillas adicionales; por defecto, M13.10 y M14.

Justificación del 5 %: es del orden de la ventaja de 4b en D10 (6,3 %) y del ruido de semilla de 4a en D10 (4,7 %).

## Secundarios (descriptivos; lectura fijada de antemano)
- **S0:** φ(D10) − φ(D1–D3). Si el IC excluye 0, la inestabilidad es específica de las señales largas.
- **S1:** MSE_4b − E[r_a·r_b]_4a en D10. Si el IC queda por debajo de 0, la ventaja de 4b no es alcanzable promediando 4a (es de sesgo o estructura).
- **S2:** si la correlación de residuos entre arquitecturas es parecida a la de dentro de 4a, el residuo es común a los sesgos inductivos probados. Es un indicio de límite de datos o de información, no una prueba.
- **S3:** pendiente de log(error común) frente a log(SNR) por terciles. Una pendiente ≈ −2 indica error dominado por el ruido (no distingue óptimo de subóptimo); ≈ 0, error sistemático.
- **S4:** el par 6a como contraste.
- **S5:** chirp_mass y total_mass.

## Regla de parada
Una ejecución. Sin GPU ni semillas nuevas. Después viene M13.9, sea cual sea el resultado.
