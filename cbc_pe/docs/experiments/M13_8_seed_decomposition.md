# M13.8 — Descomposición del error de χ_eff entre semillas

**Estado:** cerrado (4-11-2026). Pre-registrado antes de calcular; resultados en §Resultados. La sección exploratoria es post hoc.

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

## Resultados (4-11-2026)

Notebook: `notebooks/m13/M13_8_seed_decomposition.ipynb`. Resúmenes: `<data_root>/results/M13_3/M13_8/M13_8_summary_20261104T135215Z.json` (pre-registrado) y `M13_8_exploratory_20261104T142718Z.json` (post hoc); figuras en `M13_8/figures/`. GATE superado (60 comprobaciones; RMSE por decil idéntico al de M13_6).

### Verificado (pre-registrado; val G3, 40 000 filas)
- **Primario:** χ_eff, D10, par 4a-aug s123/s124: φ = 11,6 %, G∞ = 6,0 % [5,5; 6,5] → **varianza relevante**. Por poco: un solo par de semillas, y el IC es de muestreo de filas.
- Del MSE de un modelo es común a las dos semillas el 88 % en D10 y el 95 % en D1–D3. Ensemble de 2: −3,0 % de RMSE en D10; de 5: −4,8 % (estimado).
- La media de d en D10 aporta el 0,3 % de E[d²]: la varianza es fila a fila, no un desplazamiento propio de cada semilla.
- **S0:** φ(D10) − φ(D1–D3) = +7,1 puntos [6,2; 8,1] (4a); +3,0 [2,5; 3,7] (6a). La inestabilidad es específica de las señales largas y mayor en el transformer.
- **S1:** MSE de 4b − error común de 4a en D10 = −0,0003 [−0,0018; +0,0013]: 4b (una semilla) iguala al ensemble ∞ estimado de 4a. En D1–D3, 4b queda un 5,9 % por encima de ese error común.
- **S2:** correlación de residuos de χ_eff, dentro de arquitectura 0,96 / 0,95 / 0,90 y entre arquitecturas 0,95 / 0,94 / 0,85 (D1 / D5 / D10).
- **S3:** pendiente de log(error común) frente a log(SNR): −2,10 (D10), −1,97 (D1–D3); φ casi constante con el SNR. Error común y discrepancia entre semillas dominados por el ruido. No distingue un estimador óptimo de uno subóptimo.
- **S5:** G∞ en D10: 4,6 % (chirp) y 4,9 % (total).

### Exploratorio (post hoc, no pre-registrado, solo descriptivo)
- **E1, ventana de 4 s:** filas truncadas 0 % en D1–D9 y 0,95 % (38/4000) en D10. Pierden ~11 % de la duración (la banda 30–31 Hz) y pesan < 1 % del MSE de D10. **Hipótesis de la ventana descartada.**
- El error de χ_eff crece de forma continua con τ30, desde ~0,6 s (RMSE de 4a: 0,15 → 0,26 en el bin de τ 1,7–5,9 s). El aparente salto de D10 se debe a la anchura del decil (τ 0,56–5,9 s).
- En D10, la duración real / τ30 correlaciona 0,90 (Spearman) con χ_eff verdadera.
- **E2, ensembles de modelos existentes:** en D10, RMSE de χ_eff ≈ 0,195 para 4b solo, para el ensemble ∞ estimado de 4a y para los ensembles de arquitecturas (4a×2+4b, 4a×2+4b+4c, 4b+4c). Ninguna combinación lo mejora.
- Global, 4a×2+4b+4c frente a la media de 4a: −2,6 / −3,1 / −3,4 % (chirp / total / χ_eff); frente a 4b solo: −2,1 / −2,4 / −1,7 %. Incluir la CNN 6a empeora el ensemble.

### Lectura y consecuencias
- La ganancia de estabilidad que señala el veredicto ya está disponible sin entrenar (4b, o el ensemble). EMA/SWA pasa a ser opcional y de baja prioridad. **Esto es una reinterpretación post hoc, no un cambio del criterio.**
- Ensemble de arquitecturas: candidato; se decidirá al congelar el pipeline (mejora pequeña frente a una sola arquitectura).
- Hipótesis abiertas sobre el suelo común de χ_eff en señales largas: (A) límite de información (degeneración q–χ_eff con SNR 10–25 y el prior usado); (B) estimador que no acumula bien la fase. No se distinguen sin referencia externa. Siguiente paso: estudio de viabilidad de una referencia por PE bayesiana; después, M13.9 y M13.10.
- Limitación: dos semillas solo en 4a y 6a; 4b y 4c, una.
