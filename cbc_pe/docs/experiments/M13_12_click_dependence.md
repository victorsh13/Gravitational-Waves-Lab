# M13.12 — Dependencia de la red del escalón inicial (clic)

**Estado:** cerrado (5-11-2026): dependencia relevante.

## Contexto (verificado en M13.11)
Las señales de G1/G3 (SEOBNRv4_opt con f_lower = 30 Hz, sin taper) empiezan con un escalón. Tras el procesado es un transitorio de banda ancha al inicio de la señal, con SNR de red de ~2–7 (mayor en masas altas). Su posición respecto al merger depende de los parámetros, y en datos reales no existe.

## Pregunta
Si se elimina el clic y se deja todo lo demás idéntico (fuente, ruido, PSD, procesado), ¿cambian las predicciones y el RMSE de los modelos entrenados?

## Diseño
- 4000 filas de val, 400 por decil de τ30 (default_rng(131201)). Reconstrucción exacta.
- X_orig (igual a G3, gate G-REC) y X_noclick: la señal se sustituye por S, SEOBNRv4_opt desde 20 Hz con taper hasta 28 Hz, alineada con A en fase y tiempo (gates G-ALIGN y G-DIFF).
- Modelos: 4b-aug s123, 4a-aug s123 y 4a-aug s124. Gate: las predicciones sobre X_orig reproducen las guardadas en val, |Δ| ≤ 1e-3 en unidades estandarizadas (fp16).

## Medidas
Por grupo de τ (D1–D3, D4–D7, D8–D10), global y por etiqueta:
- ΔRMSE % (sin clic frente a con clic), con IC por bootstrap emparejado de filas (B = 2000, semilla 131202);
- desplazamiento medio de las predicciones (sin clic − con clic), relativo al RMSE;
- RMS del cambio de predicción, relativo al RMSE.
Descriptivo: relación del cambio de predicción con el SNR del clic.

## Criterio
- **Dependencia relevante:** en algún grupo y etiqueta, el límite inferior del IC de ΔRMSE es > +3 % o |desplazamiento medio| > 0,1 × RMSE (en cualquiera de los modelos). Consecuencia: la red usa el artefacto. La generación futura (G4) debe llevar taper antes de M14, y la lectura de los eventos reales debe revisarse.
- **Inocuo:** todos los IC de ΔRMSE dentro de ±3 % y todos los desplazamientos < 0,1 × RMSE. Se documenta; G4 lleva taper igualmente.
- **No concluyente:** el resto.

## No se puede concluir
- Cuánto se ganaría o perdería reentrenando sin clic.
- Nada sobre otras diferencias entre simulación y datos reales.
- Que una mejora sin clic implique mejor rendimiento en datos reales.

## Regla de parada
Una ejecución. Sin reentrenamientos. Si falla algún gate, se para.

## Enmienda 1 (4-11-2026, antes de ver ninguna predicción)
G-ALIGN: la comparación de SNR del clic (alineado frente a fase maximizada) usa la tolerancia max(0,2 · SNR_phasemax; 0,3) en lugar de 0,2 relativo. Motivo: la ejecución abortó en la fila 405978 (D10, SNR del clic 0,16; residuo de fase ~1e-17; G-DIFF ~5e-13). Con clics casi nulos (filas truncadas, cuyo escalón cae fuera de la ventana final) la tolerancia relativa no es aplicable. La condición de fase no cambia.

## Resultados (5-11-2026)
Notebook `notebooks/m13/M13_12_click_dependence.ipynb`; resumen `<data_root>/results/M13_3/M13_12/M13_12_summary_20261105T114640Z.json`.
- Gates: build PASS (4000 filas; G-REC 0; G-DIFF ≤ 2,5e-12); reproducción de predicciones sobre X_orig |Δ| ≤ 2,5e-6 (estandarizado).
- **Veredicto: dependencia relevante** (30/36 combinaciones por ΔRMSE; 22/36 por desplazamiento).
- ΔRMSE % al quitar el clic (rango de 4b-aug s123, 4a-aug s123 y s124): global +28–30 (chirp), +16 (total), +11–13 (χ_eff); D1–D3 +38–44 / +24–28 / +28–30; D8–D10 +22–30 / +6–9 / +1–2.
- Desplazamiento medio relativo: D1–D3 −0,38 a −0,48 en las tres etiquetas (sin clic, las predicciones bajan); D8–D10, chirp mass +0,13 a +0,18.
- El cambio de predicción crece de forma monótona con el SNR del clic.
- Lectura: las redes entrenadas en G1/G3 usan el escalón inicial, que en la simulación indica el instante en que la señal pasa por 30 Hz y no existe en datos reales. Los RMSE de simulación de M10–M13 están inflados, sobre todo en masa alta.
- Hipótesis (sin test, n = 9): explica parte del sesgo a la baja en eventos reales de M13.7 (GW191230; chirp mass negativa en 7/9).
- No se puede concluir: el rendimiento de una red reentrenada sin escalón (→ M13.13).
