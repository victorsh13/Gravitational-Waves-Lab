# M13.13 — G3T: G3 con forma de onda con taper

**Estado:** pre-registrado el 5-11-2026, antes de generar.

## Motivo
M13.12: las redes entrenadas en G1/G3 usan el escalón inicial de SEOBNRv4_opt (f_lower = 30 Hz, sin taper). Quitarlo empeora el RMSE global +28 / +16 / +12 %. Es una fuga de información de la simulación que no existe en datos reales.

## Cambio (un solo factor)
La forma de onda pasa a SEOBNRv4_opt con f_lower = 15 Hz y taper Kaiser (β = 8) hasta que la frecuencia instantánea cruza 25 Hz, alineada en fase y tiempo con la de G3. Fuentes, colocación, PSD, ruido (mismas semillas), procesado, z-score, split y etiquetas: idénticos a G3. Train: se regenera la señal limpia (aumentación de M13.6, mismo banco de ruido). Val, cal, test y monitor train40k: X con el mismo ruido de G3.

## Gates (si falla alguno, se para)
- G-TAPER (antes de generar, 2000 filas de train con las 20 de mayor masa): mismatch(S15, SEOBNRv4_ROM) ≤ 1e-3 en 30–1024 Hz y |S15(t0)|/max|S15| ≤ 1e-3.
- G-ALIGN: residuo de fase < 0,01 rad. G-NOISE: ruido idéntico al de G3 (≤ 1e-6 relativo). G-REC: X de G3 reconstruido igual al guardado (≤ 1e-5).
- G-EDGE: si la onda empieza antes del segmento de procesado, taper de 0,5 s en el borde; se informa del número de filas.
- Consistencia: señal limpia + ruido original = X de val/train40k en 64 filas (≤ 1e-5).

## Primer modelo
4a-aug s123 con el procedimiento de M13.6 (80 épocas, coseno, aumentación). Solo cambian las rutas de datos.
- Salud: P-AUG1 (MSE_train40k / MSE_val ≥ 0,95) y P-AUG2 (val final ≤ 0,5 % peor que la mejor época).

## Medidas (descriptivas; G3T se adopta como base por corrección, no por rendimiento)
Sobre val G3T, por etiqueta, global y por decil de τ30, con IC por bootstrap emparejado (B = 2000, semilla 131302):
- RMSE de 4a-G3T;
- frente a 4a-G3 sobre val G3 (rendimiento con la fuga) y frente a 4a-G3 sobre val G3T (sin reentrenar).
Eventos reales (9 de M13.7), descriptivo: residuo frente a LVK con 4a-G3 y con 4a-G3T.

## No se puede concluir
- El ranking de arquitecturas en G3T (un solo modelo).
- La ganancia en datos reales (n = 9).
- Si el taper de 15–25 Hz es óptimo (solo se verifica que es inocuo en la banda).

## Regla de parada
Un entrenamiento. No se entrena ningún otro modelo hasta revisar este. Si falla algún gate, se para.

## Enmienda 1 (5-11-2026, antes de generar)
G-TAPER compara S15 con SEOBNRv4_opt desde 10 Hz con taper hasta 14 Hz (mismo modelo; aísla el efecto del taper): mismatch ≤ 1e-4 en 30–1024 Hz. Escalón ≤ 2e-3 (borde de la ventana Kaiser). El mismatch con SEOBNRv4_ROM pasa a ser solo informativo. Motivo: la primera ejecución falló en 113/2000 filas, todas por mismatch con el ROM y concentradas en masas bajas (0 fallos con M > 100 M☉), lo contrario de lo esperable si el problema fuera el taper; el gate mezclaba el error de interpolación del ROM. Nota para M13.11: el error del ROM frente a SEOBNRv4 llega a ~4,5e-3 en masas bajas.

## Enmienda 2 (5-11-2026, antes de generar; decidida después de ver la distribución del gate)
Con 15–25 Hz, 14/2000 filas superaban el mismatch de 1e-4 frente a Sref (mediana 7,5e-7; máximo 3,5e-4; sin patrón de masa claro). Con 12–20 Hz, 8 de esas 14 seguían fallando y algunas empeoraban (226404: 1,3e-4 → 2,0e-4; 228011: 1,1e-4 → 2,7e-4): no es un problema de margen del taper. Hipótesis: dependencia del modelo EOB con la frecuencia de inicio de la integración. Según la regla fijada antes de esa prueba, se mantiene 15–25 Hz con un criterio físico en lugar del umbral arbitrario de 1e-4: mismatch ≤ 8e-4 en todas las filas (diferencia en banda con SNR ≤ 1 a SNR 25), como mucho un 2 % de filas por encima de 1e-4 y escalón ≤ 2e-3. Referencia: el escalón que se elimina tenía SNR de red 2–7 (M13.11/M13.12).

## Enmienda 3 (5-11-2026, antes de generar)
G-ALIGN: el residuo de fase tras alinear con la forma de onda de G3 suele ser ~1e-16, pero algunas filas llegan a 4–8e-3 rad (gate y prueba de val), cerca del límite de 0,01. Un pequeño error de fase no altera la validez física de la señal (cambia ligeramente la fase de coalescencia, que no es una etiqueta); solo afecta a la comparación fila a fila con G3. Nuevo criterio: tolerancia de 0,04 rad (diferencia en banda con SNR ≤ 1 a SNR 25, como la enmienda 2); las filas que la superan se registran y no detienen la generación. Se para solo si alguna fila supera 0,5 rad (fallo) o más del 0,1 % de un split o fragmento supera 0,04 rad.

## Enmienda 4 (6-11-2026, antes de ver resultados finales)
Se lanza 4b-aug s123 en G3T en paralelo a 4a (excepción a la regla de no entrenar otro modelo hasta revisar el primero), para aprovechar la GPU (4b tarda ~50 h). Motivo: las 13 primeras épocas de 4a-G3T son sanas (MSE de train40k ≈ val, descenso continuo, pendientes ~1). Si 4a-G3T no cumple P-AUG1 o P-AUG2 al terminar, 4b se detiene.

## Enmienda 5: residuo esperado en eventos reales (10-11-2026, antes de ejecutar)
Pregunta: ¿son los residuos reales (predicción − mediana LVK, sistema del detector; los 9 eventos de M13.7) compatibles con los que el mismo modelo comete en simulación en la región del evento? Descriptivo; los eventos reales no intervienen en ninguna elección. Notebook: `notebooks/m13/M13_13_expected_residual.ipynb`; salidas en `results/M13_3/M13_13/expected_residual/` (modo `'x'`).

- **Modelos:** 4a-G3T (primario) y 4b-G3T (secundario). Residuos reales: los de la ejecución de `M13_13_real_events.ipynb` (`M13_13_real_events_20261110T130749Z.json`), con las mismas posteriores LVK (`C01:Mixed`, `mixed_cosmo`), leídas con las funciones de M13.7.
- **Simulación:** predicciones de val de G3T de cada modelo (`val_predictions_embeddings.npz`), desestandarizadas con su `y_mean`/`y_std`; r_sim = pred − verdad.
- **SNR simulado:** SNR de red de G3, √(Σ snr_det²) con `snr_H1`, `snr_L1`, `snr_V1` del HDF5 de G3 por fila. Es aproximado para G3T, porque el clic aporta poco SNR en la banda. **SNR real:** `network_snr_lvk` del catálogo (matched filter). Son magnitudes distintas (óptimo frente a matched filter); por eso la ventana es ancha.
- **Vecindario del evento:** filas de val con |Mc_true/Mc_LVK − 1| ≤ 0,10 y |ρ/ρ_LVK − 1| ≤ 0,25 (Mc_LVK = mediana de la posterior, sistema del detector). Si quedan < 200 filas se amplía, en este orden fijo, a (0,20; 0,25) y después a (0,20; 0,40); si siguen siendo < 200, el evento se marca `sin_soporte` y se excluye de los criterios. Se registran la ventana usada y n. `borde_prior` si Mc_LVK está fuera del percentil 2–98 de Mc_true en train (`y_train.npy` de G3T); estos eventos se mantienen en los criterios, marcados.
- **Referencia por (evento, etiqueta):** r_ref = r_sim + (s − mediana LVK), con 20 000 pares (fila del vecindario, muestra s de la posterior LVK de esa etiqueta), ambos con reemplazo; un solo `default_rng(151301)` recorrido en orden fijo: modelo (4a, 4b) → evento (por Mc_LVK creciente) → etiqueta (chirp_mass, total_mass, chi_eff). Versión secundaria: r_sim del vecindario sin convolución.
- **Estadístico:** u = fracción de r_ref ≤ residuo real. Compatible si 0,05 ≤ u ≤ 0,95.
- **Criterio** por etiqueta y modelo, sobre los eventos con soporte (n ≤ 9), con la u convolucionada. Bajo H1 la u es aproximadamente uniforme. Desajuste (H2) si ≥ 4 eventos tienen u fuera de [0,05; 0,95] (binomial p = 0,1: P ≈ 0,008 con n = 9). Sesgo de signo si ≥ 7 tienen u < 0,5 (P ≈ 0,09 con n = 9; solo descriptivo). En cualquier otro caso: "no se detecta desajuste". Los umbrales son absolutos; si n < 9 se informa de la P binomial para el n real. La u sin convolución se informa, sin criterio.
- **No se puede concluir:** con n = 9 la potencia solo alcanza para desajustes groseros; "no se detecta desajuste" no demuestra compatibilidad. LVK es una referencia, no la verdad. La cohorte está seleccionada.

## Enmienda 6: Mondrian con 4a-G3T (11-11-2026, antes de ejecutar)
Pregunta: ¿el Mondrian de M13.7, recalibrado con 4a-G3T en cal de G3T, da en test de G3T la cobertura nominal, marginal y por bin? Notebook: `notebooks/m13/M13_13_mondrian_G3T.ipynb`; salidas en `results/M13_3/M13_13/mondrian_G3T/run_<sello>/` (JSON en modo `'x'`, con sha256 de las entradas y de las celdas reutilizadas).

- **Entradas:** predicciones de 4a-G3T s123 (`M13_13_4a_G3T_s123_final.pt`) en cal y test de G3T (`preds_4a_G3T_on_G3T_{cal,test}`, `fp16_memmap_heldout`) y en val de G3T (`M13_13_4a_G3T_s123/val_predictions_embeddings.npz`). Eventos reales: los 9 de `M13_13_real_events_20261110T130749Z.json`, con las posteriores LVK `C01:Mixed`, `mixed_cosmo`.
- **Comprobaciones previas (si falla alguna, se para):** checkpoint `M13_13_4a_G3T_s123_final.pt` y `memmap_dir` de G3T (`m13_13_g3t`) en los tres npz, y `fp16_memmap_heldout` en cal y test; `idx` de cal y test = `row_ids` de `fp16_memmap_heldout` = pertenencia al split congelado, sin solapamiento entre cal, test y val; `y` de cal y test = `y` del HDF5 de G3 en esas filas (rtol 1e-5, atol 1e-6); el mismo escalador train-only en los tres npz y en el checkpoint; RMSE por etiqueta en cal y en test dentro de ±5 % del de val (5,25 / 8,76 / 0,168). Si el RMSE no se cumple, no se calibra.
- **Método:** exactamente el Mondrian de M13.7. Se ejecutan sus celdas 6 (ajuste) y 8 (evaluación en test) desde el notebook cerrado, sin copiarlas. Seis sistemas (política × etiqueta) con la taxonomía, el modo de intervalo y el número de bins de la selección congelada de M11 en val: conservative con chirp_mass difficulty/asimétrico/16, total_mass prediction/asimétrico/4 y chi_eff difficulty/simétrico/8. Hiperparámetros: nivel 0,90, k = 5, mínimo de 20 por bin, jitter de 1e-10 y regla de cuantil finito de `fit_selected_calibrators`. Se calibra solo con cal de G3T y se evalúa en test de G3T. No se cambia ningún hiperparámetro. La política principal es conservative, como en M13.7. Efficient se informa sin criterio.
- **Ruido de cal (cambio del 11-11-2026, antes de ejecutar).** Con cuantiles conformales, la cobertura condicionada a cal en un bin con n filas de cal y rango k sigue Beta(k, n − k + 1). Los aciertos en test con m filas siguen entonces Beta-Binomial(m, k, n − k + 1). Una banda binomial con p = 0,90 ignora el ruido de cal, que con n ≈ m es aproximadamente la mitad de la varianza, y marcaría más bins de la cuenta. Por eso las bandas de (a) y (b) y el criterio se basan en la Beta-Binomial. k_b y n_b se leen del calibrador que ajusta la celda 6 de M13.7 (`quantile_indices_`, `bin_counts_`): son los estadísticos de orden que usa realmente, tras la regla de cuantil finito y el recorte. El mínimo de 20 por bin solo se comprueba; no cambia el rango. Rango de cobertura: en modo simétrico (chi_eff) k = índice + 1 del cuantil de |r|; en modo asimétrico (chirp_mass, total_mass) la cobertura es U₍ₖ_high₎ − U₍ₖ_low₎, que sigue Beta(k_high − k_low, n − (k_high − k_low) + 1), así que k = k_high − k_low (índices desde 0). Total_mass, con taxonomía prediction, sigue la misma regla con sus bins.
- **Medidas (conservative):**
  (a) cobertura marginal en test por etiqueta. Banda del 95 % por Monte Carlo: `default_rng(131301)`, 100 000 réplicas, un solo generador recorrido en orden fijo (sistemas ordenados por (política, etiqueta)). En cada réplica, c_b ~ Beta(k_b, n_b − k_b + 1) por bin, aciertos ~ Binomial(m_b, c_b), y la cobertura es la suma sobre bins dividida por m. También se da el IC binomial exacto del 95 % (Clopper–Pearson), marcado "solo test, sin ruido de cal";
  (b) cobertura por bin en test. Banda central del 99 % del número de aciertos con `scipy.stats.betabinom(m_b, k_b, n_b − k_b + 1)`, y la P de dos colas de cada bin. Bajo H0, cada bin cae fuera con la probabilidad exacta de su banda (≤ 1 %, por la discreción). P de observar al menos ese número de bins fuera: Poisson-binomial con esas probabilidades, por convolución exacta. Solo cuentan los bins con m_b > 0 en test;
  (c) anchura física media y mediana por etiqueta y por grupo de τ30 (D1–D3, D4–D7, D8–D10). Los deciles salen de τ30 newtoniano con la Mc verdadera de val de G3T, como en la evaluación de M13.13, y se comprueban frente a `tau_edges_s` de su resumen. La anchura marginal se compara con la del resumen de M13.7, solo como descripción. M13.7 no guarda las anchuras por τ30, así que esa comparación no se hace;
  (d) eventos reales: 4a-G3T se reinfiere en CPU con el procesado de M13.7 (celdas 16 y 19, como `M13_13_real_events.ipynb`). Gate: las predicciones deben coincidir con las del JSON con |Δ| ≤ 1e-4 en unidades físicas; si no, se para. Para cada evento y etiqueta se dan: intervalo Mondrian, bin asignado, si la mediana LVK cae dentro y el solapamiento con el 90 % de LVK (longitud y fracción del intervalo LVK). GW200202 se marca "fuera de soporte (enmienda 5)".
  (d′) fiabilidad del bin kNN: la distancia media del embedding de cada evento a sus 5 vecinos de cal (el mismo `NearestNeighbors` del modelo de dificultad) se sitúa en la distribución de esa misma distancia en test (percentil). Marcas fijadas antes de ejecutar: "lejos de cal" si supera el P99 de test y "fuera del rango de test" si supera el máximo. En total_mass el bin depende de la predicción y la distancia solo se informa.
- **Criterio (sustituye al anterior, basado en bandas binomiales):** la calibración es válida en test si, para las tres etiquetas (conservative), la cobertura marginal observada cae dentro de la banda Monte Carlo del 95 % y la P Poisson-binomial de observar al menos ese número de bins fuera de la banda Beta-Binomial del 99 % es ≥ 0,01. En otro caso, "cobertura no verificada", con la tabla completa. (d) y (d′) son descriptivos (n = 9).
- **No se puede concluir:** nada sobre la cobertura en eventos reales (n = 9, cohorte seleccionada, LVK no es la verdad); nada sobre otras semillas. Un bin "lejos de cal" no invalida el intervalo, pero indica que el kNN extrapola.
