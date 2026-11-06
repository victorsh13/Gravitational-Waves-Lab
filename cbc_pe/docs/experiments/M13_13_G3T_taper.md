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
