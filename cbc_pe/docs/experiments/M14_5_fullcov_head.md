# M14.5 — Cabeza heterocedástica con covarianza completa
**Estado:** pre-registrado el 11-11-2026.
**Motivo (M14.4):** la correlación de los residuos normalizados cambia fuertemente con la τ30 predicha (𝓜–M_tot de 0,985 a 0,25); una R por decil reduce el volumen de los elipsoides conformales un 34,5 % con cobertura conjunta válida y casi plana por grupos.
**Pregunta:** ¿una Σ(x) completa aprendida por la red da regiones conjuntas válidas y menores que D R_g D (P de M14.4), sin empeorar μ?

## Diseño
- **M14.5a:** 4a-G3T s123 con covarianza completa; un factor frente a M14.1 (σ diagonal).
- **M14.5b:** 4a-G3T s123 + banco de plantillas de M14.2 + covarianza completa; un factor frente a M14.2. Solo se lanza si M14.2 cumple su criterio primario (enmienda 2 de M14.2).
- **Cabeza:** μ (3) y L(x) triangular inferior (6). Diagonal L_ii = exp(log L_ii) con log L_ii limitado a [−7, 3]; fuera de la diagonal, lineal sin restricción; Σ = L Lᵀ en unidades estandarizadas.
- **Pérdida:** e = L⁻¹ r por sustitución hacia adelante (r = y − μ). NLL = Σ_i [e_i²/2 + log L_ii]. β-NLL por término, con peso sg(L_ii^(2β)), β = 0,5; con L diagonal coincide con la pérdida de M14.1. Orden de etiquetas [chirp_mass, total_mass, chi_eff].
- Procedimiento de entrenamiento idéntico al de la referencia (80 épocas, mismo protocolo).

## Criterios (val G3T; radios ajustados en cal G3T)
1. **Salvaguarda de μ:** ΔRMSE% por etiqueta frente a la referencia (M14.1 para a; M14.2 para b), bootstrap emparejado B = 2000, default_rng(145001). PASA si IC sup ≤ +4 %; FALLA si el punto > +4 %; INDETERMINADO en otro caso.
2. **Calibración conjunta (con veredicto, no decide):** s = rᵀ Σ⁻¹ r frente a χ²₃. mean(s)/3 ∈ [0,9; 1,1] y fracción con s ≤ 6,251 (cuantil 0,90 de χ²₃) en [0,88; 0,92], en 7 celdas: global, D1–D3 / D4–D7 / D8–D10 de τ30 predicha (cortes en cal) y terciles de σ_chirp (cortes en cal).
3. **Primario:** elipsoide conformal 3D con la Σ aprendida (radio = cuantil k-ésimo de s en cal, k = ⌈(n + 1)·0,9⌉) frente a P de M14.4 (mismas filas de val). Validez: cobertura conjunta global en la banda binomial al 95 % de 0,90. Δ = 100·(exp(mean log V_Σ − mean log V_P) − 1), IC por bootstrap emparejado (B = 2000, default_rng(145002)).
   - **Adopción:** Σ aprendida si C1 no FALLA, es válida y el IC superior de Δ < 0. Si no, se usa P (D R_g D) con la D del nuevo modelo.

## Descriptivos
- Cobertura conjunta por grupos (D1–D3 / D4–D7 / D8–D10; terciles de SNR nominal).
- 2D (𝓜, M_tot).
- Correlaciones implícitas de Σ(x) por decil de τ30 predicha frente a las empíricas de M14.4.
- Mondrian v2 (M14.3) con σ_i = √Σ_ii.

## No se puede concluir
- Una semilla; sin datos reales.
- b frente a a mezcla dos factores (banco y cabeza): solo descriptivo.

## Regla de parada
Un entrenamiento por variante; sin variantes de β ni de parametrización.

## Enmienda 1 (12-11-2026, antes de ver resultados de M14.5a y M14.5b)
- **M14.5b se lanza por decisión del usuario** aunque M14.2 no cumplió su criterio primario (enmienda 2 de M14.2: sin mejora resuelta frente a 2 × ruido de semilla). Motivo: mejora del banco en las 13 celdas de M14.2, creciente en señales largas (χ_eff D7–D10 −3,6 a −5,6 %; chirp D10 −10 %), coherente con la hipótesis de acumulación de fase. El veredicto de M14.2 no cambia.
- **Elección del modelo final entre M14.5a y M14.5b (val G3T), fijada ahora:** se elige M14.5b si cumple C1 frente a M14.2, es válido en C3, y en val tiene (i) RMSE de μ en D8–D10 menor o igual que M14.5a en chirp y en χ_eff (punto) y (ii) volumen medio geométrico del elipsoide conformal 3D menor o igual que M14.5a (punto). En otro caso, M14.5a.
- **Limitación declarada:** es una elección por estimación puntual. Con una semilla por modelo, una diferencia del orden del ruido de semilla (3,5–4,7 % en D8–D10) no está resuelta. Se acepta porque los dos modelos son válidos, tienen latencia de inferencia similar (ms) y el coste extra del banco es solo de entrenamiento.

## Enmienda 2 (12-11-2026, tras ver val; antes de mirar test)
- **Motivo:** en val, la Σ aprendida (S) da Δ volumen 3D −28,1 % [−28,4; −27,9] frente a P, con C1 PASA y C2 calibrada en 7/7, pero C3 la declara no válida por cobertura 0,9030 frente a un límite superior de 0,9029: una banda binomial bilateral sin el ruido de cal (defecto ya señalado en M14.4 y no corregido al escribir este pre-registro). La validez conformal solo exige no cubrir de menos.
- **Criterio corregido:** banda Monte Carlo al 95 % con el ruido de cal (procedimiento de la enmienda 6: c ~ Beta(k, n_cal − k + 1), aciertos ~ Binomial(n_eval, c)); válido si la cobertura no está por debajo del límite inferior.
- **Confirmación en test (30 000 filas; M14.5a en test no se ha mirado):** se adopta S si en test (i) la cobertura 3D de S no está por debajo del límite inferior de la banda corregida y (ii) el IC superior de Δ volumen 3D de S frente a P (P con M14.1, como en val) es < 0. En otro caso, P con la D del nuevo modelo.
- **Para M14.5b y para la regla de la enmienda 1:** se aplica el mismo criterio corregido de validez.
- Decisión post hoc, documentada como tal; el veredicto pre-registrado (P) queda registrado.