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
