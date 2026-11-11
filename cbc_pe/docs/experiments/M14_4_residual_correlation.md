# M14.4 — Estructura de correlación de los residuos (M14.1)
**Estado:** pre-registrado el 11-11-2026.
**Pregunta:** ¿cambia con X la correlación entre etiquetas de los residuos normalizados z = (y − μ)/σ lo bastante como para que una covarianza por evento Σ(x) estreche las regiones conjuntas frente a Σ(x) = D(x) R D(x) con R global (D = diag(σ(x)))?
**Decide:** si se entrena una cabeza heterocedástica con covarianza completa (VM, después de la CPAN).

## Diseño (cal G3T para ajustar, val G3T para evaluar; modelo M14.1)
- z por etiqueta en unidades estandarizadas.
- Grupos calculados desde X: deciles de τ30 predicha (cortes en cal), agrupados también en D1–D3, D4–D7 y D8–D10.
- **G (global):** R = correlación de z en cal; Σ(x) = D R D.
- **P (por grupo):** R_g = correlación de z en cal dentro de cada decil de τ30 predicha; Σ(x) = D R_g(x) D.
- Score de Mahalanobis s = (y − μ)ᵀ Σ(x)⁻¹ (y − μ), en 3D (las tres etiquetas) y en 2D (chirp, total). Radio conformal: cuantil k-ésimo de s en cal, k = ⌈(n + 1)·0,9⌉. Volumen ∝ sqrt(det Σ(x))·r^(d/2).

## Criterio (val)
- **Validez:** cobertura conjunta de G y de P dentro de la banda binomial al 95 % de 0,90 con n = 40 000 (y por grupo D1–D3, D4–D7, D8–D10, descriptivo).
- **Ganancia:** Δ = 100·(exp(mean log V_P − mean log V_G) − 1), con IC 95 % por bootstrap emparejado de filas de val (B = 1000, default_rng(144001)), en 3D (primario) y 2D.
- **Decisión:** si G y P son válidas y el IC superior de Δ en 3D es ≤ −10 %, se justifica la cabeza con covarianza completa. En otro caso basta D R D con R global.

## Descriptivos
- Correlaciones de z por pares en cada decil de τ30 predicha, con IC bootstrap; también por tercil de σ_chirp y por tercil de SNR nominal (este último, por la verdad: solo informativo).
- Cobertura simultánea de los intervalos v2 de M14.3 (fracción de filas con las tres etiquetas cubiertas), global y por grupo.
- Volumen medio de los elipsoides frente al de los rectángulos v2 (producto de anchuras), sabiendo que no tienen la misma cobertura conjunta.

## No se puede concluir
- La ganancia exacta de una cabeza con covarianza completa entrenada: R por grupo es una aproximación por trozos de Σ(x), así que el resultado es orientativo (más bien una cota inferior).
- Una semilla. Sin datos reales.

## Regla de parada
Un notebook.
