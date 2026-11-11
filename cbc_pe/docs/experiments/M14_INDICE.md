# Índice de la fase M14

Este índice recoge los experimentos de la fase M14, con su estado y su documentación.

## 1. Experimentos

| id | qué hace | dónde corre | estado | doc |
|---|---|---|---|---|
| M14.1 | 4a-G3T s123 con cabeza heterocedástica (μ y log σ por etiqueta) y pérdida β-NLL (β = 0,5); referencia 4a-G3T s123 | entrenamiento GPU | cerrado (11-11): C1 PASA, C2 FALLA; σ no adoptada por pre-registro | M14_1_heteroscedastic.md |
| M14.2 | 4a-G3T s123 con banco de K = 32 filtros adaptados aprendibles en cuadratura (correlación FFT en toda la ventana, \|c\| por detector y plantilla) concatenado con la entrada; referencia 4a-G3T s123 | entrenamiento GPU | pre-registrado (10-11) | M14_2_template_bank.md |
| M14.3 | Mondrian v2 con score r/σ de M14.1, mismos bins que v1, decisión en val | CPU + inferencia GPU | cerrado (11-11): v2 adoptado en las tres etiquetas (enmienda 2, validez condicional al SNR) | M14_3_mondrian_v2.md |
| M14.4 | Estructura de correlación de z = (y − μ)/σ de M14.1: regiones conjuntas conformales de Mahalanobis (3D y 2D) con Σ(x) = D R D, R global (G) frente a R por decil de τ30 predicha (P); decide si se entrena una cabeza con covarianza completa | CPU | pre-registrado (11-11) | M14_4_residual_correlation.md |
| M14.5 | Cabeza heterocedástica con covarianza completa (μ y L(x) triangular inferior, Σ = L Lᵀ) y β-NLL multivariante (β = 0,5): a) 4a-G3T s123 frente a M14.1; b) + banco de M14.2 frente a M14.2 (solo si M14.2 cumple su primario); primario: elipsoide conformal 3D con Σ aprendida frente a P de M14.4 | entrenamiento GPU | pre-registrado (11-11) | M14_5_fullcov_head.md |
