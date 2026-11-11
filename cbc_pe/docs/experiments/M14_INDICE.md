# Índice de la fase M14

Este índice recoge los experimentos de la fase M14, con su estado y su documentación.

## 1. Experimentos

| id | qué hace | dónde corre | estado | doc |
|---|---|---|---|---|
| M14.1 | 4a-G3T s123 con cabeza heterocedástica (μ y log σ por etiqueta) y pérdida β-NLL (β = 0,5); referencia 4a-G3T s123 | entrenamiento GPU | cerrado (11-11): C1 PASA, C2 FALLA; σ no adoptada por pre-registro | M14_1_heteroscedastic.md |
| M14.2 | 4a-G3T s123 con banco de K = 32 filtros adaptados aprendibles en cuadratura (correlación FFT en toda la ventana, \|c\| por detector y plantilla) concatenado con la entrada; referencia 4a-G3T s123 | entrenamiento GPU | pre-registrado (10-11) | M14_2_template_bank.md |
| M14.3 | Mondrian v2 con score r/σ de M14.1, mismos bins que v1, decisión en val | CPU + inferencia GPU | pre-registrado (11-11) | M14_3_mondrian_v2.md |
