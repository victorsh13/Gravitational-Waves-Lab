# M14.7 — Re-selección de la configuración del Mondrian para el modelo final
**Estado:** pre-registrado el 12-11-2026.
**Motivo:** la configuración actual (taxonomía, número de bins, simetría) se eligió para M11, con score r y el criterio de ±2 puntos (mal fijado para bins pequeños). El modelo y el score han cambiado (M14.5: σ = √diag Σ; M14.3: r/σ).

## Espacio de búsqueda (por etiqueta)
- El mismo de la selección de M11 (taxonomías, número de bins, simetría e hiperparámetros del kNN de dificultad), más:
  - la opción de 1 bin (sin taxonomía);
  - score r o r/σ.
- Una sola política (desaparece la distinción conservative/efficient).

## Datos
Ajuste en cal; selección en val; test una sola vez para la configuración elegida. Modelo: M14.5a (si el 13-11 se elige M14.5b, se repite sin cambios).

## Validez (val), con bandas Monte Carlo que incluyen el ruido de cal (procedimiento de la enmienda 6 de M13.13 y la enmienda 2 de M14.5)
1. Cobertura marginal no por debajo del límite inferior de su banda al 95 %.
2. Ningún bin por debajo del límite inferior de su banda Beta-Binomial al 99 %.
3. Validez condicional en particiones calculadas desde X: máx. |cobertura − 0,90| ≤ 0,015 en terciles de τ30 predicha (cortes en cal) y en terciles de SNR nominal (cortes de val de M14.3).

## Eficiencia
Entre las configuraciones válidas, la de menor anchura media en val. Si varias están dentro del 1 % de la mejor, la de menos bins (y simétrica antes que asimétrica).

## Confirmación en test
La configuración elegida debe cumplir 1–3 en test (bandas con el n de test). Si falla, se informa y se usa la configuración heredada de M11 con score r/σ (validada en M14.3 y M14.5).

## No se puede concluir
- Optimalidad fuera del espacio de búsqueda.
- La selección entre muchos candidatos en val es optimista: lo confirma test.
- Una semilla.

## Regla de parada
Una ejecución por modelo final.

## Resultado con M14.5b (13-11-2026, run 20261113T152305Z)
- Configuraciones válidas en val: 34 (𝓜), 32 (M_tot) y 33 (χ_eff) de 68; todas las de score r son condicionalmente inválidas.
- Seleccionada para las tres etiquetas: 1 bin, simétrica, r/σ (la misma que con M14.5a). Confirmada en test: cobertura 0,904 / 0,901 / 0,904; desviación condicional máxima 0,010 / 0,007 / 0,010; anchura media 13,72 / 25,05 / 0,513.
- Mondrian de M11 en test: con score r, inválido (desviación 0,035 / 0,077 / 0,041); con r/σ, válido y con anchura casi igual (< 0,5 %).
- Eventos reales con 5b (run 20261113T154408Z, n = 9, descriptivo): mediana de LVK dentro del intervalo 9/9 en las tres etiquetas y dentro del elipsoide S 9/9 (3D y 2D).
- Run con M14.5a: 20261112T161005Z (M14.7) y 20261112T170136Z (eventos reales).
