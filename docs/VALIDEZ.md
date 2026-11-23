# Validez: qué está validado y qué no

## Validado (frente a G3T)

- **La receta por defecto** (`configs/default_g3t.json`, `click_free: true`) reproduce el camino de
  X_T de G3T (`m13_13.noisy_row`).
- **El criterio de equivalencia**, fijado antes de ejecutar la prueba
  (`tools/check_equivalence_g3t.py`): en 64 filas equiespaciadas de val de G3T, la diferencia máxima
  en float16 debe ser ≤ 1 ulp y las etiquetas deben coincidir.
  > **Estado:** pendiente de ejecutar. Este documento no afirma que la prueba haya pasado; el
  > resultado estará en su informe JSON.
- **La señal S15**, para M_tot ≤ 180 M☉: puertas G-TAPER, G-ALIGN y G-EDGE de M13.13. Los umbrales
  son un desajuste de S15 frente a Sref ≤ 8·10⁻⁴ por fila y una fase residual de alineación
  ≤ 0,04 rad. `align_residual_rad` permite filtrar; una fila con |residuo| > 0,5 rad (umbral
  `ALIGN_ABORT` de producción) se guarda con `status = 2` y no se usa.
- **La red de detectores**: H1, L1 y V1, en ese orden.

## Pendiente

- **S10 (M_tot > 180 M☉).** Es la regla de banda de la enmienda 2 de M15, pero su validación
  (G-TAPER-G4b) está pendiente. gwsim avisa si el prior puede dar M_tot > 180.

## No validado

- **M_tot > 250 M☉.**
- **M_tot < 16 M☉.** SEOBNRv4_opt puede no generarse a 4096 Hz; esas filas quedan con `status = 2`.
- **Cualquier cambio en `simulation` o `signal_processor`** respecto a G3T (frecuencia de muestreo,
  ventanas, banda, filtro…). Hay aviso.
- **Otros aproximantes con `click_free: true`.** Hay aviso.
- **El ruido analítico (`analytic`)**: sus PSD son curvas de diseño, no las de O3b. La receta es la
  misma, pero no se ha comparado con producción.
- **Priors y SNR fuera de los de G1** (SNR en [10, 25), masas en [5, 90]). La física es la misma,
  pero los modelos entrenados con G1/G3T no han visto esas regiones.
- **Las salidas distintas de `X`** (`X_noise`, `signal_whitened`, …) salen de la misma cadena, pero
  la prueba de equivalencia solo compara `X`.

## El ruido no es ruido real

El ruido es **gaussiano, estacionario y coloreado**, con la PSD empírica de un bloque de O3b
(Welch sobre 128 s fuera del intervalo simulado). **No** es ruido real de los detectores:

- **no hay glitches** (transitorios no gaussianos: blips, scattered light, etc.);
- **no hay no estacionariedad**: la PSD es fija dentro de la muestra;
- **no hay líneas que varíen** ni correlaciones entre detectores, más allá de que comparten bloque
  temporal.

Esto importa especialmente en un estudio **rank-based model-agnostic**: un detector entrenado o
calibrado solo con este ruido verá sus primeros glitches reales en la evaluación. Un rango o un
p-valor calculado con muestras de `noise_only` de gwsim describe el fondo gaussiano, no el fondo real
de O3b. Cualquier conclusión sobre falsas alarmas en datos reales necesita ruido real, por ejemplo
time slides sobre datos de GWOSC.

## Espines alineados

Solo hay χ₁z y χ₂z: no hay precesión ni modos superiores (SEOBNRv4_opt es solo el modo (2,2)).
Los aproximantes con precesión están prohibidos en la configuración: con espines alineados
producirían en silencio una señal sin precesión.

## Equivalencia y redondeo

- **Ventana de procesado.** Con un `gps` dado, gwsim deriva el inicio de la ventana de procesado
  como `gps − duration/2 − contexto`. G3 lo obtiene desplazando la ventana de G1, y en coma flotante
  los dos valores pueden diferir en el último bit (≈ 2·10⁻⁷ s a GPS ≈ 1,26·10⁹). La prueba de
  equivalencia pasa el valor de G3 (`RowOptions.processing_start`) para no depender de ese redondeo.
- **Entorno.** La equivalencia bit a bit supone el mismo entorno que la producción (versiones de
  `environment.yml` y el mismo backend de FFT de PyCBC). Con otras versiones son de esperar
  diferencias pequeñas.

## Datos de GWOSC: licencia y cita

Las PSD del paquete se estiman a partir de datos públicos de O3b de GWOSC (Gravitational Wave Open
Science Center), distribuidos bajo **CC BY 4.0**. Si publicas resultados que usen estos datos,
debes citarlos:

- R. Abbott et al. (LIGO Scientific Collaboration, Virgo Collaboration and KAGRA Collaboration),
  *Open Data from the Third Observing Run of LIGO, Virgo, KAGRA and GEO*, ApJS 267, 29 (2023).
- La frase de reconocimiento de GWOSC que indica https://gwosc.org/acknowledgement/.
