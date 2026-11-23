# Validez: qué está validado y qué no

Este documento te dice hasta dónde puedes fiarte de los datos de gwsim. Léelo antes de sacar
conclusiones.

## Resultado verificado

**Equivalencia bit a bit con la producción de referencia del proyecto de origen** (prueba ejecutada
el 23-11-2026):

- 64 filas equiespaciadas del split de validación de la referencia;
- modo `snr`, señal sin clic con la banda de arranque baja (todas con M_tot ≤ 180 M☉), PSD de O3b del
  paquete y física por defecto;
- resultado: **PASS en 64/64 filas, con diferencia de 0 ulp**. Es decir, X es idéntico bit a bit tras
  convertirlo a float16, y las etiquetas `[chirp_mass, total_mass, chi_eff]` coinciden;
- el criterio (≤ 1 ulp de float16 y etiquetas iguales en todas las filas) se fijó antes de ejecutar
  la prueba.

Por tanto, con la configuración por defecto gwsim produce exactamente los mismos datos que la
referencia. Esa referencia, a su vez, pasó sus propias comprobaciones de la forma de onda, la
alineación de S con A y el taper de borde, con estos umbrales:

- desajuste de la señal sin clic frente a una referencia más larga ≤ 8·10⁻⁴ por fila;
- fase residual de alineación ≤ 0,04 rad.

`align_residual_rad` te permite filtrar. Una fila con |residuo| > 0,5 rad se considera un fallo de la
alineación y se guarda con `status = 2`.

## La banda de arranque alta (M_tot > 180 M☉)

**Validada a nivel de forma de onda en el proyecto de origen hasta M_tot = 250 M☉.** Para estos
sistemas, la señal sin clic arranca a 10 Hz con un taper hasta 14 Hz, y en el proyecto de origen se
comparó con una referencia que arranca a más baja frecuencia, con el mismo criterio que la banda baja.
Superó la comprobación en todo el rango, hasta M_tot = 250 M☉.

Lo que **no** se ha comprobado es la equivalencia bit a bit de gwsim con la producción en esta región.
La prueba de equivalencia solo cubre M_tot ≤ 180 M☉, porque no hay datos de producción con la banda
alta con los que comparar. Por encima de 250 M☉ no hay ninguna validación.

**Por qué el taper tiene que quedar por debajo de la banda analizada.** El taper es la subida suave con
la que la señal sin clic pasa de cero a su amplitud completa; mientras dura, la señal está «deformada»
respecto a la física real. Esa deformación es inofensiva solo si ocurre a frecuencias que el análisis
no mira. El procesado elimina todo lo que hay por debajo de 30 Hz, así que el taper tiene que haber
terminado antes de que la frecuencia de la señal llegue a 30 Hz.

Para M_tot > 180 M☉ se usa una señal que arranca a 10 Hz con taper hasta 14 Hz, porque con
15 → 25 Hz una fracción del efecto del taper llegaba a la banda ≥ 30 Hz. Esta variante está validada a
nivel de forma de onda hasta M_tot = 250 M☉.

*Interpretación, no comprobada aparte:* en un sistema pesado la señal recorre 15 → 25 Hz en muy pocos
ciclos, así que el taper es corto en tiempo y, por tanto, ancho en frecuencia (anchura del orden de
1/duración); además, la fusión ocurre cerca de esa banda (f_ISCO ≈ 4400 Hz/(M_tot/M☉), unos 22 Hz
para 200 M☉).

## Lo que la prueba NO cubre

- **El modo `distance`.** Usa la misma cadena, pero sin el reescalado de la distancia, y no se ha
  comparado con la referencia.
- **El modo `noise_only`.** El ruido y el procesado son los mismos que en las filas validadas, pero
  las muestras de solo ruido no se han comparado una a una.
- **Las PSD analíticas** (`noise.source = "analytic"`). Son curvas de diseño, no las de O3b.
- **La banda de arranque alta (M_tot > 180 M☉), en cuanto a la equivalencia con la producción.** Su
  forma de onda sí está validada hasta M_tot = 250 (sección anterior), pero la igualdad bit a bit de
  gwsim con la producción solo se ha comprobado con M_tot ≤ 180. gwsim avisa si el prior puede dar
  M_tot > 180.
- **Priors fuera de masas 5–90 M☉ y espines de −1 a 1** (y SNR fuera de [10, 25)). La física es la
  misma, pero:
  - con M_tot > 250 M☉ no hay ninguna validación;
  - con M_tot < 16 M☉, SEOBNRv4_opt puede no generarse (la fila queda con `status = 2`).
- **Cualquier cambio en `simulation` o `signal_processor`**, otro aproximante o `click_free: false`.
  Hay aviso en todos los casos.
- **Las salidas distintas de `X`** (`X_noise`, `signal_whitened`, …). Salen de la misma cadena, pero la
  prueba solo compara `X`.

## El ruido es gaussiano, no ruido real

El ruido es **gaussiano, estacionario y coloreado**, con la PSD empírica de un bloque de O3b (estimada
con Welch sobre 128 s fuera del intervalo simulado). **No** es ruido real de los detectores:

- **no hay glitches** (transitorios no gaussianos: blips, scattered light, etc.);
- **no hay no estacionariedad**: la PSD es fija dentro de la muestra;
- **no hay líneas que varíen** ni correlaciones entre detectores, más allá de que comparten bloque
  temporal.

Esto importa especialmente en un estudio **basado en rangos y agnóstico al modelo**. Un estadístico
calibrado solo con este ruido verá sus primeros glitches reales en la evaluación, y un rango o un
p-valor calculado con muestras de `noise_only` describe el fondo gaussiano, no el fondo real de O3b.
Cualquier conclusión sobre falsas alarmas en datos reales necesita ruido real, por ejemplo time slides
sobre datos de GWOSC.

## Espines alineados

Solo hay χ₁z y χ₂z: no hay precesión ni modos superiores (SEOBNRv4_opt es solo el modo (2,2)). Los
aproximantes con precesión están prohibidos en la configuración: con espines alineados producirían en
silencio una señal sin precesión.

## Reproducibilidad numérica

- **Ventana de procesado.** Con un `gps` dado, gwsim deriva el inicio de la ventana de procesado como
  `gps − duration/2 − contexto`. La referencia lo obtiene de otra forma, y en coma flotante los dos
  valores pueden diferir en el último bit (≈ 2·10⁻⁷ s a GPS ≈ 1,26·10⁹). La prueba de equivalencia
  pasa el valor de la referencia (`RowOptions.processing_start`) para no depender de ese redondeo.
- **Entorno.** La igualdad bit a bit supone las versiones de `environment.yml` y el mismo backend de
  FFT de PyCBC. Con otras versiones son de esperar diferencias en los últimos bits, sin efecto físico.

## Datos de GWOSC: licencia y cita

Las PSD del paquete se estiman a partir de datos públicos de O3b de GWOSC (Gravitational Wave Open
Science Center), distribuidos bajo **CC BY 4.0**. Si publicas resultados que usen estos datos, debes
citarlos:

- R. Abbott et al. (LIGO Scientific Collaboration, Virgo Collaboration and KAGRA Collaboration),
  *Open Data from the Third Observing Run of LIGO, Virgo, KAGRA and GEO*, ApJS 267, 29 (2023).
- La frase de reconocimiento de GWOSC que indica https://gwosc.org/acknowledgement/.
