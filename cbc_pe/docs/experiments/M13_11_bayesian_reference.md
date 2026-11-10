# M13.11 — Referencia bayesiana (PE) por régimen de duración

**Estado:** en pausa (5-11-2026). G-REC PASA; G-MM FALLA por el escalón inicial de las señales (ver M13.12); se replantea sobre datos sin clic (M13.13).

## Pregunta
Para χ_eff y la chirp mass, en señales cortas (τ30 D1–D2), medias (D5–D6) y largas (D9–D10): ¿cuánto margen de mejora del RMSE queda en la red frente al estimador bayesiano óptimo con la misma información (ventana final de 4 s, banda 30–1024 Hz, mismo prior)?

## Datos
60 filas de val por régimen (default_rng(131101)), reconstruidas de forma exacta con las semillas del manifest (placement_seed, gaussian_seed) y la PSD de procesado del bloque.
- Gate G-REC: el procesado de la serie reconstruida reproduce X con max|ΔX| ≤ 1e-5.
- Se usan las predicciones de la red ya existentes (sin nueva inferencia).

## PE
pycbc.inference con el modelo "relative" y dynesty; forma de onda SEOBNRv4_ROM, f_lower = 30 Hz, banda 30–1024 Hz; verosimilitud gaussiana con la PSD con la que se generó el ruido.
- Gate G-MM: mismatch entre SEOBNRv4_opt y SEOBNRv4_ROM ≤ 0,003 (informado por régimen).
- Priors iguales a los de generación: m1, m2 U[5,90] ordenadas, sin redshift; χ1z, χ2z U[−1,1]; cielo e inclinación isótropos; polarización U[0,2π]; fase de coalescencia fijada al valor por defecto de pycbc (como en la generación); t_c U[±0,1 s].
- Distancia: p(d) ∝ 1/d² en un rango amplio. Es una aproximación al prior inducido por el SNR objetivo U[10,25], cuyos límites dependen de θ; afecta sobre todo a la amplitud.
- Valores por defecto de pycbc 2.9.0 (src/waveform.py:33-43 no pasa coa_phase ni f_ref; get_td_waveform y get_fd_waveform los toman de default_args vía props(), pycbc/waveform/waveform.py:64-66 y 424-436): coa_phase = 0 rad (pycbc/waveform/parameters.py:401-403), f_ref = 0 Hz (pycbc/waveform/parameters.py:342-344).

## Cantidades
Por régimen y etiqueta (χ_eff, chirp_mass; total_mass descriptivo):
- R = MSE de la red (todas las filas de val del régimen) / media de la varianza posterior (filas con PE).
- H = 1 − 1/√R (mejora máxima posible del RMSE).
- Secundaria: R emparejado (MSE de la red / MSE de la media posterior en las mismas filas).
- Modelos: 4b-aug s123 y la media de 4a-aug s123/s124.
- IC por bootstrap: filas con PE para el denominador; filas de val para el numerador.

## Criterios
- **Gate de validez de la PE:** cobertura de la verdad en los intervalos creíbles del 50 % y el 90 % de χ_eff y de la chirp mass (180 inyecciones), dentro del intervalo binomial del 95 %. Si falla, la referencia no es válida y se para.
- **Margen relevante:** límite inferior del IC de H ≥ 15 % (R ≳ 1,38).
- **Sin margen:** límite superior del IC de H < 7 % (R ≲ 1,16).
- **No concluyente:** el resto.

## Se puede concluir / no se puede concluir
**Se puede concluir:** en qué regímenes y para qué etiqueta queda margen para M14 (plantillas, especialistas por régimen).

**No se puede concluir:**
- nada sobre ruido real;
- nada sobre la información perdida por la banda de 1024 Hz o por la ventana de 4 s;
- si R está entre 1,16 y 1,38.

## Regla de parada
PE de prueba de tiempo sobre la fila más larga del régimen largo.
- Si tarda más de 1,5 h, se reduce a 40 filas por régimen (informando de la resolución) o se descarta.
- Si falla algún gate, se para.

## Diagnóstico del gate G-MM (4-11-2026)
Script `scripts/m13/m13_11_prepare_rows.py` (opciones --diag-mm, --diag-intrinsic, --diag-click); notebook `notebooks/m13/M13_11_click_visualization.ipynb`; 180 filas en `<data_root>/results/M13_3/M13_11/rows/20261104T172343Z/gates.json`.
- G-REC PASA: reconstrucción exacta de filas de val (max|ΔX| = 0).
- G-MM FALLA: SEOBNRv4_opt frente a SEOBNRv4_ROM, mismatch 1–3 % (máximo en masa alta). No es la banda ni el arranque a 30 Hz (D1: subir f_low del match a 40 Hz apenas cambia nada) ni la conversión de dominio (D3).
- Intrínseco: SEOBNRv4_opt = SEOBNRv4 (mismatch ≤ 4e-5); mismos argumentos en las dos llamadas.
- Causa: la forma de onda generada con f_lower = 30 Hz sin taper arranca con un escalón (|h(t0)|/max|h| hasta 0,73). Con S (f_lower = 20 Hz, taper Kaiser hasta 28 Hz): A frente a S 0,6–3,7 %; S frente a ROM 0,01–0,09 %. El ROM es válido sobre señales sin escalón.
- SNR de red del escalón (fase maximizada, 180 filas): mediana ~3,1 (cortas), 2,6 (medias), 1,8 (largas); máximo ~7.

## Enmienda 2 (6-11-2026, antes de calcular)
Datos: las mismas 180 filas de val (selección original, default_rng(131101)), con su ruido exacto, pero con la señal generada con SEOBNRv4_ROM (dominio del tiempo, f_lower = 15 Hz, alineada en fase y tiempo con la S15 de G3T) en lugar de SEOBNRv4_opt. Motivo: el escalón inicial de G1/G3 (M13.12) se ha eliminado en G3T, pero el ROM difiere de SEOBNRv4 hasta ~4,5e-3 de mismatch en masas bajas (M13.13, G-TAPER), y eso sesgaría la PE en el régimen largo. Inyectar con el mismo modelo que usa la PE hace la referencia exactamente consistente.
Red: 4a-G3T s123 (y 4b-G3T cuando termine), evaluada sobre las mismas filas con señal del ROM. Gate G-NET: para esas filas, |ΔRMSE| entre las predicciones con señal ROM y con señal S15 (X de val G3T) ≤ 2 % por etiqueta y régimen; si no se cumple, la comparación no es válida y se informa.
El resto del diseño (regímenes, cantidades R y H, criterios, gate de cobertura, regla de parada) no cambia.

## Enmienda 3 (6-11-2026, antes de calcular)
Datos de la PE: 8 s centrados en la ventana final de 4 s (2 s de margen a cada lado, solo ruido salvo en filas truncadas), con una ventana Tukey cuyas rampas caen en los márgenes. Así la señal queda entera en la zona plana y se usa el modelo relative estándar, sin aproximación. Los márgenes no aportan información de la señal. Las filas cuya señal no cabe entera en la zona plana (truncadas) se excluyen del primario y se informan aparte. Prior de distancia: p(d) ∝ 1/d² en [5, 50000] Mpc (el inducido por el SNR objetivo uniforme; los límites dependientes de los parámetros se aproximan por un rango amplio). Fase de coalescencia marginalizada (la señal del ROM lleva el desfase de la alineación). Masas muestreadas en (mchirp, q) con prior uniforme en m1, m2 ∈ [5, 90].
Primario: solo filas con is_truncated = False (G3) y gate de zona plana superado; las filas truncadas pasan por la PE pero se informan aparte (fijado el 6-11, antes de calcular).

## Enmienda 4 (7-11-2026, antes de ver resultados del banco)
Gate de zona plana: la fracción de SNR² medida sobre la señal blanqueada sale ~0,998 en filas no truncadas, porque el filtro de blanqueado de 16 s extiende la energía de la señal fuera de la ventana; el umbral de 0,999 clasificaba mal 6/7 filas de prueba. Primario: filas con is_truncated = False en G3 (onda desde 30 Hz contenida en la ventana final de 4 s). La parte de 15–30 Hz de la señal inyectada queda fuera de la banda de la PE (30–1024 Hz). Las fracciones de SNR² se informan como dato.

## Aplicación de la regla de parada (6-11-2026)
PE de prueba (fila 414223, la más larga del régimen largo): 1,0 h de reloj con 3 procesos, 2,8 h de CPU, 2,2·10⁶ evaluaciones; relbin max|Δloglr| = 0,017. Supera las 1,5 h de la regla, así que el banco se reduce a 40 filas por régimen (las 40 primeras de cada régimen en selection.json). Resolución esperada del IC de R: ±0,10–0,15 (frente a ±0,08–0,12 con 60). loglr_truth (109) es menor que en la posterior (~165) porque la evaluación en el valor verdadero no aplicaba el desplazamiento de la alineación ROM–S15 (~1 ms); se añade loglr_truth_aligned como comprobación.

## Plan del banco (6-11-2026, antes de calcular)
Análisis intermedio con 40 filas por régimen (para las Jornadas CPAN, 16–18-11-2026), etiquetado como intermedio. Después, el banco continúa, con independencia del resultado intermedio, hasta 60 filas por régimen (toda la selección); una ampliación posterior con filas nuevas al azar de val (semilla nueva) se fijará en número antes de ver el resultado intermedio. El análisis final usa todas las filas hasta el final fijado; se informan el intermedio y el final. Todas las PE usan la misma configuración (nlive 500, walks 100, epsilon 0,03, dlogz 0,1, priors, banda y ventana) y el mismo entorno (dynesty 2.1.5, pycbc 2.9.0).

## Enmienda 5 (6-11-2026, antes del banco)
La PE de prueba (fila 414223) no encontró el modo: el log-likelihood en la verdad alineada (179,55) supera al máximo encontrado (174,27), con t_c y el cielo en otro modo (multimodalidad extrínseca). Se marginalizan los parámetros extrínsecos dentro del modelo relative (tiempo, cielo, polarización, fase y, si es posible, distancia) y se muestrean solo los intrínsecos; la posterior de chirp_mass, total_mass y chi_eff es la misma en teoría. Prueba antes del banco: 10 filas (3 por régimen de las posiciones 41–60 de la selección, fuera del banco intermedio, y la 414223). Criterio: en todas, el máximo encontrado ≥ loglr en la verdad alineada − 1. Las filas de prueba no se usan para comparar con la red ni para elegir la configuración por su resultado. Indicador sampler_missed_mode en todas las filas del banco.
