# La física de gwsim, paso a paso

Este documento explica qué le pasa a cada muestra desde los parámetros de la fuente hasta el array
`X` de forma `(3, 16384)`. Los valores entre paréntesis son los de la configuración por defecto
(la que reproduce la producción del proyecto de origen). El código está en [gwsim/recipe.py](../gwsim/recipe.py) (`simulate_row`) y en
[gwsim/physics/](../gwsim/physics/).

## 0. Ventanas de tiempo

Hay dos ventanas, centradas en el mismo GPS:

- la **ventana final**, de 4 s (16384 muestras a 4096 Hz), que es la que se guarda;
- la **ventana de procesado**, que añade un contexto de 33408 muestras (≈ 8,16 s) a cada lado:
  20,3125 s en total (83200 muestras).

El contexto existe porque el blanqueado y los filtros FIR estropean los bordes del segmento. Con un
filtro de blanqueado de 16 s se pierden 8 s por lado, más 256 + 256 muestras de los dos FIR. El
contexto (33408 ≥ 32768 + 512 muestras) hace que todo lo estropeado quede fuera de la ventana final.

## 1. Parámetros y forma de onda A

Una fuente tiene masas m₁ ≥ m₂ (en M☉, marco del detector), espines alineados χ₁z y χ₂z,
inclinación ι, ascensión recta α, declinación δ y ángulo de polarización ψ. A partir de ellos se
derivan:

- la masa de chirp 𝓜 = (m₁m₂)^{3/5} / M^{1/5};
- la masa total M = m₁ + m₂;
- el cociente de masas q = m₂/m₁;
- el espín efectivo χ_eff = (m₁χ₁z + m₂χ₂z)/M.

Con `get_td_waveform` de PyCBC se genera **A**: SEOBNRv4_opt en el dominio del tiempo desde
f_lower = 30 Hz, a una distancia de referencia de 1000 Mpc. Se obtienen h₊(t) y h×(t), con t = 0 en
el pico.

## 2. Proyección sobre los detectores

`DetectorProjector` aplica, para cada detector, h(t) = F₊(α, δ, ψ, t) h₊ + F×(α, δ, ψ, t) h×,
desplazada en el tiempo por el retraso de propagación del geocentro al detector. Usa
`Detector.project_wave(method="lal")`, de modo que el patrón de antena depende del GPS por la
rotación de la Tierra.

## 3. Colocación `random_contained`

La señal no está siempre centrada en la ventana. Primero se proyecta A en el centro de la ventana
final y se recorta su último tramo, de como mucho 4 s (`keep_last_segment`: si la señal dura más de
4 s, se queda la parte final, con la fusión). Después se elige al azar, con `placement_seed`, un
inicio de ventana tal que toda esa red quepa dentro. Ese desplazamiento se traslada al tiempo de
coalescencia geocéntrico, t_c = centro + desplazamiento, y la señal se **reproyecta en el GPS
real**. Así, el patrón de antena y los retrasos corresponden al instante verdadero.

## 4. SNR óptima y reescalado de la distancia

La SNR óptima de un detector, con S_n(f) su PSD unilateral, es

ρ² = 4 Re ∫_{30 Hz}^{1024 Hz} |h̃(f)|² / S_n(f) df

(`pycbc.filter.sigma`, sobre la señal de la ventana final rellenada con ceros). La SNR de red es la
suma en cuadratura: ρ_red = √(ρ_H1² + ρ_L1² + ρ_V1²).

Como h ∝ 1/d, la SNR es inversamente proporcional a la distancia. Con la SNR ρ_ref medida a
d_ref = 1000 Mpc, la distancia que da la SNR objetivo ρ_obj es

d = d_ref · ρ_ref / ρ_obj.

Se regenera A a esa distancia y se comprueba que |ρ_final − ρ_obj| / ρ_obj ≤ 0,05. Si no se cumple,
la fila falla (`status = 2`).

En el modo `distance` se usa la distancia dada, sin reescalar, y se guarda la SNR que resulta. La
SNR y la distancia guardadas son siempre las de **A**, también cuando se inyecta S (paso 6), igual
que en la referencia. En la banda de 30–1024 Hz, A y S difieren muy poco: la validación de la forma de onda exigía
un desajuste ≤ 8·10⁻⁴ por fila.

## 5. Ruido gaussiano coloreado y las dos PSD

### Qué es la PSD de Welch

La densidad espectral de potencia S_n(f) dice cuánta potencia de ruido hay en cada frecuencia. El
método de Welch la estima a partir de un tramo de datos reales: se trocea en segmentos de 8 s
solapados al 50 %, se enventana cada segmento, se hace la FFT, se toma |·|² y se promedia. Promediar
reduce la varianza de la estimación. Con segmentos de 8 s, la resolución es Δf = 1/8 Hz. En gwsim,
cada bloque de ruido tiene una PSD de Welch estimada sobre 128 s de datos de O3b de GWOSC, tomados
fuera del intervalo que se simula.

### Por qué hay dos PSD

Una serie de N muestras con paso Δt tiene una FFT con Δf = 1/(N Δt). Cada operación necesita la PSD
en la rejilla de su segmento:

- la PSD **"snr"** usa el segmento de 4 s, con Δf = 0,25 Hz;
- la PSD **"proc"** usa el segmento de procesado de 20,3125 s, con Δf ≈ 0,0492 Hz. Sirve para
  generar el ruido y para blanquear.

Las dos se obtienen interpolando linealmente la misma PSD de Welch, de Δf = 1/8 Hz, y recortándola
hasta la frecuencia de Nyquist (`interpolate_raw_psd`).

### Ruido

`pycbc.noise.gaussian.noise_from_psd` genera, en cada detector, 83200 muestras de ruido gaussiano
estacionario con la PSD "proc". Las semillas por detector se derivan de `gaussian_seed`:
`default_rng(gaussian_seed).integers(0, 2³² − 1)` para H1, L1 y V1, en ese orden.

## 6. La señal sin clic S (bandas de arranque baja y alta)

A empieza bruscamente a 30 Hz: su primera muestra no es cero. Ese escalón, el **clic**, tiene
potencia en todas las frecuencias y, tras el blanqueado, deja un artefacto que no es física de la
fuente. Por eso se sustituye A por **S**, que es el mismo modelo empezado a una frecuencia más baja y
con una subida suave (media ventana de Kaiser, β = 8) desde su inicio hasta que la frecuencia
instantánea alcanza el final del taper:

- **banda de arranque baja**: desde 15 Hz, con taper hasta 25 Hz, si M ≤ 180 M☉;
- **banda de arranque alta**: desde 10 Hz, con taper hasta 14 Hz, si M > 180 M☉. En sistemas pesados la
  frecuencia sube tan deprisa que un taper breve (15 → 25 Hz) ensancha su espectro y deja parte de
  la deformación por encima de 30 Hz; arrancar a 10 Hz lo aleja de la banda analizada. Validada a nivel
  de forma de onda hasta M = 250 M☉ (ver VALIDEZ.md).

S se alinea con A en fase y en tiempo submuestra (`_align_to_A`). Para ello se generan las dos de
cara (ι = 0), se busca el máximo de la correlación compleja sobre su tramo común, se adelanta S esa
fracción de muestra por FFT y se corrige la fase de coalescencia. Después se proyecta S con los
mismos parámetros y el mismo t_c que A. En la banda de 30–1024 Hz, S coincide con A; solo cambia lo
que ocurre por debajo de 30 Hz, que el procesado elimina, y desaparece el escalón.
`align_residual_rad` guarda la fase residual de la alineación.

### Taper de borde

S empieza antes que A (15 Hz frente a 30 Hz) y puede empezar antes que la ventana de procesado. Al
inyectarla, el recorte en el borde crearía otro escalón. En ese caso se aplica a S una subida de
Kaiser (β = 8) de 0,5 s desde el inicio de la ventana de procesado. `edge_samples_<ifo>` guarda
cuántas muestras quedaban fuera; si vale 0, no se aplicó taper.

## 7. Inyección y procesado

Se suma s(t) + n(t) en la ventana de procesado (`SignalInjector.inject`) y se procesa
(`SignalProcessor`):

1. **Blanqueado.** x̃_w(f) = x̃(f) / √S_n^c(f), donde S_n^c es la PSD "proc" acondicionada por
   `inverse_spectrum_truncation`: el filtro 1/√S_n se lleva al dominio del tiempo, se trunca a 16 s
   (65536 muestras, con bordes de Hann) y se vuelve a la frecuencia. Truncar el filtro limita cuánto
   se extiende en el tiempo la respuesta a un evento, y por tanto cuánto borde se estropea. Tras
   blanquear, el ruido es aproximadamente blanco: todas las frecuencias pesan igual y la señal se ve
   donde el detector es sensible.
2. **Banda.** Un filtro FIR paso alto de 30 Hz y otro paso bajo de 1024 Hz (orden 256, Kaiser β = 5).
   Es la misma banda que la de la SNR.
3. **Recorte** a la ventana final de 4 s.

Todo este procesado es lineal: L(s + n) = L(s) + L(n). Por eso `signal_whitened` es exactamente la
parte de señal de `X_unnormalized`.

## 8. z-score

Por muestra y por detector: X = (x − media) / (desviación típica + 10⁻⁶), en float32. Así cada canal
tiene media 0 y desviación 1, y la red no puede usar la amplitud absoluta, que depende de la PSD del
momento. Es el único paso no lineal. `X_noise` se normaliza con su propia media y desviación, no
con las de `X`.
