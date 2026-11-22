# M15 — G4: reformulación del prior de entrenamiento (pre-registro)

**Estado:** pre-registrado el 20-11-2026, con el paso 0 hecho (§5.1). Se commitea **antes** de generar nada. El paso 0 solo usó modelos y datos ya existentes y vistos (val de G3T). No se ha generado nada de G4.

**Diseño de partida:** `claude/LINEAS_FUTURAS_CBC_PE.md` §3 (20-11). Este documento lo concreta y corrige un punto (T_F, §2.3).

---

## 1. Pregunta e hipótesis

**Contexto.** Con pérdida cuadrática, la red aproxima la media posterior E[θ | X] bajo el prior de entrenamiento. Cerca de un borde del prior, la posterior queda cortada y su media se desplaza hacia dentro. Eso produce un **sesgo condicionado al valor verdadero** b(θ) = E[μ̂ − θ | θ] ≠ 0 que la red no puede evitar mientras el borde exista: es el precio de minimizar el MSE bajo ese prior.

**H-RANGO (G4-R frente a la referencia).** Ampliar el rango de masas de [5, 90] a [5, 125] por componente, sin cambiar la densidad dentro del cuadrado original, reduce de forma relevante el sesgo condicionado en el borde alto original (𝓜 y M_tot).

**H-FORMA (G4-F frente a G4-R).** Con el mismo rango, aplanar la densidad marginal de 𝓜 (uniforme) sin tocar la distribución de q dado 𝓜 reduce el sesgo condicionado en el borde bajo de 𝓜, donde el prior actual tiene poca densidad.

**Fuera del alcance:** χ_eff (los espines no cambian), el SNR, el procesado y la arquitectura.

---

## 2. Datos

### 2.1. Qué se mantiene igual que G3T

Se mantienen igual: la forma de onda S15 (SEOBNRv4_opt desde 15 Hz, con taper de Kaiser hasta 25 Hz, alineada como en M13.13), la banda de 30 a 1024 Hz, el blanqueado de 16 s, el z-score, la proyección y la colocación temporal. También se mantienen los espines (χ1z y χ2z uniformes en [−1, 1]), el SNR de red uniforme en [10, 25) medido en 30–1024 Hz con la distancia derivada de él, los bloques de PSD de cada split (los mismos que G1) y el banco de ruido de M13.6 (K = 64 por bloque) para la aumentación en train. Las etiquetas son [𝓜, M_tot, χ_eff] en el sistema del detector.

**SNR.** El paso lineal L(s) se aplica antes del z-score y, en train, la entrada es zscore(L(s) + ruido del banco). Por tanto, una señal limpia guardada a un SNR se puede reescalar a cualquier otro multiplicándola, lo que equivale a cambiar la distancia. Por eso no se genera un SNR más ancho en la propuesta: un estudio futuro de SNR puede reescalar estas señales.

### 2.2. Objetivo T_R (solo rango)

m1 y m2 son i.i.d. U[5, 125] y se ordenan (m2 ≤ m1). La densidad es p_R(m1, m2) = 2/120² en el triángulo.

Restringido al cuadrado [5, 90]², T_R es exactamente la distribución de G3T. El 50,3 % de las filas de T_R cae en el cuadrado, así que 800k filas de T_R dan en el cuadrado la misma densidad que las 400k de G3T.

### 2.3. Objetivo T_F (rango y forma de 𝓜) — corrección del diseño del 20-11

**Corrección.** El diseño del 20-11 definía T_F como 𝓜 uniforme y q uniforme en [q_min(𝓜), 1], y lo justificaba porque así se conservaba la distribución de q de G3T. Una simulación con 2·10⁶ muestras (20-11) muestra que esa justificación es falsa.

| fracción con q < 0,25 | 𝓜 10–20 | 𝓜 20–35 | 𝓜 35–50 | marginal |
|---|---|---|---|---|
| G3T | 0,49 | 0,29 | 0,01 | 0,17 |
| T_R | 0,54 | 0,48 | 0,22 | 0,19 |
| T_F del 20-11 (q uniforme) | 0,19 | 0,18 | 0,09 | 0,06 |
| **T_F (esta definición)** | **0,51** | **0,48** | **0,21** | 0,15 |

**Por qué falla.** Con masas uniformes, el jacobiano |∂(m1, m2)/∂(𝓜, q)| = m1²/𝓜 hace que p(q | 𝓜) ∝ m1(𝓜, q)² ∝ (1 + q)^{2/5} q^{−6/5}. Esa densidad se concentra en q pequeño, de modo que poner q uniforme dado 𝓜 cambia también la mezcla de relaciones de masa. No sería un solo cambio.

**Definición adoptada:**

p_F(m1, m2) = p_R(m1, m2) · u(𝓜) / p_R,𝓜(𝓜)

donde u es la densidad uniforme en [𝓜_min, 𝓜_max] = [4,353; 108,82] y p_R,𝓜 es la marginal de 𝓜 bajo T_R, calculada numéricamente así:

p_R,𝓜(𝓜) = (2/120²) ∫_{q_min(𝓜)}^{1} m1(𝓜, q)²/𝓜 dq

Así la marginal de 𝓜 es uniforme y p(q | 𝓜) es exactamente la de T_R. Solo cambia la marginal de 𝓜.

**Muestreo directo de T_F:** se toma 𝓜 ~ U y luego q por inversión numérica de la CDF de p(q | 𝓜) ∝ m1² en [q_min(𝓜), 1]. Se exige que q quede estrictamente dentro del soporte y que las masas cumplan 5 ≤ m2 ≤ m1 ≤ 125.

**Consecuencia, que se mide y no se oculta:** T_F tiene mucha más densidad a 𝓜 alto. En 800k filas, ~326k quedan por encima del P95 de 𝓜 de G3T (frente a 222k en T_R) y un 22 % tiene M_tot > 200 (frente a un 9 % en T_R). En el interior hay, por tanto, menos filas que en T_R. Es un coste de esta forma, y se mide en §5.3.

### 2.4. Propuesta y submuestreo por rechazo

**Propuesta:** q(θ) = ½ p_R + ½ p_F. Para cada fila se elige la componente con probabilidad ½ y se muestrea de ella.

**Aceptación.** Con c = sup p_R/q = 2, la probabilidad de aceptar la fila para T_R es a_R = p_R / (p_R + p_F), y para T_F es a_F = p_F / (p_R + p_F) = 1 − a_R. Se usa la densidad de la **mezcla**, nunca la etiqueta de la componente de la que salió la fila.

**Acoplamiento.** Cada fila lleva un único u ~ U(0, 1), común a los dos objetivos. Se acepta para T_R si u < a_R y para T_F si u < a_F.
- Cada marginal es exacta, porque cada indicador es Bernoulli con la probabilidad correcta.
- La probabilidad de que una fila sea común a los dos objetivos es min(a_R, a_F), el máximo posible. En la simulación, el 79,5 % de las filas de cada objetivo son compartidas, lo que sirve para las comparaciones emparejadas en los tests propios.

**Corte.** Las filas se recorren en el orden de la propuesta y se toman las primeras N aceptadas por cada objetivo. El corte es exacto: los valores aceptados forman una secuencia i.i.d. de la distribución objetivo, y cortar por número de filas no los condiciona.

**Aceptación antes de generar.** La aceptación depende solo de θ, así que se decide antes de generar ninguna forma de onda. Solo se generan las filas aceptadas por al menos un objetivo.

**Tamaños:**

| split | propuesta | aceptadas por objetivo (media) | margen | filas que se generan (unión, simulación) |
|---|---|---|---|---|
| train | 1 620 000 | 810 000 (de las que se usan 800 000) | 15,8σ | ≈ 0,96 M de señales limpias |
| val | 82 000 | 41 000 (40 000) | 11σ | ≈ 49k ruidosas |
| cal | 62 000 | 31 000 (30 000) | 12,7σ | ≈ 37k ruidosas |
| test | 62 000 | 31 000 (30 000) | 12,7σ | ≈ 37k ruidosas |

A esto se suman las filas del monitor train40k de cada objetivo, ruidosas: ≈ 78k en la unión.

**Identificadores.** El identificador de fila es el índice en la propuesta, de modo que una fila compartida tiene el mismo identificador en los dos objetivos.

**Resto de parámetros y asignación de ruido.** Se hacen como en G1 (`scripts/prepare_m11_final_manifest.py`):
- espines, inclinación, ra, dec y polarización con `ParameterSampler` de `src/sampling.py`, con su `PriorConfig` de G1. Solo las masas vienen de §2.2 y §2.3;
- distancia de referencia de 1000 Mpc, y SNR objetivo uniforme en [10, 25) con un rng propio;
- asignación de recorte de ruido con `sample_assignments`: entorno, después bloque y después recorte, uniformes, dentro del banco congelado de recortes del split, con una semilla nueva;
- en train, solo bloques presentes en `bank_psd_key.npy` (el banco de M13.6 se construyó con las claves de train de G1);
- `placement_seed` y `gaussian_seed` con bases disjuntas de las de G1, así que el ruido de G4 no reproduce el de G1 ni el de G3.

### 2.5. Puertas de la generación (todas antes de entrenar)

1. **G-PRIOR.**
   - Prueba KS de las marginales de 𝓜, q, M_tot y m1 de cada objetivo frente a su CDF analítica o numérica (n = 800k): p > 10⁻³ en las cuatro.
   - Tasa de aceptación de cada objetivo y split dentro de 0,5 ± 5σ binomial.
   - En T_R, la fracción dentro de [5, 90]² entre 0,500 y 0,506.
2. **G-TAPER-G4.** La misma prueba y los mismos criterios que la enmienda 2 de M13.13, sobre 2000 filas de la unión de train:
   - ≥ 600 filas con M_tot ∈ [180, 250], estratificadas por M_tot;
   - ≥ 150 filas con χ_eff < −0,5 y M_tot > 180, donde el inicio a 15 Hz queda más cerca de la fusión.
   - **Hipótesis sin comprobar:** para M_tot ≈ 250 con espines antialineados, 15 Hz queda cerca o por encima de la frecuencia de la ISCO (≈ 17,6 Hz para M = 250 sin espín). SEOBNRv4_opt podría fallar, o el taper hasta 25 Hz podría alcanzar la fusión. G3T solo se validó hasta M_tot = 180.
   - **Matiz:** la construcción S15 necesita también la señal A (SEOBNRv4_opt desde 30 Hz), que fija la referencia de fase y de tiempo. En G3 se generó A hasta M_tot = 180, cuya ISCO está en ≈ 24 Hz, por debajo de 30 Hz. Por tanto, empezar por encima de la ISCO de Schwarzschild no basta para que el modelo falle, y el argumento no decide nada: lo decide esta prueba, que genera A y S15 en las mismas filas.
   - Los bins de masa del informe se amplían con 180–220 y 220–250; en m13_13, las filas por encima de 180 no caen en ningún bin.
3. **G-ALIGN, G-EDGE y la consistencia** de 64 filas: los mismos de M13.13.

**Si alguna puerta falla, se para antes de entrenar y se decide en el chat.** No se ajustan los umbrales en bucle.

---

## 3. Entrenamiento

| | Referencia (existente) | G4-R | G4-F |
|---|---|---|---|
| modelo | 5a (`SimpleCNN_ResidualDilatedTransformerAgg`, Σ completa, β-NLL con β = 0,5) | igual | igual |
| datos | G3T, 400k | T_R, 800k | T_F, 800k |
| épocas | 80 | 40 | 40 |
| pasos | 80 × 1562 = 124 960 | `total_steps` = 124 960 (39 épocas completas + 3085 pasos) | igual que G4-R |
| calentamiento | 2 épocas = 3124 pasos | `warmup_steps` = 3124 | igual |
| dropout apagado desde | época 64 (paso 99 968) | época 32 (paso 100 000) | igual |
| semilla | 123 | 123 | 123 |
| estadísticos de etiquetas | G1 train | **los mismos** (el mismo fichero) | igual |
| modelo registrado | última época | última época | igual |

- Con los mismos estadísticos, la estandarización y los pesos relativos de la pérdida por etiqueta son idénticos.
- Con la misma semilla, la inicialización es idéntica; el orden de los lotes difiere necesariamente.
- No se ajusta ningún hiperparámetro. Val solo monitoriza, no selecciona.

**Confusión conocida (se declara antes).** Con los mismos pasos, la región del cuadrado original recibe en G4-R la mitad de las exposiciones (≈ 16 M frente a 32 M): hay el mismo número de filas distintas, pero cada una se ve la mitad de veces.
- Si el coste en el interior (§5.3) sale INFERIOR, no se podrá separar «menos pasos en la región» de «capacidad empleada fuera». Las dos cosas son costes reales de la reformulación a igual cómputo.
- Un reentrenamiento con el doble de pasos solo se considera tras ver el resultado, como experimento nuevo con su propio pre-registro.

---

## 4. Evaluación

**Banco común:** el test de G3T (30 000 filas; `fp16_memmap_heldout` de `m13_13_g3t`). Las predicciones de 5a sobre él ya existen (M14.5a). G4-R y G4-F se predicen con el mismo script y la misma configuración, cambiando solo el checkpoint. Todo es emparejado por fila. Se usa μ̂, la media de la cabeza.

**Grupos, definidos por la verdad.** Agrupar por θ es correcto aquí: el sesgo condicionado es una propiedad del estimador, no una garantía de cobertura. Los cortes son cuantiles de la verdad en G3T train (`m13_13_g3t/fp16_memmap/y_train.npy`, 400k), fijados en el paso 0:

| grupo | definición | etiqueta primaria | filas esperadas en test |
|---|---|---|---|
| B_𝓜+ | 𝓜 > Q95(𝓜) (≈ 66,3) | 𝓜 | ≈ 1500 |
| B_Mt+ | M_tot > Q95(M_tot) (≈ 153,1) | M_tot | ≈ 1500 |
| B_𝓜− | 𝓜 < Q05(𝓜) (≈ 13,3) | 𝓜 | ≈ 1500 |
| I (interior) | Q10 ≤ 𝓜 ≤ Q90 y Q10 ≤ M_tot ≤ Q90 | las tres | — |

Los valores aproximados provienen de la simulación del prior; los exactos, del paso 0.

**Estadístico por grupo:** b_X,g = media en g de (μ̂_X − θ) para la etiqueta primaria, en M☉.

**Diferencia emparejada:** D_g(X, Y) = |b_Y,g| − |b_X,g|, donde Y es el modelo de comparación. D > 0 significa que X tiene menos sesgo.

**IC:** bootstrap emparejado por fila, 2000 réplicas, semilla 151001, percentiles del 95 %.

---

## 5. Criterios (fijados antes de generar)

### 5.1. Relevancia y certificabilidad (paso 0, sobre val, con modelos existentes)

Para cada grupo g se calcula sobre val de G3T (40k, ya usado):
- b_ref,g^val: sesgo de 5a.
- s_mod,g: rango (máximo − mínimo) de b_g entre 4a-G3T, 5a y 5b. Es una cota conservadora de la variación entre entrenamientos, porque incluye además diferencias de arquitectura y de pérdida.
- h_g: semiancho del IC de D_g(5b, 5a) en val, escalado por √(n_val,g / n_test,g). Es una estimación de la resolución en test.

Umbral de relevancia: **δ_g = ½ |b_ref,g^val|**, es decir, quitar al menos la mitad del sesgo.

Un grupo es **certificable con una semilla** si δ_g ≥ 2 s_mod,g y δ_g ≥ h_g. Un grupo no certificable pasa a ser descriptivo, y eso se declara antes de entrenar.

**Resultado del paso 0 (20-11-2026, val de G3T, 40 000 filas).**

Cortes (cuantiles de la verdad en G3T train): Q95(𝓜) = 66,259; Q95(M_tot) = 153,085; Q05(𝓜) = 13,287; Q10/Q90(𝓜) = 16,328 / 61,049; Q10/Q90(M_tot) = 47,967 / 142,077.

Sesgo b (M☉) y su error típico en val:

| modelo | B_𝓜+ (n 2006) | B_Mt+ (n 2011) | B_𝓜− (n 1983) |
|---|---|---|---|
| 4a-G3T (`M13_13_4a_G3T_s123`) | −3,843 ± 0,110 | −7,487 ± 0,210 | +0,224 ± 0,023 |
| 5a (`M14_5a_4a_G3T_fullcov_s123`), referencia | −4,079 ± 0,111 | −7,940 ± 0,213 | +0,223 ± 0,025 |
| 5b (`M14_5b_4a_G3T_tbank_fullcov_s123`) | −3,824 ± 0,110 | −7,463 ± 0,211 | +0,170 ± 0,019 |

Criterios derivados:

| grupo | b_ref^val | s_mod (rango de 4a-G3T, 5a, 5b) | h (semiancho previsto en test) | δ = ½·\|b_ref\| | 2·s_mod | ¿certificable? |
|---|---|---|---|---|---|---|
| B_𝓜+ | −4,079 | 0,255 | 0,105 | **2,040** | 0,510 | sí (δ/2s_mod = 4,0) |
| B_Mt+ | −7,940 | 0,477 | 0,189 | **3,970** | 0,954 | sí (4,2) |
| B_𝓜− | +0,223 | 0,054 | 0,043 | **0,112** | 0,108 | sí, **en el límite** (1,04) |

**Sensibilidad (descriptiva, no cambia la regla).** Hay más modelos de G3T con el protocolo completo de 400k y 80 épocas: 4b, M14.1 y M14.2. Si el rango se calcula con los seis, s_mod vale 0,324 / 0,603 / 0,093. B_𝓜+ y B_Mt+ seguirían siendo certificables, pero B_𝓜− no (2 × 0,093 = 0,186 > δ = 0,112). La regla fijada usa el conjunto de tres modelos y se aplica tal cual. Aun así, **cualquier conclusión sobre H-FORMA se informará con esta advertencia**: el efecto relevante en B_𝓜− (0,11 M☉) es del orden de la variación entre modelos.

**Observación (descriptiva).** El sesgo de borde depende del entrenamiento:
- es mayor con menos datos (n100k: −3,98 / −7,78) y con el protocolo corto (short20: −4,26 / −8,27);
- es menor con el banco de filtros (M14.2: −3,76 / −7,34).

La magnitud es ±0,3 M☉ en B_𝓜+, muy por debajo del δ de 2,04 M☉.

### 5.2. Veredicto por grupo (en test, comparación X frente a Y)

- **RELEVANTE:** D̂_g ≥ δ_g y el extremo inferior del IC es > 0.
- **NO RELEVANTE:** el extremo superior del IC es < δ_g. Se excluye una reducción relevante.
- **INDETERMINADO:** cualquier otro caso.

### 5.3. Coste en el interior (no inferioridad)

ΔRMSE_rel = RMSE_X / RMSE_Y − 1 en I, por etiqueta, con IC por bootstrap emparejado (como en §4).

Márgenes: **3 %** para 𝓜 y M_tot, **5 %** para χ_eff (del orden del ruido de semilla de 4a en D8–D10: 3,55 / 1,61 / 4,73 %).
- **NO INFERIOR:** el extremo superior del IC es ≤ al margen.
- **INFERIOR:** el extremo inferior del IC es > al margen.
- **INDETERMINADO:** en otro caso.

### 5.4. Decisiones

| comparación | primario | resultado y lectura |
|---|---|---|
| G4-R frente a 5a | B_𝓜+ y B_Mt+ | **H-RANGO APOYADA** si los dos dan RELEVANTE (unión-intersección: sin corrección por multiplicidad). **REFUTADA** si los dos dan NO RELEVANTE. Mixto o indeterminado: se informa tal cual. |
| G4-F frente a G4-R | B_𝓜− | **H-FORMA APOYADA** si da RELEVANTE; **REFUTADA** si da NO RELEVANTE. |
| coste | I, las tres etiquetas | Si una hipótesis queda apoyada con coste INFERIOR, el compromiso se documenta y la adopción se decide en el chat. No es automática. |

Si un grupo primario resultó no certificable en el paso 0, su fila de esta tabla queda descriptiva.

### 5.5. Secundarias (descriptivas, sin criterio)

- Curva b(θ) en 20 bins de cuantiles de la verdad para las tres etiquetas y los tres modelos.
- MSE en los grupos de borde. **Se espera que G4 empeore el MSE en el test de G3T:** la referencia es, aproximadamente, la regla óptima en MSE bajo el prior de ese test, que tiene el borde en 90. Un MSE peor no es un fallo.
- χ_eff en los grupos de borde.
- G4-R contra G4-F en B_𝓜+ y B_Mt+.
- Test propio de cada objetivo (30k), incluido el sesgo en el borde nuevo (M_tot ≈ 250, fuera de la región de interés).
- Eventos reales: μ̂ de 5a, G4-R y G4-F para GW191230, GW200208, GW200219 y GW200202, frente a las medianas de LVK. Con n = 4 no se contrasta nada.
- La calibración de σ y Σ no se evalúa aquí: si se adopta un G4, el conformal se recalibra en la fase «Mondrian final».

---

## 6. Procedimiento y regla de parada

1. **Paso 0** (CPU local, segundos): cálculo de §5.1 sobre val con las predicciones existentes. Se rellenan las tablas y se commitea el pre-registro.
   - **Si B_𝓜+ y B_Mt+ salen no certificables, no se entrena con una semilla:** se vuelve al chat con la opción de 2 o 3 semillas por objetivo y su coste.
2. **Fase A de CC:** coste de la generación, formato que exige el entrenador y tiempo por época (solo lectura del código; las cifras de los logs las saca el usuario con comandos).
3. **Fase B de CC:** implementación del generador. Revisión del diff en el chat. Generación en local con las puertas de §2.5. Si alguna puerta falla, se para.
4. **Entrenamiento:** una ejecución por objetivo con la semilla 123, en la VM, con al menos una GPU libre. Solo se reanuda (`--resume`) tras un fallo técnico. Ante NaN o divergencia, se para y se informa. No se cambia ningún hiperparámetro.
5. **Evaluación en test, una sola vez,** con un notebook escrito antes de tener las predicciones (`notebooks/m15/M15_G4_evaluation.ipynb`). Los puntos ambiguos se fijan como enmienda antes de ejecutarlo.
6. **Tras el veredicto, no se entrenan más variantes de G4** sin un pre-registro nuevo.

---

## 7. Qué puede y qué no puede concluir

**Puede concluir:**
- si, en simulación con el mismo ruido, forma de onda y procesado, el sesgo de borde de 𝓜 y M_tot a masa alta se debe al corte del prior en 90 M☉ (H-RANGO);
- si la escasez de densidad a 𝓜 bajo produce un sesgo medible (H-FORMA);
- qué coste tiene cada cambio en el interior, a igual cómputo.

**No puede concluir:**
- que el sesgo observado en eventos reales se deba al prior: n = 4, y además hay sistemáticos de forma de onda y de ruido real;
- nada sobre el borde nuevo: se mueve a 125 M☉ por componente, no desaparece;
- resultados válidos para otras semillas más allá de lo que justifica s_mod;
- nada sobre el efecto de remuestrear las filas: G4 usa fuentes nuevas, no las de G3T. Se supone que ese efecto es pequeño frente a δ, pero no se mide;
- nada sobre la calibración de los intervalos.

---

## 8. Coste (con las cifras medidas en G3T, 20-11)

- **Generación local (CPU).** ≈ 0,96 M señales limpias más ≈ 0,20 M ruidosas (heldout y monitores).
  - **Medido en G3T, con 3 procesos:**
    - limpia: 0,341 s por fila y proceso de media (0,114 s de reloj por fila). El tiempo baja con la masa: mediana de 0,52 s con M_tot de 10 a 40 y de 0,30–0,32 s con M_tot ≥ 80;
    - ruidosa: 0,61–0,72 s por fila y proceso (≈ 0,2 s de reloj por fila).
  - **Estimación para G4 con 3 procesos y la máquina libre:**
    - limpias: 0,96 M × 0,114 s ≈ 30 h;
    - ruidosas: 0,20 M × 0,2 s ≈ 11 h. Es una cota superior, porque el camino ruidoso de G3T reconstruía además la fila de G3 para G-REC.
    - Total ≈ 40 h. Con 2 procesos, ≈ 60 h.
  - **Memoria:** 0,7–0,85 GB por proceso.
- **Disco local:**
  - unión de limpias ≈ 95 GB y ruidosas ≈ 20 GB (98 304 B por fila);
  - conjuntos materializados por objetivo ≈ 2 × 86 GB;
  - en total ≈ 290 GB de los 12 TB libres.
- **VM:** ≈ 86 GB por experimento.
  - Después de la limpieza quedan ≈ 150 GB libres: cabe un experimento cada vez.
  - Para entrenar los dos a la vez habría que borrar también `m13_13_g3t/memmap/X_clean_train.npy` (37 GB; el original está en local). Se decide aparte.
- **GPU.** 5a: 538 s por época de 1562 pasos (0,344 s por paso), con espera de datos de ≈ 1 s por época; 11,9 h de entrenamiento en total.
  - G4 tiene los mismos 124 960 pasos: ≈ 11,9 h de entrenamiento más ≈ 0,4 h de val y monitor (40 épocas × 2 × 16 s).
  - Total ≈ 12,3 h por modelo, en una GPU.

## Enmienda 1 (21-11-2026, antes de generar): puerta G-VIS

Sobre la salida de `signal --limit 64` (filas espaciadas de un fragmento de la unión de train), para cada fila r:
E_r = Σ_d Σ_t L(s)_{r,d,t}² (float64, desde el float16 guardado, antes del z-score) y κ_r = E_r / ρ_r²,
con ρ_r el SNR objetivo de la fila (proposal/train.npz).
- G-VIS PASA si (i) la mediana de κ por bin de M_tot (< 80, 80–180, 180–250) está dentro de ±10 % de la mediana global
  y (ii) ninguna fila tiene κ < 0,5 × mediana global.
- Descriptivo: κ en 64 filas de G3T (X_clean_train de m13_13_g3t, SNR objetivo de la fila de G3), y figuras de L(s) y del
  X ruidoso de filas de masa baja, ~150 y > 220 M☉, por detector.
- Si G-VIS falla, no se lanza la generación completa; se decide en el chat.
Motivo: el blanqueado es lineal y normalizado, así que E ∝ ρ²; un κ que dependiera de la masa indicaría que el procesado
pierde energía de la señal de forma dependiente de la masa (en especial a M_tot > 180, que G3T no cubría).

## Enmienda 2 (22-11-2026, tras el FAIL de G-TAPER-G4 y antes de generar señales)

**Hechos.** Informe del 21-11 (archivado como `gate_taper_v1_FAIL_20261121`); diagnóstico del 22-11.
- M_tot ≤ 180: 9/1398 filas con mismatch > 1e-4 (0,64 %; G3T 0,7 %) y ninguna > 8e-4.
- M_tot > 180: fracción > 1e-4 de 16 / 39 / 53 / 60 % en 180–200 / 200–220 / 220–235 / 235–251; 31 filas > 8e-4,
  todas con χ_eff < −0,5 y M_tot > 220; con χ_eff ≥ 0,2, 0 %. El escalón inicial cumple en todas las filas.
- 2 errores (M_tot 10,4 y 11,5; q ≈ 1; χ_eff +0,68 y +0,82): LAL «Ringdown frequency > Nyquist frequency»
  con Δt = 1/4096 s y f_lower 30, 15 y 10 Hz. Es el mismo límite por el que G1 rechazó 4 de 500 000 fuentes
  (`scripts/repair_m11_final_invalid_waveforms.py`).

**Cambio 1: construcción de la señal según la masa.**
- S15 (desde 15 Hz, taper hasta 25 Hz) si M_tot ≤ 180, idéntica a G3T.
- S10 (desde 10 Hz, taper hasta 14 Hz, misma ventana de Kaiser β = 8 y misma alineación con A) si M_tot > 180.
- Motivo: a q y espines fijos, la forma de onda depende de la frecuencia a través de M·f. S10 a M_tot ≤ 250 tiene
  M·f del taper ≤ que S15 a M_tot = 140, donde la puerta pasó con todos los espines, y acaba 16 Hz por debajo
  de la banda (S15, 5 Hz).
- Corte en 180: dentro del rango de G3T, la construcción de G4 es la de G3T. En 150–180, el mismatch S15–Sref
  tiene mediana 6,6e-6 y máximo 1,9e-4.
- El corte depende solo de θ (M_tot en el sistema del detector).

**Cambio 2: G-GEN, fuentes generables.**
- Una fila es generable si SEOBNRv4_opt se genera con el Δt de producción. Se comprueba con f_lower = 100 Hz,
  porque la condición de Nyquist no depende de f_lower (verificado a 30, 15 y 10 Hz).
- Se comprueban todas las filas de la propuesta con M_tot < 16. Ajustes de remanente y QNM sitúan la última fila
  no generable en M_tot ≈ 13,1. Por encima de 16 se supone generable; si una no lo fuera, la generación se para con error.
- Objetivos efectivos: p_T · 1[generable] / Z_T. La aceptación de cada objetivo exige además que la fila sea
  generable, y el corte toma las primeras N filas aceptadas y generables, en orden de propuesta.
- G-PRIOR (KS) se calcula sobre el corte sin la exclusión. G-GEN informa, por objetivo y split, del número de filas
  excluidas y de su región (M_tot, q, χ_eff).
- Criterio: exclusión ≤ 0,5 % por objetivo. Esperado (ajustes aproximados): T_F ≈ 0,06 %, T_R ≈ 0,001 %,
  G3T ≈ 0,001 % (G1: 0,0008 %).
- La región excluida es la misma región física, ausente también de G3T, así que no sesga la comparación.
  Trunca la uniformidad de T_F en 𝓜 ≈ 4,4–5,8; queda documentado.

**Cambio 3: G-TAPER-G4b.**
- La misma regla de selección de §2.5.2 (semilla 151105) sobre la propuesta nueva.
- Cada fila compara su construcción de producción con su referencia: S15 frente a Sref (10 → 14 Hz) si M_tot ≤ 180;
  S10 frente a S7 (7 → 9,8 Hz) si M_tot > 180.
- Criterios de la enmienda 2 de M13.13: ≤ 8e-4 por fila, escalón ≤ 2e-3, como mucho un 2 % > 1e-4 en las 2000
  filas; un error cuenta como fallo. Predicción: PASA. Si falla, se para.

**Archivo.** La propuesta y la puerta v1 se renombran y se conservan. `sample` se repite con las mismas semillas.