# M12.5.2 — Protocolo de referencia de información (diseño, sin ejecutar)

Estado: propuesta científica; no se ha implementado, entrenado ni ejecutado una referencia. M12.5.1-A es el piloto de 576 observaciones. M12.5.1-B amplía el diagnóstico condicional a 25 fuentes diseñadas y 12.800 observaciones. Ninguno identifica por sí mismo el error irreducible de G1. Este documento fija las distinciones, controles y criterios que deberán registrarse antes de evaluar una referencia.

## 1. Contrato auditado y trazabilidad

Las fuentes normativas son `configs/generation/generate_m11_final_G1_500k.json`, `scripts/prepare_m11_final_manifest.py`, `scripts/generate_m11_final_G1.py`, `src/sampling.py`, `src/waveform.py`, `src/detectors.py`, `src/windowing.py`, `src/injection.py`, `src/snr.py`, `src/processing.py` y `src/models/dataset.py`. Los hashes y versiones auditados se comprueban en `scripts/m12/m12_5_common.py`. El gate `reproduce_m12_5_1_g1.py` verifica las filas train 0, 1 y 5830, incluyendo una reparación histórica; comprueba también los adaptadores `prepare_signal` y `observation` del piloto.

El gate exitoso no demuestra validez universal del generador, de una función de verosimilitud o de una referencia posterior. Sus tolerancias permanecen `atol=rtol=1e-5`, error L2 relativo ≤1e-6 y las comprobaciones temporales originales. Los módulos incluidos en su identidad no se modifican para ampliar el estudio. Los archivos congelados, escaladores train-only y checkpoints M11, M12.1, M12.2 y M12.3-S1 permanecen inmutables.

## 2. Distribución conjunta: qué significa «G1»

Definir una variable latente Z que contenga masas, espines, orientación, cielo, polarización, entorno PSD, bloque/crop, semilla o variable de colocación y objetivo de SNR. Una realización de ruido N completa el simulador. Los targets, en el orden histórico, son

- chirp_mass = (m1 m2)^(3/5) / (m1+m2)^(1/5), en masas solares del marco detector;
- total_mass = m1+m2, en las mismas unidades;
- chi_eff = (m1 spin1z + m2 spin2z)/(m1+m2), adimensional.

El mecanismo G1 no es un prior astrofísico volumétrico estándar:

1. Las dos masas se muestrean uniformemente en [5,90], se ordenan y se permutan coherentemente sus espines. Los espines alineados se muestrean independientemente en [-1,1]. Esto induce correlaciones y una distribución no uniforme sobre los tres targets. No sustituirla por tres priors marginales independientes.
2. cos(inclinación), RA, sin(dec) y polarización siguen el muestreador histórico. Fase, espines transversos y excentricidad tienen los valores cero históricos; no introducir marginalizaciones ajenas a G1 como si preservaran su prior.
3. El muestreador consume un sorteo de distancia, pero la distancia de referencia es 1000 Mpc. Para cada fuente se sortea un SNR objetivo uniforme [10,25] y se determina d = 1000 rho_ref/rho_objetivo. La distancia final depende de masas, espines, orientación, PSD y ventana; no se recorta al intervalo nominal [200,5000] del muestreador. Un prior independiente en distancia cambia la distribución conjunta. Una parametrización alternativa con distancia necesita el Jacobiano y soporte inducidos; es preferible retener la variable SNR del generador.
4. El banco de 29 entornos, 292 bloques y 4380 crops tiene pertenencia disjunta por split. La asignación histórica es uniforme en entorno, luego bloque y crop, no necesariamente uniforme en todos los crops. La mezcla de entornos de train y los entornos de evaluación son distintos: especificar el dominio de cada riesgo.
5. La colocación `random_contained` depende de duración y proyección. El GPS real se determina con el orden de operaciones histórico, se reproyecta y se comprueba la ventana físicamente admisible. Un prior independiente uniforme sobre tiempo de coalescencia no reproduce automáticamente este mecanismo. Condicionar en el GPS observado puede informar sobre la duración; fijarlo mediante una intervención es otra distribución.
6. SEOBNRv4_opt, proyección H1/L1/V1, señal completa en contexto y tratamiento histórico de truncación se mantienen. Las cuatro reparaciones train forman parte del archivo finito congelado, no definen por sí mismas una ley poblacional continua. Diferenciar reproducción de ese archivo y simulación de la ley generativa idealizada; registrar explícitamente cómo se tratarían futuros fallos físicos sin descartes selectivos ocultos.
7. El ruido es gaussiano condicionado a PSD empíricas estimadas en 128 s, mediante Welch de 8 s, Hann y la implementación histórica. Esas PSD son entradas fijas del simulador G1, no una PSD verdadera desconocida que pueda marginalizarse sin cambiar el modelo.

Definir antes de ejecutar si el riesgo objetivo corresponde a una mezcla train-G1, a la mezcla de entornos held-out, o a una distribución condicional de control. No presentar estas mezclas como una sola distribución idéntica ni usar la selección de 25 fuentes como prior poblacional.

## 3. Observables y auxiliares: contrato explícito

D denota las tres series de strain inyectado **antes** de whitening/filtros/z-score: contexto de 4,8125 s a 4096 Hz, 3×19712 valores float64. Incluye ruido gaussiano y la proyección de señal completa en ese contexto; no es una observación real de GWOSC.

U denota la salida de whitening, filtros y crop antes del z-score. X = T(D,E) es el tensor CNN: 3×16384 valores float32, 4 s, normalización temporal por detector `(U-mean(U))/(std(U)+1e-6)` con desviación ddof=0. T depende del entorno PSD E. El contexto extra, filtros, crop, pérdida de medias y normalización pueden cambiar información. La invariancia de escala no es exacta por epsilon y redondeo. No asumir que toda información de amplitud desaparece, ni que toda diferencia D–X se debe al z-score.

Registrar en cada brazo una lista exhaustiva de auxiliares A: PSD y su resolución, entorno, GPS/crop, cielo, inclinación, polarización, objetivo SNR, distancia, escalas de normalización y cualquier verdad usada para fijar nuisances. El CNN congelado recibe únicamente X; no recibe de forma explícita ninguno de estos auxiliares. Aunque pueda inferir parcialmente algunos, eso no equivale a proporcionarle sus valores verdaderos.

| Brazo | Entrada disponible al estimador | Pregunta autorizada |
|---|---|---|
| CNN congelado | X | Riesgo efectivo del estimador histórico |
| Referencia X primaria | X solamente | Eficiencia respecto de la misma información y prior declarado |
| Control X,A | X y un A explícito | Valor adicional de auxiliares; no sustituye la comparación primaria |
| Referencia D,A | D y exactamente el mismo A que el control X,A | Efecto conjunto del procesamiento para un experimento común |
| Referencia U,A | U y el mismo A | Separación adicional del z-score respecto de filtros/crop |
| Control D con nuisances verdaderas | D más las nuisances autorizadas | Diagnóstico condicional/oráculo; no baseline directo del CNN X-only |

Si E no es observable para la comparación primaria, la referencia X debe integrar la mezcla de E; entrenarla sobre un solo entorno conocido cambia el prior. Para comparar D y X sin A, T depende de una E oculta y no basta invocar una relación determinista D→X. La desigualdad de procesamiento de datos sí se aplica a (D,E,A)→(X,E,A), bajo la misma ley. Separar explícitamente esa comparación del problema original X-only.

## 4. Identidades de riesgo y lo que no son

Para cada target escalar Y con segundo momento finito y un predictor f, bajo una **misma distribución conjunta fija**, el **mismo observable X** y el **mismo prior**:

\[
\mathbb E[(Y-f(X))^2]
=\mathbb E[\operatorname{Var}(Y\mid X)]
+\mathbb E[(\mathbb E[Y\mid X]-f(X))^2].
\]

La esperanza condicional m_X=E[Y|X] minimiza el riesgo cuadrático. La primera parte es el riesgo Bayes bajo ese experimento, no una constante física universal. La identidad vale por target; un riesgo sumado requiere escalas fijadas de antemano, por ejemplo las desviaciones train-only, sin ocultar resultados individuales. Si hay auxiliares, reemplazar X por (X,A) en **todos** los términos. Cambiar prior, selección SNR, nuisances, resolución temporal o información de PSD cambia el experimento. Los objetivos ponderados de M12.1/M12.2 y el muestreo de M12.3 pueden cambiar el óptimo de entrenamiento; ninguna discrepancia frente a m_X prueba capacidad arquitectónica insuficiente.

Si X es transformación de D dado el mismo A, la ley de varianza total permite R*_D,A ≤ R*_X,A. La diferencia cuantifica pérdida de información para esas variables, ese prior y esa transformación. Un orden entre dos riesgos de estimadores aproximados no demuestra el orden de sus riesgos Bayes ni estima automáticamente su diferencia.

Para una fuente theta, SNR y entorno fijos, M12.5.1 estima otra identidad:

\[
\widehat{MSE}_{\theta}=\widehat b_{\theta}^{\,2}
+\frac{R-1}{R}s^2_{f,\theta},\qquad
\widehat b_{\theta}=\overline f-y(\theta).
\]

`s²` es la varianza muestral **insesgada** entre ruidos; el factor finito R es obligatorio. La varianza describe fluctuación del predictor al cambiar ruido con verdad fija. Una posterior varía los parámetros posibles con observación fija. Predicciones estables no implican posteriors estrechas. Sesgo persistente no prueba falta de capacidad: puede reflejar prior, selección, objetivo de entrenamiento, optimización, normalización o información limitada. Tampoco el cuadrado del sesgo muestral es un estimador insesgado del sesgo poblacional al cuadrado.

## 5. Referencia física con strain y verosimilitud gaussiana

Primero desarrollar un control con entorno PSD fijo y nuisances de orientación conocidas; inferir masas/espines, no fijar los targets verdaderos. Definir si la ley es G1 **condicionada** en esos auxiliares o una intervención controlada. En particular, no confundir fijar GPS con condicionar en el GPS dependiente de duración del generador. La rejilla de 25 fuentes es una selección de puntos de evaluación; no convertirla en un prior discreto que revele al estimador las respuestas posibles.

Después marginalizar progresivamente cielo, inclinación, polarización, colocación y SNR latente conforme al mecanismo G1 declarado. La distancia se deriva de SNR y demás parámetros, no se introduce como nuisance independiente. Fase permanece cero en la referencia G1; una extensión con fase libre se reporta como otro experimento. El entorno se fija o marginaliza según la columna A declarada, nunca de forma implícita.

La forma gaussiana formal es `-1/2 (D-h)^T C_E^{-1}(D-h)` más normalización y prior. Implementarla exige comprobar que C_E corresponde a las series finitas generadas: PSD unilateral, delta_f, normalización FFT, DC/Nyquist, duración, periodicidad, bins anulados, correlaciones, rango de frecuencias y tratamiento de segmentos. C_E puede requerir trabajar en un subespacio no singular. El inner product estándar de una likelihood en frecuencia no queda validado por escribir esa fórmula. Un corte 30–512 Hz debe declararse como selección de observable; no usar implícitamente otros bins o tiempos que un brazo comparador no recibe.

Validar likelihood con ruido conocido, residuos normalizados, log-likelihood ratios en fixtures controlados, recuperación de parámetros y comparación con una implementación independiente. No usar los filtros de whitening de G1 otra vez sobre una likelihood que ya incorpora la misma covarianza sin derivar el resultado. La señal del template debe respetar el contexto y proyección históricos. La PSD empírica del banco es fija bajo G1; estudiar incertidumbre de PSD sería una extensión separada.

Usar como predictor la media posterior física de cada target, calculada transformando cada muestra conjunta; no la transformación de la media de parámetros ni el MAP/mediana como sustituto del óptimo MSE. En masas/espines degenerados verificar modos, límites y soporte, no sólo un resumen marginal de chirp_mass.

## 6. Referencia con la misma X

Una opción es SBI entrenada con simulaciones del **mismo** generador G1 y la transformación T congelada, con entrada X-only. Otra es una integración o likelihood numérica independiente si se logra derivar y validar la distribución inducida por T. La normalización no invertible y el filtrado impiden identificar sin demostración una likelihood gaussiana de D con una likelihood de X. No reconstruir strain mediante escalas verdaderas y llamarlo X-only.

Una posterior aprendida o un regresor alternativo no son automáticamente Bayes óptimos. Un predictor independiente puede demostrar que una CNN concreta tiene margen de mejora si reduce riesgo con la misma información; no mide por sí solo el riesgo irreducible. Para usar la media/varianza posterior como referencia se necesita evidencia de exactitud de la posterior, no sólo menor RMSE. SBC correcto tampoco demuestra por sí solo informatividad: una salida igual al prior puede superar ciertos controles marginales y seguir siendo inútil.

Evaluar al menos dos métodos o aproximaciones independientes donde sea factible, incluyendo casos reducidos con integración verificable. Auditar sensibilidad al tamaño de simulación, capacidad y optimización de la referencia y replicaciones de semillas de referencia. No reutilizar las CNN congeladas como único mecanismo de validación de una referencia que se pretende independiente. Toda simulación nueva de M12.5.2 requerirá una fase autorizada separada: este documento no la ejecuta.

## 7. Separación de roles y dominios

- **Train:** ajustar parámetros, referencias de prior/entorno y estadísticas permitidas; conservar los escaladores históricos cuando se expresen riesgos normalizados. Nuevas simulaciones de referencia deben tener IDs, seeds y pertenencia explícitos, sin colisiones con evaluación.
- **Val:** elegir presupuesto y configuración de referencia, diagnosticar fallos y congelar criterios. La selección adaptativa no se reporta como confirmación held-out. Mantener intactos checkpoints CNN y decisiones históricas.
- **Cal:** después de congelar referencia y predictor, calibrar sólo los ajustes de cobertura declarados. No usar cal para seleccionar arquitecturas. Separar calibración posterior y conformal: una no prueba la otra; no reutilizar cuantiles conformales M11 para otro predictor.
- **Test:** evaluación final una vez bloqueados diseño, referencias y reglas. Los entornos test distintos exigen reportar generalización de entorno, además de error de estimación; no llamarlo riesgo iid de la mezcla train sin justificación. Nunca seleccionar variantes en test ni eventos reales.

Un estudio poblacional nuevo necesita fuentes sorteadas de la ley declarada, con ponderación explícita si se estratifica. Si se reutilizan entornos o fuentes, usar inferencia jerárquica por fuente/entorno acorde al estimando; 128 ruidos de una fuente no son 128 sistemas independientes. Un único entorno no permite estimar varianza entre entornos. El estudio diseñado 1-B sirve para localizar fallos, no para estimar riesgo Bayes G1 por promedio uniforme de su rejilla.

## 8. Evaluación, precisión Monte Carlo y criterios previos

Reportar por target: riesgo MSE físico y normalizado train-only, sesgo, colas, cobertura/anchura posterior, diagnóstico de soporte y diferencias emparejadas de pérdida. MSE y RMSE no se intercambian en descomposiciones de riesgo. Registrar también el punto de predicción (media posterior), ESS relevante por target y errores Monte Carlo de medias, varianzas y riesgos.

Controles posteriores: SBC por rankings en simulaciones independientes, cobertura a 50/80/90/95%, estratificada por SNR, masas, spin y entorno donde el tamaño lo permita; PIT o ranks aleatorizados si corresponde; comprobaciones predictivas y recuperación en casos verificables. Usar intervalos binomiales y corrección predeclarada por multiplicidad para cobertura; la cobertura puntual no garantiza cobertura condicional uniforme. No evaluar sólo la cobertura en la rejilla diseñada ni omitir casos fallidos.

Para referencias de muestreo: múltiples inicializaciones/cadenas o ejecuciones independientes, diagnóstico de mezcla y modos, ESS efectivo (no número bruto de draws), estabilidad al presupuesto y comparación de estimadores. Para SBI: semillas independientes, simulación held-out, controles de prior y posterior y sensibilidad a capacidad/presupuesto. No interpretar ESS o pérdida de entrenamiento como validación completa.

Criterios propuestos que deben bloquearse antes de la evaluación confirmatoria:

1. Gate físico y contrato de observables superados; ningún fallo de soporte o dato excluido silenciosamente. Si falla, no hay conclusión de información.
2. En casos verificables, acuerdo de medias posteriores entre referencias dentro de dos errores Monte Carlo combinados, sin patrones sistemáticos por target/SNR. Si varias discrepancias aparecen, investigar multiplicidad y dependencia; este criterio solo no demuestra exactitud.
3. Error Monte Carlo del contraste de riesgo ≤10% de su error de muestreo de fuentes, y estabilidad de conclusiones al duplicar presupuesto en una muestra val preespecificada. Si no se alcanza, informar contraste no resuelto; no ocultar el error numérico dentro de bootstrap de fuentes.
4. Intervalos simultáneos sobre diferencias de riesgo de los tres targets: Bonferroni con intervalos marginales de 98,333…% para una familia de tres. Declarar otra familia para análisis condicionados adicionales; no usar 1200 comparaciones puntuales como descubrimientos confirmatorios.
5. Margen práctico propuesto por target: tau_j = 1% del MSE M11 en la **distribución de evaluación declarada**, estimado y fijado con val antes de test. Es una propuesta nueva para riesgo MSE, no el guardrail histórico de 2% de RMSE y no una certeza estadística. Fijar N mediante precisión planificada con val, no detención opcional al obtener significación. No adoptar ese margen si la precisión de referencia no puede resolverlo.
6. La equivalencia requiere que todo el intervalo esté dentro de [-tau_j,+tau_j]; ausencia de significación no basta. Una mejora práctica requiere que el límite inferior del contraste favorable supere tau_j. Reportar además efecto continuo y sensibilidad al margen predeclarada.

## 9. Decisiones científicas permitidas

| Evidencia observada tras validación | Conclusión limitada permitida | Lo que falta antes de una recomendación |
|---|---|---|
| Referencia X fiable mejora CNN con mismo prior/observable; exceso de riesgo >tau | Limitación del estimador CNN concreto en ese dominio | Separar objetivo ponderado, optimización, datos y capacidad; no identifica arquitectura como causa |
| Referencias D,A y X,A fiables, diferencia >tau con mismo A y ley | Pérdida asociada a T en ese experimento | Control U,A para atribuirla a normalización en vez de contexto/filtros/crop |
| CNN y media posterior X equivalentes, posterior validada mantiene riesgo sustancial | Evidencia compatible con limitación de información de X bajo ese prior | Cotas/precisión de referencia y cobertura poblacional; no extrapolar a otros observables o priors |
| CNN estable entre ruidos o sesgo alto a SNR 24,5 sin referencia | Robustez del punto estimado o sesgo condicional persistente | No hay separación de error irreducible y error de estimador |
| D con PSD/cielo verdaderos supera CNN X-only | Beneficio combinado de información/estimador | Control X con idénticos auxiliares o referencia X-only antes de atribuir causas |
| Dos referencias discrepan o fallan calibración | Resultado inconcluso | Validar la referencia antes de comparar arquitecturas |

No recomendar rediseño arquitectónico sólo por la rejilla, pendientes de regresión o contracción hacia medias train. El contrato mínimo es: misma información/prior, referencia validada, mejora resoluble y reproducible de riesgo en fuentes independientes, análisis de fallos por etiqueta/SNR/entorno, y controles de objetivo/optimización. Cualquier experimento de arquitectura posterior cambia un único factor y respeta la política M12; requiere autorización propia.

## 10. Etapas concretas, factibilidad y condiciones de parada

**Etapa 0 — Congelar experimento.** Publicar especificación versionada de p(Z,E,N), observables/auxiliares por brazo, target de riesgo, particiones, seeds, fallos físicos, presupuesto máximo y criterios anteriores. Auditar el prior inducido de distancia/tiempo con el generador. No reutilizar la selección del piloto como validación poblacional.

**Etapa 1 — Validación numérica reducida.** En controles train con pocas dimensiones inferidas, comparar likelihood contra fixtures y una integración suficientemente resuelta. Incluir anti-aligned spins y degeneraciones masa-spin; verificar soporte temporal y bordes. Sólo después fijar rangos de presupuesto usando val. Un fallo detiene interpretaciones, no se resuelve relajando el gate.

**Etapa 2 — Raw D condicionado.** Aplicar referencia física a subconjunto val predefinido, con nuisances fijas declaradas. Medir evaluaciones de waveform/likelihood, memoria y ESS por hora; esos tiempos se medirán entonces, no se inventan ahora. Estudiar coste de marginalizar nuisances por etapas. Conservar todos los fallos y límites de presupuesto.

**Etapa 3 — Referencia X independiente.** Evaluar viabilidad de SBI u otra aproximación. Entrenar sólo dentro de una futura autorización y presupuesto. Validar posterior, calibración y medias contra casos reducidos y referencias independientes, bloqueando hiperparámetros con val. Si no hay validación suficiente, reportarla como estimador alternativo, no como Bayes óptimo.

**Etapa 4 — Controles de información.** Comparar D,A/U,A/X,A con auxiliares igualados, y distinguirlos de X-only. Mantener los mismos datos subyacentes y pares de realizaciones. No asignar a z-score toda diferencia cuando cambian duración o bandas.

**Etapa 5 — Evaluación bloqueada.** Tras cal cuando corresponda, ejecutar una vez el protocolo sobre fuentes poblacionales independientes y entornos declarados; reportar riesgo, intervalos simultáneos, error MC, fallos y reproducibilidad. La rejilla completa puede complementar con diagnóstico condicionado, sin inferencias poblacionales.

**Etapa 6 — Decisión.** Aplicar la tabla de decisiones. Si la referencia o precisión falla, el resultado es inconcluso y se documenta. Si hay evidencia de limitación del estimador, formular después una hipótesis de intervención única; ninguna nueva CNN se implementa como parte de este documento.

El coste dominante de una likelihood física es número de observaciones × evaluaciones por observación × costo de waveform/proyección; no se estima tiempo sin medición. SBI desplaza coste hacia simulación, entrenamiento y validación, con riesgo de error de aproximación. Registrar hashes de código, configuración, PSD, datos, checkpoints de referencia, seeds y resultados de ejecuciones independientes. Considerar imposibilidad práctica de referencia exacta, likelihood mal especificada, priors incorrectos, multimodalidad, soporte truncado, errores de escala/unidades, calibración marginal engañosa y cambios de entorno como fallos explícitos, no como evidencia de límite informacional.

## Apéndice A. Ejecución de M12.5.1-B (implementada, no ejecutada aquí)

Los nuevos scripts son `scripts/m12/run_m12_5_1_full.py` y `scripts/m12/analyze_m12_5_1_full.py`. Se mantienen sin cambios el gate, common, piloto, su análisis y notebook. La generación utiliza el entorno histórico `gw-env`; `load_frozen` exige las versiones y la procedencia exactas del gate. No hay override. Inferencia usa `cbc_torch`, batches compartidos, `eval`/`no_grad`, carga compatible con torch 1.9.1 y verificaciones SHA256/estado antes y después. No se importa física durante inferencia ni se abren HDF5/checkpoints durante análisis.

Desde la raíz del repositorio, con `CBC_PE_DATA_ROOT` ya definido mediante el mecanismo del proyecto:

```bash
# Sólo cálculo de tamaños y operaciones: no carga datos ni modelos.
conda run -n gw-env python scripts/m12/run_m12_5_1_full.py plan --batch-size 16

# Ejecutar por el usuario en el mismo entorno/rutas del gate exitoso.
conda run -n gw-env python scripts/m12/run_m12_5_1_full.py generate --run \
  --gate "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_1/gate_v1/gate.json" \
  --output-dir "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_1/full_v1"

# Tras transferir los artefactos verificados al root de la VM (transferencia manual).
conda run -n cbc_torch python scripts/m12/run_m12_5_1_full.py infer --run \
  --study "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_1/full_v1" \
  --device cuda --batch-size 16 \
  --output-dir "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_1/full_inference_v1"

# Local CPU: devolver receipts y CSV/JSON de full_v1, y full_inference_v1.
conda run -n gw-env python scripts/m12/analyze_m12_5_1_full.py \
  --study "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_1/full_v1" \
  --inference "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_1/full_inference_v1" \
  --bootstrap 2000 \
  --output-dir "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_1/full_analysis_v1"

# Tests sintéticos dirigidos; no generación/inferencia ni notebooks.
PYTHONDONTWRITEBYTECODE=1 conda run -n gw-env python -m unittest discover \
  -s tests -p 'test_m12_5*.py' -v
```

Todos los destinos son nuevos y exclusivos. Si existen, el comando falla: usar otro tag revisado, nunca sobreescribir el piloto. No hay reanudación automática ni borrado de una ejecución parcial. Un recibo `complete=False` impide inferencia/análisis posterior. El análisis exige también la versión exacta del código registrada por la generación; conservarla junto con el experimento.

## Apéndice B. Salidas, recursos y límites de 1-B

Diseño determinista: orden masas, espines, SNR y realización. IDs `M1251B_SRC_00..24`; condition_id 0..99; observation_id = 128 condition_id + realization_id. Seed maestro 125101, PCG64, rechazo global de colisiones y de los 1728 seeds del piloto. Se registran los 38400 seeds de detectores, hashes y rechazos. Esto evita reutilización A/B; no prueba independencia matemática del PRNG ni garantiza disjunción con todos los seeds históricos G1, que no se escanean. El bootstrap usa seed 125102 y namespaces por condición, etiqueta y R.

| Array | Shape y dtype | Bytes lógicos sin comprimir |
|---|---|---:|
| X | 12800×3×16384 float32 | 2.516.582.400 |
| raw_strain | 12800×3×19712 float64 | 6.055.526.400 |
| signal_context | 100×3×19712 float64 | 47.308.800 |
| seeds, tres IDs, y, PSD | tipos explícitos en el script | 1.201.200 |
| Total arrays | sin cabeceras HDF5 ni CSV/JSON | 8.620.618.800 |

Aproximadamente 8,03 GiB lógicos. LZF y chunks de una observación permiten I/O acotado; no se promete compresión de ruido gaussiano. Se exige espacio libre de al menos dos veces esos arrays (17.241.237.600 bytes ≈16,06 GiB) antes de generar; es margen operacional, no cota garantizada del filesystem. Cada copia local/VM requiere espacio propio. Se hacen lecturas secuenciales completas para SHA256 al finalizar generación y antes de inferencia, sin cargar todo en RAM.

Generación produce seis archivos: `design.csv` (100 filas), `design.json`, `conditions.csv` (100 filas con parámetros físicos, distancia, SNR de detectores/red, GPS y PSD), `observations.csv` (12800 filas con IDs/seeds/hashes raw y X), `observations.h5` y `generation.json`. Inferencia produce `predictions.csv` (153600 filas: 12800×4×3) e `inference.json`. El tamaño textual exacto depende de representación numérica y rutas; se registra después de escribir, no se presenta como tamaño exacto previo. Presupuestar decenas de MB para predicciones y tablas escalares, además del margen HDF5.

Análisis produce cinco archivos: `conditional_metrics.csv` (1200 filas, R=128), `paired_differences.csv` (1800), `convergence_metrics.csv` (4800), `convergence_paired.csv` (7200) y `analysis.json`. Las tablas de convergencia incluyen R=128, por lo que son deliberadamente redundantes con las principales. Métricas: media, sesgo firmado/absoluto, varianza insesgada, desviación, MSE/RMSE condicional, componentes finitos y fracción de MSE atribuida al sesgo cuadrado; cuantiles 5/50/95 del error, 90/95 del error absoluto y máximo. Fracción indefinida si MSE=0 se exporta como `nan`. Unidades de varianza/MSE son el cuadrado de las unidades de etiqueta. Seis contrastes de checkpoints por condición/etiqueta, signo `a_minus_b`.

Los CI son percentiles bootstrap puntuales aproximados al 95%, remuestreando IDs de ruido dentro de condición y conservando emparejamiento entre modelos. Para |sesgo| se transforma el intervalo del sesgo firmado; el contraste de |sesgo| usa envolvente Bonferroni de dos intervalos al 97,5%, posiblemente amplia. No se reclaman intervalos simultáneos para toda la rejilla. Cuantiles de cola y fracciones de componentes son descriptivos, sin CI automáticos. R=16/32/64/128 son prefijos anidados, no cuatro experimentos independientes. El bloque temporal máximo de bootstrap es 128 remuestreos: 128×128×4 float64 ≈0,5 MiB por array de muestras, además de temporales y distribuciones de métricas. Los CSV completos caben en memoria de CPU ordinaria; no se recorren arrays de strain en análisis.

El modelo congelado tiene 236435 parámetros; cuatro copias float32 ocupan 3.782.960 bytes. Con batch 16, la entrada ocupa 3.145.728 bytes y la mayor activación individual, 16×128×2049, ocupa 16.785.408 bytes. Su suma con pesos, 23.714.096 bytes (~22,6 MiB), es **sólo un suelo de memoria**, no la memoria pico GPU. Faltan intermediarios simultáneos, workspace cuDNN, contexto CUDA y allocator. No hay gradientes ni estado de optimizador. La compatibilidad real P100/torch 1.9.1 no se ha probado en esta implementación; comenzar con batch 16 y registrar memoria medida en la ejecución autorizada.

El forward tiene 476.287.936 MAC de convoluciones/densas por ejemplo (excluye normalización, pooling y activaciones). Son 51200 ejemplos-modelo, 3200 forwards a batch 16 y 24.385.942.323.200 MAC (~48,77 billones de operaciones aritméticas si se cuentan dos por MAC). Generación requiere 100 preparaciones de señal, 200 llamadas de waveform en el adaptador histórico y 38400 realizaciones de ruido/procesamientos de detector. No se asignan segundos u horas a estos conteos sin mediciones.

Limitaciones: un entorno PSD/extrínseco train fijo, 25 fuentes diseñadas y un checkpoint por intervención; no muestreo poblacional, ni evidencia de reproducibilidad entre semillas de entrenamiento. Validez física de cada punto y memoria pico quedan pendientes de ejecución. El código aborta en errores, no adapta la rejilla a predicciones. Los raw arrays preservados posibilitan investigar una futura likelihood; no garantizan que una likelihood estándar coincida con el proceso finito de ruido. La separación entre error irreducible y de estimador sigue pendiente de M12.5.2.
