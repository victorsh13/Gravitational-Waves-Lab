# M14.6 — Información incremental de un banco de filtros adaptados (sondas sobre el residuo)

Escrito el 19-11-2026, antes de calcular.

## Pregunta
¿Hay información en X, accesible con filtros adaptados de formas de onda físicas, que el modelo final (M14.5b) no usa? ¿Crece con el número de plantillas? Objetivo decisional: si merece la pena invertir en un banco mayor o con inicialización física.

## Fundamento
La red aproxima la media posterior, μ(X) ≈ E[θ|X]. Si fuera exacta, E[y − μ | f(X)] = 0 para cualquier función f de X y ninguna sonda evaluada fuera de muestra predeciría el residuo. Un R² fuera de muestra claramente positivo del residuo a partir de variables f(X) demuestra error evitable recuperable con esas variables; mide la fracción del MSE de μ que se podría eliminar. Es una cota inferior: R² ≈ 0 no demuestra ausencia de información.
Parte del error evitable puede ser una descalibración de μ corregible con una función de la propia salida del modelo; la sonda A la mide y ΔR² aísla la información aportada por el banco.

## Variables
Para cada fila de val (40 000) y plantilla k: matched filter en cuadratura de los datos blanqueados con la plantilla blanqueada igual que los datos (entorno PSD y detector de la fila), normalizada, ρ_kd(t). Resúmenes: pico por detector (3) y pico de red sqrt(Σ_d pico_d²) (1).
Bancos: P32 ⊂ P128 ⊂ P512, plantillas SEOBNRv4_opt muestreadas del prior de entrenamiento (semilla 20261119, anidados); L32, banco aprendido de M14.5b.
Sonda de referencia A: variables de salida del modelo cuyo residuo se estudia: μ (3), σ (3) y las 3 correlaciones de Σ. Sondas con banco (B): las variables de A + las del banco.

## Residuos
Primario: r = y − μ de M14.5b en val, por etiqueta, en unidades estandarizadas. Secundario: M14.5a.

## Sonda
HistGradientBoostingRegressor con hiperparámetros fijos (max_iter 300, learning_rate 0,05, max_leaf_nodes 31, min_samples_leaf 100, l2_regularization 1,0, early_stopping con 10 % interno, random_state 0). Validación cruzada en 5 particiones aleatorias de val (semilla 0). R²_oof = 1 − Σ(r − r̂)²/Σ r² (frente a predecir 0). IC 95 % por bootstrap de filas (2000) sobre las predicciones fuera de partición. Celdas: global y por grupos de τ30 predicha D1–D3, D4–D7, D8–D10 (deciles de val como en M14.5). Test no se usa.
- Una sonda por (residuo, etiqueta, conjunto de variables), ajustada una sola vez con las 5 particiones sobre las 40 000 filas; en cada grupo, R²_g = 1 − Σ_g (r − r̂)²/Σ_g r² sobre las predicciones fuera de partición de las filas del grupo; bootstrap de filas dentro del grupo.
- Las 5 particiones son idénticas (misma semilla) para todos los conjuntos de variables y ambos residuos. Las diferencias entre conjuntos (ΔR² = R²_B − R²_A, y entre bancos) se dan con bootstrap emparejado: las mismas filas remuestreadas para ambos.
- Grupos de τ30: cortes = cuantiles de τ30 predicha por M14.5b en cal (como TAU_EDGES_CAL de M14.5), aplicados a la τ30 predicha por M14.5b en val; los mismos grupos de filas para el residuo de 5b y el de 5a.
- Se informa R²_A por sí mismo: fracción recuperable recalibrando la salida, sin información nueva.

## Comprobaciones previas (si falla alguna, se corrige y se repite sin mirar el resultado principal)
G1: con P512, R²_oof de la etiqueta 𝓜 estandarizada ≥ 0,8.
G2: residuo de M14.5b con las filas de las variables del banco permutadas (las de A sin permutar): |ΔR²_oof| < 0,005 en las tres etiquetas.
G3: la capa del banco reproducida coincide con el forward del checkpoint de M14.5b (diferencia relativa ≤ 1e-4 en 64 filas).

## Criterio primario (M14.5b, P512, 𝓜 y χ_eff en D8–D10)
Sobre ΔR² = R²_B − R²_A, con IC 95 % por bootstrap emparejado:
- Margen relevante: IC inferior del ΔR² ≥ 0,04 en 𝓜 o en χ_eff → se pre-registra un entrenamiento con banco mayor o inicialización física.
- Sin margen relevante: IC superior < 0,02 en ambas → no se invierte más en el banco.
- En otro caso: no concluyente. Regla de parada: no se hacen más sondas ni se cambian los hiperparámetros.
Umbral: 0,04 del MSE ≈ −2 % de RMSE, del orden de la ganancia de M14.2 en D8–D10 no resuelta frente al ruido de semilla.

## Secundarios (descriptivos; R² y ΔR²)
R²_oof y ΔR² frente a K (32, 128, 512); L32 frente a P32; residuo de M14.5a (pregunta de M14.2 sin ruido de semilla); M_tot; R²_A.

## Qué no puede concluir
Ausencia de información si R² ≈ 0; que el margen sea específicamente información de fase; nada sobre otras semillas.

## Coste
Plantillas en local (CPU); correlaciones de val en la VM (1 GPU, ≲ 1 h); sondas en local (CPU, 1 hilo, en paralelo al banco de M13.11).

## Implementación

### Datos y entornos PSD de val
- X de val: `processed/M13_G3_band1024_w16_500k/m13_13_g3t/fp16_memmap/X_val.npy` (40 000 × 3 × 16384, fp16, ya z-scoreado por fila y detector), filas `row_ids_val.npy` (= `idx_val` de `preds_val_G3T` de 5a y 5b, comprobado).
- Entorno PSD = `block_id` del manifiesto de G1 (`m11_final_G1_500k_manifest_repaired.csv`): una PSD de procesado fija por bloque (`psd_key = str(block_id)`, comprobado por `check_psd_key_per_block`). Val usa **28 bloques** (977–2774 filas por bloque; 292 en todo el dataset).
- PSD de procesado (`psds["proc"]` de `get_block_psds` en `scripts/generate_m11_final_G1.py`): Welch con segmentos de 8 s sobre los 128 s de strain O3b de `[psd_start, psd_end]` (`gwosc_cache/m11`), interpolada a `processing_delta_f`. El ruido gaussiano de G3/G3T se genera con esa misma PSD y se blanquea con ella.

### Blanqueado (reproducido en las plantillas)
- `SignalProcessor` de G3 (`src/processing.py`, spec de G1 con los overrides de `configs/generation/generate_m13_G3_band1024_w16.json`): `whitening_method = "psd"`, `inverse_spectrum_truncation` con filtro de 16 s (`hann`, f_low 30 Hz), división por √PSD, se quitan 8 s corruptos por lado; high-pass FIR 30 Hz y low-pass FIR 1024 Hz (orden 256, β 5); recorte a la ventana final de 4 s (16384 muestras); sin estandarización (el z-score por fila y detector se hace después).
- Las plantillas pasan por **la misma función** que la señal limpia de G3T: `m13_11_prepare_rows.process_clean` (inyección en un strain nulo de longitud de procesado + `g1.PROCESSOR.process_network` con las PSD del bloque), tras `g6.setup` (que inicializa la física de G3 y comprueba el contrato del manifiesto). El z-score no se reproduce: cada plantilla blanqueada se normaliza a norma unidad, y como X tiene varianza ≈ 1 por muestra, |c| ≈ SNR con ruido de escala 1.
- Forma de onda: S15 de G3T = `m13_11._opt_click_free` con las constantes de `m13_13_build_g3t` (f_lower 15 Hz, f_ref 30 Hz, taper Kaiser β = 8 hasta que la frecuencia instantánea cruza 25 Hz), SEOBNRv4_opt, Δt = 1/4096, de cara (ι = 0), distancia 1, coa_phase 0; G-EDGE (`edge_taper`, 0,5 s Kaiser) si empieza antes del segmento de procesado. La alineación de fase/tiempo con A de G3T no se aplica: en cuadratura el pico de |c| no depende de la fase, y la latencia se maximiza.

### Banco de M14.5b (L32)
- `TemplateBank` (`src/models/network.py`): K = 32, L = N = 16384, `template_i` y `template_q` **independientes y aprendidos** (inicializados en M14.2 con filas de train: I = L(s) z-scoreada del detector de mayor energía, Q = Hilbert(I); semilla 142001, rejilla 8 τ30 × 4 χ_eff; registrado en `run_info.json → procedure.template_bank`). Mismas plantillas para los 3 detectores.
- Cada plantilla se normaliza a norma unidad en el forward; correlación **circular** de longitud N con la muestra final (merger) en el índice 0.
- Salida: **|c| = sqrt(c_I² + c_Q² + 1e-12)**, no I y Q por separado; (B, 3·32, N) con canales (detector, plantilla).
- L32 se calcula reproduciendo esa operación (circular, como la capa) con las plantillas del checkpoint `M14_5b_4a_G3T_tbank_fullcov_s123_final.pt`; G3 compara la reproducción con `model.template_bank(x)` en las 64 primeras filas de val.

### Decisiones de implementación (no afectan al criterio; fijadas antes de calcular)
- **Parámetros de P512:** `default_rng(20261119).choice(train_idx, 512, replace=False)` sobre las filas de train del split congelado (`train_idx` del npz de split de M11): cada fila es un draw del prior de entrenamiento; se toman (mass_1, mass_2, spin_1z, spin_2z) del manifiesto (las masas que recibe `get_td_waveform`), ordenando m1 ≥ m2. P32 y P128 = las 32 y 128 primeras del orden de extracción.
- **Cuadratura:** h_Q = transformada de Hilbert de la h+ blanqueada (|c|² = c_I² + c_Q² se calcula como |señal analítica de c_I|², ifft con espectro unilateral). El blanqueado es un filtro real lineal e invariante, así que conmuta con Hilbert; para SEOBNRv4_opt (solo (2,2)) de cara equivale a h×. Solo se guarda h+ blanqueada (ahorra 2,8 GB).
- **Colocación:** el pico de h+ (t = 0 de SEOBNRv4_opt) en la muestra p = N − 1 − 512 de la ventana (0,125 s de margen para el ringdown y la cola del filtro de blanqueado). El segmento de procesado es sintético (`processing_start` = 1e9 s); el procesado es invariante en el tiempo salvo el redondeo de inyección.
- **Sin borde circular:** FFT de longitud 2N (datos y plantilla con ceros): ρ(q) es el ajuste de la plantilla con su pico en la muestra q de la ventana, q = 0…N − 1 (todas las latencias con el pico dentro de la ventana); fuera de la ventana los datos valen 0, sin renormalizar por el solapamiento parcial. Los picos por detector se maximizan por separado (sin restricción de coincidencia), como dice el pre-registro.
- **G3:** diferencia relativa = max|a − b| / max|b| sobre las 64 × 96 × 16384 salidas.
- **Semillas no pre-registradas:** bootstrap `default_rng([146001, celda])` (los mismos índices para todas las sondas, etiquetas y ambos residuos de una celda: emparejado); permutación de G2 `default_rng(146002)`.
- **Particiones:** `KFold(5, shuffle=True, random_state=0)`.
- **G1** usa solo las variables del banco P512 (sin A) y R² frente a 0 sobre 𝓜 estandarizada (global).

## Ejecución
- Plantillas (local, CPU): `scripts/m14/m14_6_make_templates.py` → `results/M13_3/M14_6/templates_<UTC>.npz`.
- Variables (VM, GPU): `scripts/m14/m14_6_bank_features.py` → `results/M13_3/M14_6/features_val_<UTC>.npz`.
- Sondas (local, CPU, 1 hilo): `notebooks/m14/M14_6_probes.ipynb` → `results/M13_3/M14_6/M14_6_summary_<UTC>.json` y `M14_6_oof_<UTC>.npz`.

## Resultado (19-11-2026, run 20261119T172902Z)
Gates: G3 diferencia 0; G1 R² de 𝓜 con P512 0,813 (≥ 0,8); G2 |ΔR²| ≤ 0,0015. Criterio primario: **sin margen relevante**. M14.5b, P512, D8–D10: ΔR² −0,0014 [−0,0056; 0,0025] (𝓜) y −0,0043 [−0,0090; 0,0003] (χ_eff). ΔR² ∈ [−0,005; +0,004] en todas las celdas, bancos y en M14.5a; sin crecimiento con K; L32 ≈ P32; R²_A ≈ 0. Decisión: no se invierte más en el banco de filtros. Cota inferior: no demuestra ausencia de información en X. Resumen `results/M13_3/M14_6/M14_6_summary_20261119T172902Z.json`.
