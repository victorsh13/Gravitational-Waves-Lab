# M13.1 — Diagnósticos en CPU de M11 sobre validación

Estado: **protocolo fijado antes de cualquier resultado.** Descriptivo y exploratorio; no selecciona modelos.

## Objetivo

Caracterizar dónde y cómo falla el estimador puntual M11 (final G1 500k) en validación, para informar
qué hipótesis de M13 merece una intervención controlada. Tres partes:

- **(a)** sesgo y RMSE por etiqueta condicionados a variables físicas derivadas de las etiquetas verdaderas;
- **(b)** acoplamiento entre los residuos de chi_eff y de eta (degeneración);
- **(c)** contenido espectral de los kernels convolucionales con stride del checkpoint de M11.

## Datos y procedencia (solo lectura)

- Predicciones de val: `$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/m11_final_G1_500k_val_predictions_embeddings.npz`,
  SHA256 fijado `586b95f2cb2cfaf150fbdd05760f50f47a95a555451fcb46d89492b7d00ed88b` (aborta si difiere).
  Se pasa a unidades físicas con `npz_to_physical` (M12.5.3); se repiten las comprobaciones de contrato de M12.5.3
  (`load_contract`, escalador train-only, pertenencia de `idx_val` al split val).
- HDF5 de G1, indexado por `idx_val`: `mass_1`, `mass_2`, `spin_1z`, `spin_2z`, `target_network_snr`,
  `file_group_id`. **Nunca se lee X.** Comprobación: (Mc, M, chi_eff) recalculados coinciden con `y_val`
  (tolerancia relativa 1e-5; suelo absoluto 1e-5·y_std por etiqueta, solo relevante para chi_eff ≈ 0).
- Checkpoint M11: `$CBC_PE_DATA_ROOT/models/checkpoints/M11_final_G1_500k/M11_final_G1_500k_SimpleCNN_ResidualDilated_M11_final_G1_inputzscore_resdilated_emb64_d124_train400k_MSELoss_seed123_checkpoint.pt`
  (`map_location='cpu'`). Si no existe, se omite solo (c).
- Cal, test y eventos reales no se tocan. Etiquetas en orden [chirp_mass, total_mass, chi_eff].

## Variables derivadas (desde las etiquetas verdaderas)

- q = m_mayor/m_menor ≥ 1; eta = m1·m2/M².
- tau30 = (5/256)·(G·Mc/c³)^(−5/3)·(π·30 Hz)^(−8/3), con G·Msun/c³ = 4.925490947e-6 s (orden newtoniano).
- f220 (Hz) con qr = m_menor/m_mayor, nu = qr/(1+qr)², chi1 = espín de la masa mayor:
  a_t = (chi1 + chi2·qr²)/(1+qr²); a_f = a_t + a_t·nu·(s4·a_t + s5·nu + t0) + nu·(2√3 + t2·nu + t3·nu²)
  (s4=−0.1229, s5=0.4537, t0=−2.8904, t2=−3.5171, t3=2.5763);
  a_e = (chi1 + chi2·qr²)/(1+qr)²; r_isco de Bardeen; E_isco = √(1 − 2/(3 r_isco(a_e)));
  E_rad = (1−E_isco)·nu + 4nu²·(4·0.04827 + 16·0.01707·a_e·(a_e+1) + E_isco − 1); M_f = M·(1 − E_rad);
  f220 = [1.5251 − 1.1568·(1−a_f)^0.1292] / (2π·M_f·G·Msun/c³).
  Clases: `<256` (f < 256), `256-512` (256 ≤ f ≤ 512), `>512` (f > 512).
- eta_pred = (Mc_pred/M_pred)^(5/3), recortado a ≤ 0.25.

## Análisis

**(a) M13.1a.** Por etiqueta: n, sesgo = media(pred − verdad), RMSE, e IC 95 % percentil bootstrap
(1000 réplicas, semilla 131001, remuestreo con reemplazo dentro de cada entorno `file_group_id`), por:
deciles de tau30 (cuantiles de val); clase de f220; q ∈ [1,1.5), [1.5,2.5), [2.5,4), [4,∞);
SNR ∈ [10,15), [15,20), [20,25]. Cruces tau30-decil × SNR y clase f220 × SNR para chi_eff y total_mass.
Grupos con n < 200 se marcan `sparse`.

**(b) M13.1b.** r_chi = chi_pred − chi_true, r_eta = eta_pred − eta_true. Pearson, Spearman y R² de la
regresión OLS de r_chi sobre r_eta: global, por tercil de Mc verdadera y por clase de f220. Histograma 2D
de (r_eta, r_chi) (60×60, rango percentiles 0.5–99.5), guardado en `analysis.json`.

**(c) M13.1c.** Para cada kernel 1-D (salida, entrada) de `block1.conv` (16×3×16), |H(f)|² con FFT de
2048 puntos a fs = 4096 Hz y fracción de energía en [0,256), [256,512) y [512, Nyquist] Hz
(espectro unilateral, bins interiores ×2). Igual con `block2.conv` (32×16×16, entrada a 2048 Hz) y
`block3.conv` (64×32×16, entrada a 1024 Hz; su banda superior se reduce al bin de Nyquist).

## Qué hipótesis informa cada parte

- (a) **campo receptivo / contexto** vía tau30; **banda alta** vía clase de f220; **degeneración** vía q.
- (b) **degeneración** masa–espín (eta–chi_eff).
- (c) **representación de la banda 256–512 Hz** en los primeros filtros.

## Lecturas fijadas

- **HIP-CONTEXTO:** a igual SNR, si el RMSE de chi_eff crece con tau30 (la física espera que mejore con
  más ciclos en banda), lo apoya.
- **HIP-BANDA-ALTA:** un escalón de error en la clase `256-512` frente a `<256` a igual SNR; y en (c),
  menos del 10 % de los kernels de block1 con más del 30 % de su energía por encima de 256 Hz.
- **Degeneración:** |Spearman(r_eta, r_chi)| ≥ 0.3 global, o mayor en el tercil de Mc bajo.

Las lecturas cuantitativas de (b) y (c) se registran en `analysis.json` (`degeneracy_readout`,
`part_c.block1_readout`); las de (a) se valoran en revisión a partir de los CSV y del notebook.

## Salidas

Directorio exclusivo `$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M13_1_diagnostics/` (aborta si existe):
`a_bins.csv`, `a_cross.csv`, `b_corr.csv`, `c_filters.csv` (ausente si se omite (c)) y `analysis.json`
(hashes de npz, script y checkpoint; comprobaciones superadas; ajustes; bordes de bins; histograma 2D).

Script: `scripts/m13/m13_1_diagnostics.py`. Notebook: `notebooks/m13/M13_1_diagnostics.ipynb`.

## Limitaciones

- Descriptivo: sin control de multiplicidad entre grupos; los IC bootstrap son de muestreo sobre val, no
  de variación entre semillas de entrenamiento.
- Los grupos se definen sobre la verdad: el sesgo condicionado a la verdad refleja en parte la regresión a
  la media y no es por sí solo evidencia de una causa física.
- La clase de f220 está fuertemente correlacionada con la masa total (y tau30 con Mc): los efectos de
  banda, contexto y masa no se separan completamente; los cruces con SNR solo controlan la SNR.
- Los ajustes de a_f, E_rad y f220 tienen errores de pocos por ciento; eventos cerca de 256/512 Hz pueden
  cambiar de clase. tau30 es de orden newtoniano.
- El ajuste de Barausse–Rezzolla puede dar a_f ≥ 1 con q extremo y espín casi máximo: antes de calcular
  E_isco y f220, a_f y a_e se recortan a [−0.998, 0.998] (límite de Thorne); el número de filas
  recortadas se registra en `analysis.json` (`n_af_clipped`, `n_ae_clipped`).
- (c) mide la respuesta lineal de cada kernel aislado: ignora GroupNorm, no linealidades, la combinación
  entre canales, el espectro de la entrada y el aliasing introducido por el stride 2.
