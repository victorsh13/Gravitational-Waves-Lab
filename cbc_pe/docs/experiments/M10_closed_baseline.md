# M10 closed baseline

This document records the closed **M10-500k** baseline for the `cbc_pe` gravitational-wave parameter-estimation project.

M10 is the current reference pipeline and should be treated as a fixed scientific baseline.

Future methodological changes should be introduced under a new experiment identifier rather than silently modifying the M10 definition.

## 1. Motivation

The M10 branch was introduced after real-data tests revealed a major synthetic-to-real scale mismatch in the previous M08 pipeline.

M08 performed well on held-out synthetic data, but real GWOSC inputs showed detector-dependent amplitude scales that were substantially different from those seen during synthetic training.

The dominant M10 change was therefore not a new parameter-estimation target or conformal method, but a revised input-normalization contract:

```text
per_sample_per_detector_zscore
```

Each detector time series is normalized independently for every sample before being passed to the CNN.

The objective was to preserve the useful synthetic performance of the existing residual/dilated CNN family while improving transfer to real detector strain.

## 2. Closed M10 dataset

Dataset identifier:

```text
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000
```

Main properties:

```text
number of samples:       500000
duration:                4 s
sampling frequency:      4096 Hz
detectors:               H1, L1, V1
waveform approximant:    SEOBNRv4_opt
target network SNR:      10–25
```

Regression targets:

```text
chirp_mass
total_mass
chi_eff
```

Generation config:

```text
configs/generation/generate_500k_bbh_4s.json
```

## 3. Closed split

The final split is:

```text
train: 400000
val:    40000
cal:    30000
test:   30000
```

Config:

```text
configs/splits/
splits_500k_train400_val40_cal30_test30_seed123.json
```

The label mean and standard deviation used for target standardization are computed from the training split only.

Calibration and test data remain isolated from CNN training.

## 4. Model definition

Training config:

```text
configs/experiments/
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

Reference architecture:

```text
SimpleCNN_ResidualDilated
```

Main model/training characteristics:

```text
embedding dimension:       64
residual channels:         64
dilations:                 [1, 2, 4]
residual kernel size:      7
dropout_conv:              0.05
dropout_dense:             0.1
num_groups:                8
loss:                      MSELoss
batch size:                256
seed:                      123
```

The JSON configuration is the authoritative source for the complete definition.

## 5. M10 input normalization

The defining preprocessing change is:

```text
per_sample_per_detector_zscore
```

For each sample and detector channel:

```text
x_norm =
    (x - mean_time(x))
    / std_time(x)
```

This transformation is applied independently to:

```text
H1
L1
V1
```

and consistently during:

```text
training
validation
synthetic prediction
real-data inference
```

The repository contains regression tests verifying equivalence between sample-wise and batch-wise implementations.

## 6. Training outcome

The closed 500k M10 training reached a best standardized validation loss of approximately:

```text
best val loss ≈ 0.11
```

This should be interpreted as the optimization metric in standardized target space rather than as a directly physical error scale.

The final scientific evaluation is performed in physical units on the held-out test subset.

## 7. Prediction artifact

Prediction config:

```text
configs/predictions/
predict_500k_M10_inputzscore_cal_test.json
```

The closed calibration/test prediction artifact is:

```text
m10_inputzscore_500k_cal_test_predictions_embeddings.npz
```

stored under:

```text
<data_root>/results/<dataset_id>/
```

It contains the calibration/test predictions, labels, embeddings, and associated normalization metadata required by the final synthetic and Mondrian analyses.

## 8. Synthetic regression performance

The final synthetic evaluation is performed in:

```text
notebooks/10_m10_inputzscore_500k_evaluation.ipynb
```

The approximate closed physical test metrics are:

| target | RMSE | MAE | bias | R2 |
|---|---:|---:|---:|---:|
| `chirp_mass` | 5.37 M_sun | 3.74 M_sun | -0.02 M_sun | 0.895 |
| `total_mass` | 10.16 M_sun | 7.18 M_sun | -0.02 M_sun | 0.916 |
| `chi_eff` | 0.155 | 0.122 | 0.002 | 0.845 |

These values show substantially stronger predictive performance for the mass parameters than for effective spin.

The global bias is small, but global bias alone does not exclude structured residual effects across the target range.

## 9. Regression-to-the-mean behavior

Synthetic evaluation shows the characteristic tendency of point-regression models to predict toward the conditional mean.

This can lead to:

```text
low true values
→ predictions biased upward

high true values
→ predictions biased downward
```

The effect is particularly relevant near the boundaries of the training distribution.

This behavior should be considered when interpreting:

- mass-tail errors;
- `chi_eff` extremes;
- real-event predictions near training-domain limits;
- local conformal interval width.

M10 does not explicitly solve regression-to-the-mean behavior.

It establishes a stronger synthetic-to-real baseline on top of which future modeling improvements can be tested.

## 10. Mondrian conformal analysis

Final notebook:

```text
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
```

The analysis evaluates combinations of:

```text
taxonomy mode:
    prediction
    difficulty

interval mode:
    symmetric
    asymmetric

number of bins:
    multiple candidate values
```

The calibration subset is used to fit the conformal systems.

The test subset is used to evaluate coverage, local validity, and interval efficiency.

## 11. Mondrian selection policies

Two final operating policies are retained:

```text
conservative
efficient
```

The conservative policy prioritizes local calibration robustness.

The efficient policy searches for narrower intervals while satisfying predefined validity constraints.

If no configuration satisfies the efficient criteria, the selection procedure falls back to the conservative solution.

Selection logic is implemented in:

```text
src/conformal/selection.py
```

Selected calibrators are reconstructed through:

```text
src/conformal/selected_calibrators.py
```

## 12. Closed conservative Mondrian configurations

The closed conservative selections are:

| target | taxonomy | interval mode | bins |
|---|---|---|---:|
| `chirp_mass` | difficulty | asymmetric | 4 |
| `total_mass` | prediction | asymmetric | 6 |
| `chi_eff` | prediction | symmetric | 8 |

These configurations prioritize local calibration behavior over minimum interval width.

## 13. Closed efficient Mondrian configurations

The closed efficient selections are:

| target | taxonomy | interval mode | bins |
|---|---|---|---:|
| `chirp_mass` | difficulty | symmetric | 4 |
| `total_mass` | difficulty | symmetric | 4 |
| `chi_eff` | difficulty | asymmetric | 12 |

These represent the more interval-efficient operating point under the closed M10 selection criteria.

## 14. Mondrian interval interpretation

The M10 intervals are conformal prediction intervals calibrated using synthetic residuals.

They are not Bayesian posterior credible intervals.

Their synthetic test interpretation is based on calibration/test exchangeability under the synthetic generation distribution.

Application to real GWOSC events constitutes a domain-transfer experiment.

Nominal synthetic coverage is therefore not automatically a formal real-data coverage guarantee.

## 15. Single-event real-data validation

Final notebook:

```text
notebooks/12_real_event_inference_m10_500k_clean.ipynb
```

GW170814 was used as the main controlled real-event validation case.

The real-data processing chain includes:

```text
GWOSC detector strain
→ off-source PSD estimation
→ whitening/filtering
→ processing context
→ final 4 s network window
→ M10 per-detector z-score
→ CNN prediction
→ embedding extraction
→ selected Mondrian intervals
```

The purpose of this stage was to verify that the complete synthetic-trained pipeline can operate on real detector data without target truth.

## 16. GW170814 closed M10 predictions

A representative M10 prediction for GW170814 is approximately:

```text
chirp_mass ≈ 25.4 M_sun
total_mass ≈ 60.4 M_sun
chi_eff ≈ -0.21
```

For the conservative Mondrian policy, the corresponding representative intervals are approximately:

```text
chirp_mass:
    [13.46, 36.13] M_sun

total_mass:
    [47.34, 68.65] M_sun

chi_eff:
    [-0.535, 0.124]
```

These values should be interpreted as outputs of the fixed M10 synthetic calibration applied to real GWOSC strain.

## 17. Event-window sensitivity

Real-event tests included controlled shifts of the 4 s event window.

For the M10 real-data pipeline, representative predictions across center offsets remained approximately within:

```text
chirp_mass:
    27.6–28.5 M_sun

total_mass:
    65.7–67.9 M_sun

chi_eff:
    0.12–0.16
```

for the tested window-sensitivity configuration.

This indicates that the M10 normalization substantially stabilizes the gross detector-scale problem seen in the previous baseline, although timing and PSD choices remain relevant sources of systematic variation.

## 18. Multi-event real-data evaluation

Final notebook:

```text
notebooks/13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

This notebook applies the M10 pipeline to multiple compatible GWOSC/LVK events.

The workflow includes:

```text
catalog query
→ event selection
→ detector-file resolution
→ GWOSC strain acquisition
→ PSD-window validation
→ real-data preprocessing
→ CNN inference
→ Mondrian application
→ LVK reference conversion
→ comparison metrics
```

The final implementation avoids retaining complete long-duration raw event strain for every processed event simultaneously.

Only lightweight scientific result tables are accumulated across the multi-event run.

## 19. LVK detector-frame comparison

The synthetic model predicts detector-frame masses.

Published source-frame LVK masses are converted through:

```text
m_detector =
    m_source * (1 + z)
```

before comparison.

`chi_eff` is not redshift transformed.

Reusable implementation:

```text
src/real_data/lvk_reference.py
```

## 20. LVK comparison diagnostics

The closed evaluation uses diagnostics including:

```text
delta_lvk
normalized_delta_lvk
lvk_central_inside_cnn
cnn_point_inside_lvk
interval_overlap_fraction_lvk
```

The normalized point discrepancy is:

```text
normalized_delta_lvk =
    (CNN point - LVK central)
    / (0.5 * (LVK upper - LVK lower))
```

The interval-overlap metric is:

```text
interval_overlap_fraction_lvk =
    overlap length
    / LVK interval width
```

This quantity is not an intersection-over-union metric.

An overlap fraction of one means that the conformal interval covers the entire LVK reference interval, even if the conformal interval itself is wider.

## 21. Original versus clipped intervals

Original conformal interval bounds are retained for scientific metrics.

Optional physically clipped mass bounds may be generated for visualization.

The following quantities must use the original intervals:

```text
coverage
interval width
LVK overlap
interval-membership flags
```

Clipped columns are plotting aids only.

## 22. M08 to M10 transition

The transition from M08 to M10 was driven primarily by real-data transfer behavior.

### M08

M08 established a strong synthetic baseline and the residual/dilated model family.

However, direct application to real GWOSC strain exposed a substantial detector-scale mismatch.

Synthetic performance alone was therefore insufficient to validate the real-data pipeline.

### M10

M10 retained the established model family while introducing per-sample/per-detector z-score normalization.

This substantially improved consistency between synthetic and real network inputs.

The key lesson from M08 → M10 is that preprocessing-domain compatibility can be at least as important as architecture-level changes when transferring a neural network from simulated to real gravitational-wave strain.

## 23. What M10 establishes

M10 establishes that:

1. the complete HDF5 training chain is reproducible at 500k scale;

2. dedicated train/validation/calibration/test splits are available;

3. the residual/dilated CNN provides useful synthetic estimates for mass and `chi_eff`;

4. per-detector input z-score normalization resolves the dominant real-data amplitude-scale mismatch observed in M08;

5. Mondrian conformal calibration can be fitted on dedicated synthetic calibration data and applied to real events without target truth;

6. reusable real-data processing, inference, LVK-reference, and evaluation code now exists outside notebooks;

7. multi-event real-data comparisons can be performed reproducibly against LVK/GWTC-3 summaries.

## 24. Known limitations

M10 should not be interpreted as a final parameter-estimation solution.

Important limitations remain.

### Synthetic-noise domain

The training distribution does not fully reproduce:

```text
real detector non-stationarity
non-Gaussian noise
glitches
PSD evolution
instrumental artifacts
```

### Synthetic-to-real conformal transfer

Conformal calibration is performed on synthetic calibration data.

Nominal synthetic coverage does not imply guaranteed real-data coverage under distribution shift.

### Waveform assumptions

The dataset uses:

```text
SEOBNRv4_opt
```

The model therefore inherits the physical assumptions and limitations of the waveform model and training parameterization.

### Spin modeling

`chi_eff` remains the weakest target of the three.

Future work should examine whether waveform assumptions, training priors, degeneracies, target representation, and point-regression loss contribute to this limitation.

### Point-regression behavior

The MSE objective encourages conditional-mean predictions and can produce regression-to-the-mean effects.

A point regressor cannot fully represent multimodal or highly degenerate posterior structure.

### Broad conformal intervals

Some Mondrian configurations remain relatively broad, especially for difficult targets.

Increasing the number of bins does not automatically improve the result because finite calibration counts reduce local stability.

## 25. Future work beyond M10

Future improvements should be introduced as a new experimental phase.

Priority directions include:

```text
real off-source detector noise injections
domain-gap characterization
non-stationary noise studies
glitch robustness
waveform-systematics studies
spin-prior consistency
alternative regression objectives
distributional or posterior models
improved uncertainty estimation
revised Mondrian strategies
```

These changes should be evaluated against the closed M10 baseline rather than incorporated into M10 itself.

## 26. Reproducibility chain

The closed M10 chain is:

```text
configs/generation/
generate_500k_bbh_4s.json

→

configs/splits/
splits_500k_train400_val40_cal30_test30_seed123.json

→

configs/experiments/
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json

→

M10 checkpoint

→

configs/predictions/
predict_500k_M10_inputzscore_cal_test.json

→

m10_inputzscore_500k_cal_test_predictions_embeddings.npz

→

notebooks/10_m10_inputzscore_500k_evaluation.ipynb

→

notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb

→

notebooks/12_real_event_inference_m10_500k_clean.ipynb

→

notebooks/13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

This chain defines the scientific reference state of M10.

## 27. Closed-baseline policy

M10 is now considered closed.

Bug fixes that do not alter scientific behavior may still be applied if they are regression-tested and documented.

Changes that alter:

```text
training data
physical priors
waveforms
preprocessing
normalization
model architecture
loss
calibration
selection policy
real-data processing
```

should receive a new experiment identifier.

The purpose of preserving M10 is to provide a stable reference against which future improvements can be measured.