# CNN evaluation workflow

This document describes the evaluation workflow for the closed **M10-500k** CBC parameter-estimation baseline.

Evaluation is divided into two complementary domains:

1. held-out synthetic BBH data;
2. real GWOSC events compared against published LVK parameter estimates.

The objective is not only to report global regression metrics, but also to characterize residual structure, parameter-dependent performance, conformal intervals, and synthetic-to-real behavior.

## 1. Closed M10 evaluation chain

The reference model is defined by:

```text
configs/experiments/
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

The calibration/test prediction artifact is generated using:

```text
configs/predictions/
predict_500k_M10_inputzscore_cal_test.json
```

and stored under:

```text
<data_root>/results/<dataset_id>/
```

The main artifact is:

```text
m10_inputzscore_500k_cal_test_predictions_embeddings.npz
```

The final evaluation notebooks are:

```text
notebooks/10_m10_inputzscore_500k_evaluation.ipynb
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
notebooks/12_real_event_inference_m10_500k_clean.ipynb
notebooks/13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

Their roles are:

- `10`: synthetic regression evaluation;
- `11`: Mondrian conformal evaluation and configuration selection;
- `12`: controlled single-event real-data validation;
- `13`: multi-event GWOSC/LVK comparison.

## 2. Evaluation targets

The model predicts:

```text
chirp_mass
total_mass
chi_eff
```

Mass targets are expressed in solar masses.

`chi_eff` is dimensionless.

The CNN is trained in standardized label space, but final scientific interpretation is performed primarily in physical units.

## 3. Standardized-space evaluation

The training targets are standardized using mean and standard deviation computed from the training split only.

For each target:

```text
y_std = (y_phys - y_mean) / y_scale
```

Standardized-space evaluation is useful for:

- comparing optimization behavior across targets;
- interpreting the training loss;
- checking whether one standardized target dominates the regression objective;
- comparing model variants under the same label transformation.

Typical standardized-space metrics include:

```text
MSE
RMSE
MAE
bias
R2
```

These metrics are useful diagnostically but should not replace physical-space reporting.

## 4. Physical-space evaluation

Predictions are transformed back through:

```text
y_phys = y_std * y_scale + y_mean
```

For each physical target, the main regression metrics are:

- RMSE
- MAE
- bias
- R2
- residual distributions
- prediction-versus-truth structure

The residual convention should be kept consistent within each analysis.

A typical convention is:

```text
residual = prediction - truth
```

The sign convention must be stated whenever residual bias or asymmetric tails are interpreted.

## 5. M10 synthetic test set

The closed synthetic split is:

```text
train: 400000
val:    40000
cal:    30000
test:   30000
```

The `test` subset is reserved for final synthetic performance evaluation.

It must not be used to:

- train the CNN;
- perform early stopping;
- fit conformal quantiles;
- tune the final model architecture.

The calibration subset is kept separate because it is required to fit conformal prediction intervals.

## 6. Main synthetic diagnostics

Final synthetic evaluation should include more than one aggregate metric.

Recommended diagnostics include:

```text
prediction versus truth
residual histograms
residual versus true parameter
residual versus predicted parameter
absolute error versus true parameter
absolute error versus SNR
bias across the target range
tail behavior
```

These diagnostics are particularly important because a low global MSE can coexist with systematic regression-to-the-mean effects at the edges of the training distribution.

For parameter estimation, this matters especially in regions such as:

```text
low masses
high masses
extreme chi_eff values
lower-SNR events
```

## 7. Regression-to-the-mean behavior

Neural-network regression trained with point-estimation losses such as MSE tends to predict conditional averages.

This can produce characteristic residual structure:

```text
low true values  → predictions biased upward
high true values → predictions biased downward
```

Such behavior should be inspected explicitly rather than inferred only from global metrics.

This effect can contribute to larger errors near target-distribution boundaries and can influence conformal interval widths in those regions.

It is therefore useful to inspect:

- residual versus true target;
- predicted versus true target;
- conditional bias by parameter bin.

## 8. Input normalization and evaluation consistency

The closed M10 model uses:

```text
per_sample_per_detector_zscore
```

The same input-normalization rule must be applied during:

```text
training
synthetic prediction
real-data prediction
```

The repository includes regression tests ensuring equivalence between the sample-wise and batch-wise normalization implementations.

Any evaluation that bypasses the M10 normalization contract is not directly comparable with the closed M10 baseline.

## 9. Real-data evaluation

Real-data evaluation uses GWOSC strain.

The reusable implementation is located under:

```text
src/real_data/
```

The real-data chain includes:

```text
GWOSC strain acquisition
→ HDF5 validation
→ detector-window validation
→ off-source PSD estimation
→ whitening/filtering
→ final event-window construction
→ M10 input normalization
→ CNN prediction
→ embedding extraction
→ Mondrian interval application
```

Real-event inference must use exactly the same model checkpoint and input-normalization definition as the synthetic evaluation.

## 10. Single-event validation

The main controlled single-event notebook is:

```text
notebooks/12_real_event_inference_m10_500k_clean.ipynb
```

Its purpose is to verify the complete processing path on one event before applying the pipeline systematically to a catalog sample.

The notebook should be treated as a pipeline-validation and interpretability tool rather than as the primary source of population-level conclusions.

Relevant checks include:

- detector strain availability;
- PSD-window validity;
- final input shapes;
- detector-scale behavior;
- sensitivity to event-window placement;
- CNN point predictions;
- selected conformal intervals.

## 11. Multi-event LVK comparison

The final multi-event evaluation is:

```text
notebooks/13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

The notebook processes multiple compatible GWOSC events and compares M10 outputs against LVK/GWTC-3 reference summaries.

The reusable comparison utilities are located in:

```text
src/evaluation/lvk.py
src/real_data/lvk_reference.py
```

## 12. Detector-frame convention

The synthetic model predicts detector-frame masses.

Published LVK catalog quantities may be reported in source frame.

For a source-frame mass:

```text
m_detector = m_source * (1 + z)
```

where `z` is the published redshift.

The same transformation is applied consistently to:

- chirp mass;
- component masses;
- total mass derived from component masses.

`chi_eff` is dimensionless and is not transformed by redshift.

This conversion is required before comparing the CNN mass predictions against LVK values.

## 13. LVK uncertainty convention

LVK reference summaries are represented by:

```text
central value
lower bound
upper bound
```

When asymmetric uncertainties are present, the full lower and upper interval bounds should be preserved.

The comparison should not silently symmetrize the published interval except where an explicit scalar width is required by a diagnostic.

## 14. Point-estimate comparison

A useful signed discrepancy is:

```text
delta_lvk =
    CNN point - LVK central
```

The normalized discrepancy used in the closed M10 analysis is:

```text
normalized_delta_lvk =
    (CNN point - LVK central)
    / (0.5 * (LVK upper - LVK lower))
```

Interpretation:

```text
normalized_delta_lvk = 0
```

means that the CNN point estimate equals the LVK central value.

Values with magnitude around one correspond roughly to a displacement comparable to one LVK average half-width.

This is a diagnostic normalization, not a posterior z-score.

## 15. Interval-membership diagnostics

Useful binary diagnostics include:

```text
lvk_central_inside_cnn
cnn_point_inside_lvk
```

These answer different questions.

### LVK central inside CNN interval

This checks whether:

```text
CNN lower
<= LVK central
<= CNN upper
```

### CNN point inside LVK interval

This checks whether:

```text
LVK lower
<= CNN point
<= LVK upper
```

Neither quantity alone is sufficient to characterize interval agreement.

## 16. Interval overlap fraction

The closed M10 analysis uses:

```text
interval_overlap_fraction_lvk =
    overlap_length(CNN interval, LVK interval)
    / LVK interval width
```

where:

```text
overlap_length =
    max(
        0,
        min(CNN upper, LVK upper)
        - max(CNN lower, LVK lower)
    )
```

and:

```text
LVK interval width =
    LVK upper - LVK lower
```

Interpretation:

```text
0
```

means that the intervals do not overlap.

```text
1
```

means that the full LVK interval is covered by the CNN/conformal interval.

This metric is **not** an intersection-over-union or Jaccard index.

Because the denominator is the LVK interval width, a CNN interval substantially wider than the LVK interval can still obtain an overlap fraction of one.

Interval width must therefore be inspected separately.

## 17. Original versus plotting-clipped intervals

Physical visualization may clip mass lower bounds to avoid plotting unphysical negative masses.

These plotting columns must remain separate from the original conformal intervals.

Scientific metrics such as:

```text
coverage
LVK interval overlap
interval membership
interval width
```

must use the original conformal bounds.

Clipped intervals are visualization aids only.

The evaluation utilities preserve the original columns and create separate explicitly named clipped columns.

## 18. Interpreting LVK comparisons

LVK comparisons should not be interpreted as direct supervised ground truth.

Published LVK values are posterior summaries obtained using a different inference framework and different assumptions.

The comparison is useful for assessing whether the low-latency CNN/conformal pipeline produces physically plausible estimates broadly compatible with established parameter-estimation results.

Important distinctions include:

```text
CNN point estimate
conformal frequentist prediction interval
LVK posterior central value
LVK posterior credible interval
```

These objects do not have identical statistical meanings.

Therefore, conclusions should focus on:

- consistency;
- discrepancy structure;
- systematic bias;
- interval overlap;
- parameter-dependent behavior;
- synthetic-to-real degradation.

## 19. Known M10 evaluation limitations

The closed M10 evaluation should be interpreted with several limitations in mind.

### Synthetic training distribution

The model is trained on synthetic BBH signals under a defined waveform/noise/preprocessing setup.

Real detector noise can contain:

```text
non-stationarity
non-Gaussian transients
glitches
PSD variability
instrumental artifacts
```

that are not fully represented by the synthetic baseline.

### Waveform assumptions

The synthetic dataset uses:

```text
SEOBNRv4_opt
```

The resulting model inherits the physical assumptions and domain limitations of the waveform family used for training.

### Point-regression limitations

MSE-trained regression can display regression-to-the-mean behavior and can lose information about multimodal or degenerate parameter structure.

### Conformal distribution shift

Conformal coverage guarantees rely on exchangeability or approximately matched calibration/test distributions.

Intervals calibrated on synthetic data do not automatically retain nominal coverage under real-data distribution shift.

Real-event intervals should therefore be treated as an empirical transfer study rather than as a formally guaranteed real-data coverage result.

## 20. Historical evaluation notebooks

Earlier evaluation work is retained for scientific traceability under:

```text
notebooks/_archive/architecture_search/
notebooks/_archive/m08_baseline/
notebooks/_archive/m10_development/
```

Reusable lightweight examples are under:

```text
notebooks/demos/
```

The authoritative closed M10 synthetic evaluation is:

```text
notebooks/10_m10_inputzscore_500k_evaluation.ipynb
```

and the authoritative multi-event real-data analysis is:

```text
notebooks/13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

## 21. Evaluation reproducibility

Scientific evaluation should preserve:

- dataset ID;
- split ID;
- checkpoint;
- normalization mode;
- prediction artifact;
- label statistics;
- conformal calibration configuration;
- real-event preprocessing configuration;
- event-selection criteria;
- LVK reference convention.

Changes to any of these should be treated as a new experimental condition rather than silently folded into the closed M10 baseline.