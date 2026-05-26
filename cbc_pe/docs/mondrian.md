# Mondrian conformal workflow

This document describes the Mondrian conformal prediction workflow used in the `cbc_pe` project.

The goal is to build calibrated prediction intervals for CNN-based BBH parameter estimation.

The current implementation is used to analyze CNN prediction errors and construct Mondrian conformal intervals conditioned on different grouping scores.

## Current status

Mondrian analysis is currently notebook-based.

The main notebook is:

```text
cbc_pe/notebooks/04_mondrian_hdf5_demo.ipynb
```

A future script should be added:

```text
cbc_pe/scripts/run_mondrian.py
```

The current notebook can be used in two modes:

```text
debug mode: pseudo calibration/test split created from validation predictions
final mode: real calibration/test splits loaded from saved prediction files
```

Debug mode is useful for checking that the pipeline runs, but it must not be reported as final conformal performance.

## Required inputs

Mondrian conformal analysis needs saved CNN outputs, typically produced by:

```text
cbc_pe/scripts/train_cnn_hdf5.py
```

The expected prediction file is usually stored under:

```text
<data_root>/results/
```

Typical filename pattern:

```text
*_predictions_embeddings.npz
```

The file should contain arrays such as:

```text
pred_train
y_train
emb_train
idx_train

pred_val
y_val
emb_val
idx_val

pred_cal
y_cal
emb_cal
idx_cal

pred_test
y_test
emb_test
idx_test

y_mean
y_std
label_names
available_splits
```

For the current 80/20 architecture-selection split, only `train` and `val` may exist.

For final conformal evaluation, `cal` and `test` must exist.

## Split requirements

### Current architecture-selection setup

The current architecture-selection setup is:

```text
train = 80000
validation = 20000
calibration = 0
test = 0
```

This split is acceptable for CNN model comparison.

It is not sufficient for final conformal reporting.

If Mondrian is run using a pseudo split of validation predictions, those results should be marked as debugging results only.

### Final conformal setup

After selecting the CNN architecture, use a split such as:

```text
train = 70%
validation = 10%
calibration = 10%
test = 10%
```

The intended role of each split is:

- `train`: fit CNN weights
- `validation`: model selection and early stopping
- `calibration`: compute conformal quantiles
- `test`: evaluate final interval coverage and width

The calibration set must not be used to train the CNN.

The test set must not be used to choose the CNN architecture, tune Mondrian settings, or select the final configuration.

## Basic conformal idea

The CNN produces point predictions:

```text
y_pred
```

For each target label, the residual is:

```text
residual = y_true - y_pred
```

Conformal calibration uses residuals on the calibration set to construct intervals around future predictions.

For a target confidence level such as:

```text
confidence_level = 0.90
```

the desired marginal coverage is approximately:

```text
P(y_true in prediction interval) ≈ 0.90
```

## Mondrian conformal idea

Standard conformal prediction computes one global residual quantile.

Mondrian conformal prediction instead splits the calibration data into bins and computes a separate conformal interval per bin.

The goal is to obtain more locally adaptive intervals.

The generic flow is:

```text
1. Compute a score for each calibration sample.
2. Bin calibration samples according to that score.
3. Compute residual quantiles inside each bin.
4. For a test sample, compute its score.
5. Assign the test sample to a bin.
6. Use the interval calibrated for that bin.
7. Evaluate empirical coverage and interval width.
```

## Taxonomy modes

The project currently supports two main taxonomy modes.

### Prediction-based Mondrian

Prediction-based Mondrian uses the CNN prediction itself as the binning score.

Example interpretation:

```text
bin samples by predicted chirp_mass
bin samples by predicted total_mass
bin samples by predicted chi_eff
```

This can adapt intervals to regions where the target value itself changes the difficulty of the problem.

This mode is simple, interpretable, and useful as a baseline.

Potential weakness:

```text
prediction-based bins may miss uncertainty structure that is not directly captured by the predicted value.
```

### Difficulty-based Mondrian

Difficulty-based Mondrian uses a difficulty score estimated from model embeddings and calibration residuals.

The idea is:

```text
samples close in embedding space may have similar prediction difficulty
```

The typical procedure is:

```text
1. Extract embeddings from the trained CNN.
2. For each sample, find nearby calibration samples in embedding space.
3. Estimate difficulty from neighboring calibration residuals.
4. Use this difficulty score for binning.
```

This can produce intervals that are more directly related to local model uncertainty.

Potential weakness:

```text
difficulty scores depend on embedding quality and the KNN/difficulty estimator.
```

If embeddings are poor, difficulty-based Mondrian may not improve over simpler prediction-based binning.

## Interval modes

The project supports two interval modes.

### Symmetric intervals

Symmetric intervals use absolute residuals.

For each bin, compute a quantile of:

```text
abs(y_true - y_pred)
```

The interval is:

```text
[y_pred - q, y_pred + q]
```

Advantages:

- simple
- stable
- easy to interpret

Potential weakness:

- cannot represent asymmetric errors

### Asymmetric intervals

Asymmetric intervals use signed residuals and estimate lower and upper errors separately.

The interval is:

```text
[y_pred + q_lower, y_pred + q_upper]
```

where `q_lower` and `q_upper` are calibrated from signed residuals.

Advantages:

- can capture biased or asymmetric residuals
- may be more efficient if errors are asymmetric

Potential weakness:

- can be noisier than symmetric intervals
- needs enough calibration samples per bin

## Binning

The current Mondrian implementation uses quantile binning.

For a chosen number of bins:

```text
n_bins
```

the calibration scores are split into approximately equal-count bins.

Typical values to test are:

```text
n_bins = 3
n_bins = 6
n_bins = 12
n_bins = 24
n_bins = 36
```

Too few bins produce intervals that may be too global.

Too many bins produce unstable quantiles because each bin has fewer calibration samples.

## Minimum samples per bin

Every bin must contain enough calibration samples.

If a bin has too few samples, the conformal quantile becomes unstable.

A configuration should be treated with caution or rejected if:

```text
min_count_per_bin is too small
```

For final analysis, the acceptable minimum count depends on the calibration set size and the number of bins.

With small calibration sets, aggressive binning is usually a bad idea.

## Metrics to report

For each Mondrian configuration, report at least:

- global empirical coverage
- average interval width
- median interval width
- coverage per bin
- width per bin
- sample count per bin
- undercoverage per bin
- worst-bin coverage
- maximum undercoverage gap
- number of bad bins
- comparison with target confidence level

For a target confidence level of 0.90, the empirical coverage should be close to:

```text
0.90
```

But global coverage is not enough.

A configuration with good global coverage can still be locally bad if some bins under-cover strongly.

## Bin-wise coverage

For each bin:

```text
coverage_bin = number of covered samples in bin / number of test samples in bin
```

Coverage should be checked per bin.

Useful diagnostics:

```text
coverage per bin
target coverage line
binomial uncertainty bands
sample count per bin
interval width per bin
```

A bin with low coverage and low count should be interpreted carefully.

A bin with low coverage and high count is a serious warning sign.

## Interval width

Coverage alone is not sufficient.

A trivial method can get high coverage by making intervals too wide.

For each configuration, compare:

```text
coverage
width
```

A good configuration should achieve coverage close to the target with intervals that are not unnecessarily wide.

Useful width metrics:

```text
mean width
median width
width per bin
width per label
```

## Configuration comparison

When comparing Mondrian configurations, do not choose only by global coverage.

A better comparison includes:

- global coverage close to target
- low average width
- stable bin-wise coverage
- no severe undercoverage bins
- reasonable minimum bin count
- no pathological width spikes
- robustness across labels
- robustness across taxonomy modes
- robustness across interval modes

A configuration is suspicious if it has:

- good global coverage but poor bin-wise coverage
- low width but severe undercoverage
- extremely wide intervals in a few bins
- many bins with too few samples
- unstable behavior across labels

## Recommended configuration grid

During development, test combinations such as:

```text
taxonomy_mode:
  prediction
  difficulty

interval_mode:
  symmetric
  asymmetric

n_bins:
  3
  6
  12
  24
  36

confidence_level:
  0.90
```

Start with fewer bins.

Only increase `n_bins` if the calibration set is large enough.

## Current debug workflow

While only train/validation predictions are available, the notebook may use:

```text
debug_split_val = True
```

This creates pseudo calibration and pseudo test sets from validation predictions.

This is useful for:

- checking that the pipeline works
- comparing rough behavior of taxonomy modes
- debugging plots and metrics
- validating code paths before final splits exist

It is not valid for final reporting.

Any result produced with a pseudo split should be clearly marked as debug.

## Final workflow

The final Mondrian workflow should be:

```text
1. Select CNN architecture using train/validation results.
2. Create or use a 70/10/10/10 split.
3. Train the selected CNN architecture.
4. Save predictions and embeddings for train, validation, calibration, and test.
5. Calibrate conformal intervals using the calibration split.
6. Evaluate coverage and width using the test split.
7. Compare Mondrian configurations.
8. Report only results based on real calibration/test splits.
```

## Recommended reporting table

A useful reporting table should contain:

```text
model_id
dataset_id
split_id
taxonomy_mode
interval_mode
label
n_bins
confidence_level
global_coverage
mean_width
median_width
min_bin_count
worst_bin_coverage
max_undercoverage_gap
n_bad_bins
notes
```

## Important warnings

Do not use the test set for selecting the CNN architecture.

Do not use the test set for selecting the Mondrian configuration.

Do not report validation pseudo-split results as final conformal results.

Do not trust global coverage alone.

Do not increase the number of bins without checking bin counts.

Do not compare interval widths without also checking coverage.