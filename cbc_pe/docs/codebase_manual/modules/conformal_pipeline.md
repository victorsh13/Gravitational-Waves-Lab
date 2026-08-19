# Conformal / Mondrian pipeline manual

> **Codebase manual — closed M10 reference**
>
> Reference snapshot:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```
>
> This document explains the conformal-prediction layer used in the closed M10 baseline: taxonomy construction, difficulty estimation, quantile binning, residual grouping, conformal calibration, interval application, local-validity evaluation, final configuration selection, truth-free deployment, and the historical hybrid experiment.

---

## 1. Scope

```text
src/conformal/
├── taxonomy.py
├── difficulty.py
├── binning.py
├── calibration.py
├── apply.py
├── metrics.py
├── pipeline.py
├── selection.py
├── selected_calibrators.py
└── hybrid.py
```

The CNN prediction/embedding pipeline that feeds this layer is documented in:

```text
docs/codebase_manual/modules/cnn_pipeline.md
```

The closed M10 conformal analysis is performed in:

```text
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
```

---

## 2. High-level conceptual map

```text
CNN predictions + CNN embeddings
             │
             ▼
       CALIBRATION SET
             │
             ▼
      define taxonomy score
      ├── prediction
      └── difficulty
             │
             ▼
        quantile binning
             │
             ▼
 calibration residuals grouped by bin
             │
             ▼
 conformal offsets per target/bin
             │
             ▼
     fitted Mondrian system
             │
        ┌────┴────┐
        ▼         ▼
 synthetic test   real target
 truth available  truth unavailable
        │         │
        ▼         ▼
   intervals    intervals
        │
        ▼
 coverage / width evaluation
```

The key architectural property is the separation:

```text
fit
apply
evaluate
```

This allows a fitted calibrator to be applied to real events without requiring target truth.

---

## 3. Core data contracts

The conformal layer consumes arrays produced by the CNN prediction artifact.

```text
pred_cal : (N_cal, 3)
y_cal    : (N_cal, 3)
emb_cal  : (N_cal, 64)

pred_test: (N_test, 3)
y_test   : (N_test, 3)
emb_test : (N_test, 64)
```

Target order:

```text
0 = chirp_mass
1 = total_mass
2 = chi_eff
```

Calibration operates in standardized target space. Physical-unit intervals can be reconstructed downstream using train-only target mean/std.

---

## 4. Critical residual-sign convention

The conformal layer defines residuals as:

\[
r = y - \hat y.
\]

That is:

```text
truth - prediction
```

This differs from some CNN diagnostic residual plots, which use:

\[
\hat y-y.
\]

### Critical contract

```text
[CRITICAL CONFORMAL CONTRACT]

residual_cal = y_cal - pred_cal
```

The calibrated offsets are later added to the point prediction:

\[
[\hat y + q_{low},\; \hat y + q_{high}].
\]

---

# 5. `src/conformal/taxonomy.py`

## 5.1 Responsibility

The main entry point is:

```python
compute_binning_scores(...)
```

Supported active modes:

```text
prediction
difficulty
```

It returns:

```text
scores_cal
scores_target
difficulty_model or None
```

---

## 5.2 Prediction taxonomy

For prediction-based taxonomy:

```text
scores_cal    = pred_cal
scores_target = pred_target
```

Interpretation for one target:

```text
low prediction  → low-prediction bins
mid prediction  → middle bins
high prediction → high-prediction bins
```

The assumption is that error behavior may vary across the predicted parameter range, especially under regression-to-the-mean, heteroscedasticity, or boundary effects.

---

## 5.3 Difficulty taxonomy

Difficulty-based taxonomy uses:

```text
calibration embedding
target embedding
calibration residuals
```

It estimates local expected error from nearby calibration samples in embedding space.

```text
target embedding
    ↓
nearest calibration embeddings
    ↓
calibration errors in that neighborhood
    ↓
difficulty score
    ↓
Mondrian bin
```

Target/test truth is not used.

---

# 6. `src/conformal/difficulty.py`

## 6.1 `DifficultyEstimator`

Purpose:

> Estimate local prediction difficulty from calibration residuals in CNN embedding space.

Important defaults:

```text
n_neighbors             = 5
distance_weighted       = True
distance_eps            = 1e-8
metric                  = euclidean
standardize_embeddings  = False
```

The score is computed independently for each target.

---

## 6.2 Calibration

The estimator receives:

```text
cal_embedding
cal_residuals = y_cal - pred_cal
```

and stores absolute residuals:

```text
abs(cal_residuals)
```

It fits `sklearn.neighbors.NearestNeighbors` on the calibration embeddings.

Closed M10 therefore uses the CNN 64-D latent space as the geometry for local similarity.

---

## 6.3 Difficulty-score formula

For the `k` nearest calibration neighbors of a target sample \(x\):

\[
N_k(x)
\]

and target-specific absolute residuals \(|r_{n,j}|\), the unweighted score would be:

\[
d_j(x)=\frac{1}{k}\sum_{n\in N_k(x)}|r_{n,j}|.
\]

By default the code uses inverse-distance weighting:

\[
w_n =
\frac{1/(\delta_n+\epsilon)}
{\sum_m 1/(\delta_m+\epsilon)}
\]

and:

\[
d_j(x)=\sum_n w_n |r_{n,j}|.
\]

Thus the same sample can be easy for one target and difficult for another.

---

## 6.4 Self-exclusion for calibration samples

When computing difficulty for the calibration set itself, the estimator requests `n_neighbors + 1` and removes the self-match.

This avoids:

```text
sample i
→ nearest neighbor = itself
→ distance 0
→ own residual contaminates difficulty estimate
```

Target/test samples do not require self-exclusion.

---

## 6.5 Embedding standardization

The class supports:

```text
standardize_embeddings = True
```

but the standard M10 `fit_mondrian()` path leaves the default `False`.

Therefore:

```text
closed M10 difficulty geometry
=
Euclidean distance on raw 64-D embeddings
without an additional embedding z-score
```

### Scientific assumption

```text
[SCIENTIFIC ASSUMPTION]

Nearby points in CNN embedding space are assumed to have
informative local prediction-error behavior.
```

---

# 7. `src/conformal/binning.py`

Contains:

```text
QuantileBinner
BinGrouper
```

It performs:

```text
continuous score → discrete Mondrian bin

residual + bin index → residual groups per target/bin
```

---

## 7.1 `QuantileBinner`

Bin edges are fitted from calibration scores only, independently for each target.

For:

```text
n_bins = 4
```

quantile edges correspond to:

```text
0%, 25%, 50%, 75%, 100%
```

In general:

\[
q_k = k/K,
\qquad
k=0,\ldots,K.
\]

This tends to create approximately balanced calibration groups and is more robust than equal-width bins when score distributions are non-uniform.

---

## 7.2 Calibration-only edge fitting

Correct flow:

```text
calibration scores
    ↓
fit quantile edges
    ↓
freeze edges
    ↓
assign calibration
    ↓
assign test
    ↓
assign real targets
```

The test/real score distribution does not redefine the edges.

---

## 7.3 Bin assignment

After fitting internal thresholds, `np.digitize()` assigns every sample/target to:

```text
0 ... n_bins-1
```

Output:

```text
bin_indices.shape = (N, n_labels)
```

A single sample may belong to different bin numbers for different targets.

---

## 7.4 Degenerate edges and jitter

Repeated scores can produce repeated quantile edges and empty or uneven bins.

The binner supports:

```text
apply_jitter
jitter_variation = 1e-10
```

If enabled:

\[
s_i' = s_i + \epsilon_i,
\qquad
\epsilon_i \sim U(-10^{-10},10^{-10}).
\]

The jitter is used only while fitting bin edges and acts as a tie-breaker.

### M10 reproducibility caveat

If no RNG is provided, the class creates:

```python
np.random.default_rng()
```

without an explicit seed. `fit_mondrian()` does not pass a seeded RNG.

```text
[REPRODUCIBILITY NOTE]

When apply_jitter=True, exact tie resolution is not guaranteed
to be bitwise reproducible in the standard M10 path.
```

The jitter magnitude is negligible scientifically, but this remains a reproducibility detail of the closed baseline.

---

## 7.5 `BinGrouper`

Inputs:

```text
residuals
bin_indices
n_bins
```

Output:

```text
grouped_residuals
shape = (n_labels, n_bins)
dtype = object
```

Conceptually:

```text
chirp_mass
├── bin 0 → [r1, r5, r9, ...]
├── bin 1 → [...]
└── ...

total_mass
├── bin 0 → [...]
└── ...

chi_eff
└── ...
```

This is exactly what the conformal calibrator consumes.

---

# 8. `src/conformal/calibration.py`

## 8.1 `ConformalIntervalCalibrator`

Purpose:

> Convert grouped calibration residuals into lower/upper offsets for every target/bin pair.

Input:

```text
grouped_residuals
shape = (n_labels, n_bins)
```

Output:

```text
intervals_
shape = (n_labels, n_bins, 2)
```

with:

```text
[...,0] = lower offset
[...,1] = upper offset
```

It also stores:

```text
bin_counts_
quantile_indices_
interval_widths_
```

---

## 8.2 Symmetric intervals

For one target/bin with \(m\) calibration residuals, the code sorts:

\[
|r_1|,\ldots,|r_m|.
\]

It chooses the conformal order statistic:

\[
k = \left\lceil (m+1)\cdot c \right\rceil -1
\]

with zero-based indexing, where \(c\) is the confidence level.

For M10:

\[
c=0.90.
\]

If the selected absolute residual is \(q\), the offsets are:

\[
[-q,+q]
\]

and the interval is:

\[
[\hat y-q,\hat y+q].
\]

---

## 8.3 Asymmetric intervals

For nominal coverage:

\[
1-\alpha
\]

the code assigns:

\[
\alpha/2
\]

to each tail.

At 90% coverage:

\[
\alpha=0.10,
\qquad
\alpha/2=0.05.
\]

Signed residuals are sorted and lower/upper order statistics are selected:

\[
q_{low}, q_{high}.
\]

The interval becomes:

\[
[\hat y+q_{low},\hat y+q_{high}].
\]

Conceptually:

```text
symmetric:
       prediction
           |
      <----|---->

asymmetric:
       prediction
           |
    <------|--->
```

This allows systematic asymmetry in the residual distribution to be represented.

---

## 8.4 Finite-sample order statistics

The implementation selects discrete residual order statistics rather than interpolated percentiles.

This is appropriate for finite-sample split-conformal calibration.

---

## 8.5 Minimum samples per bin

Default technical requirement:

```text
min_samples_per_bin = 10
```

If a label/bin group contains fewer residuals, fitting fails.

For asymmetric calibration the code also warns when the selected lower/upper statistic lies at the minimum or maximum residual, indicating limited tail support.

---

# 9. `src/conformal/apply.py`

Main function:

```python
apply_indices(...)
```

Inputs:

```text
point predictions
bin indices
calibrated offsets
```

For each sample/target:

\[
L_i = \hat y_i + \Delta^{low}_{b(i)}
\]

\[
U_i = \hat y_i + \Delta^{high}_{b(i)}.
\]

No target truth is used.

This is the fundamental truth-free deployment operation.

---

# 10. `src/conformal/pipeline.py`

The preferred modern interface exposes:

```text
fit_mondrian()
apply_mondrian()
evaluate_mondrian()
```

and retains:

```text
run_mondrian_regression()
```

as a combined convenience/backward-compatible API.

---

## 10.1 `FittedMondrianRegressor`

Stores calibration-derived state:

```text
binner
calibrator

taxonomy_mode
interval_mode
confidence_level
n_bins

calibration bin indices
calibration residual groups
calibration binning scores

difficulty model, if applicable
```

Critical property:

> It contains only calibration-derived state and can be applied to arbitrary targets without target labels.

---

## 10.2 `fit_mondrian()`

Inputs:

```text
pred_cal
y_cal
n_bins

cal_embedding, if difficulty
n_neighbors, if difficulty

confidence_level
apply_jitter
jitter_variation
interval_mode
taxonomy_mode
min_samples_per_bin
```

Workflow:

```text
pred_cal + y_cal
    ↓
residuals_cal = y_cal - pred_cal
    ↓
taxonomy score
    ↓
fit quantile binner
    ↓
assign calibration bins
    ↓
group calibration residuals
    ↓
fit conformal offsets
    ↓
FittedMondrianRegressor
```

No test or real-event truth is involved.

---

## 10.3 `apply_mondrian()`

Inputs:

```text
fitted system
pred_target
target_embedding, if difficulty
```

Prediction taxonomy:

```text
pred_target → score → bin → offset
```

Difficulty taxonomy:

```text
target embedding → difficulty score → bin → offset
```

Output:

```text
MondrianPrediction
├── lower
├── upper
├── bin_indices
└── binning_scores
```

No `y_target` argument exists.

---

## 10.4 `evaluate_mondrian()`

This function receives already-built intervals plus:

```text
y_true
```

and computes coverage/width diagnostics.

Thus:

```text
INTERVAL CONSTRUCTION
    does not use target truth

EVALUATION
    uses target truth
```

This separation is essential for real-event use.

---

## 10.5 `run_mondrian_regression()`

Combined wrapper:

```text
fit + apply + evaluate
```

It requires `y_test` because it evaluates immediately.

Classification:

```text
ACTIVE CONVENIENCE / BACKWARD-COMPATIBILITY API
```

For deployment, the clearer abstraction is the separate fit/apply/evaluate path.

---

# 11. `src/conformal/metrics.py`

## 11.1 `CoverageEvaluator`

Purpose:

> Measure global validity, local validity, and interval efficiency.

Global outputs include:

```text
global_coverage
miscoverage
global_coverage_gap
covered_count_global
global_undercoverage_pvalue

global_mean_width
global_median_width

global_lower_miss_rate
global_upper_miss_rate
global_tail_miss_imbalance
```

Bin-wise outputs include:

```text
coverage_per_bin
counts_per_bin
covered_count_per_bin
min_coverage_per_label
max_undercoverage_gap
bin_undercoverage_pvalue

mean_width_per_bin
median_width_per_bin

lower_miss_rate_per_bin
upper_miss_rate_per_bin
```

It also computes nominal tolerance bands.

---

## 11.2 Why global coverage is insufficient

Example:

```text
bin A coverage = 0.99
bin B coverage = 0.81

global coverage ≈ 0.90
```

Global coverage can appear nominal while local coverage is poor.

Mondrian evaluation therefore explicitly measures local calibration behavior.

---

## 11.3 Nominal normal tolerance bands

For nominal coverage \(p\) and sample count \(n\):

\[
\sigma = \sqrt{\frac{p(1-p)}{n}}.
\]

The evaluator reports:

\[
p \pm k\sigma
\]

for configured values such as 1σ, 2σ, 3σ.

These are not error bars around measured coverage. They answer:

> If true coverage were nominal, what empirical fluctuation is expected for this sample count?

---

## 11.4 One-sided binomial undercoverage tests

Null hypothesis:

\[
H_0: p=p_{nominal}.
\]

Alternative:

\[
H_1: p<p_{nominal}.
\]

Small p-values indicate unusually severe undercoverage under the nominal assumption.

---

## 11.5 Tail miss imbalance

Defined as:

\[
|P(y<L)-P(y>U)|.
\]

A small value means lower- and upper-tail misses are relatively balanced.

This is particularly informative for asymmetric intervals.

---

# 12. `src/conformal/selection.py`

This module selects among already evaluated conformal configurations.

Candidate grid varies:

```text
taxonomy:
    prediction
    difficulty

interval mode:
    symmetric
    asymmetric

n_bins:
    multiple tested values
```

Selection is target-specific.

---

## 12.1 Common admissibility criteria

A candidate must satisfy:

```text
requested label

global coverage inside nominal 2σ tolerance band

minimum test count in every bin >= 200
```

This count threshold is much stricter than the technical calibrator minimum and protects against locally unstable candidate configurations.

---

## 12.2 Conservative policy

Goal:

> Prioritize local validity while avoiding unnecessary width.

Steps:

1. common admissibility;
2. prefer `n_bins_under_2sigma == 0`;
3. otherwise retain minimum number of undercovered bins;
4. find minimum global median physical width;
5. treat candidates within 2% of that minimum as practically tied;
6. among tied candidates prefer:

```text
more bins
smaller max undercoverage gap
smaller tail-miss imbalance
smaller median width
```

This is not simply a narrowest-interval rule.

---

## 12.3 Efficient policy

Additional local-validity constraints:

```text
fraction bins below 2σ <= 0.10
max undercoverage gap <= 0.05
```

Among acceptable candidates, width is prioritized.

If no configuration satisfies these conditions:

```text
fallback → conservative policy
```

---

# 13. Closed M10 selected configurations

## Conservative

| target | taxonomy | interval mode | bins |
|---|---|---|---:|
| `chirp_mass` | difficulty | asymmetric | 4 |
| `total_mass` | prediction | asymmetric | 6 |
| `chi_eff` | prediction | symmetric | 8 |

## Efficient

| target | taxonomy | interval mode | bins |
|---|---|---|---:|
| `chirp_mass` | difficulty | symmetric | 4 |
| `total_mass` | difficulty | symmetric | 4 |
| `chi_eff` | difficulty | asymmetric | 12 |

Important interpretation:

> No single taxonomy or interval mode dominates for all targets.

---

## 13.1 Why few bins can be optimal

Tradeoff:

```text
more bins
    → more local resolution
    → fewer samples per bin
    → noisier quantiles
    → less stable local coverage

fewer bins
    → less local resolution
    → more samples per bin
    → more stable quantiles
```

Therefore small selected `n_bins` values can be the correct finite-sample compromise.

---

# 14. `src/conformal/selected_calibrators.py`

Purpose:

```text
selected configuration table
        ↓
fit selected calibration systems
        ↓
apply to arbitrary target predictions
```

This bridges synthetic model selection and real-event deployment.

---

## 14.1 `SelectedMondrianCalibrator`

Stores:

```text
final_policy
label
label_index
selection_policy
taxonomy_mode
interval_mode
n_bins
fitted Mondrian system
```

Typical dictionary keys:

```text
(conservative, chirp_mass)
(conservative, total_mass)
(conservative, chi_eff)
(efficient, chirp_mass)
(efficient, total_mass)
(efficient, chi_eff)
```

---

## 14.2 `fit_selected_calibrators()`

For every selected row it calls `fit_mondrian()` using:

```text
pred_cal
y_cal
emb_cal, when required
```

Only calibration data are needed.

---

## 14.3 Important implementation detail: all labels are fitted at once

`fit_mondrian()` calibrates all model outputs simultaneously under the chosen configuration.

A selected row may target only `chirp_mass`, but the fitted object contains all three target columns. `label_index` specifies which one is extracted later.

This is valid but important for understanding object structure.

---

## 14.4 `apply_selected_calibrators()`

Inputs:

```text
selected fitted calibrators
pred_target
label_names
target_embedding, if needed
event_name, optional
```

No target truth is required.

Output is a long-format DataFrame containing fields such as:

```text
event
sample_index
final_policy
label
label_index
selection_policy
taxonomy_mode
interval_mode
n_bins
pred_std
lower_std
upper_std
width_std
mondrian_bin
```

This is the truth-free interface appropriate for real-event application.

---

# 15. Critical truth-separation contract

```text
CALIBRATION
pred_cal
y_cal
emb_cal
    ↓
truth allowed
    ↓
fit conformal system

TEST
pred_test
emb_test
    ↓
construct intervals without truth
    ↓
y_test used only for evaluation

REAL EVENT
pred_real
emb_real
    ↓
construct intervals
    ↓
no target truth required
```

```text
[CRITICAL DATA-SEPARATION CONTRACT]

Calibration truth:
    used for fitting

Test truth:
    evaluation only

Real-event truth:
    not required
```

---

# 16. `src/conformal/hybrid.py`

## 16.1 Status

```text
EXPERIMENTAL / NOT ADOPTED IN CLOSED M10
```

The hybrid method combines:

```text
prediction bin × difficulty bin
```

It was implemented and evaluated as a methodological experiment but did not show significant enough improvement to be retained in the final M10 pipeline.

It should not be classified as dead code.

---

## 16.2 `HybridQuantileBinner`

Fits two independent quantile binners:

```text
prediction binner
difficulty binner
```

and combines indices as:

\[
b_{joint}
=
b_{pred}N_{difficulty}+b_{difficulty}.
\]

If:

```text
n_prediction_bins = P
n_difficulty_bins = D
```

then:

\[
N_{joint}=PD.
\]

Example:

```text
P=4
D=4
→ 16 joint bins
```

---

## 16.3 Hybrid hypothesis

The motivation is sensible:

```text
same prediction, different difficulty

or

same difficulty, different prediction region
```

may correspond to different error distributions.

The hybrid taxonomy attempts to preserve both pieces of information.

---

## 16.4 Finite-sample cost

The number of groups grows multiplicatively.

For 30k calibration samples, idealized average counts are:

```text
4×4   = 16 bins  → 1875/bin
6×6   = 36 bins  → 833/bin
8×8   = 64 bins  → 469/bin
12×12 = 144 bins → 208/bin
```

Actual joint occupancy need not be uniform.

Potential consequences:

```text
less stable quantiles
more extreme order statistics
weaker asymmetric-tail estimates
noisier local coverage estimates
```

This creates a plausible reason why increased taxonomic complexity did not automatically improve performance.

---

## 16.5 Hybrid reproducibility

Unlike the standard `fit_mondrian()` path, the hybrid runner accepts an explicit `random_seed` and derives separate RNG streams for prediction and difficulty binners.

This is a useful implementation detail, although the method is not part of final M10.

---

# 17. Active vs experimental classification

## Closed-M10 active conformal path

```text
taxonomy.py
difficulty.py
binning.py
calibration.py
apply.py
metrics.py
pipeline.py
selection.py
selected_calibrators.py
```

## Experimental / not adopted

```text
hybrid.py
```

Reason:

```text
prediction × difficulty experiment
implemented and evaluated
no significant improvement observed
not retained in final M10 selection
```

---

# 18. Scientific assumptions

## 18.1 Difficulty locality

```text
nearby embeddings
→ similar local prediction difficulty
```

This is a modeling choice used to define Mondrian groups, not a guarantee of conformal theory itself.

## 18.2 Prediction locality

```text
prediction region
→ informative about error distribution
```

Again, this is a taxonomy choice.

## 18.3 Exchangeability

Split-conformal guarantees are tied to calibration/target exchangeability under the same distribution.

Applying synthetic calibration to real GWOSC events is a domain-transfer experiment.

Therefore:

```text
synthetic nominal coverage
≠
automatic formal real-data coverage guarantee
```

---

# 19. Finite-sample tradeoffs

The central design tension is:

```text
taxonomic resolution
vs
samples per bin
```

Increasing `n_bins` may improve localization but can also:

```text
reduce bin counts
increase quantile variance
force extreme order statistics
create local undercoverage instability
widen intervals
```

This is why M10 selection considers count and local-validity diagnostics, not only interval width.

---

# 20. Reproducibility notes

## 20.1 Jitter

Tiny but unseeded in the standard path when enabled.

## 20.2 Selection defaults

Important closed-M10 values include:

```text
minimum count per bin            = 200
width tie fraction               = 0.02
efficient max under-bin fraction = 0.10
efficient max undercoverage gap  = 0.05
```

These are part of the closed selection definition.

---

# 21. Audit findings

### `[ACTIVE M10]`

```text
taxonomy.py
difficulty.py
binning.py
calibration.py
apply.py
metrics.py
pipeline.py
selection.py
selected_calibrators.py
```

### `[EXPERIMENTAL / NOT SELECTED]`

```text
hybrid.py
```

### `[CRITICAL CONFORMAL CONTRACT]`

```text
residual = truth - prediction
```

### `[CRITICAL DATA-SEPARATION CONTRACT]`

```text
calibration truth → fitting allowed
test truth        → evaluation only
real-event truth  → not required
```

### `[SCIENTIFIC ASSUMPTION]`

Difficulty relies on local residual behavior in CNN embedding space.

### `[SCIENTIFIC ASSUMPTION]`

The standard M10 difficulty path uses Euclidean distance on raw 64-D embeddings without extra embedding standardization.

### `[FINITE-SAMPLE TRADEOFF]`

More bins increase taxonomic resolution but reduce local sample support.

### `[REPRODUCIBILITY NOTE]`

Standard-path jitter uses an unseeded default RNG.

### `[ARCHITECTURAL NOTE]`

`run_mondrian_regression()` remains useful for end-to-end synthetic evaluation, but separate `fit_mondrian()` / `apply_mondrian()` / `evaluate_mondrian()` is the cleaner deployment abstraction.

---

# 22. Change-impact guide

## If changing taxonomy

Review:

```text
src/conformal/taxonomy.py
src/conformal/difficulty.py
src/conformal/pipeline.py
notebook 11 candidate grid
selection results
selected calibrators
real-event application
```

Requires recalibration and re-evaluation.

## If changing embeddings

Affected:

```text
CNN checkpoint
prediction artifact embeddings
DifficultyEstimator neighborhoods
difficulty scores
Mondrian bins
calibration offsets
selected configurations
real-event intervals
```

A new embedding invalidates old difficulty-based calibration.

## If changing `n_neighbors`

Affected:

```text
difficulty scores
difficulty bins
residual grouping
conformal offsets
selection results
```

Treat as a conformal hyperparameter requiring recalibration.

## If changing `n_bins`

Affected:

```text
quantile edges
bin counts
residual groups
order statistics
coverage stability
interval widths
selection result
```

More bins are not automatically better.

## If changing interval mode

Switching symmetric/asymmetric changes the calibrated residual statistic and requires full conformal refit.

## If changing confidence level

Affected:

```text
order-statistic indices
interval widths
coverage target
tolerance bands
selection interpretation
```

Requires complete recalibration.

## If changing jitter behavior

Review:

```text
QuantileBinner
fit_mondrian
selected_calibrators
reproducibility tests
closed-M10 compatibility
```

A seeded implementation should be introduced prospectively, not silently inside M10.

## If changing selection policy

Review:

```text
src/conformal/selection.py
model_summary columns
selected_configurations.csv
selected calibrators
real-event reported intervals
```

Selection changes may leave candidate systems unchanged but alter which systems are designated final.

---

# 23. Closed M10 conformal workflow summary

```text
CNN calibration predictions
CNN calibration truth
CNN calibration embeddings
        │
        ▼
residuals = truth - prediction
        │
        ├───────────────────────┐
        │                       │
        ▼                       ▼
prediction score        difficulty score
                        via kNN in 64-D embedding
        │                       │
        └───────────┬───────────┘
                    ▼
              QuantileBinner
                    │
                    ▼
             calibration bins
                    │
                    ▼
               BinGrouper
                    │
                    ▼
      residual distributions per bin
                    │
                    ▼
    ConformalIntervalCalibrator
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     symmetric            asymmetric
          │                   │
          └─────────┬─────────┘
                    ▼
       FittedMondrianRegressor
                    │
          ┌─────────┴────────────┐
          ▼                      ▼
 synthetic test             real target
          │                      │
     pred + emb               pred + emb
          │                      │
          └─────────┬────────────┘
                    ▼
             apply_mondrian
                    │
                    ▼
              lower / upper
                    │
          ┌─────────┴────────┐
          ▼                  ▼
    evaluate test        truth-free use
          │
          ▼
 coverage / width / local validity
```

---

# 24. Mental model to retain

```text
taxonomy.py
    = choose what defines similarity

difficulty.py
    = estimate local error difficulty from embeddings

binning.py
    = convert continuous scores into Mondrian groups

calibration.py
    = convert calibration residuals into conformal offsets

apply.py
    = add the correct offsets to target predictions

metrics.py
    = evaluate global and local validity

pipeline.py
    = clean fit / apply / evaluate interface

selection.py
    = choose which evaluated configurations become final

selected_calibrators.py
    = rebuild and deploy the selected systems

hybrid.py
    = historical prediction × difficulty experiment,
      not adopted in final M10
```

Shortest accurate description:

```text
M10 conformal
=
dedicated synthetic calibration split
+
prediction- or embedding-difficulty-based Mondrian bins
+
symmetric/asymmetric finite-sample residual calibration
+
explicit local coverage diagnostics
+
conservative / efficient target-specific selection
+
truth-free application to real-event CNN predictions
```

---

# 25. Status of this manual section

This document describes the **closed M10 conformal behavior**.

The hybrid method is retained as an implemented but non-adopted experiment.

Audit notes and future improvements are intentionally separated from active baseline behavior.

Future changes to taxonomy, embedding geometry, number of neighbors, number of bins, interval construction, jitter reproducibility, or selection thresholds should be treated as new conformal methodology and evaluated against the closed M10 reference.
