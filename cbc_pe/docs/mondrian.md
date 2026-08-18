# Mondrian conformal workflow

This document describes the Mondrian conformal prediction workflow used by the closed **M10-500k** CBC parameter-estimation baseline.

The objective is to construct calibrated prediction intervals around CNN point estimates while allowing interval widths to adapt to different regions of the regression problem.

The conformal implementation is located under:

```text
src/conformal/
```

The final M10 analysis is performed in:

```text
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
```

## 1. Statistical objective

For a target miscoverage level:

```text
alpha
```

the desired prediction interval satisfies approximately:

```text
P(
    y_true ∈ [lower, upper]
) = 1 - alpha
```

For the closed M10 analysis:

```text
confidence level = 0.90
alpha            = 0.10
```

Conformal intervals are frequentist prediction intervals.

They are not Bayesian posterior credible intervals.

## 2. Calibration and test separation

The closed synthetic split is:

```text
train: 400000
val:    40000
cal:    30000
test:   30000
```

Roles:

```text
train → fit CNN parameters
val   → training diagnostics / model selection
cal   → fit conformal calibration quantities
test  → evaluate conformal performance
```

The test set must not be used to fit conformal quantiles.

The calibration set must remain independent of CNN training.

## 3. Required CNN outputs

The closed M10 prediction artifact is:

```text
m10_inputzscore_500k_cal_test_predictions_embeddings.npz
```

stored under:

```text
<data_root>/results/<dataset_id>/
```

The conformal pipeline uses:

```text
calibration predictions
calibration targets
calibration embeddings

test predictions
test targets
test embeddings
```

Embeddings are required only for taxonomy modes based on learned representation-space difficulty.

## 4. Basic conformal regression

Let:

```text
y_hat
```

be the CNN point prediction and:

```text
y
```

the true target.

A residual can be defined as:

```text
r = y - y_hat
```

or equivalently under the opposite sign convention if used consistently.

Conformal calibration estimates empirical residual quantiles on the calibration set.

These quantiles are then applied to future predictions without access to target truth.

## 5. Mondrian conformal prediction

Standard split conformal regression uses one global calibration distribution.

Mondrian conformal prediction partitions samples into groups and calibrates separate interval corrections within each group.

Generic procedure:

```text
1. compute a taxonomy score for calibration samples
2. partition calibration scores into bins
3. compute conformal residual quantiles within each bin
4. compute the taxonomy score for a target sample
5. assign the sample to the corresponding bin
6. apply the bin-specific conformal correction
```

The objective is to adapt interval width to local prediction difficulty while preserving adequate calibration support in every bin.

## 6. Taxonomy modes

The closed M10 implementation supports two principal taxonomy modes.

### Prediction taxonomy

The taxonomy score is based on the CNN point prediction itself.

Conceptually:

```text
score = y_hat
```

Samples are therefore grouped according to their predicted physical value.

Examples:

```text
low predicted chirp mass
intermediate predicted chirp mass
high predicted chirp mass
```

Advantages:

- simple;
- interpretable;
- inexpensive;
- applicable without embeddings.

Limitations:

- prediction value may not capture all sources of local model difficulty.

### Difficulty taxonomy

Difficulty-based Mondrian uses the CNN embedding space and calibration residual information.

The central idea is:

```text
nearby samples in embedding space
may have similar regression difficulty
```

For a target embedding, local calibration neighbors are identified and their residual behavior is used to estimate a difficulty score.

Conceptually:

```text
embedding
→ local calibration neighbors
→ local residual scale
→ difficulty score
```

The resulting difficulty score is then used for Mondrian binning.

Advantages:

- can adapt directly to local error structure;
- can capture difficulty not explained only by the predicted target value.

Limitations:

- depends on embedding quality;
- depends on calibration-neighbor structure;
- requires target embeddings at application time.

## 7. Prediction-time truth independence

A fitted conformal calibrator must be applicable without target truth.

This requirement is explicitly protected in the current implementation.

For a new real event, the application stage may use:

```text
CNN point prediction
CNN embedding
fitted calibration objects
```

but it must not require:

```text
true event parameter
target residual
```

This separation is important because real-event truth is unavailable.

## 8. Fit, apply, and evaluate stages

The reusable conformal pipeline explicitly separates:

```text
fit
apply
evaluate
```

### Fit

Calibration data are used to construct:

- taxonomy bins;
- bin thresholds;
- residual quantiles;
- difficulty estimators where required.

### Apply

A fitted calibrator receives new predictions and, for difficulty-based taxonomies, embeddings.

It returns prediction intervals without requiring true labels.

### Evaluate

When true labels are available, such as on the synthetic test set, interval outputs are evaluated for coverage and efficiency.

This design avoids coupling real-event inference to synthetic test labels.

## 9. Interval modes

Two interval modes are supported.

### Symmetric intervals

Symmetric intervals use absolute residual magnitude.

For a bin-specific conformal quantile:

```text
q
```

the interval is:

```text
[y_hat - q, y_hat + q]
```

Advantages:

- simple;
- stable;
- robust when residual asymmetry is weak.

Limitations:

- cannot explicitly capture asymmetric residual structure.

### Asymmetric intervals

Asymmetric intervals estimate lower and upper residual corrections separately.

Conceptually:

```text
[y_hat + q_lower, y_hat + q_upper]
```

Advantages:

- can represent asymmetric residuals;
- can compensate for systematic direction-dependent errors;
- may provide more efficient intervals.

Limitations:

- requires stable calibration in both residual tails;
- can be more sensitive to finite calibration counts.

## 10. Quantile binning

Mondrian taxonomy scores are partitioned using quantile-based bins.

The objective is to obtain approximately balanced calibration counts across bins.

Increasing the number of bins produces more local adaptivity but reduces the number of calibration samples available in each group.

This produces a fundamental trade-off:

```text
few bins
→ stable calibration
→ weak local adaptivity

many bins
→ stronger local adaptivity
→ noisier calibration quantiles
```

Therefore, the largest possible number of bins is not automatically the best configuration.

## 11. Minimum bin support

Every Mondrian bin must contain enough calibration samples for meaningful quantile estimation.

Configurations with very small calibration counts per bin can produce:

- unstable interval widths;
- noisy local coverage;
- extreme tail behavior;
- misleading apparent adaptivity.

Minimum bin count is therefore part of the configuration-selection criteria.

## 12. Finite-sample quantiles

Conformal calibration uses empirical finite-sample quantiles rather than an asymptotic population quantile.

The exact quantile convention is part of the implementation and should remain unchanged for a closed baseline.

Changes to quantile indexing or interpolation can alter coverage behavior and therefore constitute a methodological change.

## 13. Closed M10 jitter behavior

The closed M10 Mondrian pipeline uses the existing quantile-binning jitter behavior.

A very small random perturbation may be applied to break exact score ties during bin construction.

This jitter is numerically negligible relative to the physical taxonomy scores and is intended only as a tie-breaking mechanism.

Its behavior is part of the closed M10 implementation and should not be silently changed during repository cleanup.

Any future change to deterministic tie handling should be introduced as a new methodological variant and validated separately.

## 14. Coverage

Global empirical coverage is:

```text
coverage =
    number of covered test targets
    / total number of test targets
```

For 90% nominal intervals, the target is approximately:

```text
0.90
```

However, global coverage alone is not sufficient.

A configuration can achieve nominal global coverage while under-covering significantly in specific regions of parameter space.

This is the main motivation for local Mondrian diagnostics.

## 15. Coverage per bin

For each Mondrian test bin:

```text
coverage_bin =
    covered samples in bin
    / samples in bin
```

This provides a local calibration diagnostic.

Important quantities include:

```text
coverage_per_bin
counts_per_bin
min_coverage_per_bin
```

A configuration with acceptable global coverage but poor minimum-bin coverage should be treated cautiously.

## 16. Binomial coverage uncertainty

Observed coverage fluctuates statistically around the nominal target.

For a bin containing:

```text
n
```

test samples and nominal coverage:

```text
p
```

a simple binomial standard deviation is approximately:

```text
sigma =
    sqrt(
        p * (1 - p) / n
    )
```

Local coverage diagnostics can therefore be compared against one- or two-sigma tolerance bands.

The closed analysis records quantities such as:

```text
n_bins_under_2sigma
n_bins_outside_2sigma
```

to identify locally problematic configurations.

These are diagnostics, not independent conformal guarantees.

## 17. Undercoverage diagnostics

A particularly important quantity is the maximum local deficit relative to the nominal target.

Conceptually:

```text
undercoverage_gap_bin =
    max(
        0,
        target_coverage - coverage_bin
    )
```

and:

```text
max_undercoverage_gap =
    maximum undercoverage gap across bins
```

This captures the severity of the worst local undercoverage.

A configuration with a small average interval width but a large worst-bin undercoverage gap may be scientifically undesirable.

## 18. Interval width

Coverage should always be interpreted jointly with interval efficiency.

For each sample:

```text
width = upper - lower
```

Useful summaries include:

```text
median width
mean width
90th percentile width
95th percentile width
width per bin
```

Narrow intervals are desirable only when adequate coverage is preserved.

## 19. Tail-miss behavior

For asymmetric intervals, it is useful to distinguish misses below and above the interval.

For example:

```text
lower misses:
    y_true < lower

upper misses:
    y_true > upper
```

Strong imbalance can indicate:

- asymmetric regression bias;
- imperfect tail calibration;
- parameter-boundary effects.

The closed M10 diagnostics include tail-miss imbalance as one of the interval-quality indicators.

## 20. Candidate configuration grid

The final M10 notebook evaluates combinations of:

```text
target label
taxonomy mode
interval mode
number of bins
```

This creates a grid of possible conformal configurations.

The objective is not simply to select the configuration with:

```text
largest n_bins
```

or:

```text
smallest width
```

but to balance local validity and interval efficiency.

## 21. Selection module

Reusable configuration-selection logic is implemented in:

```text
src/conformal/selection.py
```

Selection is performed independently for each target label.

The repository currently exposes two selection policies:

```text
conservative
efficient
```

These policies are designed to provide two scientifically interpretable operating points rather than one opaque "best" configuration.

## 22. Conservative policy

The conservative policy prioritizes local calibration robustness.

Its ranking favors configurations with:

```text
acceptable global coverage
good local-bin coverage
small undercoverage deficits
stable calibration support
```

before prioritizing interval width or a larger number of bins.

The conservative policy should therefore be interpreted as the safer conformal operating point when local validity matters more than maximal interval efficiency.

## 23. Efficient policy

The efficient policy searches among configurations that satisfy predefined validity limits and prioritizes narrower intervals.

Its objective is to obtain a more compact interval while remaining within acceptable local-calibration constraints.

If no candidate satisfies the efficient-policy requirements, the implementation falls back to the conservative selection rather than returning an invalid configuration.

This fallback behavior is regression-tested.

## 24. Selected configurations

The exact selected configurations are generated from the closed M10 analysis rather than hard-coded into conceptual documentation.

The authoritative selections are those produced by:

```text
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
```

and the associated saved selection artifact.

The current closed M10 selections include one configuration per:

```text
policy
× target label
```

for:

```text
conservative
efficient
```

and:

```text
chirp_mass
total_mass
chi_eff
```

The exact taxonomy, interval mode, and number of bins should be read from the saved closed selection table.

This avoids duplicating experiment-specific numerical results across multiple documentation files.

## 25. Reusable selected calibrators

Selected rows are converted into fitted reusable calibrators using:

```text
src/conformal/selected_calibrators.py
```

A selected calibrator records the association between:

```text
policy
target label
fitted Mondrian model
```

The application stage validates:

- label-index consistency;
- prediction shapes;
- embedding requirements;
- duplicate policy/label definitions.

This protects real-event inference against silent configuration mismatches.

## 26. Real-event application

For a real event:

```text
processed H1/L1/V1 strain
→ M10 CNN
→ point predictions
→ embeddings
→ selected fitted Mondrian calibrators
→ prediction intervals
```

No target truth is required.

For prediction-based taxonomies:

```text
point prediction
```

is sufficient.

For difficulty-based taxonomies:

```text
point prediction
+ target embedding
```

are required.

## 27. Synthetic versus real-data interpretation

The synthetic test set is drawn from the same experimental generation framework as the calibration set.

Under that setting, conformal coverage can be evaluated directly.

Real GWOSC data represent a domain-shifted application.

Potential differences include:

```text
real detector noise
non-stationarity
non-Gaussian transients
glitches
PSD mismatch
waveform mismatch
selection effects
```

Therefore, the nominal synthetic conformal coverage should not be interpreted as a formal real-data coverage guarantee.

Real-event intervals are an empirical transfer application of the synthetic calibration.

## 28. Relationship to LVK intervals

The M10 Mondrian interval and an LVK posterior interval have different statistical meanings.

Mondrian conformal interval:

```text
frequentist prediction interval
calibrated from held-out synthetic residuals
```

LVK interval:

```text
posterior credible interval
derived from Bayesian parameter estimation
```

Direct comparison is still informative, but agreement should be described in terms of:

```text
central-value consistency
interval overlap
relative width
systematic discrepancy
```

rather than assuming the intervals are statistically equivalent.

## 29. Physical clipping

Mass prediction intervals can mathematically extend below physical lower bounds.

For plots, physically clipped versions may be created.

However:

```text
original conformal bounds
```

must be retained for scientific coverage and overlap calculations.

Plotting transformations must not alter the reported conformal statistics.

## 30. Historical Mondrian analyses

Earlier Mondrian development notebooks are retained under:

```text
notebooks/_archive/m08_baseline/
notebooks/_archive/m10_development/
notebooks/_archive/architecture_search/
```

The lightweight reusable demo is:

```text
notebooks/demos/04_mondrian_hdf5_demo.ipynb
```

The authoritative closed M10 conformal analysis is:

```text
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
```

## 31. Reproducibility contract

A reported Mondrian result should preserve:

- CNN checkpoint;
- calibration/test prediction artifact;
- target label;
- taxonomy mode;
- interval mode;
- number of bins;
- confidence level;
- binning behavior;
- difficulty-estimation configuration;
- selection policy;
- fitted calibration state.

Changing any of these can change interval behavior.

Methodological changes should therefore be introduced under a new experimental identifier rather than silently modifying the closed M10 baseline.

## 32. Main interpretation principle

Mondrian conformal prediction is useful only if increased local adaptivity does not destroy calibration stability.

The relevant optimization problem is therefore:

```text
local validity
+ sufficient calibration support
+ acceptable interval width
```

not:

```text
maximize number of bins
```

and not:

```text
minimize interval width at any cost
```

The conservative and efficient policies are intended to expose that trade-off explicitly.