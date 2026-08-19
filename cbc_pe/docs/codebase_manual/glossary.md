# Glossary

## Analysis window

Final time interval presented to the CNN.

Closed M10:

```text
4 s
```

---

## Processing context

Extra samples before/after the final analysis window used to absorb whitening/filter edge effects.

---

## Physical waveform support

The actual portion of the gravitational waveform retained before processing.

Not necessarily identical to processing context.

---

## Geocentric coalescence time

Reference GPS-like time defined at Earth's center before detector-specific propagation delay.

---

## Detector arrival time

Geocentric reference time plus detector-specific geometric delay.

---

## Projection

Conversion of \(h_+\) and \(h_\times\) into detector strain using antenna response and propagation delay.

---

## Windowing

Selection/truncation of the projected physical GW network before later embedding/injection.

---

## Injection

Placement and addition of a projected signal into a strain segment using absolute time coordinates.

---

## PSD

Power spectral density.

Synthetic M10 uses analytical design PSDs; real M10 estimates PSD from off-source GWOSC strain.

---

## Whitening

Frequency-dependent normalization by the square root of a PSD to approximately flatten colored noise.

---

## Optimal SNR

Matched-filter-like signal norm computed from a signal-only segment and PSD.

---

## Network SNR

\[
\rho_\mathrm{net}
=
\sqrt{\sum_d \rho_d^2}.
\]

---

## Chirp mass

\[
\mathcal{M}
=
\frac{(m_1m_2)^{3/5}}
{(m_1+m_2)^{1/5}}.
\]

---

## Total mass

\[
M = m_1 + m_2.
\]

---

## Effective aligned spin

\[
\chi_\mathrm{eff}
=
\frac{
m_1\chi_{1z}
+
m_2\chi_{2z}
}{
m_1+m_2
}.
\]

---

## Source-frame mass

Intrinsic mass before cosmological redshift scaling.

---

## Detector-frame mass

Observed redshifted mass:

\[
M_\mathrm{det}
=
(1+z)M_\mathrm{source}.
\]

---

## Label standardization

Train-only transformation:

\[
y_\mathrm{std}
=
\frac{y-\mu_\mathrm{train}}
{\sigma_\mathrm{train}}.
\]

---

## M10 input z-score

Per-sample/per-detector temporal normalization:

\[
X'_{c,t}
=
\frac{X_{c,t}-\mu_c}
{\sigma_c+\epsilon}.
\]

---

## Embedding

Latent CNN representation before the final regression head.

Closed M10:

```text
64 dimensions
```

---

## Regression to the mean

Tendency of a point regressor trained with squared-error-like losses to predict values closer to the conditional mean, often producing low-target overprediction and high-target underprediction.

---

## Calibration split

Dedicated held-out synthetic data used to fit conformal uncertainty, not CNN weights.

Closed M10:

```text
30000 samples
```

---

## Test split

Held-out data used for final synthetic evaluation.

Closed M10:

```text
30000 samples
```

---

## Conformal residual

Closed implementation:

\[
r = y-\hat y.
\]

---

## Split conformal prediction

Uncertainty method using a held-out calibration set to derive finite-sample prediction intervals under exchangeability assumptions.

---

## Mondrian conformal prediction

Conformal calibration performed separately inside groups/bins defined by a taxonomy.

---

## Prediction taxonomy

Mondrian grouping based on the CNN point prediction.

---

## Difficulty taxonomy

Mondrian grouping based on an estimated local prediction-error difficulty derived from neighboring calibration embeddings.

---

## Quantile binning

Creating groups using quantiles of calibration taxonomy scores to obtain approximately balanced calibration groups.

---

## Symmetric conformal interval

\[
[\hat y-q,\hat y+q].
\]

---

## Asymmetric conformal interval

\[
[\hat y+q_\mathrm{low},
 \hat y+q_\mathrm{high}].
\]

---

## Coverage

Fraction of targets whose true value lies inside the prediction interval.

---

## Local coverage

Coverage measured inside a Mondrian bin or subgroup.

---

## Undercoverage

Empirical coverage below nominal target coverage.

---

## Tail miss imbalance

Absolute difference between lower-tail and upper-tail miss rates.

---

## Conservative conformal policy

Final-selection strategy prioritizing robust local validity while controlling interval width.

---

## Efficient conformal policy

Final-selection strategy prioritizing narrower intervals subject to explicit local-validity constraints.

---

## Hybrid taxonomy

Historical experiment combining prediction-bin and difficulty-bin indices into joint groups.

Not adopted in closed M10.

---

## Off-source PSD

PSD estimated from detector data away from the event window.

---

## GWOSC

Gravitational Wave Open Science Center.

Used for public detector strain and catalog metadata.

---

## LVK

LIGO-Virgo-KAGRA Collaboration.

In this project, "LVK reference" means published event parameter summaries used as an external comparison reference.

---

## `project_root`

Repository root containing code/configs/notebooks/docs.

---

## `data_root`

External storage root containing large generated and downloaded artifacts.

---

## HDF5

Hierarchical Data Format used for the large synthetic dataset.

---

## Split NPZ

NumPy archive storing train/val/cal/test physical HDF5 indices.

---

## Checkpoint

Serialized PyTorch training artifact containing model weights plus model/training metadata and target statistics.

---

## Prediction artifact

NPZ containing cal/test point predictions, embeddings, truths, physical indices, and provenance metadata.

---

## Characterization test

Test that freezes existing behavior before refactoring, even if the behavior is not being claimed as theoretically optimal.

---

## Audit notebook

Notebook used to investigate or justify a methodological choice rather than to define the final active pipeline.

---

## Final notebook

Notebook considered part of the authoritative closed-M10 analysis/reporting workflow.

---

## Domain gap

Difference between the synthetic training/calibration distribution and real detector data.

Relevant sources include:

```text
noise statistics
PSD mismatch
nonstationarity
glitches/transients
waveform mismatch
processing-context differences
```

---

## Nonstationarity

Time variation in detector-noise statistical properties.

A PSD estimated at one time may not perfectly represent the noise around another time.

---

## Transient / glitch

Short-duration non-Gaussian detector-noise feature not described well by stationary Gaussian noise.

---

## Closed M10 baseline

Immutable reference pipeline identified by:

```text
tag: m10-closed-baseline
commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
```

Future methodology should be compared against, not silently merged into, this reference.
