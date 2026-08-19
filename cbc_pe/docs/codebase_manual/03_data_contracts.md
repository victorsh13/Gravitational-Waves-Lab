# Data contracts

> Closed M10 reference:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Purpose

This document records shapes, ordering, units, semantic conventions, and artifact identities that must remain consistent across modules.

---

# 2. Detector order

Closed model detector order:

```text
0 = H1
1 = L1
2 = V1
```

This ordering is semantic.

A tensor with channels:

```text
L1,H1,V1
```

is not equivalent.

---

# 3. Time-series sampling contract

Closed M10:

```text
sampling frequency = 4096 Hz
delta_t = 1/4096 s
final duration = 4 s
final length = 16384 samples
```

Final detector channels must share:

```text
length
delta_t
aligned final start time
```

---

# 4. Model input contract

Dataset:

```text
X.shape = (N,3,16384)
```

Single sample:

```text
(C,T) = (3,16384)
```

Batch:

```text
(B,C,T)
```

Real single-event batch:

```text
(1,3,16384)
```

---

# 5. Label contract

Fixed order:

```text
0 = chirp_mass
1 = total_mass
2 = chi_eff
```

Physical units:

```text
chirp_mass = solar masses
total_mass = solar masses
chi_eff = dimensionless
```

---

# 6. CBC parameter contract

`CBCParameters` enforces:

```text
mass_1 >= mass_2
```

If masses are swapped, corresponding aligned-spin components are swapped too.

Angles:

```text
RA normalized modulo 2π
polarization normalized modulo 2π
```

Spins in current generator:

```text
spin_1z
spin_2z
```

only.

No in-plane spin components are part of this contract.

---

# 7. Time-reference contract

Detector projection uses an absolute geocentric coalescence time.

Closed default path uses a fixed geocentric GPS-like reference unless explicitly overridden.

Physical detector delays are introduced during detector projection, not injection.

---

# 8. Windowing contract

`windowing.py` selects a physical projected network of at most:

```text
config.duration
```

for closed M10.

For long signals:

```text
late network segment retained
```

to preserve merger/ringdown.

This is distinct from later processing-context cropping.

---

# 9. Processing-context contract

Processing input length:

```text
final 4 s
+
context before
+
context after
```

`SignalProcessor` eventually outputs exactly:

```text
config.length
```

samples.

Synthetic and real pipelines differ in physical signal support inside context; see the synthetic/real module manuals.

---

# 10. Noise contract

Synthetic:

```text
Gaussian colored noise
analytical design PSD
detector-specific PSD model
```

Real:

```text
actual GWOSC strain
empirical off-source PSD
```

These are not statistically equivalent domains.

---

# 11. SNR contract

Synthetic optimal detector SNR is computed from:

```text
signal-only final 4-s segment
+
compatible PSD
```

Network SNR:

\[
\rho_\mathrm{net}
=
\sqrt{\sum_d \rho_d^2}.
\]

Distance targeting:

\[
D_\mathrm{new}
=
D_\mathrm{old}
\frac{\rho_\mathrm{current}}
{\rho_\mathrm{target}}.
\]

---

# 12. Label standardization contract

Train-only statistics:

```text
y_mean.shape = (3,)
y_std.shape  = (3,)
```

Training target:

\[
y_\mathrm{std}
=
\frac{y_\mathrm{phys}-\mu_\mathrm{train}}
{\sigma_\mathrm{train}}.
\]

Inverse:

\[
y_\mathrm{phys}
=
y_\mathrm{std}\sigma_\mathrm{train}
+
\mu_\mathrm{train}.
\]

No validation/calibration/test samples should influence target statistics.

---

# 13. M10 input-normalization contract

Per sample, per detector:

\[
X'_{c,t}
=
\frac{X_{c,t}-\mu_c}
{\sigma_c+\epsilon}.
\]

Closed epsilon:

```text
1e-6
```

No global training-set input mean/std are used.

This removes absolute per-channel scale from the model input.

---

# 14. CNN output contract

Model output:

```text
pred_std.shape = (N,3)
```

Embedding:

```text
emb.shape = (N,64)
```

for closed M10.

The raw model output is standardized target space.

---

# 15. Prediction artifact ordering

Batch-mode prediction orders samples by:

```text
sorted physical HDF5 index
```

Therefore mapping must use:

```text
idx_cal
idx_test
```

saved in the artifact.

Never infer row identity from original split ordering alone.

---

# 16. Conformal residual contract

Calibration residual:

\[
r = y - \hat y.
\]

Not:

\[
\hat y - y.
\]

Offsets are added to point predictions.

---

# 17. Conformal binning contract

Quantile edges are fitted on:

```text
calibration scores only
```

then frozen and applied to target/test/real samples.

No target distribution is used to refit edges.

---

# 18. Difficulty-score contract

Uses:

```text
calibration embeddings
absolute calibration residuals
nearest-neighbor geometry
```

Target truth is not needed.

Closed standard path uses raw 64-D embedding with Euclidean distance and no extra embedding standardization.

---

# 19. Conformal application contract

Inputs:

```text
fitted calibration system
target point prediction
target embedding if difficulty
```

No target truth.

Outputs:

```text
lower
upper
bin_indices
binning_scores
```

---

# 20. Real-data PSD contract

PSD is estimated from an off-source interval using:

```text
raw long detector strain
event-relative offsets
segment-duration PSD estimator
interpolation
inverse spectrum truncation
```

A common valid offset window is selected across detectors.

---

# 21. Real preprocessing contract

`build_real_input_like_training()` outputs:

```text
X_real_raw
```

already:

```text
processed
cropped
stacked
```

but not M10-z-scored.

`event_runner.py` applies the model normalization afterward.

---

# 22. LVK frame contract

CNN masses:

```text
detector frame
```

Published LVK source-frame masses must be converted using:

\[
M_\mathrm{det}
=
(1+z)M_\mathrm{source}.
\]

`chi_eff` is dimensionless and not redshift-transformed.

---

# 23. LVK comparison metric contracts

Point difference:

\[
\Delta
=
\mathrm{CNN}
-
\mathrm{LVK}.
\]

Normalized delta:

\[
\Delta_\mathrm{norm}
=
\frac{\Delta}
{(U_\mathrm{LVK}-L_\mathrm{LVK})/2}.
\]

Overlap:

\[
O
=
\max(0,\min(U_C,U_L)-\max(L_C,L_L)).
\]

Overlap fraction:

\[
O/(U_L-L_L).
\]

This is not IoU.

---

# 24. Visualization clipping contract

For mass targets, optional clipped display bounds may use:

```text
lower = max(0, lower)
```

but original conformal intervals remain authoritative for scientific metrics.

---

# 25. Artifact contract summary

## HDF5

Contains:

```text
X
y
parameters
SNR metadata
placement metadata
windowing metadata
projection metadata
injection metadata
processing/provenance metadata
```

## Split NPZ

Contains train/val/cal/test indices.

## Label-stats NPZ

Contains train-only target mean/std.

## Checkpoint

Contains:

```text
weights
optimizer state
model config
training config
target stats
history
```

## Prediction NPZ

Contains:

```text
predictions
embeddings
truth
physical HDF5 indices
target stats
normalization metadata
checkpoint/data references
```

---

# 26. Breaking-change rule

Any change to:

```text
detector order
input shape
label order
normalization definition
embedding dimension
residual sign
artifact ordering
```

should be treated as a breaking scientific/data-contract change.
