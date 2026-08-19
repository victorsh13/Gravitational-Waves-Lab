# Change impact guide

> Closed M10 reference:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Purpose

Use this document before changing code.

For each change, answer:

```text
What source files change?
Which configs become stale?
Which artifacts must be regenerated?
Which tests must pass?
Which notebooks/results stop being comparable?
```

---

# 2. Change: input normalization

Primary files:

```text
src/models/dataset.py
src/models/hdf5_batch_dataset.py
src/real_data/event_runner.py
```

Tests:

```text
test_input_normalization_equivalence.py
test_real_data_inference.py
test_real_data_event_runner.py
```

Regenerate:

```text
checkpoint
predictions
embeddings
conformal calibration
real-event inference
```

Scientific effect:

```text
changes information available to CNN
changes synthetic↔real domain mapping
```

Classification:

```text
FULL MODEL VERSION CHANGE
```

---

# 3. Change: synthetic SignalProcessor

Primary:

```text
src/processing.py
src/dataset.py
generation config/script
```

Regenerate:

```text
HDF5
splits/stats if dataset recreated
checkpoint
predictions
embeddings
conformal
real comparisons
```

Tests to add/run:

```text
processing characterization
dataset characterization
real processing tests
```

Scientific effect:

```text
changes model input distribution
```

---

# 4. Change: processing-context physical signal support

Current M10 issue:

```text
synthetic context may contain noise only outside retained <=4s GW
real context preserves physical GW if present
```

Future change would touch:

```text
windowing.py
dataset.py
injection.py
possibly processing.py
```

Must regenerate entire synthetic chain.

This should be introduced as a new pipeline ID, not patched into M10.

---

# 5. Change: waveform approximant

Primary:

```text
src/waveform.py
generation config/default
```

Regenerate:

```text
dataset
model
predictions
conformal
```

Scientific effect:

```text
waveform morphology
duration
spin assumptions
SNR
domain agreement
```

---

# 6. Change: spin parameterization

Current:

```text
aligned z-spin components only
```

Adding in-plane spins/precession affects:

```text
parameters.py
sampling.py
waveform.py
possibly labels
configs
dataset metadata
```

Full synthetic regeneration required.

---

# 7. Change: detector set/order

Primary:

```text
generation config
detectors.py
dataset.py
CNN n_detectors
real event runner
```

Breaking contracts:

```text
X channel meaning
model first-layer dimensions
real event eligibility
```

Requires new dataset/model.

---

# 8. Change: SNR targeting

Primary:

```text
src/snr.py
src/dataset.py
generation config
```

Review:

```text
distance distribution
selection/rejection effects
SNR metadata
```

Regenerate dataset/model chain.

---

# 9. Change: noise model

Synthetic Gaussian→real-noise or other change touches:

```text
src/noise.py
src/dataset.py
generation configs
```

Potentially major scientific change.

Regenerate full chain.

---

# 10. Change: CNN architecture

Primary:

```text
src/models/network.py
training config
```

Check:

```text
return_embedding=True
embedding dimension
checkpoint reconstruction
prediction script
real inference
```

Regenerate:

```text
checkpoint
predictions
embeddings
conformal
real-event inference
```

---

# 11. Change: embedding dimension

Direct downstream impact:

```text
prediction artifact
DifficultyEstimator
difficulty Mondrian
selected calibrators
real-event difficulty intervals
```

Old conformal calibration is invalid.

---

# 12. Change: training loss

Current code risk:

```text
training.loss config is descriptive
train.py hardcodes MSELoss
```

Any real loss change requires editing `train.py` or adding a loss factory.

New experiment ID required.

---

# 13. Change: label order

Current:

```text
chirp_mass
total_mass
chi_eff
```

Affected everywhere:

```text
HDF5
stats
checkpoint
evaluation
conformal
real inference
LVK comparison
```

Classification:

```text
BREAKING DATA CONTRACT
```

Avoid unless there is a strong reason.

---

# 14. Change: split sizes/seed

Primary:

```text
split config
```

Regenerate:

```text
split NPZ
train-only stats
checkpoint
predictions
conformal
```

Dataset HDF5 itself can remain unchanged.

---

# 15. Change: HDF5 loading strategy only

Potentially infrastructure-only if data values/order semantics remain identical.

Review:

```text
dataset.py
hdf5_batch_dataset.py
samplers.py
train/predict scripts
```

Validate:

```text
same X
same y
same normalization
same split membership
correct prediction indices
```

No scientific change should occur if implementation is correct.

---

# 16. Change: prediction artifact ordering

Any ordering change must preserve explicit:

```text
idx_<split>
```

Downstream code should map by saved physical indices.

Silent ordering changes are dangerous.

---

# 17. Change: conformal taxonomy

Primary:

```text
taxonomy.py
difficulty.py
pipeline.py
```

Regenerate:

```text
calibration
candidate evaluation
selection
selected calibrators
real-event intervals
```

CNN checkpoint can remain fixed.

---

# 18. Change: `n_neighbors`

Affects:

```text
difficulty scores
difficulty bins
calibration offsets
selection
real difficulty intervals
```

Requires conformal recalibration but not CNN retraining.

---

# 19. Change: `n_bins`

Affects:

```text
quantile edges
bin counts
residual groups
coverage stability
widths
selection
```

Requires conformal recalibration/evaluation.

---

# 20. Change: confidence level

Affects:

```text
conformal order statistics
widths
target coverage
tolerance bands
selection interpretation
```

Requires full conformal recalibration.

---

# 21. Change: selection thresholds

Primary:

```text
selection.py
```

Candidate systems may remain unchanged.

Must regenerate:

```text
selected configuration table
selected calibrators
real-event final intervals
```

---

# 22. Change: hybrid conformal

Hybrid is historical experimental code.

If revisited:

```text
treat as new conformal experiment
do not silently merge into active M10 path
```

---

# 23. Change: real PSD strategy

Primary:

```text
real_data/psd.py
event_runner.py
signal_processing.py
```

Regenerate only real-event inputs/predictions.

Synthetic model artifacts can remain unchanged.

Scientific effect can be large because whitening changes.

---

# 24. Change: real PSD window policy

Does not require retraining but changes:

```text
X_real_raw
X_real_z
pred_real
emb_real
intervals
LVK comparison
```

Re-run real notebooks.

---

# 25. Change: LVK frame conversion

Primary:

```text
real_data/lvk_reference.py
```

CNN outputs unchanged.

Re-run:

```text
LVK reference table
comparison metrics
Notebook 13
```

---

# 26. Change: LVK comparison metric definition

Primary:

```text
evaluation/lvk.py
```

No upstream model/conformal regeneration.

Re-run comparison/reporting only.

Document metric redefinition clearly.

---

# 27. Change: paths/layout

Primary:

```text
src/paths.py
configs
scripts
```

Scientific artifacts can remain identical.

Run:

```text
test_paths.py
script smoke tests
```

Do not confuse path migration with scientific-method change.

---

# 28. Change: notebook only

If only:

```text
plot
table formatting
report text
```

changes, scientific artifacts remain intact.

If notebook contains duplicated scientific logic, consider extracting that logic to `src/` first.

---

# 29. Minimum safe workflow for a major change

```text
1. identify change category
2. read relevant module manual
3. read dependency map
4. add characterization tests if missing
5. create new branch / experiment ID
6. implement minimal change
7. run targeted tests
8. regenerate only required artifacts
9. compare against closed M10
10. update manual
```

---

# 30. Major-change matrix

| Change | Dataset | Retrain CNN | Refit conformal | Re-run real | New experiment ID |
|---|---:|---:|---:|---:|---:|
| input z-score | no | yes | yes | yes | yes |
| synthetic processing | yes | yes | yes | yes | yes |
| waveform model | yes | yes | yes | yes | yes |
| noise model | yes | yes | yes | yes | yes |
| CNN architecture | no | yes | yes | yes | yes |
| embedding dim | no | yes | yes | yes | yes |
| split seed | no | yes | yes | yes | yes |
| conformal bins | no | no | yes | yes | yes/conformal |
| selection thresholds | no | no | maybe candidates reused | yes | recommended |
| real PSD policy | no | no | no | yes | real-analysis version |
| LVK metric | no | no | no | comparison only | report version |

---

# 31. Mental model

```text
upstream data change
    → everything downstream invalidated

model change
    → predictions/embeddings/conformal/real invalidated

conformal change
    → intervals/real uncertainty invalidated

real-processing change
    → real results only invalidated

evaluation/report change
    → upstream scientific artifacts intact
```
