# Dependency map

> Closed M10 reference:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Purpose

This document maps important dependency directions so changes can be reasoned about before editing code.

---

# 2. Synthetic dependency chain

```text
config.py
    ↓
sampling.py
parameters.py
    ↓
waveform.py
    ↓
detectors.py
    ↓
windowing.py
    ↓
injection.py
    ↓
noise.py
snr.py
processing.py
labels.py
    ↓
dataset.py
    ↓
generate_bbh_dataset_hdf5.py
```

`dataset.py` is the main synthetic composition point.

---

# 3. Model dependency chain

```text
HDF5
    ↓
models/dataset.py
models/hdf5_batch_dataset.py
models/samplers.py
    ↓
models/network.py
    ↓
models/train.py
models/evaluate.py
    ↓
train_cnn_hdf5.py
predict_cnn_hdf5.py
```

---

# 4. Conformal dependency chain

```text
prediction artifact
    ↓
taxonomy.py
difficulty.py
    ↓
binning.py
    ↓
calibration.py
    ↓
apply.py
    ↓
pipeline.py
    ↓
metrics.py
selection.py
selected_calibrators.py
```

`hybrid.py` is a parallel experimental branch.

---

# 5. Real-data dependency chain

```text
catalog.py
    ↓
gwosc_utils.py
    ↓
psd.py
signal_processing.py
    ↓
models normalization
    ↓
inference.py
    ↓
event_runner.py
    ↓
selected conformal calibrators
```

LVK comparison branch:

```text
catalog parameter summaries
    ↓
lvk_reference.py
    ↓
evaluation/lvk.py
```

---

# 6. Cross-layer dependencies

## `DatasetBuilder`

Depends on:

```text
SimulationConfig
ParameterSampler
WaveformGenerator
DetectorProjector
ProjectedNetworkWindowSelector
SignalInjector
NoiseModel
SignalProcessor
LabelTransformer
SNR helpers
```

This is the highest-coupling synthetic module.

---

## M10 real runner

Depends on:

```text
GWOSC utilities
real PSD
real signal processing
model input normalization
CNN inference
selected conformal application
```

This is the highest-coupling real-data orchestrator.

---

# 7. Config dependencies

```text
generation config
→ generator
→ synthetic modules

split config
→ split script

training config
→ loaders
→ network
→ train loop

prediction config
→ checkpoint reconstruction
→ loaders
→ evaluation extraction
```

---

# 8. Artifact dependencies

```text
dataset HDF5
    ↓
split/stat artifacts
    ↓
checkpoint
    ↓
prediction/embedding artifact
    ↓
conformal selection
    ↓
real-event selected calibrators
```

Each downstream artifact depends on upstream scientific definitions.

---

# 9. If normalization changes

Direct dependencies:

```text
models/dataset.py
models/hdf5_batch_dataset.py
real_data/event_runner.py
```

Downstream:

```text
trained checkpoint
predictions
embeddings
conformal calibration
real-event predictions
LVK comparison
```

---

# 10. If synthetic processing changes

Direct:

```text
processing.py
dataset.py
generation config/script
```

Downstream:

```text
HDF5
label-independent input distribution
CNN checkpoint
embeddings
conformal calibration
real-data comparability
```

This is a full-pipeline change.

---

# 11. If waveform model changes

Direct:

```text
waveform.py
generation config/default
```

Downstream:

```text
synthetic morphology
SNR
distance targeting
CNN training distribution
real-domain comparison
```

---

# 12. If detector projection changes

Direct:

```text
detectors.py
dataset.py
```

Downstream:

```text
channel timing
network morphology
SNR
CNN features
```

---

# 13. If embedding changes

Direct:

```text
network.py
checkpoint
prediction artifact
```

Downstream difficulty-based components:

```text
DifficultyEstimator
Mondrian binning
selected calibrators
real-event conformal intervals
```

Prediction-based Mondrian is less directly affected by embedding geometry, but a new CNN checkpoint still changes point predictions.

---

# 14. If conformal selection changes

Direct:

```text
selection.py
selected configuration table
```

Downstream:

```text
selected_calibrators.py
real-event intervals
reported final comparisons
```

Candidate calibration systems need not change if only policy thresholds change.

---

# 15. If real PSD changes

Direct:

```text
real_data/psd.py
event_runner.py
signal_processing.py
```

Downstream:

```text
X_real_raw
X_real_z
CNN prediction
embedding
Mondrian interval
LVK comparison
```

Synthetic artifacts remain unchanged.

---

# 16. If LVK transformation changes

Direct:

```text
real_data/lvk_reference.py
```

Downstream:

```text
evaluation/lvk.py outputs
Notebook 13 tables/figures
reported real-event agreement
```

CNN predictions do not change.

---

# 17. Tests as dependency guards

Strong guard coverage:

```text
paths
M10 normalization
conformal
real data
LVK evaluation
```

Weak direct guard coverage:

```text
synthetic core
```

See:

```text
06_tests_map.md
```

---

# 18. Notebook dependency map

```text
prediction artifact
    ↓
Notebook 10
Notebook 11

selected conformal configs
    ↓
Notebook 12
Notebook 13

real-data code + checkpoint
    ↓
Notebook 12
Notebook 13

LVK reference/evaluation
    ↓
Notebook 13
```

---

# 19. Dependency-risk hotspots

Highest-coupling areas:

```text
DatasetBuilder
SignalProcessor
input normalization
SimpleCNN_ResidualDilated embedding
event_runner
selected conformal calibrators
```

Changes here should trigger broad regression review.

---

# 20. Mental model

```text
upstream scientific changes
    propagate through artifacts

downstream analysis changes
    may leave upstream artifacts intact
```

Always ask:

```text
Does this change alter data?
Does it alter model?
Does it alter calibration?
Does it alter only presentation/evaluation?
```
