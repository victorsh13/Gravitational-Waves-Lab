# End-to-end workflows

> Closed M10 reference:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Purpose

This document describes the complete artifact-producing and analysis workflows of the repository.

---

# 2. Synthetic generation workflow

```text
generation JSON
    ↓
SimulationConfig
    ↓
PriorConfig / ParameterSampler
    ↓
CBCParameters
    ↓
WaveformGenerator
    ↓
h+, h×
    ↓
DetectorProjector
    ↓
H1/L1/V1 projected strains
    ↓
ProjectedNetworkWindowSelector
    ↓
retained physical GW <= final duration
    ↓
SignalInjector placement
    ↓
signal-only final 4-s embedding
    ↓
NoiseModel PSD
    ↓
optimal network SNR
    ↓
optional distance rescaling
    ↓
rebuild signal network
    ↓
build processing-context noise/zero strain
    ↓
inject retained GW
    ↓
SignalProcessor
    ↓
whitening / filters / crop
    ↓
X sample
+
LabelTransformer
    ↓
y sample
    ↓
HDF5 persistence
```

---

# 3. Closed synthetic-context caveat

Closed M10 does:

```text
full GW
    ↓
window to <= 4 s
    ↓
inject that retained window into longer processing context
```

Therefore processing context outside the retained signal does not restore physical inspiral that was already discarded.

This is documented as a future pipeline-improvement candidate.

---

# 4. Dataset persistence workflow

```text
DatasetBuilder sample
    ↓
metadata dictionary
    ↓
generate_bbh_dataset_hdf5.py
    ↓
HDF5 schema
    ↓
processed/<dataset_id>/
├── dataset.h5
└── dataset.metadata.json
```

---

# 5. Split/stat workflow

```text
complete HDF5
    ↓
split JSON
    ↓
validate dataset completeness
    ↓
shuffle indices with fixed seed
    ↓
train
val
cal
test
    ↓
compute train-only y_mean/y_std
    ↓
save:
    split NPZ
    stats NPZ
    metadata JSON
```

---

# 6. CNN training workflow

```text
HDF5
+
train/val indices
+
train-only y_mean/y_std
    ↓
HDF5BatchIterableDataset
    ↓
per-sample/per-detector X z-score
+
target standardization
    ↓
SimpleCNN_ResidualDilated
    ↓
64-D embedding
    ↓
Linear 64→3
    ↓
standardized predictions
    ↓
MSELoss
    ↓
AdamW
    ↓
validation monitoring
    ↓
best checkpoint
```

Calibration and test are excluded from training/model selection.

---

# 7. CNN prediction extraction workflow

```text
checkpoint
+
cal/test indices
+
dataset
    ↓
reconstruct model from checkpoint model_config
    ↓
recover input normalization
    ↓
HDF5 batch loaders
    ↓
prediction + embedding extraction
    ↓
save:
pred_cal
emb_cal
y_cal
idx_cal
pred_test
emb_test
y_test
idx_test
metadata
```

---

# 8. Synthetic evaluation workflow

```text
prediction artifact
    ↓
Notebook 10
    ↓
standardized metrics
physical metrics
bias diagnostics
residual plots
error-vs-target/SNR/etc.
```

Notebook 10 evaluates but does not retrain.

---

# 9. Mondrian calibration workflow

```text
pred_cal
y_cal
emb_cal
    ↓
residuals = y_cal - pred_cal
    ↓
taxonomy:
    prediction
    or
    difficulty
    ↓
QuantileBinner
    ↓
bin indices
    ↓
BinGrouper
    ↓
residuals per label/bin
    ↓
ConformalIntervalCalibrator
    ↓
symmetric or asymmetric offsets
    ↓
FittedMondrianRegressor
```

---

# 10. Mondrian test evaluation workflow

```text
fitted Mondrian
+
pred_test
+
emb_test if difficulty
    ↓
apply_mondrian()
    ↓
lower/upper intervals
    ↓
y_test
    ↓
evaluate_mondrian()
    ↓
global coverage
local coverage
bin counts
widths
tail imbalance
undercoverage diagnostics
```

---

# 11. Final conformal-selection workflow

```text
candidate summary
    ↓
selection.py
    ↓
per target:
    conservative
    efficient
    ↓
selected_configurations table
```

Final M10 target-specific configurations are then rebuilt from calibration data.

---

# 12. Real-event discovery workflow

```text
GWOSC catalog
    ↓
flatten event metadata
    ↓
filter required detectors
    ↓
resolve H1/L1/V1 4096-s HDF5 URLs
    ↓
event configs
```

---

# 13. Real-event strain workflow

```text
event config
    ↓
download/cache long HDF5 files
    ↓
read PyCBC TimeSeries
    ↓
center_time = gps + center_offset
    ↓
validate event processing window
    ↓
select valid off-source PSD window
    ↓
estimate one empirical PSD per detector
    ↓
extract event + processing context
    ↓
SignalProcessor
    ↓
X_real_raw
```

---

# 14. Real M10 normalization/inference workflow

```text
X_real_raw
    ↓
same batch-wise M10 z-score used in training
    ↓
X_real_z
    ↓
SimpleCNN_ResidualDilated
    ↓
pred_std
+
64-D embedding
    ↓
inverse target standardization
    ↓
pred_phys
```

---

# 15. Real conformal workflow

```text
pred_real_std
+
emb_real
+
selected fitted calibrators
    ↓
apply selected Mondrian systems
    ↓
real-event intervals
```

No real-event truth is required.

---

# 16. LVK reference workflow

```text
GWOSC published parameter summaries
    ↓
source-frame masses
+
redshift
    ↓
M_det = (1+z) M_source
    ↓
first-order asymmetric uncertainty propagation
    ↓
detector-frame LVK references
```

`chi_eff` passes through unchanged.

---

# 17. LVK comparison workflow

```text
CNN point estimates
+
CNN conformal intervals
+
LVK detector-frame references
    ↓
evaluation/lvk.py
    ↓
point differences
normalized differences
membership checks
absolute overlap
LVK-normalized overlap fraction
```

---

# 18. Final notebook workflow

```text
10
synthetic CNN evaluation
    ↓
11
Mondrian evaluation/selection
    ↓
12
single-event real application
    ↓
13
multi-event LVK comparison
```

This is conceptual ordering; notebooks consume script/module artifacts rather than passing mutable notebook state directly.

---

# 19. Closed M10 reproducibility path

```bash
python scripts/generate_bbh_dataset_hdf5.py \
  --config configs/generation/generate_500k_bbh_4s.json

python scripts/create_hdf5_splits.py \
  --config configs/splits/splits_500k_train400_val40_cal30_test30_seed123.json

python scripts/train_cnn_hdf5.py \
  --config configs/experiments/train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json

python scripts/predict_cnn_hdf5.py \
  --config configs/predictions/predict_500k_M10_inputzscore_cal_test.json
```

Then run the final analysis notebooks.

---

# 20. Mental model

```text
scripts
    produce artifacts

modules
    define transformations

tests
    protect contracts

notebooks
    analyze/report artifacts
```
