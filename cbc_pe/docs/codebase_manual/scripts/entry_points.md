# Script entry points

> **Codebase manual — closed M10 reference**
>
> Reference snapshot:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Scope

This document maps executable scripts to the artifacts they consume and produce.

Primary active scripts:

```text
scripts/
├── generate_bbh_dataset_hdf5.py
├── create_hdf5_splits.py
├── train_cnn_hdf5.py
├── predict_cnn_hdf5.py
├── inspect_hdf5_dataset.py
├── benchmark_hdf5_io.py
└── run_generation_campaign.py
```

Historical:

```text
scripts/legacy_npz/
```

---

# 2. Entry-point philosophy

The current repository architecture follows:

```text
src/
    = reusable scientific / ML logic

scripts/
    = orchestration and artifact production

notebooks/
    = analysis, validation, visualization, reporting
```

A notebook should not be the only way to reproduce a core artifact.

---

# 3. Canonical closed-M10 chain

From `cbc_pe/`:

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

---

# 4. `generate_bbh_dataset_hdf5.py`

## Responsibility

```text
parse CLI
load generation JSON
resolve roots
construct SimulationConfig / DatasetBuilder
allocate HDF5
generate samples
persist X/y/metadata
support resume/overwrite
write sidecar metadata
```

Important boundary:

```text
DatasetBuilder
=
scientific construction of one sample

generate_bbh_dataset_hdf5.py
=
large-scale orchestration + HDF5 persistence
```

---

# 5. HDF5 schema ownership

The generation script defines persistent schema elements such as:

```text
PARAMETER_KEYS
LABEL_NAMES
PLACEMENT_KEYS
WINDOWING_KEYS
PROJECTION_NETWORK_KEYS
PROJECTION_DETECTOR_KEYS
SNR_EXTRA_KEYS
INJECTION_KEYS
```

Therefore the HDF5 data contract is shared between:

```text
src/dataset.py
and
scripts/generate_bbh_dataset_hdf5.py
```

---

# 6. Generation CLI

Main arguments:

```text
--project-root
--data-root
--config
--overwrite
--resume
```

### `--overwrite`

Recreate/replace an existing output.

### `--resume`

Continue a compatible incomplete HDF5.

These are semantically different operations.

---

# 7. Generation artifact outputs

Conceptually:

```text
generation JSON
      ↓
generator script
      ↓
processed/<dataset_id>/
├── dataset.h5
└── dataset.metadata.json
```

The HDF5 contains both training arrays and extensive provenance metadata.

---

# 8. `create_hdf5_splits.py`

## Responsibility

Consumes:

```text
complete HDF5
+
split JSON
```

and creates:

```text
split indices NPZ
train-only label stats NPZ
split metadata JSON
```

---

# 9. Split validation

Before partitioning it verifies:

```text
X exists
y exists
X/y sample counts match
num_written == dataset size
dataset_status == complete
requested split count <= dataset size
```

This prevents downstream use of partial datasets.

---

# 10. Split naming

Names encode:

```text
train size
val size
cal size
test size
seed
```

Example:

```text
train400000_val40000_cal30000_test30000_seed123
```

This strongly improves artifact traceability.

---

# 11. Train-only target statistics

The split script also computes:

```text
y_mean
y_std
```

using the train split only.

This is the correct stage because split membership is already defined.

---

# 12. `train_cnn_hdf5.py`

## Responsibility

```text
load experiment JSON
resolve dataset/split/stats
validate split overlap
validate HDF5 completeness
construct loaders
run input-normalization sanity checks
construct model
train
track validation
save best checkpoint
save history
```

The model/training mathematics live under:

```text
src/models/
```

The script is orchestration.

---

# 13. Closed M10 training outputs

Conceptually:

```text
experiment JSON
      ↓
train_cnn_hdf5.py
      ↓
models/checkpoints/<dataset_id>/
      └── checkpoint.pt

results/<dataset_id>/
      └── training history / metadata
```

The checkpoint contains more than weights:

```text
model_state_dict
optimizer state
model_config
y_mean/y_std
training metadata
history
```

---

# 14. `predict_cnn_hdf5.py`

## Responsibility

```text
load prediction JSON
load checkpoint
reconstruct model
recover normalization
build requested split loaders
extract predictions
extract embeddings
save truth + physical indices + metadata
```

This script converts a checkpoint into a stable downstream analysis artifact.

---

# 15. Prediction output contract

For closed M10:

```text
pred_cal
emb_cal
y_cal
idx_cal

pred_test
emb_test
y_test
idx_test
```

plus metadata.

This NPZ is the stable input to:

```text
synthetic evaluation
Mondrian conformal
```

---

# 16. Why prediction extraction is a script

Preferred architecture:

```text
checkpoint
    ↓
predict script
    ↓
stable NPZ
    ↓
many notebooks/analyses
```

rather than:

```text
every notebook
    ↓
rebuild model
    ↓
run inference again
```

This reduces accidental divergence between analyses.

---

# 17. `inspect_hdf5_dataset.py`

Classification:

```text
ACTIVE INFRASTRUCTURE / DIAGNOSTIC
```

Purpose:

```text
inspect HDF5 tree
inspect root attrs
preview datasets
inspect parameters/SNR
inspect sidecar metadata
inspect generation config
```

It does not modify the dataset.

Use it when the question is:

```text
"What is actually inside this HDF5?"
```

---

# 18. `benchmark_hdf5_io.py`

Classification:

```text
INFRASTRUCTURE BENCHMARK
NOT SCIENTIFIC PIPELINE
```

Measures:

```text
sequential reads
random reads
contiguous-slice reads
DataLoader throughput
worker-count behavior
```

It does not train a model.

This benchmark helped motivate locality-aware HDF5 access.

---

# 19. Local benchmark duplication

The benchmark contains local classes such as:

```text
HDF5SplitDataset
SortedBlockBatchSampler
```

These are benchmark-local implementations.

Authoritative production equivalents live under:

```text
src/models/
```

### Classification

```text
[LOCAL DUPLICATION / BENCHMARK CODE]
```

Do not treat the benchmark classes as canonical production code.

---

# 20. `run_generation_campaign.py`

Classification:

```text
ACTIVE SPECIAL-CAMPAIGN ORCHESTRATOR
NOT PART OF MAIN M10-500k CHAIN
```

Purpose:

```text
base generation config
+
campaign specification
    ↓
generate multiple temporary configs
    ↓
call generate_bbh_dataset_hdf5.py repeatedly
```

---

# 21. Fixed-mass campaign behavior

Per job it may modify:

```text
output filename
seed
mass_1
mass_2
spin_1z = 0
spin_2z = 0
```

while reusing the standard generator.

This avoids duplicating scientific signal-generation logic.

---

# 22. `scripts/legacy_npz/`

Contains historical NPZ-era workflows such as:

```text
generate_bbh_dataset.py
load_dataset.py
merge_chunks.py
split_dataset.py
train_cnn.py
```

Classification:

```text
HISTORICAL
```

New work should use the HDF5 pipeline.

---

# 23. Entry-point map

| Script | Main input | Main output | Status |
|---|---|---|---|
| `generate_bbh_dataset_hdf5.py` | generation JSON | HDF5 + metadata | active |
| `create_hdf5_splits.py` | split JSON + HDF5 | split NPZ + train stats | active |
| `train_cnn_hdf5.py` | experiment JSON | checkpoint + history | active |
| `predict_cnn_hdf5.py` | prediction JSON + checkpoint | preds/embeddings NPZ | active |
| `inspect_hdf5_dataset.py` | HDF5 | inspection output | diagnostic |
| `benchmark_hdf5_io.py` | HDF5 + split | throughput stats | benchmark |
| `run_generation_campaign.py` | campaign JSON | many generation runs | special campaign |
| `legacy_npz/*` | old NPZ workflow | historical artifacts | historical |

---

# 24. Artifact flow

```text
generation JSON
      ↓
generate_bbh_dataset_hdf5.py
      ↓
dataset.h5
      ↓
create_hdf5_splits.py
      ↓
splits + label stats
      ↓
train_cnn_hdf5.py
      ↓
checkpoint
      ↓
predict_cnn_hdf5.py
      ↓
predictions + embeddings
      ↓
final notebooks
```

---

# 25. Change-impact notes

## Changing HDF5 schema

Review:

```text
generator script
inspection script
split script
model loaders
data-contract docs
```

## Changing split artifact naming

Review:

```text
training configs
prediction configs
notebooks
path resolution
```

## Changing checkpoint schema

Review:

```text
predict script
real-data model reconstruction
manual docs
```

## Changing generation CLI behavior

Review campaign script and reproducibility commands.

---

# 26. Mental model

```text
generate
    = create scientific dataset artifact

split
    = define experimental partition and train-only scaling

train
    = optimize one model definition

predict
    = freeze reusable model outputs

inspect
    = diagnose

benchmark
    = measure infrastructure

campaign
    = orchestrate many generator runs

legacy_npz
    = historical only
```
