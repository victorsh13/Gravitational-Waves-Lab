# Configuration and path-resolution system

> **Codebase manual — closed M10 reference**
>
> Reference snapshot:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Scope

This document explains how runtime paths and JSON experiment definitions control the `cbc_pe` pipeline.

Primary files:

```text
src/paths.py
configs/README.md

configs/
├── generation/
├── splits/
├── experiments/
├── predictions/
└── historical / benchmark / campaign configs
```

The configuration system should be understood as part of the reproducibility contract, not as incidental boilerplate.

---

## 2. Project root vs data root

The repository deliberately distinguishes:

```text
project_root
=
source code
configs
notebooks
docs

data_root
=
large generated artifacts
datasets
splits
checkpoints
predictions
GWOSC cache
LVK references
```

This separation allows the same Git checkout to run against different storage locations.

---

## 3. `src/paths.py`

### `DATA_ROOT_ENV_VAR`

```text
CBC_PE_DATA_ROOT
```

is the canonical environment-variable override for external data storage.

---

## 4. `resolve_project_root()`

Precedence:

```text
1. CLI --project-root
2. config project_root, if that path exists on the current machine
3. auto-detect repository root from src/paths.py
```

Important behavior:

```text
config project_root
```

is treated as portable historical metadata rather than an unconditional path.

If it does not exist on the current machine, automatic repository detection is used instead.

### Classification

```text
ACTIVE INFRASTRUCTURE
```

---

## 5. `resolve_data_root()`

Precedence:

```text
1. CLI --data-root
2. CBC_PE_DATA_ROOT
3. data_root in JSON config
```

If none exists:

```text
ValueError
```

This is stricter than project-root resolution because large data cannot be inferred safely.

### Operational consequence

The path written inside a historical JSON config is not necessarily the runtime path actually used.

Effective path resolution is:

```text
CLI
>
ENV
>
CONFIG
```

This should always be remembered when reproducing old runs.

---

## 6. Recommended external data layout

The modern layout is:

```text
<data_root>/
├── processed/
│   └── <dataset_id>/
├── models/
│   └── checkpoints/
│       └── <dataset_id>/
├── results/
│   └── <dataset_id>/
├── gwosc_cache/
└── lvk_references/
```

Recommended portable setup:

```bash
export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
```

---

## 7. `dataset_processed_dir()`

Canonical processed-data location:

```text
<data_root>/processed/<dataset_id>/
```

This directory contains:

```text
dataset HDF5
dataset metadata
split NPZ
train-only label statistics
split metadata
```

---

## 8. `resolve_processed_artifact()`

Preferred modern lookup:

```text
processed/<dataset_id>/<file>
```

Legacy fallback:

```text
processed/<file>
```

if:

```text
allow_legacy_flat = True
```

### Classification

```text
ACTIVE INFRASTRUCTURE
+
LEGACY COMPATIBILITY BOUNDARY
```

The legacy fallback should not be removed casually because historical artifacts may depend on it.

---

# 9. JSON configs as experiment definitions

Configs in this repository are not merely parameter files.

They function as:

```text
reproducible experiment definitions
```

A config should capture enough context to identify:

```text
dataset
split
architecture
normalization
major hyperparameters
seed
artifact naming
```

This is why descriptive filenames are preferred over vague names such as:

```text
train_final.json
train_new.json
test_config.json
```

---

# 10. Closed M10 configuration chain

The closed synthetic M10 chain is:

```text
configs/generation/
generate_500k_bbh_4s.json
        ↓
configs/splits/
splits_500k_train400_val40_cal30_test30_seed123.json
        ↓
configs/experiments/
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
        ↓
configs/predictions/
predict_500k_M10_inputzscore_cal_test.json
```

This corresponds to:

```text
dataset generation
→ split/stat creation
→ CNN training
→ calibration/test prediction + embedding extraction
```

---

# 11. Closed M10 dataset definition

Dataset ID:

```text
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000
```

Main properties:

```text
samples:               500000
duration:              4 s
sampling frequency:    4096 Hz
detectors:             H1, L1, V1
waveform approximant:  SEOBNRv4_opt
target network SNR:    10–25
```

---

# 12. Closed M10 split definition

```text
train: 400000
val:    40000
cal:    30000
test:   30000
seed:   123
```

The split definition and train-only statistics are written under:

```text
<data_root>/processed/<dataset_id>/
```

---

# 13. Closed M10 training definition

Reference config:

```text
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

Key settings:

```text
model class:
    SimpleCNN_ResidualDilated

input normalization:
    per_sample_per_detector_zscore

embedding dimension:
    64

dilations:
    [1,2,4]

loss:
    MSELoss

batch size:
    256

seed:
    123
```

The authoritative values are those in the exact JSON + exact code snapshot.

---

# 14. Closed M10 prediction definition

Reference config:

```text
predict_500k_M10_inputzscore_cal_test.json
```

Purpose:

```text
load the trained checkpoint
extract cal/test predictions
extract cal/test embeddings
save reusable downstream artifact
```

Main artifact:

```text
m10_inputzscore_500k_cal_test_predictions_embeddings.npz
```

under:

```text
<data_root>/results/<dataset_id>/
```

---

# 15. Configuration directory classification

```text
configs/generation/
    generation experiment definitions

configs/splits/
    reproducible data partition definitions

configs/experiments/
    CNN training experiment definitions

configs/predictions/
    prediction / embedding extraction definitions

configs/generation/processing_benchmark/
    audit / benchmark

configs/generation/foundation/
    special generation campaigns

configs/_archive/
    historical / compatibility
```

---

# 16. Historical configuration retention

Historical configs are retained for:

```text
architecture search
M00–M10 experiment traceability
processing benchmarks
foundation-model data studies
older data-root layouts
```

They should not be interpreted as part of the active closed M10 chain unless explicitly referenced by final documentation.

---

# 17. M07 verification

The repository contains config names of the form:

```text
train_500k_M07_multihead_...
predict_500k_M07_multihead_...
```

Therefore the historical mapping is:

```text
M07 = SimpleCNN_MultiHead
```

This mapping is now verified by repository configuration naming.

---

# 18. Configuration/code boundary

Not every scientific default is necessarily exposed in JSON.

For example, the generation entry point constructs `SimulationConfig` using a selected subset of config fields while other behavior may still come from dataclass defaults.

### Review note

```text
[CONFIGURATION REVIEW]

A JSON config is not necessarily the complete physical definition by itself.
Reproduction depends on:
    config
    +
    code version
```

This is one reason the M10 Git tag matters.

---

# 19. Potential non-functional config fields

Some fields may remain from older layouts or interfaces.

Example from split creation:

```text
output.output_dir
```

is read, but the modern path is built canonically through:

```text
dataset_processed_dir(data_root, dataset_id)
```

### Classification

```text
[CONFIGURATION / CLEANUP REVIEW]
```

Do not remove historical fields before checking compatibility.

---

# 20. Portable execution pattern

Preferred pattern:

```bash
export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
cd cbc_pe
```

Then run configs without editing machine-specific paths.

Optional CLI override:

```bash
python <script> \
  --data-root /other/storage/location \
  --config <config>
```

---

# 21. Change-impact guide

## Changing `project_root`

Usually affects only code/config discovery.

## Changing `data_root`

Changes artifact lookup/storage but should not alter scientific behavior if the exact artifacts are equivalent.

## Changing `dataset_id`

Affects:

```text
processed paths
checkpoint namespace
results namespace
split/stat resolution
```

## Changing config naming

Does not change science directly, but can damage experiment traceability.

## Removing legacy path fallback

Can break historical artifact loading even if active M10 remains unaffected.

---

# 22. Mental model

```text
src/paths.py
    = where are code and data?

configs/
    = what experiment is being defined?

scripts/
    = execute the experiment definition

artifacts/
    = persistent result of that definition
```

The reproducibility unit is:

```text
Git snapshot
+
JSON config
+
input artifact identities
+
runtime data-root resolution
```
