# CBC-PE codebase manual

> **Closed M10 reference**
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

This manual documents the structure, scientific contracts, workflows, and maintenance boundaries of the `cbc_pe` repository.

It is intended as an internal engineering/scientific guide for:

```text
understanding the codebase
reproducing the closed M10 pipeline
auditing scientific assumptions
planning safe future changes
distinguishing active vs historical code
```

It is not a replacement for the scientific report.

---

## Manual structure

```text
docs/codebase_manual/
├── README.md
├── 01_repository_overview.md
├── 02_end_to_end_workflows.md
├── 03_data_contracts.md
├── 04_configuration_system.md
├── 05_dependency_map.md
├── 06_tests_map.md
├── 07_active_vs_historical.md
├── 08_change_impact_guide.md
├── glossary.md
│
├── modules/
│   ├── synthetic_generation.md
│   ├── cnn_pipeline.md
│   ├── conformal_pipeline.md
│   └── real_data_pipeline.md
│
├── scripts/
│   └── entry_points.md
│
└── notebooks/
    └── notebook_map.md
```

---

## Recommended reading order

For a first pass:

```text
README
→ 01_repository_overview
→ 02_end_to_end_workflows
→ 03_data_contracts
→ modules/*
→ 05_dependency_map
→ 08_change_impact_guide
```

For a maintenance task:

```text
08_change_impact_guide
→ relevant module manual
→ 05_dependency_map
→ 06_tests_map
```

For experiment reproduction:

```text
04_configuration_system
→ scripts/entry_points
→ 02_end_to_end_workflows
```

For code archaeology:

```text
07_active_vs_historical
→ notebooks/notebook_map
→ configs and archive
```

---

## Closed M10 scientific chain

```text
synthetic BBH generation
        ↓
HDF5 dataset
        ↓
train / val / cal / test split
        ↓
M10 CNN training
        ↓
cal/test predictions + embeddings
        ↓
Mondrian conformal calibration/selection
        ↓
real GWOSC inference
        ↓
LVK detector-frame comparison
```

---

## Closed M10 configuration chain

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

---

## Core scientific modules

### Synthetic generation

See:

```text
modules/synthetic_generation.md
```

Covers:

```text
config
parameters
sampling
waveform
detector projection
windowing
injection
noise
SNR
processing
labels
DatasetBuilder
```

### CNN / ML

See:

```text
modules/cnn_pipeline.md
```

Covers:

```text
HDF5 loading
M10 input z-score
target standardization
architectures
M08/M10 residual-dilated model
training
prediction extraction
embeddings
```

### Conformal / Mondrian

See:

```text
modules/conformal_pipeline.md
```

Covers:

```text
prediction taxonomy
difficulty taxonomy
quantile binning
symmetric/asymmetric calibration
coverage metrics
selection
truth-free deployment
hybrid experiment
```

### Real data

See:

```text
modules/real_data_pipeline.md
```

Covers:

```text
GWOSC catalog
strain download/cache
off-source PSD
real preprocessing
M10 normalization
CNN inference
selected Mondrian
LVK reference conversion
comparison metrics
```

---

## Key contracts to remember

```text
detector order:
    H1, L1, V1

final input shape:
    (N, 3, 16384)

label order:
    chirp_mass, total_mass, chi_eff

M10 input normalization:
    per sample, per detector z-score

target standardization:
    train-only global mean/std

conformal residual:
    truth - prediction

real Mondrian application:
    truth-free
```

---

## Audit status labels

This manual uses:

```text
[SAFE]
[REVIEW]
[SCIENTIFIC RISK]
[LEGACY]
```

and code-role classifications:

```text
ACTIVE CORE
ACTIVE INFRASTRUCTURE
ACTIVE ANALYSIS
TEST
DEMO
AUDIT
HISTORICAL
ASSET
PLACEHOLDER
UNKNOWN / REVIEW
```

---

## Closed baseline rule

The closed tag must be treated as immutable scientific history.

Future changes should:

```text
branch from the closed reference
add tests first when possible
introduce a new experiment/pipeline ID
avoid silently redefining M10
```

The manual documents current M10 behavior and future review points separately.
