# Repository overview

> Closed M10 reference:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Purpose of the repository

`cbc_pe` is a compact-binary coalescence parameter-estimation research codebase combining:

```text
synthetic BBH signal generation
CNN regression
latent embeddings
Mondrian conformal uncertainty
real GWOSC inference
LVK comparison
```

The repository contains both active closed-M10 code and historical experiment layers.

---

## 2. Three conceptual layers

### Scientific core

Reusable scientific transformations:

```text
src/config.py
src/parameters.py
src/sampling.py
src/waveform.py
src/detectors.py
src/windowing.py
src/injection.py
src/noise.py
src/processing.py
src/snr.py
src/labels.py
src/dataset.py

src/models/*
src/conformal/*
src/real_data/*
src/evaluation/*
```

### Orchestration / infrastructure

```text
src/paths.py
scripts/*
configs/*
tests/*
```

### Analysis / presentation

```text
notebooks/*
docs/*
figures
reports
```

---

## 3. Main active directories

```text
cbc_pe/
├── configs/
├── docs/
├── notebooks/
├── scripts/
├── src/
└── tests/
```

Large scientific artifacts live outside Git under `data_root`.

---

## 4. Active scientific source map

### Synthetic

```text
src/config.py
src/parameters.py
src/sampling.py
src/waveform.py
src/detectors.py
src/windowing.py
src/injection.py
src/noise.py
src/processing.py
src/snr.py
src/labels.py
src/dataset.py
```

### Models

```text
src/models/dataset.py
src/models/hdf5_batch_dataset.py
src/models/samplers.py
src/models/network.py
src/models/train.py
src/models/evaluate.py
src/models/plots.py
src/models/utils.py
```

### Conformal

```text
src/conformal/taxonomy.py
src/conformal/difficulty.py
src/conformal/binning.py
src/conformal/calibration.py
src/conformal/apply.py
src/conformal/metrics.py
src/conformal/pipeline.py
src/conformal/selection.py
src/conformal/selected_calibrators.py
src/conformal/hybrid.py
```

### Real data

```text
src/real_data/catalog.py
src/real_data/gwosc_utils.py
src/real_data/psd.py
src/real_data/signal_processing.py
src/real_data/inference.py
src/real_data/event_runner.py
src/real_data/lvk_reference.py
```

### Evaluation

```text
src/evaluation/lvk.py
```

---

## 5. Entry points

Main closed-M10 executable chain:

```text
generate_bbh_dataset_hdf5.py
create_hdf5_splits.py
train_cnn_hdf5.py
predict_cnn_hdf5.py
```

Supporting scripts:

```text
inspect_hdf5_dataset.py
benchmark_hdf5_io.py
run_generation_campaign.py
```

Historical NPZ scripts live under:

```text
scripts/legacy_npz/
```

---

## 6. Final notebooks

```text
10_m10_inputzscore_500k_evaluation.ipynb
11_mondrian_m10_inputzscore_500k_final.ipynb
12_real_event_inference_m10_500k_clean.ipynb
13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

These form the final closed-M10 analysis layer.

---

## 7. Data-root organization

```text
<data_root>/
├── processed/
│   └── <dataset_id>/
├── models/checkpoints/
│   └── <dataset_id>/
├── results/
│   └── <dataset_id>/
├── gwosc_cache/
└── lvk_references/
```

---

## 8. Main closed dataset

```text
dataset_id:
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000
```

Contract:

```text
500000 samples
4 s
4096 Hz
H1/L1/V1
X shape = (N,3,16384)
y shape = (N,3)
labels = chirp_mass,total_mass,chi_eff
```

---

## 9. Model lineage

Verified/high-confidence experiment lineage:

```text
M00  baseline
M01  pool experiment
M02  pool_size=4
M04  deeper dense head
M06  wide CNN
M07  multi-head
M08  residual-dilated
M09  residual-dilated + multi-attention
M10  M08 architecture + input z-score
```

Closed active model:

```text
SimpleCNN_ResidualDilated
embedding_dim = 64
dilations = [1,2,4]
```

---

## 10. Conformal lineage

Final active taxonomies:

```text
prediction
difficulty
```

Interval modes:

```text
symmetric
asymmetric
```

Historical experiment:

```text
hybrid prediction × difficulty
```

The hybrid method was implemented and tested but not adopted in the final M10 pipeline.

---

## 11. Real-data philosophy

The real-data pipeline tries to preserve:

```text
same final shape
same detector order
same SignalProcessor
same input z-score
same CNN checkpoint
same label scaling
same selected conformal systems
```

while replacing synthetic inputs with:

```text
real GWOSC detector strain
empirical off-source PSD
```

This makes domain-gap effects more interpretable.

---

## 12. Repository maturity assessment

### Strongly modularized

```text
CNN
conformal
real data
paths
LVK evaluation
```

### Older core with weaker direct test protection

```text
synthetic generation
waveform/projection
windowing/injection
noise/SNR
DatasetBuilder
```

This does not imply poor correctness; it identifies where future refactors require more characterization tests.

---

## 13. Navigation

Use:

```text
02_end_to_end_workflows.md
```

for pipeline flow.

Use:

```text
03_data_contracts.md
```

for shapes/order/units.

Use:

```text
05_dependency_map.md
```

for module coupling.

Use:

```text
08_change_impact_guide.md
```

before changing behavior.
