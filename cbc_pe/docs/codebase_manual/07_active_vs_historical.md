# Active vs historical code

> Closed M10 reference:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Purpose

The repository contains several generations of work. This document prevents historical code from being mistaken for the closed active M10 pipeline.

---

# 2. Classification vocabulary

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

# 3. Active synthetic core

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

Classification:

```text
ACTIVE CORE
```

---

# 4. Active CNN path

```text
src/models/dataset.py
src/models/hdf5_batch_dataset.py
src/models/network.py
src/models/train.py
src/models/evaluate.py
src/models/utils.py
```

Closed model class:

```text
SimpleCNN_ResidualDilated
```

Classification:

```text
ACTIVE CORE
```

Alternative but valid:

```text
ArrayRegressionDataset
SortedBlockBatchSampler
sample-wise loading
```

Classification:

```text
ACTIVE INFRASTRUCTURE / ALTERNATIVE
```

---

# 5. Historical CNN architectures

```text
SimpleCNN_Baseline
SimpleCNN_Pool
SimpleCNN_PoolDeep
WideCNN_Pool
SimpleCNN_MultiHead
SimpleCNN_ResidualDilatedMultiAttention
```

These remain instantiable but are not used by closed M10.

Classification:

```text
HISTORICAL ARCHITECTURE
```

Verified experiment mapping includes:

```text
M07 = SimpleCNN_MultiHead
M08 = SimpleCNN_ResidualDilated
M09 = SimpleCNN_ResidualDilatedMultiAttention
M10 = M08 architecture + input z-score
```

---

# 6. Active conformal code

```text
taxonomy.py
difficulty.py
binning.py
calibration.py
apply.py
metrics.py
pipeline.py
selection.py
selected_calibrators.py
```

Classification:

```text
ACTIVE CORE
```

---

# 7. Hybrid conformal

```text
src/conformal/hybrid.py
```

Purpose:

```text
prediction × difficulty joint taxonomy
```

Outcome:

```text
implemented
experimentally assessed
no meaningful enough improvement to adopt
```

Classification:

```text
HISTORICAL / EXPERIMENTAL
```

Not dead code; retained for methodological history.

---

# 8. Active real-data code

```text
src/real_data/catalog.py
src/real_data/gwosc_utils.py
src/real_data/psd.py
src/real_data/signal_processing.py
src/real_data/inference.py
src/real_data/event_runner.py
src/real_data/lvk_reference.py
src/evaluation/lvk.py
```

Classification:

```text
ACTIVE CORE
```

---

# 9. Active infrastructure

```text
src/paths.py

scripts/generate_bbh_dataset_hdf5.py
scripts/create_hdf5_splits.py
scripts/train_cnn_hdf5.py
scripts/predict_cnn_hdf5.py
scripts/inspect_hdf5_dataset.py
scripts/run_generation_campaign.py
```

Classification:

```text
ACTIVE INFRASTRUCTURE
```

`benchmark_hdf5_io.py`:

```text
AUDIT / INFRASTRUCTURE BENCHMARK
```

---

# 10. Historical scripts

```text
scripts/legacy_npz/*
```

Classification:

```text
HISTORICAL
```

The active large-scale data pipeline is HDF5.

---

# 11. Final notebooks

```text
10_m10_inputzscore_500k_evaluation.ipynb
11_mondrian_m10_inputzscore_500k_final.ipynb
12_real_event_inference_m10_500k_clean.ipynb
13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

Classification:

```text
ACTIVE ANALYSIS
```

---

# 12. Demo notebooks

```text
notebooks/demos/*
```

Classification:

```text
DEMO
```

Do not use them as authoritative M10 result sources.

---

# 13. Audit notebooks

```text
notebooks/audits/*
```

Classification:

```text
AUDIT
```

These document methodological checks and benchmarks.

---

# 14. Archive notebooks

```text
notebooks/_archive/*
```

Classification:

```text
HISTORICAL
```

Important archive families:

```text
architecture_search
m08_baseline
m10_development
```

---

# 15. Config status

Active closed-M10 chain:

```text
generate_500k_bbh_4s.json
splits_500k_train400_val40_cal30_test30_seed123.json
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
predict_500k_M10_inputzscore_cal_test.json
```

Other configs may be:

```text
historical experiment
benchmark
foundation campaign
development
```

Status should be inferred from explicit workflow references, not filename recency alone.

---

# 16. Local/generated assets

Examples:

```text
GWOSC HDF5 cache
checkpoints
prediction NPZs
logs
figures
temporary configs
```

Classification:

```text
ASSET / GENERATED ARTIFACT
```

They are not source code.

---

# 17. Placeholder files

Examples:

```text
.gitkeep
```

Classification:

```text
PLACEHOLDER
```

No scientific meaning.

---

# 18. Deletion policy

Do not delete historical code solely because it is inactive.

Before removing, ask:

```text
Does it document a published/reported result?
Does a config reference it?
Does an archive notebook rely on it?
Is it needed to reconstruct an experiment lineage?
```

Prefer:

```text
archive
document
then remove only with evidence
```

---

# 19. Refactor policy

Closed M10:

```text
document first
characterize with tests
do not silently refactor behavior
```

Future pipeline:

```text
new branch
new experiment ID
explicit migration
```

---

# 20. Mental model

```text
active
    = current closed pipeline

alternative
    = valid infrastructure not selected by M10

audit
    = evidence for decisions

historical
    = past scientific path

asset
    = generated data/result

placeholder
    = structure only
```
