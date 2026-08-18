# CBC-PE workflow

This document describes the current reproducible workflow for the `cbc_pe` project.

The closed reference pipeline is **M10-500k**, which combines synthetic BBH generation, CNN regression, Mondrian conformal prediction, and validation on real GWOSC events.

## 1. Environment and external data root

Large datasets and generated artifacts are stored outside the Git repository.

Recommended structure:

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

Set the external root with:

```bash
export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
```

Path precedence is:

```text
--data-root
→ CBC_PE_DATA_ROOT
→ data_root in config
```

The repository root follows:

```text
--project-root
→ valid project_root in config
→ automatic repository-root detection
```

This allows the same configs to be used across local and VM environments without editing machine-specific paths.

## 2. Closed M10 dataset

Dataset identifier:

```text
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000
```

Main characteristics:

```text
number of samples:       500000
duration:                4 s
sampling frequency:      4096 Hz
detectors:               H1, L1, V1
waveform approximant:    SEOBNRv4_opt
target network SNR:      10–25
strain mode:             signal injected in noise
```

Physical regression targets:

```text
chirp_mass
total_mass
chi_eff
```

The generation config is:

```text
configs/generation/generate_500k_bbh_4s.json
```

Run:

```bash
python scripts/generate_bbh_dataset_hdf5.py \
  --config configs/generation/generate_500k_bbh_4s.json
```

The generated dataset is stored under:

```text
<data_root>/processed/<dataset_id>/
```

The generation pipeline includes:

```text
parameter sampling
→ waveform generation
→ detector projection and time delays
→ common network-window selection
→ injection placement
→ network-SNR evaluation
→ optional distance rescaling
→ noise generation
→ signal injection
→ whitening/filtering
→ final 4 s crop
→ labels and metadata
```

## 3. Split creation and label statistics

The closed M10 split is:

```text
train: 400000
val:    40000
cal:    30000
test:   30000
```

Config:

```text
configs/splits/
splits_500k_train400_val40_cal30_test30_seed123.json
```

Run:

```bash
python scripts/create_hdf5_splits.py \
  --config configs/splits/splits_500k_train400_val40_cal30_test30_seed123.json
```

The split file and label statistics are stored in:

```text
<data_root>/processed/<dataset_id>/
```

Label means and standard deviations are computed from the **training split only**.

Split roles:

- `train`: CNN parameter optimization
- `val`: early stopping and model-selection diagnostics
- `cal`: conformal calibration
- `test`: final synthetic reporting

Calibration and test data must remain separate.

## 4. M10 CNN training

Reference config:

```text
configs/experiments/
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

Reference architecture:

```text
SimpleCNN_ResidualDilated
```

Main parameters:

```text
n_detectors:          3
n_outputs:            3
embedding_dim:        64
residual_channels:    64
dilations:            [1, 2, 4]
residual_kernel_size: 7
dropout_conv:         0.05
dropout_dense:        0.1
num_groups:           8
loss:                 MSELoss
batch_size:           256
```

### Input normalization

M10 introduces:

```text
per_sample_per_detector_zscore
```

For each sample and detector channel, the time series is normalized independently using its own mean and standard deviation.

This change was introduced after the previous M08 pipeline showed a large synthetic-to-real detector-scale mismatch.

The M10 normalization is part of the closed model definition and must be applied consistently during:

```text
training
validation
synthetic evaluation
real-data inference
```

Run training:

```bash
python scripts/train_cnn_hdf5.py \
  --config configs/experiments/train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

Checkpoint output:

```text
<data_root>/models/checkpoints/<dataset_id>/
```

Training histories and related outputs:

```text
<data_root>/results/<dataset_id>/
```

## 5. Prediction and embedding extraction

Reference prediction config:

```text
configs/predictions/
predict_500k_M10_inputzscore_cal_test.json
```

Run:

```bash
python scripts/predict_cnn_hdf5.py \
  --config configs/predictions/predict_500k_M10_inputzscore_cal_test.json
```

The main closed artifact is:

```text
m10_inputzscore_500k_cal_test_predictions_embeddings.npz
```

stored under:

```text
<data_root>/results/<dataset_id>/
```

It contains calibration/test predictions, labels, embeddings, label statistics, and model-normalization metadata.

This file is the main bridge between CNN evaluation and Mondrian calibration.

## 6. Synthetic M10 evaluation

Final notebook:

```text
notebooks/10_m10_inputzscore_500k_evaluation.ipynb
```

The notebook evaluates the closed 500k M10 model in both standardized and physical spaces.

Main regression diagnostics include:

```text
RMSE
MAE
bias
R2
residual distributions
prediction-vs-truth structure
parameter-dependent errors
```

The synthetic test split is used only after model training and conformal configuration decisions are fixed according to the analysis protocol.

## 7. Mondrian conformal calibration

Final notebook:

```text
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
```

The conformal analysis uses:

```text
calibration predictions
calibration labels
calibration embeddings

test predictions
test labels
test embeddings
```

Supported taxonomy modes:

```text
prediction
difficulty
```

Supported interval modes:

```text
symmetric
asymmetric
```

Difficulty-based Mondrian uses local residual information in CNN embedding space.

Core reusable implementation:

```text
src/conformal/
```

The final pipeline supports explicit:

```text
fit
apply
evaluate
```

stages so fitted calibrators can be applied to real data without target truth.

### Selection policies

Final configurations are selected through:

```text
conservative
efficient
```

policies implemented in:

```text
src/conformal/selection.py
```

Selected configurations are reconstructed into reusable fitted calibrators through:

```text
src/conformal/selected_calibrators.py
```

Important reported diagnostics include:

```text
global coverage
coverage per bin
sample counts per bin
median and tail interval widths
minimum bin coverage
maximum undercoverage gap
local statistical compatibility
tail-miss asymmetry
```

The final selected systems should be interpreted as conformal prediction intervals under the calibration distribution, not as Bayesian posterior credible intervals.

## 8. Single-event real-data inference

Final notebook:

```text
notebooks/12_real_event_inference_m10_500k_clean.ipynb
```

This notebook validates the complete real-data processing and inference path on an individual GWOSC event.

Real strain processing uses:

```text
GWOSC HDF5 strain
→ off-source PSD estimation
→ whitening
→ high-pass / low-pass processing
→ processing-context crop
→ final 4 s H1/L1/V1 input
→ M10 per-detector z-score
→ CNN point prediction
→ embedding extraction
→ selected Mondrian intervals
```

The reusable implementation is under:

```text
src/real_data/
```

## 9. Multi-event GWTC-3 / LVK comparison

Final notebook:

```text
notebooks/13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

The notebook:

```text
loads GWOSC catalog metadata
→ selects compatible H1/L1/V1 BBH events
→ resolves detector strain URLs
→ downloads/caches GWOSC strain
→ validates PSD and event windows
→ preprocesses real detector data
→ applies M10
→ applies fitted Mondrian calibrators
→ builds LVK detector-frame references
→ compares CNN/Mondrian outputs against LVK summaries
```

Real-event selection includes explicit scientific criteria such as:

```text
required detector network
finite strain availability
component-mass range
minimum published network SNR
```

Events with invalid or non-finite required strain are excluded rather than interpolated.

## 10. LVK comparison convention

Source-frame LVK masses are converted to detector-frame values using the published redshift.

`chi_eff` is not redshifted.

The LVK comparison implementation is under:

```text
src/evaluation/lvk.py
src/real_data/lvk_reference.py
```

Two important metrics are:

```text
normalized_delta_lvk =
    (CNN point - LVK central)
    / (0.5 * (LVK upper - LVK lower))
```

and:

```text
interval_overlap_fraction_lvk =
    overlap(CNN interval, LVK interval)
    / LVK interval width
```

The second metric is the fraction of the LVK interval covered by the CNN/conformal interval. It is **not** an intersection-over-union metric.

For mass visualizations, physically clipped interval columns may be generated for plotting. Scientific overlap metrics must use the original, unclipped conformal intervals.

## 11. Artifact and output policy

Canonical generated-data layout:

```text
<data_root>/
├── processed/<dataset_id>/
├── models/checkpoints/<dataset_id>/
├── results/<dataset_id>/
├── gwosc_cache/
└── lvk_references/
```

New training and prediction outputs should be written directly into the dataset-specific directories.

Historical flat layouts are supported only as backward-compatible read fallbacks.

Large artifacts are excluded from Git.

## 12. Notebook organization

Current final notebooks remain visible at the top level:

```text
notebooks/10_m10_inputzscore_500k_evaluation.ipynb
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
notebooks/12_real_event_inference_m10_500k_clean.ipynb
notebooks/13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

Reusable demos:

```text
notebooks/demos/
```

Pipeline audits:

```text
notebooks/audits/
```

Historical development:

```text
notebooks/_archive/
├── architecture_search/
├── m08_baseline/
├── m10_development/
└── legacy notebooks
```

Historical notebooks are preserved for scientific traceability but are not part of the active M10 execution chain.

## 13. Tests and validation

Run the full suite with:

```bash
python -m unittest discover \
  -s tests \
  -p "test_*.py" \
  -v
```

The test suite covers:

```text
conformal pipeline equivalence
configuration selection
selected calibrators
M10 normalization equivalence
path resolution
LVK transformations
LVK comparison metrics
GWOSC file utilities
PSD selection and estimation
real-data preprocessing
real-data inference
event orchestration
```

Before closing a scientific phase:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
git diff --check
```

Final notebooks should additionally be run from a clean kernel whenever their scientific logic is modified.

## 14. Closed M10 baseline

The following chain defines the closed M10 reference:

```text
500k synthetic BBH dataset

→ 400k / 40k / 30k / 30k split

→ SimpleCNN_ResidualDilated

→ per-sample/per-detector z-score

→ calibration/test prediction artifact

→ synthetic M10 evaluation

→ selected Mondrian calibration

→ real GWOSC validation

→ multi-event GWTC-3/LVK comparison
```

M10 should now be treated as a reproducible baseline.

Future work such as domain-gap studies, real-noise injection, waveform-systematics analysis, improved uncertainty modeling, or alternative architectures should be introduced as new experimental phases rather than modifying the closed M10 definition.