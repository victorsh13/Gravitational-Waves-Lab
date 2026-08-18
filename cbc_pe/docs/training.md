# CNN training workflow

This document describes the current CNN training workflow for the `cbc_pe` project.

The closed reference model is **M10-500k**.

Earlier 100k architecture-search experiments and M00/M08 runs are retained as historical experiments, but they are not the current reference workflow.

## 1. Closed M10 training setup

Dataset:

```text
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000
```

Split:

```text
train: 400000
val:    40000
cal:    30000
test:   30000
```

Training config:

```text
configs/experiments/
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

Reference model class:

```text
SimpleCNN_ResidualDilated
```

Main training characteristics:

```text
embedding dimension:       64
residual channels:         64
dilations:                 [1, 2, 4]
residual kernel size:      7
dropout_conv:              0.05
dropout_dense:             0.1
num_groups:                8
loss:                      MSELoss
batch size:                256
seed:                      123
```

The JSON config is the authoritative source for the exact model and training parameters.

## 2. Input normalization

M10 introduces:

```text
per_sample_per_detector_zscore
```

Each detector time series in each sample is normalized independently using its own time-domain mean and standard deviation.

Conceptually:

```text
for each sample:
    for each detector:
        x_norm = (x - mean(x)) / std(x)
```

This normalization is part of the model definition and must be applied consistently during:

```text
training
validation
synthetic prediction
real-data inference
```

The implementation is shared between sample-wise and batch-wise data paths, with regression tests verifying numerical equivalence.

This preprocessing change was introduced to address the detector-scale mismatch observed when transferring the previous synthetic baseline to real GWOSC strain.

## 3. Training data

The CNN reads:

- HDF5 strain data
- reproducible split indices
- train-only label statistics
- model configuration
- training hyperparameters
- input-normalization configuration

The canonical processed-data location is:

```text
<data_root>/processed/<dataset_id>/
```

Typical contents include:

```text
<dataset_id>.h5
<dataset_id>_splits_....npz
<dataset_id>_label_stats_train_only_....npz
metadata files
```

The training script resolves the dataset-specific layout first and retains legacy flat-path support only for backward compatibility.

## 4. Label standardization

Regression targets are:

```text
chirp_mass
total_mass
chi_eff
```

Target means and standard deviations are computed using the **training split only**.

The network is optimized in standardized label space.

Physical predictions are recovered through:

```text
y_physical = y_standardized * y_std + y_mean
```

Using train-only statistics avoids leakage from validation, calibration, or test data.

## 5. Launching training

From `cbc_pe/`:

```bash
python scripts/train_cnn_hdf5.py \
  --config configs/experiments/train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

The script also supports:

```text
--project-root
--data-root
```

For normal use, the external data root can be configured with:

```bash
export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
```

Path precedence is:

```text
--data-root
→ CBC_PE_DATA_ROOT
→ data_root in config
```

## 6. Output layout

Training outputs are generated outside Git.

Canonical checkpoint directory:

```text
<data_root>/models/checkpoints/<dataset_id>/
```

Canonical result directory:

```text
<data_root>/results/<dataset_id>/
```

Typical generated artifacts include:

```text
*_checkpoint.pt
*_history.npz
```

Prediction and embedding extraction for the final closed M10 chain is handled separately by:

```text
scripts/predict_cnn_hdf5.py
```

rather than relying on notebook-only inference logic.

## 7. Checkpoint policy

The best validation checkpoint should be treated as the reference model artifact.

A final epoch is not automatically preferred over the best validation epoch.

The checkpoint filename records the dataset and model configuration sufficiently to associate the artifact with the corresponding experiment config.

For the closed M10 chain, the canonical checkpoint is stored under:

```text
<data_root>/models/checkpoints/
    bbh_processed_4s_seobnrv4opt_snr10-25_n500_000/
```

Legacy checkpoints stored directly under:

```text
<data_root>/models/checkpoints/
```

may still be resolved as a backward-compatible fallback, but new training runs should always write into the dataset-specific directory.

## 8. Calibration and test isolation

The closed split contains dedicated:

```text
cal = 30000
test = 30000
```

samples.

These sets have distinct roles:

- `cal`: conformal calibration
- `test`: final synthetic evaluation of prediction intervals

Neither calibration nor test samples should be used to fit CNN weights.

Training and early stopping operate using only:

```text
train
validation
```

This separation is part of the scientific reproducibility contract of M10.

## 9. Prediction and embedding extraction

After training, run:

```bash
python scripts/predict_cnn_hdf5.py \
  --config configs/predictions/predict_500k_M10_inputzscore_cal_test.json
```

This produces the main closed M10 artifact:

```text
m10_inputzscore_500k_cal_test_predictions_embeddings.npz
```

under:

```text
<data_root>/results/<dataset_id>/
```

This file contains the arrays required for:

- synthetic evaluation
- Mondrian calibration
- difficulty estimation from embeddings
- final conformal test evaluation

Prediction extraction is therefore a separate reproducible stage between training and analysis.

## 10. Training validation

Model selection should not rely only on one global validation-loss value.

Relevant diagnostics include:

- standardized-space loss
- physical-space RMSE
- physical-space MAE
- bias
- R2
- residual structure
- parameter-dependent error
- train/validation gap
- stability across physically difficult regions

For M10, detailed final evaluation is performed in:

```text
notebooks/10_m10_inputzscore_500k_evaluation.ipynb
```

The training script is responsible for fitting the model and saving reproducible artifacts; detailed scientific interpretation belongs to the evaluation stage.

## 11. Historical architecture search

Earlier experiments explored:

- baseline CNN variants
- pooling variants
- embedding dimensions
- regularization
- width/depth changes
- batch-size effects
- residual/dilated architectures
- random-seed variability

Historical notes are retained under:

```text
docs/experiments/architecture_search_100k.md
```

Historical notebooks are under:

```text
notebooks/_archive/architecture_search/
notebooks/_archive/m08_baseline/
```

These records explain how the project evolved toward M08 and M10, but they should not be confused with the current closed pipeline.

## 12. Reproducibility requirements

A training run intended for scientific comparison should preserve:

- exact JSON config
- dataset identifier
- split identifier
- train-only label statistics
- model class and keyword arguments
- preprocessing/normalization mode
- training seed
- checkpoint
- training history
- relevant evaluation artifacts

Avoid changing a completed run's config in a way that changes its scientific meaning.

New methodological changes should receive a new experiment identifier.

## 13. Validation before closing a training phase

Run:

```bash
python -m unittest discover \
  -s tests \
  -p "test_*.py" \
  -v
```

and:

```bash
git diff --check
```

Training or preprocessing changes should additionally be validated against known closed behavior whenever possible.

For M10 specifically, input-normalization equivalence is protected by dedicated regression tests.

## 14. Closed M10 policy

M10 is now a baseline, not an active architecture-search target.

Changes such as:

- alternative waveform families
- real detector noise injections
- different preprocessing
- new architectures
- alternative uncertainty models
- revised target definitions

should be introduced as new experimental phases rather than silently modifying the M10 configuration.