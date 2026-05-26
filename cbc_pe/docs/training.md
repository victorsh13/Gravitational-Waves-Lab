# CNN training workflow

This document describes how CNN training experiments are currently run in the `cbc_pe` project.

The current training workflow is script-based and uses HDF5 datasets.

## Current phase

The project is currently in the CNN architecture-selection phase.

The current dataset setup is:

```text
dataset = bbh_processed_4s_seobnrv4opt_snr10-25_n100_000
split = 80/20 train/validation
train = 80000
validation = 20000
calibration = 0
test = 0
```

This phase is used to compare candidate CNN architectures before scaling to larger datasets.

Final conformal evaluation should not be done with this split. After selecting the CNN architecture, the intended final split is:

```text
train = 70%
validation = 10%
calibration = 10%
test = 10%
```

## Launching training

Training is launched from the repository root using:

```bash
python cbc_pe/scripts/train_cnn_hdf5.py --config cbc_pe/configs/train_simpleCNN_baseline.json
```

For the pooling architecture:

```bash
python cbc_pe/scripts/train_cnn_hdf5.py --config cbc_pe/configs/train_simpleCNN_pool.json
```

The training script reads:

- the HDF5 dataset file
- the split file
- the train-only label statistics file
- the model class and model keyword arguments
- the training hyperparameters
- output options

## Configuration files

Training configs live under:

```text
cbc_pe/configs/
```

Current examples:

```text
train_simpleCNN_baseline.json
train_simpleCNN_pool.json
```

For multiple architecture experiments, prefer adding explicit configs under:

```text
cbc_pe/configs/experiments/
```

Recommended naming convention:

```text
train_100k_<model>_<main_change>_emb<dim>_<loss>_seed<seed>.json
```

Examples:

```text
train_100k_simplecnn_baseline_emb64_mse_seed123.json
train_100k_simplecnn_pool1_emb128_mse_seed123.json
train_100k_simplecnn_pool4_emb128_mse_seed123.json
train_100k_simplecnn_pool8_emb128_mse_seed123.json
```

Avoid vague names such as:

```text
train_test.json
train_new.json
train_final.json
train_final_v2.json
```

They become useless quickly.

## Output files

The training script writes generated artifacts under the local `data_root`, not into Git.

Typical output locations are:

```text
<data_root>/models/checkpoints/
<data_root>/results/
```

Typical generated files include:

```text
*_checkpoint.pt
*_history.npz
*_predictions_embeddings.npz
```

These files are intentionally ignored by Git.

## Prediction and embedding files

If `save_predictions` is enabled in the config, the training script saves predictions and embeddings for available splits.

These files are needed for:

- CNN evaluation
- residual diagnostics
- Mondrian conformal analysis
- difficulty-based Mondrian scores using embeddings

The expected prediction file contains arrays such as:

```text
pred_train
y_train
emb_train
pred_val
y_val
emb_val
pred_cal
y_cal
emb_cal
pred_test
y_test
emb_test
y_mean
y_std
label_names
available_splits
```

Depending on the split config, `cal` and `test` may be absent.

## Architecture search tracking

Each relevant training run should be recorded in:

```text
cbc_pe/docs/architecture_search.md
```

At minimum, record:

- run ID
- config file
- model class
- embedding dimension
- pooling choice
- loss function
- train/validation sizes
- best validation loss
- notes

Do not rely only on checkpoint filenames. They are useful but not enough.

## Model selection criteria

Do not select the final architecture using only global validation loss.

For this project, compare at least:

- validation MSE
- validation RMSE
- validation MAE
- per-label metrics
- bias per label
- residual standard deviation
- prediction-vs-truth plots
- residuals versus physical parameters
- residuals versus SNR
- embedding usefulness for difficulty-based Mondrian

A model with slightly better global validation loss can still be worse for conformal analysis if it has biased residuals, unstable tails, or poor local behavior in physically important regions.

## Recommended workflow

Current architecture-selection workflow:

```text
1. Train candidate architectures on 100k using the 80/20 split.
2. Save predictions and embeddings.
3. Evaluate each candidate on validation metrics and residual diagnostics.
4. Record results in docs/architecture_search.md.
5. Select the best architecture family.
6. Regenerate or reuse a dataset with a 70/10/10/10 split.
7. Retrain the selected architecture.
8. Run proper Mondrian conformal calibration on calibration/test splits.
9. Scale to 500k or 1M only after the pipeline is stable.
```

## Important warning

The 80/20 split is acceptable for architecture selection, but it is not acceptable for final conformal reporting.

Final Mondrian conformal results require separate calibration and test sets.