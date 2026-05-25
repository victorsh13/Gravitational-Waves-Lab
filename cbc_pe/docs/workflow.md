# CBC-PE workflow

This document describes the current workflow for the `cbc_pe` project.

The project focuses on synthetic BBH gravitational-wave signals, CNN-based parameter estimation, and Mondrian conformal prediction intervals.

## Main workflow

For large-scale experiments, the preferred dataset format is HDF5.

NPZ files are still useful for small debugging datasets, but they should not be considered the main workflow for large training runs.

## 1. Generate BBH signals

Synthetic BBH datasets are generated using:

```bash
python cbc_pe/scripts/generate_bbh_dataset_hdf5.py --config cbc_pe/configs/generate_bbh_config.json
```

The script also supports:

```bash
python cbc_pe/scripts/generate_bbh_dataset_hdf5.py --config cbc_pe/configs/generate_bbh_config.json --resume
python cbc_pe/scripts/generate_bbh_dataset_hdf5.py --config cbc_pe/configs/generate_bbh_config.json --overwrite
```

The generation pipeline includes:

- BBH parameter sampling
- waveform generation
- detector projection
- noise generation
- signal injection
- preprocessing
- label generation
- network SNR control

Large generated datasets are stored locally under:

```text
cbc_pe/data/
```

The `cbc_pe/data/` directory is ignored by Git.

## 2. Create dataset splits

Train, validation, calibration, and test splits are created using:

```bash
python cbc_pe/scripts/create_hdf5_splits.py --config cbc_pe/configs/splits_bbh_config.json
```

The script also supports:

```bash
python cbc_pe/scripts/create_hdf5_splits.py --config cbc_pe/configs/splits_bbh_config.json --overwrite
```

The intended role of each split is:

- `train`: used to fit the CNN model
- `validation`: used for model selection and early stopping
- `calibration`: used to calibrate conformal intervals
- `test`: used only for final evaluation

The calibration and test sets should not be used during CNN training.

## 3. Train CNN models

CNN models are trained using:

```bash
python cbc_pe/scripts/train_cnn_hdf5.py --config cbc_pe/configs/train_simpleCNN_baseline.json
```

Alternative model configurations can be stored under:

```text
cbc_pe/configs/
```

Current examples:

```text
train_simpleCNN_baseline.json
train_simpleCNN_pool.json
```

Model checkpoints, prediction files, embeddings, histories, and logs are generated artifacts and should not be committed to Git.

## 4. Evaluate CNN predictions

CNN evaluation is currently notebook-based.

Current candidate notebook:

```text
cbc_pe/notebooks/03_evaluate_cnn.ipynb
```

The evaluation should include:

- global MSE, RMSE, MAE, and bias
- per-label metrics
- standardized-space metrics
- physical-space metrics
- residual diagnostics
- prediction versus truth plots

Planned script:

```text
cbc_pe/scripts/evaluate_cnn.py
```

## 5. Run Mondrian conformal analysis

Mondrian conformal analysis is currently notebook-based.

Current HDF5 notebook:

```text
cbc_pe/notebooks/04_mondrian_hdf5_demo.ipynb
```

The conformal analysis should use:

- CNN predictions
- true labels
- optional CNN embeddings
- calibration split
- test split

The main quantities to report are:

- empirical coverage
- miscoverage
- interval width
- coverage per bin
- width per bin
- counts per bin
- comparison between taxonomy modes
- comparison between interval modes

Planned script:

```text
cbc_pe/scripts/run_mondrian.py
```

## Dataset formats

### HDF5

HDF5 is the preferred format for large datasets, such as:

- 100k samples
- 500k samples
- 1M samples

This format is better suited for scalable training because it avoids loading the full dataset into memory.

### NPZ

NPZ is kept mainly for:

- small debugging datasets
- quick sanity checks
- early experiments

NPZ should not be used as the main format for large-scale training.

## Git policy

The repository should contain:

- source code
- scripts
- configuration files
- lightweight notebooks
- documentation
- tests

The repository should not contain:

- large datasets
- generated HDF5 files
- generated NPZ files
- model checkpoints
- prediction dumps
- temporary outputs
- logs
- cache files

The following paths and file types are ignored by Git:

```text
cbc_pe/data/
cbc_pe/outputs/
*.npz
*.h5
*.hdf5
*.pt
*.pth
*.ckpt
*.log
__pycache__/
.ipynb_checkpoints/
```

## Current status

The current large-scale workflow is based on HDF5.

Generation and CNN training are script-based.

CNN evaluation and Mondrian conformal analysis are still partly notebook-based and should eventually be converted into reproducible scripts.