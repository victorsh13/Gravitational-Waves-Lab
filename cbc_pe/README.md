# CBC parameter estimation with CNNs and Mondrian conformal intervals

This module contains the `cbc_pe` workflow for synthetic BBH gravitational-wave parameter estimation.

The current pipeline generates synthetic compact-binary coalescence signals, trains CNN regressors to predict physical parameters, and evaluates Mondrian conformal prediction intervals.

## Current status

The current large-scale workflow uses HDF5 datasets.

Generation and CNN training are script-based. CNN evaluation and Mondrian conformal analysis are currently notebook-based, with planned script versions.

The current CNN architecture-selection phase uses an 80/20 train/validation split. After selecting the final architecture, the intended final split is 70/10/10/10 for train/validation/calibration/test.

## Repository structure

```text
cbc_pe/
├── configs/        JSON configuration files
├── data/           local datasets and generated artifacts, ignored by Git
├── docs/           project documentation
├── notebooks/      lightweight demos and analysis notebooks
├── scripts/        command-line scripts for generation, splits, and training
├── src/            source code
└── tests/          tests
```

## Main workflow

### 1. Generate an HDF5 BBH dataset

```bash
python cbc_pe/scripts/generate_bbh_dataset_hdf5.py --config cbc_pe/configs/generate_bbh_config.json
```

To resume an interrupted generation:

```bash
python cbc_pe/scripts/generate_bbh_dataset_hdf5.py --config cbc_pe/configs/generate_bbh_config.json --resume
```

To overwrite an existing output file:

```bash
python cbc_pe/scripts/generate_bbh_dataset_hdf5.py --config cbc_pe/configs/generate_bbh_config.json --overwrite
```

### 2. Create HDF5 splits

```bash
python cbc_pe/scripts/create_hdf5_splits.py --config cbc_pe/configs/splits_bbh_config.json
```

The current default split is intended for CNN architecture selection:

```text
train = 80%
validation = 20%
calibration = 0%
test = 0%
```

For final conformal analysis, use a separate config with:

```text
train = 70%
validation = 10%
calibration = 10%
test = 10%
```

### 3. Train a CNN model

Baseline model:

```bash
python cbc_pe/scripts/train_cnn_hdf5.py --config cbc_pe/configs/train_simpleCNN_baseline.json
```

Pooling model:

```bash
python cbc_pe/scripts/train_cnn_hdf5.py --config cbc_pe/configs/train_simpleCNN_pool.json
```

## Notebooks

Current visible notebooks:

```text
notebooks/01_signal_generation_demo.ipynb
notebooks/02_cnn_training_hdf5_demo.ipynb
notebooks/03_cnn_evaluation_hdf5_demo.ipynb
notebooks/04_mondrian_hdf5_demo.ipynb
```

Notebook roles:

01_signal_generation_demo.ipynb: visual and numerical sanity checks for generated BBH signals.
02_cnn_training_hdf5_demo.ipynb: lightweight inspection of HDF5 training configs, dataset shapes, splits, label statistics, loaders, and model construction.
03_cnn_evaluation_hdf5_demo.ipynb: CNN prediction diagnostics using HDF5-based training outputs.
04_mondrian_hdf5_demo.ipynb: Mondrian conformal analysis using saved predictions, labels, and embeddings.

CNN training is script-based:

```bash
python cbc_pe/scripts/train_cnn_hdf5.py --config cbc_pe/configs/train_simpleCNN_baseline.json
```

A lightweight training-inspection notebook may be added later as:

```bash
notebooks/02_cnn_training_hdf5_demo.ipynb
```

Exploratory or outdated notebooks are archived under:

```text
notebooks/_archive/
```

The final goal is to keep only a small number of clean demo notebooks.

## Documentation

See:

```text
docs/workflow.md
```

for the current workflow.

## Data and artifacts

Large datasets and generated artifacts are not tracked by Git.

Ignored artifacts include:

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
```

This repository should contain code, configuration files, documentation, tests, and lightweight notebooks. It should not contain generated datasets, checkpoints, prediction dumps, or temporary outputs.