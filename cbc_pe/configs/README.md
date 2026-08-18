# Configuration files

This directory contains JSON configuration files for dataset generation, split creation, CNN training, prediction extraction, and related experiments.

## Directory structure

```text
configs/
├── generation/      synthetic HDF5 dataset generation configs
├── splits/          reproducible train/validation/calibration/test splits
├── experiments/     CNN training experiment configs
├── predictions/     prediction and embedding extraction configs
└── ...
```

Some historical or compatibility configs may remain outside these subdirectories, but new workflows should use the structured layout above.

## Closed M10 configuration chain

The current closed reference pipeline is **M10-500k**.

Its main configuration files are:

```text
generation/
    generate_500k_bbh_4s.json

splits/
    splits_500k_train400_val40_cal30_test30_seed123.json

experiments/
    train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json

predictions/
    predict_500k_M10_inputzscore_cal_test.json
```

These four configs define the reproducible synthetic M10 chain:

```text
dataset generation
→ split/stat creation
→ CNN training
→ calibration/test prediction and embedding extraction
```

The subsequent synthetic evaluation, Mondrian analysis, and real-data validation are performed through the final M10 notebooks.

## Dataset configuration

The closed M10 dataset is:

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

The corresponding generation config is:

```text
generation/generate_500k_bbh_4s.json
```

## Split configuration

The closed M10 split is:

```text
train: 400000
val:    40000
cal:    30000
test:   30000
seed:   123
```

Config:

```text
splits/splits_500k_train400_val40_cal30_test30_seed123.json
```

The split file and train-only label statistics are stored under:

```text
<data_root>/processed/<dataset_id>/
```

## M10 training configuration

Reference training config:

```text
experiments/train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

The M10 model uses:

```text
SimpleCNN_ResidualDilated
```

with per-sample, per-detector input z-score normalization.

Important model/training characteristics include:

```text
embedding dimension:   64
dilations:             [1, 2, 4]
loss:                  MSELoss
batch size:            256
seed:                  123
```

The complete authoritative settings are those stored in the JSON config itself.

## Prediction configuration

Reference prediction config:

```text
predictions/predict_500k_M10_inputzscore_cal_test.json
```

This config loads the closed M10 checkpoint and extracts predictions and embeddings for the dedicated calibration and test splits.

The main resulting artifact is:

```text
m10_inputzscore_500k_cal_test_predictions_embeddings.npz
```

stored under:

```text
<data_root>/results/<dataset_id>/
```

## External data root

Large generated artifacts are stored outside the Git repository.

The preferred layout is:

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

The recommended portable configuration is:

```bash
export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
```

Data-root precedence is:

```text
CLI --data-root
→ CBC_PE_DATA_ROOT
→ data_root in config
```

The repository root follows:

```text
CLI --project-root
→ valid project_root in config
→ automatic repository-root detection
```

This keeps historical configs usable while allowing the same experiment config to run on different machines.

## Running the closed M10 chain

From the `cbc_pe/` directory:

### Generate dataset

```bash
python scripts/generate_bbh_dataset_hdf5.py \
  --config configs/generation/generate_500k_bbh_4s.json
```

### Create split and label statistics

```bash
python scripts/create_hdf5_splits.py \
  --config configs/splits/splits_500k_train400_val40_cal30_test30_seed123.json
```

### Train M10

```bash
python scripts/train_cnn_hdf5.py \
  --config configs/experiments/train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

### Extract calibration/test predictions and embeddings

```bash
python scripts/predict_cnn_hdf5.py \
  --config configs/predictions/predict_500k_M10_inputzscore_cal_test.json
```

## Naming conventions

Prefer descriptive configuration names containing:

- dataset scale
- model/run identifier
- major architecture or preprocessing change
- embedding dimension when relevant
- batch size when relevant
- seed

For example:

```text
train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json
```

Avoid vague names such as:

```text
train_new.json
train_final.json
train_final_v2.json
test_config.json
```

A config filename should remain interpretable long after the experiment has finished.

## Historical configurations

Configs from earlier phases such as:

```text
M00
M04
M08
architecture-search runs
processing benchmarks
foundation-model dataset studies
```

are retained for scientific traceability.

They should not be interpreted as part of the active closed M10 chain unless explicitly referenced by historical documentation.

Future methodological work should use new run identifiers rather than modifying the closed M10 configs.