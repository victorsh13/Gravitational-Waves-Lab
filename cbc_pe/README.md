# CBC parameter estimation with CNNs and Mondrian conformal intervals

`cbc_pe` is a research pipeline for compact-binary coalescence parameter estimation from gravitational-wave strain using convolutional neural networks and Mondrian conformal prediction.

The current closed reference baseline is **M10-500k**, trained on synthetic BBH signals and evaluated both on held-out synthetic data and on real GWOSC events.

## Current status

The current closed M10 workflow uses:

- 500,000 synthetic BBH injections
- H1, L1, and V1 detector strain
- 4 s input windows sampled at 4096 Hz
- `SEOBNRv4_opt` waveforms
- target network SNR in the range 10–25
- train/validation/calibration/test split of:
  - 400,000 train
  - 40,000 validation
  - 30,000 calibration
  - 30,000 test
- `SimpleCNN_ResidualDilated`
- per-sample, per-detector input z-score normalization
- physical targets:
  - chirp mass
  - total mass
  - effective spin `chi_eff`
- Mondrian conformal prediction with prediction- and difficulty-based taxonomies
- real-data validation with GWOSC strain
- multi-event comparison with published LVK/GWTC-3 parameter estimates

M10 is treated as a closed baseline. Future methodological improvements should be developed in a new experimental phase rather than modifying the closed M10 chain.

## Repository structure

```text
cbc_pe/
├── configs/        JSON configurations for generation, splits, training, and prediction
├── data/           lightweight placeholder structure only; large data are external
├── docs/           workflow, methods, experiment history, and scientific documentation
├── notebooks/      final M10 notebooks, demos, audits, and archived development notebooks
├── scripts/        command-line entry points
├── src/            reusable scientific and ML implementation
└── tests/          unit and regression tests
```

Main source-code areas:

```text
src/
├── conformal/      Mondrian calibration, binning, selection, and fitted calibrators
├── evaluation/     evaluation utilities, including LVK comparison metrics
├── models/         CNN architectures, datasets, training, and inference utilities
├── real_data/      GWOSC access, PSD estimation, real-data preprocessing, and event execution
└── ...             synthetic-data generation and signal-processing modules
```

## External data root

Large datasets, checkpoints, prediction artifacts, GWOSC caches, and generated results are stored outside the Git repository.

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

For the current local setup:

```text
/data/vserrano/cbc_pe_data
```

The portable way to configure this path is:

```bash
export CBC_PE_DATA_ROOT=/data/vserrano/cbc_pe_data
```

Path precedence is:

```text
CLI --data-root
→ CBC_PE_DATA_ROOT
→ data_root in JSON config
```

The repository root can also be overridden with:

```bash
--project-root
```

but is otherwise inferred automatically if the configured path is not valid on the current machine.

## Closed M10 configuration chain

The reference M10 pipeline is defined by:

```text
configs/generation/generate_500k_bbh_4s.json

configs/splits/
    splits_500k_train400_val40_cal30_test30_seed123.json

configs/experiments/
    train_500k_M10_inputzscore_resdilated_emb64_d124_bs256_seed123.json

configs/predictions/
    predict_500k_M10_inputzscore_cal_test.json
```

The resulting analysis chain is:

```text
synthetic generation
→ train/val/cal/test split + train-only label statistics
→ M10 CNN training
→ calibration/test predictions and embeddings
→ synthetic evaluation
→ Mondrian calibration and selection
→ single-event real-data inference
→ multi-event GWTC-3 / LVK comparison
```

## Main commands

Run the following commands from the `cbc_pe/` directory.

### Generate the 500k dataset

```bash
python scripts/generate_bbh_dataset_hdf5.py \
  --config configs/generation/generate_500k_bbh_4s.json
```

Resume an interrupted generation:

```bash
python scripts/generate_bbh_dataset_hdf5.py \
  --config configs/generation/generate_500k_bbh_4s.json \
  --resume
```

### Create the final split

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

All four scripts support:

```text
--project-root
--data-root
```

and large-data paths can be configured through `CBC_PE_DATA_ROOT`.

## Final M10 notebooks

The four notebooks kept at the top level of `notebooks/` form the final M10 analysis chain:

```text
10_m10_inputzscore_500k_evaluation.ipynb
11_mondrian_m10_inputzscore_500k_final.ipynb
12_real_event_inference_m10_500k_clean.ipynb
13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

Roles:

- **10** — synthetic M10-500k regression evaluation.
- **11** — Mondrian conformal grid evaluation and final configuration selection.
- **12** — controlled single-event real-data inference.
- **13** — multi-event GWOSC inference and comparison with LVK/GWTC-3 reference values.

Development notebooks from M08, early M10, and architecture search are kept under:

```text
notebooks/_archive/
```

Reusable lightweight examples are kept under:

```text
notebooks/demos/
```

Pipeline and data-quality audits are kept under:

```text
notebooks/audits/
```

## Real-data inference

Real-event processing reuses the same signal-processing contract as the synthetic pipeline where applicable.

The reusable real-data implementation lives in:

```text
src/real_data/
```

and includes:

- GWOSC catalog and strain-file access
- HDF5 validation and caching
- PSD-window validation and selection
- off-source PSD estimation
- preprocessing of real strain into CNN-ready network input
- M10 input normalization
- CNN prediction and embedding extraction
- LVK detector-frame reference construction
- single-event execution orchestration

Real-data intervals are produced without access to target truth.

## Mondrian conformal prediction

The conformal implementation is under:

```text
src/conformal/
```

The closed M10 analysis supports:

- prediction-based taxonomy
- embedding/difficulty-based taxonomy
- symmetric intervals
- asymmetric intervals
- bin-wise local-validity diagnostics
- conservative and efficient configuration-selection policies
- reusable fitted calibrators that can be applied to real events without target truth

The main closed analysis is documented in:

```text
notebooks/11_mondrian_m10_inputzscore_500k_final.ipynb
```

## Tests

Run the complete test suite from `cbc_pe/`:

```bash
python -m unittest discover \
  -s tests \
  -p "test_*.py" \
  -v
```

The current test suite covers:

- Mondrian fit/apply equivalence
- selected calibrators
- input-normalization equivalence
- path and artifact resolution
- GWOSC utilities
- PSD handling
- real-data preprocessing
- real-event execution
- LVK reference conversion
- LVK comparison metrics

## Git and artifact policy

The repository should contain:

- source code
- scripts
- configuration files
- lightweight scientific documentation
- selected figures used for documentation
- notebooks
- tests

It should not contain:

- large HDF5 datasets
- checkpoints
- prediction dumps
- GWOSC strain caches
- transient logs
- temporary outputs

Historical NPZ scripts remain under:

```text
scripts/legacy_npz/
```

for reference only.

## Documentation

Main documentation:

```text
docs/workflow.md
docs/training.md
docs/evaluation.md
docs/mondrian.md
docs/experiments/
```

Historical architecture-search and early 500k baseline notes are retained under:

```text
docs/experiments/
```

rather than being treated as the current workflow.