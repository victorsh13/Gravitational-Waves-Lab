# 500k M00 baseline experiment

This document tracks the first large-scale 500k training run for the CBC parameter-estimation CNN baseline.

## Goal

The goal of this experiment is to test whether increasing the training dataset size improves the M00 SimpleCNN baseline more than the small architecture changes explored in the 100k architecture search.

Main question:

> Does scaling the data from 100k to 500k reduce validation error enough to justify prioritizing dataset size over architecture changes?

## Dataset

Dataset ID:

```text
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000
```

Expected split:

```text
train: 400000
val:    40000
cal:    30000
test:   30000
```

Split config:

```text
configs/splits/splits_500k_train400_val40_cal30_test30_seed123.json
```

## Model

Baseline model:

```text
M00 SimpleCNN baseline
```

Expected current training config:

```text
configs/experiments/train_500k_M00_simplecnn_emb64_mse_seed123.json
```

Historical launch path used for the current run:

```text
configs/train_simpleCNN_500k_baseline.json
```

The current run was launched before the config cleanup/reorganization. The training script reads the JSON config at startup, so moving or renaming config files after launch does not affect the running process.

## Launch command

The current training run was launched on the VM with:

```bash
nohup /usr/bin/time -v python scripts/train_cnn_hdf5.py \
  --config configs/train_simpleCNN_500k_baseline.json \
  > data/logs/train_500k_baseline_seed123_workers4.log 2>&1 &
```

Process ID at launch:

```text
1546
```

Log file:

```text
data/logs/train_500k_baseline_seed123_workers4.log
```

## Notes

* This run uses the M00 baseline as the main reference architecture.
* The purpose is not architecture tuning, but testing the effect of larger training data.
* The result should be compared against the 100k M00 seed variability, not only against one 100k seed.
* Final Mondrian/conformal analysis should use the real calibration and test splits, not pseudo-cal/test debug mode.
* The run was launched with the old config path, but the equivalent current config should be kept under `configs/experiments/`.

## Evaluation plan after training finishes

1. Confirm that the best checkpoint and training history were saved.

2. Generate train/val/cal/test predictions and embeddings, if not already produced.

3. Evaluate regression performance with:

   ```text
   notebooks/03_cnn_evaluation_hdf5_demo.ipynb
   ```

4. Compare 500k M00 against the 100k M00 seed variability.

5. If regression improves meaningfully, run real Mondrian conformal evaluation with:

   ```text
   notebooks/04_mondrian_hdf5_demo.ipynb
   ```

6. Report:

   * global regression metrics
   * per-label regression metrics
   * physical-space errors
   * conformal coverage
   * interval widths
   * undercoverage bins
   * final candidate Mondrian configurations

## Interpretation criteria

The 500k run should be considered useful if it produces a clear improvement over the 100k M00 baseline distribution.

A small improvement over only one 100k seed is not enough. The comparison should account for seed variability.

Possible outcomes:

### Case A: 500k clearly improves

Prioritize:

```text
M00 baseline + larger dataset + real Mondrian calibration/test evaluation
```

This would suggest that data scaling is more valuable than small architecture changes.

### Case B: 500k improves only marginally

Do not assume architecture is solved. Next directions may include:

```text
loss function changes
target reparameterization
preprocessing checks
residual architectures
regularization
SNR/parameter-space stratified evaluation
```

### Case C: 500k does not improve

This would suggest a bottleneck in one or more of:

```text
signal preprocessing
label choice
network architecture
training objective
information content of the generated data
difficulty of chi_eff estimation
```

In that case, further architecture work should be guided by detailed error breakdowns, not blind model search.
