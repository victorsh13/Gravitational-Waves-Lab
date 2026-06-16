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

Dataset path:

```text
/scratch/vserrano/cbc_pe_data/processed/bbh_processed_4s_seobnrv4opt_snr10-25_n500_000/
```

Split:

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

Main dataset artifacts:

```text
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000.h5
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000_label_stats_train_only_train400000_val40000_cal30000_test30000_seed123.npz
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000_splits_train400000_val40000_cal30000_test30000_seed123.npz
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000.metadata.json
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000_splits_train400000_val40000_cal30000_test30000_seed123.metadata.json
```

## Model

Baseline model:

```text
M00 SimpleCNN baseline
```

Current equivalent training config:

```text
configs/experiments/train_500k_M00_simplecnn_emb64_mse_seed123.json
```

Historical launch path used for this run:

```text
configs/train_simpleCNN_500k_baseline.json
```

This run was launched before the config cleanup/reorganization. The training script reads the JSON config at startup, so moving or renaming config files after launch does not affect the completed run.

## Launch command

The training run was launched on the VM with:

```bash
nohup /usr/bin/time -v python scripts/train_cnn_hdf5.py \
  --config configs/train_simpleCNN_500k_baseline.json \
  > data/logs/train_500k_baseline_seed123_workers4.log 2>&1 &
```

Process ID at launch:

```text
1546
```

Original log file:

```text
data/logs/train_500k_baseline_seed123_workers4.log
```

## Training outcome

The 500k M00 baseline training completed successfully.

Training history:

```text
epochs recorded:   99
best val_loss:     0.1379008364
best epoch:        74
final train_loss:  0.1349718811
final val_loss:    0.14712434
```

The best validation checkpoint should be used for evaluation, not the final epoch, since validation loss degraded after the best epoch.

Compared with the 100k M00 baseline runs:

```text
100k M00 seed123 best val_loss ≈ 0.1771
100k M00 seed124 best val_loss ≈ 0.1733
500k M00 seed123 best val_loss ≈ 0.1379
```

This is a clear improvement over the 100k baseline range. The result supports the conclusion that increasing the training dataset size substantially improves the M00 baseline.

The approximate relative improvement against the 100k M00 seed123 run is:

```text
(0.1771 - 0.1379) / 0.1771 ≈ 22%
```

This suggests that, for the current baseline, data scaling is more effective than the small architecture changes previously explored around 100k samples.

## Generated training and evaluation artifacts

Results directory:

```text
/scratch/vserrano/cbc_pe_data/results/bbh_processed_4s_seobnrv4opt_snr10-25_n500_000/
```

Main training/evaluation files:

```text
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000_SimpleCNN_Baseline_simple_emb64_mse_MSELoss_seed123_history.npz
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000_SimpleCNN_Baseline_simple_emb64_mse_MSELoss_seed123_train_val_cal_test_predictions_embeddings.npz
```

The predictions/embeddings file contains all four splits:

```text
train: pred_train, emb_train, y_train, idx_train
val:   pred_val,   emb_val,   y_val,   idx_val
cal:   pred_cal,   emb_cal,   y_cal,   idx_cal
test:  pred_test,  emb_test,  y_test,  idx_test
```

This makes the run suitable for real calibration/test Mondrian evaluation.

## Mondrian evaluation

Mondrian conformal evaluation was performed using the real calibration and test splits:

```text
cal:  30000 samples
test: 30000 samples
```

Mondrian results directory:

```text
/scratch/vserrano/cbc_pe_data/results/bbh_processed_4s_seobnrv4opt_snr10-25_n500_000/mondrian/
```

Main Mondrian files:

```text
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000_SimpleCNN_Baseline_simple_emb64_mse_MSELoss_seed123_train_val_cal_test_predictions_embeddings_mondrian_summary_real_cal_test.csv
bbh_processed_4s_seobnrv4opt_snr10-25_n500_000_SimpleCNN_Baseline_simple_emb64_mse_MSELoss_seed123_train_val_cal_test_predictions_embeddings_mondrian_final_configs_real_cal_test.csv
```

## Mondrian selection policy

The final Mondrian selection was treated as a multiobjective problem.

The goal was not simply to minimize interval width or maximize the number of bins. Mondrian binning is useful only if the increased local adaptivity does not introduce unacceptable local undercoverage.

The selection procedure first defines a base candidate set requiring:

```text
global_within_2sigma == True
min_count_per_bin >= 100
```

Then configurations are selected using a hierarchical local-validity policy.

### Strict local validity

A configuration is labelled as:

```text
strict_local_validity
```

if it satisfies:

```text
global_within_2sigma == True
n_bad_bins_p005 == 0
n_bins_under_2sigma == 0
n_bins_outside_2sigma == 0
min_count_per_bin >= 100
```

This means:

* global coverage is compatible with the nominal confidence level;
* no bin is statistically undercovered according to the binomial p-value criterion;
* no bin falls below the local 2sigma tolerance band;
* no bin falls outside the local 2sigma tolerance band;
* every bin has sufficient sample support.

Among strict candidates, the final ranking prioritizes local validity, then physical interval width, and only then the number of bins. This avoids selecting a more adaptive configuration if it has worse local behavior or unnecessarily wider intervals.

### Relaxed best available

If no strict locally valid configuration exists for a label, the fallback policy is:

```text
relaxed_best_available
```

This policy is mainly expected for difficult targets such as `chi_eff`.

The relaxed policy still requires global coverage within 2sigma and enough samples per bin, but allows a limited number of local issues. Ranking then prioritizes:

```text
1. fewer bins under local 2sigma
2. fewer bins outside local 2sigma
3. fewer statistically bad bins, n_bad_bins_p005
4. smaller max_undercoverage_gap
5. smaller median physical interval width
6. larger number of bins
```

This makes the limitation explicit instead of hiding it. A relaxed selection should not be interpreted as locally perfect.

## Final Mondrian configurations

Final selected configurations:

| label      | selection_policy       | taxonomy_mode | interval_mode | n_bins | global_coverage | global_median_width_phys | n_bad_bins_p005 | n_bins_under_2sigma | n_bins_outside_2sigma | min_coverage_per_bin | max_undercoverage_gap |
| ---------- | ---------------------- | ------------- | ------------- | -----: | --------------: | -----------------------: | --------------: | ------------------: | --------------------: | -------------------: | --------------------: |
| chirp_mass | strict_local_validity  | difficulty    | asymmetric    |      8 |        0.898967 |                14.712123 |               0 |                   0 |                     0 |             0.892549 |              0.007451 |
| total_mass | strict_local_validity  | difficulty    | asymmetric    |      4 |        0.902400 |                28.008818 |               0 |                   0 |                     0 |             0.900267 |              0.000000 |
| chi_eff    | relaxed_best_available | difficulty    | asymmetric    |      4 |        0.900233 |                 0.651478 |               1 |                   0 |                     0 |             0.893929 |              0.006071 |

## Interpretation of Mondrian results

The final selected configurations all use the `difficulty` taxonomy. This suggests that difficulty-based Mondrian binning is preferable to prediction-based binning for this run.

For `chirp_mass`, the selected configuration achieves strict local validity:

```text
taxonomy_mode: difficulty
interval_mode: asymmetric
n_bins:        8
coverage:      0.898967
median width:  14.712123
```

No bins are flagged as statistically undercovered, and all bins are inside the local 2sigma tolerance band.

For `total_mass`, the selected configuration also achieves strict local validity:

```text
taxonomy_mode: difficulty
interval_mode: asymmetric
n_bins:        4
coverage:      0.902400
median width:  28.008818
```

This configuration is preferred over configurations with more bins because it provides excellent local behavior, zero undercoverage gap, and a competitive interval width.

For `chi_eff`, no fully strict configuration is selected. The final configuration is therefore reported as the best relaxed candidate:

```text
taxonomy_mode: difficulty
interval_mode: asymmetric
n_bins:        4
coverage:      0.900233
median width:  0.651478
```

Although the global coverage is close to the nominal target and all bins are within the local 2sigma tolerance band, one bin is flagged by the binomial undercoverage p-value criterion:

```text
n_bad_bins_p005 = 1
bad_bin_fraction = 0.25
```

Therefore, `chi_eff` remains the most difficult target. Its selected Mondrian configuration should be interpreted as the best available relaxed solution, not as a fully locally valid result.

## Comparison with 100k runs

The regression comparison between 100k and 500k is direct at the validation-loss level. The 500k run clearly improves over the 100k M00 baseline range.

However, the Mondrian comparison should be interpreted carefully:

```text
100k runs used a pseudo calibration/test split from validation.
500k uses real calibration and test splits.
```

Therefore, the 100k and 500k conformal results are not strictly equivalent. The 500k Mondrian result is methodologically cleaner because it uses dedicated calibration and test splits.

## Runtime and I/O notes

The 500k training run took approximately three days.

The approximate epoch time was around:

```text
35 minutes per epoch
```

The main bottleneck appears to be data loading rather than GPU compute. The dataset was effectively accessed through a setup where data were read from local storage and transferred to the VM during training. This likely introduced a significant I/O bottleneck.

Before launching additional 500k seeds, the data-loading pipeline should be profiled and optimized.

Potential optimization directions:

```text
local copy of the HDF5 file to fast VM storage
HDF5 chunking strategy
HDF5 compression settings
sequential vs random read access
DataLoader num_workers tuning
pin_memory tuning
prefetching
batch-level reads instead of sample-level random reads
caching frequently used chunks
```

Given the current runtime cost, launching another 500k seed is not the immediate priority. The next step should be I/O benchmarking and evaluation consolidation.

## Main conclusions

The 500k M00 baseline run supports the following conclusions:

1. Scaling from 100k to 500k substantially improves validation performance.

2. The improvement is large enough to justify prioritizing dataset scaling over small architecture changes, at least for the current M00 baseline.

3. Real calibration/test Mondrian evaluation is now available for the 500k run.

4. `chirp_mass` and `total_mass` achieve strict local Mondrian validity.

5. `chi_eff` remains the limiting parameter. It achieves good global coverage, but one bin is still flagged by the binomial undercoverage test.

6. Difficulty-based Mondrian binning is preferred for all selected labels.

7. The next technical bottleneck is not necessarily model capacity, but data-loading efficiency.

## HDF5 I/O optimization debug

A set of debug runs was performed to reduce the training time of the 500k HDF5 dataset.

The original training pipeline used shuffled sample-level HDF5 access. This was inefficient because the HDF5 dataset is chunked as:

- `X chunks = (64, 3, 16384)`
- `batch_size = 64`

Two alternative data-loading modes were tested:

1. `sorted_block`: groups physically nearby HDF5 indices into batches.
2. `hdf5_batch_slices`: uses an `IterableDataset` that reads full batches from HDF5, using contiguous slices when the batch is dense and fancy indexing when the batch is sparse.

Observed debug timings:

| mode | epoch | train_loss | val_loss | train_time_s | val_time_s | epoch_time_s |
|---|---:|---:|---:|---:|---:|---:|
| sorted_block | 1 | 0.540163 | 0.381555 | 928.6 | 168.6 | 1097.2 |
| sorted_block | 2 | 0.354093 | 0.284130 | 891.2 | 166.2 | 1057.4 |
| sorted_block | 3 | 0.302040 | 0.251019 | 727.6 | 150.1 | 877.7 |
| hdf5_batch_slices | 1 | 0.518039 | 0.367898 | 779.3 | 169.3 | 948.6 |
| hdf5_batch_slices | 2 | 0.331414 | 0.282257 | 647.1 | 159.6 | 806.7 |
| hdf5_batch_slices | 3 | 0.295274 | 0.255147 | 643.5 | 154.7 | 798.2 |

The `hdf5_batch_slices` mode reduced epoch time from roughly 35 minutes in the original run to approximately 13-16 minutes in the debug run. The validation loss decreased normally during the debug run, suggesting that the new data-loading mode does not obviously break training dynamics.

GPU usage still oscillates, so data loading and CPU-to-GPU transfer remain relevant bottlenecks. Further optimization may require testing larger batch sizes, for example `batch_size=128`.

## Next steps

1. Compare 500k M00 against the 100k M00 baseline runs in regression space.

2. Interpret the 100k vs 500k Mondrian comparison carefully because the split methodology differs.

3. Benchmark HDF5/DataLoader throughput before launching any additional 500k training run.

4. Investigate `chi_eff` specifically:

   * residuals as a function of true `chi_eff`;
   * coverage by bins;
   * difficulty-score behavior;
   * possible loss or model-head modifications.

5. Prepare a concise supervisor-facing summary with:

   * validation loss comparison;
   * final Mondrian configurations;
   * coverage/width table;
   * note about `chi_eff` limitations.
