# Configuration files

This directory contains JSON configuration files for dataset generation, split creation, and CNN training.

## Main categories

- `generation/`: HDF5 dataset generation configs.
- `splits/`: reproducible train/validation/calibration/test split configs.
- `experiments/`: explicit training experiment configs.
- `experiments/processing_benchmark/`: small processing-comparison training configs.

## Current main workflows

Architecture search on 100k:

- `experiments/train_100k_*.json`
- `splits/splits_100k_train80_val20_seed123.json`

500k scaling experiment:

- `generation/generate_500k_bbh_4s.json`
- `splits/splits_500k_train400_val40_cal30_test30_seed123.json`
- `experiments/train_500k_M00_simplecnn_emb64_mse_seed123.json`

Older default configs may remain at the top level for backward compatibility with README/tutorial commands.
