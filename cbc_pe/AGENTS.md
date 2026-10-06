# CBC PE project instructions

## Scientific methodology

- This repository is used for gravitational-wave CBC parameter-estimation experiments.
- M10 is a frozen scientific baseline. Do not modify M10 retrospectively.
- M11 experiments must follow a one-variable-at-a-time methodology whenever possible.
- Do not introduce additional preprocessing, normalization, architecture changes, diagnostics, or hyperparameter changes unless explicitly requested.
- Preserve paired samples, source IDs, train/val/cal/test splits, seeds, and train-only label scalers when comparing domains.
- Do not infer scientific conclusions from code changes alone.

## Current scientific context

- Main labels:
  - chirp_mass
  - total_mass
  - chi_eff
- Input:
  - H1, L1, V1
  - 4096 Hz
  - 4 s
- Current waveform approximant:
  - SEOBNRv4_opt
- M10 uses per-sample/per-detector z-score.
- M11.6 compared:
  - G0: analytical-PSD Gaussian noise
  - G1: empirical-PSD Gaussian noise
  - R: real off-source detector noise
- M11.9 compares G1 and R without the final per-sample/per-detector z-score.
- The current scientific conclusion is that G1 and R are very similar for this regression task under the current preprocessing pipeline.

## Workflow

- Prefer notebooks for exploratory scientific analysis.
- Move only validated and reusable logic into scripts.
- Do not run expensive full datasets unless explicitly requested.
- Do not use Run All on large notebooks unless explicitly requested.
- Before editing shared training or dataset code, inspect whether the change affects historical configurations.
- Preserve backward compatibility with old JSON configs.
- Prefer minimal, reviewable changes over refactors.
- Do not create unnecessary tests or diagnostics.

## Environment

- Python 3.10 compatibility is required.
- GPU training VM currently uses:
  - PyTorch 1.9.1+cu111
  - Tesla P100 GPUs
- Main training environment:
  - cbc_torch
- Use existing src.paths infrastructure for data-root and artifact resolution.

## Git

- Active branch:
  - m11-real-noise-domain-gap
- Do not commit automatically unless explicitly requested.
- Show or summarize the diff before proposing a commit.
- Do not modify frozen scientific artifacts.

## Coding style

- Keep changes small and local.
- Reuse existing project utilities instead of duplicating logic.
- Avoid introducing new dependencies unless necessary.
- Add comments only where they clarify non-obvious scientific or technical behavior.