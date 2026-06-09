# CNN architecture search

Current phase: architecture selection on 100k samples using an 80/20 train/validation split.

Final conformal evaluation will use a separate 70/10/10/10 split after selecting the architecture.

| Run ID | Config | Model | Embedding dim | Pool size | Loss | Train size | Val size | Best val loss | Notes |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| M00_baseline_emb64_mse | train_100k_M00_simplecnn_emb64_mse_seed123.json | SimpleCNN_Baseline | 64 | n/a | MSELoss | 80000 | 20000 | 0.1771 | Best so far; lowest global val MSE and best per-label RMSE |
| M01_pool1_emb128_mse | train_100k_M01_simplecnn_pool_emb128_pool1_mse_seed123.json | SimpleCNN_Pool | 128 | 1 | MSELoss | 80000 | 20000 | 0.1785 | Larger embedding did not improve; slightly worse than M00 across labels |
| M02_pool4_emb128_mse | train_100k_M02_simplecnn_pool_emb128_pool4_mse_seed123.json | SimpleCNN_Pool | 128 | 4 | MSELoss | 80000 | 20000 | 0.1829 | Pool4 worsened all labels; no evidence that coarse temporal retention helps yet |
| M04_pooldeep_emb128_pool4_mse | train_100k_M04_simplecnn_pooldeep_emb128_pool4_mse_seed123.json | SimpleCNN_PoolDeep | 128 | 4 | MSELoss | 80000 | 20000 | 0.1762 | Improves chirp_mass and mass tails, but worsens chi_eff and bias. Candidate, not confirmed. |
| M06_widecnn_emb128_pool4_mse | train_100k_M06_widecnn_emb128_pool4_mse_seed123.json | WideCNN_Pool | 128 | 4 | MSELoss | 80000 | 20000 | 0.1810 | Wider encoder converged faster but overfit and underperformed M00/M04. Discarded for now. |
| M00_baseline_emb64_seed124 | train_100k_M00_simplecnn_emb64_mse_seed124.json | SimpleCNN_Baseline | 64 | n/a | MSELoss | 80000 | 20000 | pending | Seed repeat to estimate M00 variability. |
| M00_baseline_emb64_seed125 | train_100k_M00_simplecnn_emb64_mse_seed125.json | SimpleCNN_Baseline | 64 | n/a | MSELoss | 80000 | 20000 | pending | Seed repeat to estimate M00 variability. |
| M04_reg_pooldeep_emb128_pool4 | train_100k_M04_pooldeep_emb128_pool4_mse_dropout020_wd1e3_seed123.json | SimpleCNN_PoolDeep | 128 | 4 | MSELoss | 80000 | 20000 | pending | Tests whether stronger regularization reduces M04 bias/overfitting. |
