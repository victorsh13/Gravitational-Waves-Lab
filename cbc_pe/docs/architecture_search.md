# CNN architecture search

Current phase: architecture selection on 100k samples using an 80/20 train/validation split.

Final conformal evaluation will use a separate 70/10/10/10 split after selecting the architecture.

| Run ID | Config | Model | Embedding dim | Pool size | Loss | Train size | Val size | Best val loss | Notes |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| baseline_emb64_mse | train_simpleCNN_baseline.json | SimpleCNN_Baseline | 64 | n/a | MSELoss | 80000 | 20000 | 0.1771 | Baseline architecture |
| pool1_emb128_mse | train_simpleCNN_pool.json | SimpleCNN_Pool | 128 | 1 | MSELoss | 80000 | 20000 | 0.1785 | Pooling experiment with embedding_dim=128 |