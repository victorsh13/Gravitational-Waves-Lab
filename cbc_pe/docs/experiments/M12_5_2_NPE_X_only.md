# M12.5.2: one X-only joint neural posterior experiment

Implementation specification; **no scientific results yet**. This document supplements, and does not rewrite, `M12_5_2_information_reference_protocol.md`. The previous patch-MLP proposal is not used, imported, trained, or modified. Existing worktree changes and historical artifacts remain untouched.

## Hypothesis, domain and fixed experiment

Hypothesis: the posterior mean of a conditional density fitted by unweighted conditional NLL to the frozen G1 training observations has lower paired squared-error risk than frozen M11 on the historical validation environment mixture. A lower point-estimation risk alone does not establish posterior accuracy or irreducible risk.

Only the already normalized float32 tensor X, ordered H1/L1/V1, shape 3×16384, enters the encoder. No input normalization, augmentation, cropping, metadata, SNR, PSD, GPS, true nuisance, M11 embedding, or simulator enters it. All 400,000 train rows are consumed every epoch without weighting or dropping the final batch. Labels remain [chirp_mass, total_mass, chi_eff]. Historical train-only means/std are retained for reporting standardized units; the likelihood uses a fixed analytic physical-support transform, not a new fitted scaler. The historical loader's `std + float32(1e-8)` denominator is retained for paired truth checks. Physical M11 decoding follows the historical `inverse_standardize`: prediction × std + mean. Reported standardized moments and risks use the historical std; no statistics are refitted.

The training law is the finite frozen train-G1 mixture, including historical repairs. Evaluation is the empirical **held-out validation environment mixture**, with historical row weights. Environment membership is disjoint; neither posterior correctness nor a train-mixture Bayes decomposition transfers automatically. Source IDs are unique, so one row equals one source in this dataset. Source reuse in another archive must fail the contract rather than be treated as independent noise realizations.

One fixed conditional affine coupling flow has nine layers, 128-wide conditioners and a 128-dimensional temporal embedding. Its encoder has five stride-4 Conv1d/GroupNorm/ELU blocks (32,48,64,96,128 channels), then eight-position pooling and a linear embedding. This learned compression is part of the posterior approximation, not a change to the input observable. There is no architecture search or second CNN competitor. The same model is independently fitted at seeds 123 and 124: primary is always 123, 124 measures optimization/approximation sensitivity and is never selected as a winner. Two seeds do not estimate a training-seed population distribution. Capacity, finite-training-data and misspecification errors remain unbounded.

Implementation uses existing PyTorch, NumPy, h5py and pandas only. No nflows/SBI dependency, attention API, AMP, new Torch serialization option or modern optimizer flag is needed. Python 3.10 and Torch 1.9.1 APIs are used. Actual P100 execution remains untested until the user runs it.

## Density and physical support

Let T be total mass, d=|m1-m2|, and c be chirp mass. Ordered component masses in [5,90] imply

- 10 < T < 180 and 0 ≤ d ≤ min(T−10,180−T);
- c_lo(T)=T [(1−(min(T−10,180−T)/T)²)/4]^(3/5);
- c_hi(T)=T (1/4)^(3/5);
- −1 < chi_eff < 1.

The fixed triangular map uses logits of `(c−c_lo)/(c_hi−c_lo)`, `(T−10)/170`, and `(chi_eff+1)/2`, in historical target order. It includes the exact continuous Jacobian in physical NLL. Gaussian latent tails are mapped back into the physical joint domain; they are never physical Gaussian tails. Mean and variance are computed **after transforming every joint draw**. The transform specifies support, not three independent target priors: the flow learns the correlated G1 law.

Mass labels are stored as float32. Near equal mass, rounding can put chirp mass on or slightly beyond the analytic boundary. Conversion is float64, permits at most four relative float32 ULPs in chirp mass and moves conditional chirp coordinates into [1e−8,1−1e−8]. Every affected training/validation label is counted. Values beyond that tolerance, nonfinite values, invalid total mass/spin, or sampled numerical saturation abort without discarding observations. This is an explicit boundary approximation to quantized-label NLL, not an exact likelihood for rounded data; its effect belongs among unresolved posterior approximation errors. Do not relax tolerances to make a failed run pass. Endpoints have zero probability under the continuous G1 sampler.

## Training, selection, provenance and split gate

Fixed settings: batch 64, Adam lr 3e−4, weight decay 3e−4, gradient norm cap 5, maximum 100 epochs, patience 20, lowest full-validation physical NLL. No alternate loss, scheduler, sweep or pilot. Every improving checkpoint has a new filename; output directories are exclusive and failed partial runs are never resumed/overwritten automatically. `complete.json` names the selected checkpoint and hashes it. History contains full train/val NLL, boundary adjustment counts and elapsed time. Metadata never enters checkpoint selection.

Split/scaler SHA256 pins, historical header pins, label/detector order, all four split memberships, unique sources, disjoint environments, tensor shapes/dtype, finite consumed arrays, baseline prediction hash, exact held-out row IDs, paired truths and scaler identity are checked. Code hashes accompany each checkpoint and inference receipt. HDF5 header/size/mtime checks are **not a cryptographic full-dataset content check**; no additional 100-GB hashing pass is introduced. Use the frozen dataset with its existing external integrity controls. mtime is a within-run mutation sentinel and is omitted from portable checkpoint identity comparisons after manual transfers. This does not detect malicious changes preserving header, size and timestamps.

Only train/val X and labels are accessed. Compact split/environment/source metadata for all splits is inspected solely for leakage checks. Calibration is reserved and no calibration adjustment is declared. Test inference is intentionally unavailable in this implementation: validation must first establish feasibility and freeze a reviewed confirmatory lock. Do not manually relabel test as val. Any future test gate needs checkpoint/code hashes, chosen draw budget, fixed validation-derived margins, diagnostics/acceptance rules and a precision plan; use test once, without retuning. Real events and M11 conformal quantiles are untouched.

## Posterior sampling and numerical precision

Default 4096 IID draws per observation, generated in chunks of 256 with inference batch 32. Direct flow sampling has no chain autocorrelation; ESS equals the draw count **for the learned density**, not an accuracy claim about the true posterior. Eight independent batches estimate MCSE of physical posterior means and variances. The archive also contains 2048-draw summaries, and 8192-draw summaries for the first 1024 sorted validation IDs, predeclared before outcomes. These are nested budgets, not independent experiments. Full draws are not saved.

The analysis reports raw mean-predictor MSE, an MC-bias-corrected diagnostic `MSE−variance/n`, and a delete-one-batch jackknife MCSE for each paired risk contrast. Raw MSE is the primary endpoint. The MC error criterion is ≤10% of source-bootstrap SE per target; numerical failure means unresolved contrast, never automatic permission to increase a budget after examining test. Require each half/full and full/double paired MSE change to be ≤10% of the source-bootstrap SE on its corresponding full-validation or fixed-panel domain before freezing a validation sampling budget. Variance MCSE does not bound density approximation error. Independent fitted seed disagreement and changes in NLL/risk/variance are separate sensitivity checks. No independent posterior solver, exact recovery case, capacity sensitivity or fresh simulation is supplied: posterior approximation error cannot be certified by this implementation.

## Assessment rules fixed before runs

Global paired differences are M11 squared error minus NPE squared error, so positive favors NPE. Primary uncertainty resamples unique sources **within each observed environment**, preserving pairing and the observed environment counts. It estimates sampling uncertainty conditional on this mixture. A separate environment-then-source bootstrap is a broader-environment sensitivity analysis; few held-out environments make those intervals fragile. Neither bootstrap includes training-seed variability. Validation-selected intervals are exploratory, not confirmatory coverage guarantees.

The three global contrasts use marginal 98.333…% bootstrap intervals (Bonferroni family of three). Region contrasts have a separate Bonferroni family over every predeclared region/target. Default 2000 replicates usually cannot resolve its extreme tails: `bootstrap_tail_resolved=False` prevents significance/equivalence declarations. Region tables remain descriptive unless adequate precision is planned. The practical MSE margin is 1% of M11 validation MSE for each target/domain; this is distinct from the historical 2% RMSE guardrail. A practical improvement needs the entire interval above the margin and resolved MC error; equivalence needs the entire interval within ±margin. Ratios use paired bootstrap numerator and denominator, not a fixed estimated denominator. Sparse groups (n<200) cannot trigger conclusions. The analysis reports the continuous effects even when criteria fail.

Predeclared regions: target SNR [10,15), [15,20), [20,25]; each historical environment; train-label Q10/Q90 tails; each mass tail crossed with each spin tail; global train-Mahalanobis 0–80/80–95/95–99/99–100 percentiles. All quantile/covariance references are fitted only on compact train labels. SNR and truth-selected groups are diagnostics only, never model inputs or training weights.

Marginal ranks and a **joint physical-density level-set rank** use held-out truths and IID posterior draws. Randomized rank histograms and simultaneous DKW envelopes are diagnostics, not a proof of the joint posterior or fresh iid SBC under train-G1. Energy score in fixed historical standardized units compares the joint posterior to a joint empirical train prior, preserving mass/spin dependence. Coverage/width at 50/80/90/95% is accompanied by Bonferroni-adjusted approximate Wilson binomial intervals over the coverage family. Fixed-environment source independence is assumed; mixture heterogeneity, reused bank crops and validation selection can invalidate nominal binomial interpretation. These intervals are not guarantees conditional on X.

**Nominal coverage need not hold after conditioning on true labels or latent SNR/environment, even for an exact X-only posterior.** Those strata expose failure patterns; deviation is not automatically evidence against a correct posterior. Conversely passing their averages does not establish conditional calibration. A prior-only output can pass marginal rank checks, so informativeness and point risk must also be examined. Predictive X diagnostics are explicitly unavailable: p(X|theta) requires the correct conditional nuisance law, which this inverse density model does not supply. Generating waveforms or inventing an X decoder is outside scope.

Average learned posterior variance is compared with empirical posterior-mean MSE, including paired source-bootstrap intervals for their difference, physical and standardized units and variance MCSE. Agreement is a necessary aggregate consistency check, not sufficient conditional validation. Since independent-reference, predictive and conditional accuracy evidence is missing, **B and the irreducible-risk fraction remain UNRESOLVED even if these checks pass**. Report the risk reduction fraction as a difference between estimators, without treating its complement as irreducible. Negative differences are retained, not clipped.

## Required outputs and answers A–E

Each inference archive includes original row/source/environment IDs, targets, SNR, frozen M11 predictions, scaler, train diagnostic references, physical posterior moments/MCSE, quantiles, marginal/joint ranks, NLL, joint energy scores, boundary flags and doubled-budget panel. No embeddings or X arrays are exported.

CPU analysis writes:

| Table | Content |
|---|---|
| assessment.csv | per-target exploratory A decision with MC, budget and second-seed direction checks; posterior accuracy unresolved |
| risk.csv | paired MSE, intervals, margins, guardrail, MC error, risk reduction fraction; irreducible fraction UNRESOLVED |
| metrics.csv | RMSE, MAE, R², OLS slope/intercept, bias, signed train-Q10/Q90 bias by region |
| information.csv | average posterior variance vs empirical MSE, consistency interval, prior variance, MCSE, both units |
| coverage.csv | marginal coverage, width, multiplicity-adjusted approximate binomial intervals, prior coverage |
| ranks.csv | marginal and joint-density rank histograms, DKW diagnostic |
| informativeness.csv | joint posterior/prior energy score, physical NLL, boundary counts |
| sampling_budget.csv | nested half/full/double changes in mean, risk and variance |
| seed_sensitivity.csv | fixed seed 123 vs 124 changes in means, risk, variance and NLL |
| environment_sensitivity.csv | separate environment-cluster uncertainty, never pooled into the primary domain |
| support.csv | sampled support failures and quantized-label adjustments |
| paired_errors.csv | original IDs and per-target paired differences/MC variance |
| analysis.json | receipts, hashes, limitations and explicit A–E interpretation |

The notebook displays these tables and provides one place to record the final A–E assessment. A: demonstrate same-X risk improvement only when its uncertainty, numerical sensitivity and independent seed evidence support it; validation remains exploratory. B: UNRESOLVED under current evidence. C: report demonstrated risk reduction with bounds; irreducible fraction UNRESOLVED per target and region. D: inspect whether slopes/tail biases persist in the NPE mean; shrinkage can be optimal under squared error. E: robust same-X improvement supports a subsequent estimator-training investigation, without identifying architecture as its cause. This experiment cannot recommend information-preserving preprocessing on evidence of information loss (there is no matched D/U control), nor validated uncertainty deployment while B is unresolved.

## Budgets and exact user-operated commands

Existing X occupies 98,304,000,000 bytes (~91.55 GiB); no copy is created. Each full epoch reads ~86.5 GB of train+val X before filesystem/chunk overread: 6250 training and 625 validation steps. The two 100-epoch caps total 1.25 million optimizer steps and ~17.3 TB logical X reads. Plan **days rather than an interactive session**, potentially longer on shared/network storage. No measured P100 throughput or wall-time guarantee exists. Use recorded epoch elapsed time and peak memory to forecast remaining time; do not start a benchmark/pilot or alter the scientific configuration automatically. Fixed max epochs/patience are the compute cap.

Allow the P100's 16 GB for training; batch64 raw input is 12 MiB and the largest early activation ~32 MiB, but autograd/cuDNN/allocator make these only lower bounds. Inference draws are streamed in chunks; each 32×256 conditioner activation is ~4 MiB per 128-wide layer, with no autograd. Host X buffers and bounded overread require hundreds of MiB, not loading 400k X into RAM. Plan 4–8 GiB host RAM including Python/HDF5 caches. The two default validation inferences sample 336,068,608 joint points in total, plus joint-density evaluations; flow arithmetic can dominate their runtime. Budget hours to days, not a promise based on unmeasured throughput.

Reserve 5 GiB new external storage for two runs, improving checkpoints, summaries, histories and analysis; summaries contain about 100 MB of numeric arrays per inference before compression, no raw posterior cloud. Retaining up to 100 improving checkpoints per seed may consume several GiB. Reserve another 5 GiB if manually copying results. No downloads, installation, automatic jobs, synchronization, commits or pushes are performed.

Run from repository root, with `CBC_PE_DATA_ROOT` already configured by the existing project mechanism. New unique tags below must not already exist.

```bash
# Small deterministic CPU checks only (no real data, training or inference).
PYTHONDONTWRITEBYTECODE=1 conda run -n cbc_torch python -m unittest discover -s tests -p 'test_m12_5_2_npe_contract.py' -v

# User launches these sequentially on the GPU VM.
conda run -n cbc_torch python scripts/m12/run_m12_5_2_npe.py train --seed 123 --tag npe_seed123_v1 --device cuda --workers 2
conda run -n cbc_torch python scripts/m12/run_m12_5_2_npe.py train --seed 124 --tag npe_seed124_v1 --device cuda --workers 2

# Resolve the recorded selected checkpoint; do not guess an epoch filename.
NPE_RUN_ROOT="$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_2_NPE"
NPE_CKPT_123=$(python -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); print(p/json.loads((p/"complete.json").read_text())["checkpoint"])' "$NPE_RUN_ROOT/npe_seed123_v1")
NPE_CKPT_124=$(python -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); print(p/json.loads((p/"complete.json").read_text())["checkpoint"])' "$NPE_RUN_ROOT/npe_seed124_v1")
conda run -n cbc_torch python scripts/m12/run_m12_5_2_npe.py infer --checkpoint "$NPE_CKPT_123" --draws 4096 --sample-seed 125221 --tag npe_val123_v1 --device cuda
conda run -n cbc_torch python scripts/m12/run_m12_5_2_npe.py infer --checkpoint "$NPE_CKPT_124" --draws 4096 --sample-seed 125222 --tag npe_val124_v1 --device cuda

# After manual transfer of summary directories/receipts only, local CPU:
python scripts/m12/analyze_m12_5_2_npe.py --primary "$NPE_RUN_ROOT/npe_val123_v1" --replicate "$NPE_RUN_ROOT/npe_val124_v1" --output "$NPE_RUN_ROOT/npe_analysis_v1" --bootstrap 2000
```

The CPU-only notebook `notebooks/m12/12_5_2_NPE_X_only_analysis.ipynb` reads those finished tables and verifies their hashes. It does not train, infer, scan HDF5, or execute automatically. Failed runtime contracts require investigation; changing code after training intentionally invalidates its identity, so preserve the exact experiment source with the run.
