# M12.5.2 first X-only reference: patch MLP

Implementation specification, not a result. No training, inference or notebook
execution accompanies this change. This authorized first X-only comparison takes
precedence over the raw-likelihood-first staging in the original
`M12_5_2_information_reference_protocol.md`; that historical proposal is unchanged.

## Question and choice

Can an independently trained estimator reduce frozen M11 squared error on exactly
the same normalized X? The primary estimand is per-target risk on the **frozen
validation held-out environment mixture**, after learning on train environments.
These are different environment mixtures, not IID draws from one common mixture.
The input is X alone: float32 `(3,16384)`, H1/L1/V1, already normalized. Labels are
`[chirp_mass,total_mass,chi_eff]`, with detector-frame solar masses and dimensionless
spin. No additional normalization, physical metadata, M11 embeddings/predictions,
pretrained weights, augmentation, weighting or sampling intervention is used.

| Candidate | Independence and cost | Point precision / validation | Decision |
|---|---|---|---|
| Patch MLP, unweighted standardized MSE | New nonconvolutional weights and global token mixing; one streamed training run | Deterministic point output; directly optimizes quadratic risk; no posterior calibration claim | **Primary** |
| Conditional joint Gaussian-mixture density network | New X encoder plus mixture head, train-only likelihood; similar order of encoder cost, no simulation required | Mixture-weighted component means give an analytic posterior mean, with no sampling error. NLL/coverage need not establish conditional accuracy; Gaussian components can violate mass/spin support, and environment shift remains | Feasible as an approximate density estimator, but adequate posterior validation is not established within this single-run budget; defer |
| Kernel/ridge regression on fixed time-frequency features of X | No M11 dependence; raw 400k kernel matrix is impractical, explicit features reduce cost | Deterministic mean, but chosen feature compression can discard phase/timing information and make a negative result weak | Defer; avoid making the first reference a deliberately restrictive summary of X |

The chosen architecture partitions X into 128 contiguous temporal patches, each
containing 128 samples from all three detectors. Reshaping is lossless. A learned
384→128 linear projection, four token/channel MLP blocks, final LayerNorm, token
mean and linear three-output head form the estimator. Token MLP: 128→128→128;
channel MLP: 128→256→128; GELU and pre-normalized residual paths. The learned
projection is allowed to compress X, just as a CNN does; it does not establish
information sufficiency. All original input values enter the model. Architecture
is the intervention, not proof of the cause of any difference. There is no sweep.

This reference is independent of M11's learned representation, **not** independent
of its training observations. It is an alternative point estimator, not an exact
posterior method. Accurate conditional inference is not ruled out computationally;
we decline to equate an affordable density fit with a validated Bayes reference.

## Frozen artifacts and execution controls

The runner resolves the original M11 configuration through `src.paths` (CLI data
root, `CBC_PE_DATA_ROOT`, historical config fallback). It reuses
`HDF5BatchIterableDataset`, `train_model` and `predict_on_loader`. Shared files and
all frozen artifacts are unchanged. It checks the exact split/scaler NPZ hashes
and the frozen validation prediction NPZ hash inspected during implementation.
The expected scaler is approximately mean `[37.46355,94.98598,-0.00021922]`, std
`[16.50806,34.70329,0.44033849]`; full float32 values come from the frozen file.
Training uses the loader's historical `(y-mean)/(std+1e-8)` convention; physical
metrics use the existing `inverse_standardize`. At these float32 std values,
adding 1e-8 does not change the stored scale. No new label scaler is fitted.

Split counts are 400000/40000/30000/30000. Membership and source-ID metadata may be
read for leakage checks; calibration/test signals and labels are never opened.
Train labels are read once during the user-run inference stage solely to derive
diagnostic Q10/Q90 cutoffs and Mahalanobis references, never fitted on validation.
No generation or physics re-audit is needed for reading this frozen X archive.

Checks include all split IDs unique/disjoint/complete and consistent with HDF5,
globally unique source IDs, train membership of scaler, header normalization and
label/detector ordering, finite consumed inputs/labels/outputs/gradients, strict
checkpoint state loading and scaler/code compatibility. Inference uses zero loader
workers to make sorted row order explicit, then verifies targets and IDs against
M11. SNR/environment are never passed to the model. A completed, hash-verified
receipt is required by subsequent stages. New output directories are exclusive;
failed runs remain incomplete and are never automatically resumed or overwritten.

Large X content is not rehashed: identity relies on the frozen archive contract,
its configuration/manifest headers and size, and the user's existing immutable
artifact management. The header is not a newly computed whole-file integrity
proof. Small files and model/output artifacts are hashed. Code hashes enforce the
same training implementation at inference, including locally modified shared
utilities; transfer those exact files to the VM. No repository changes are made
to accommodate a mismatched artifact.

## Fixed budget and operations

One seed (123), AdamW lr 3e-4, weight decay 3e-4, batch 256, unweighted MSE,
shuffled train batches, no within-batch shuffle, drop-last train, complete val.
These match the M11 optimization settings. Maximum **100 epochs**, patience **25**,
minimum validation mean standardized MSE selects the checkpoint (first minimum
wins ties). The 100-epoch cap is a budget constraint versus M11's 250; reaching it
while still improving is a material limitation, not evidence of an information
limit. There is no scheduler, second seed, extension, retry with another model or
automatic follow-on experiment. Any subsequent budget change requires a new
recorded development decision.

- 447,747 trainable parameters, 56,623,488 linear-layer MACs per forward example
  (excluding normalization, activations, reductions and backward operations).
- At most 156,200 optimizer steps / 39,987,200 consumed training examples and
  4,000,000 validation forwards for selection; one final 40,000-row val pass.
- One Tesla P100, float32, PyTorch 1.9.1 APIs; no new dependencies or AMP. Budget a
  16-GB GPU and 8-GB host allocation with two loader workers. Input batch is 48 MiB;
  each channel-hidden tensor is 32 MiB. These are tensor sizes, not peak-memory
  measurements; autograd/workspaces/prefetch add memory. Stop and report OOM rather
  than silently changing batch size. GPU compatibility is reviewed, not runtime-tested.
- Existing trainio X is 98.304 GB logical. Train+val reads are at least about
  86.5 GB/epoch, up to 8.65 TB logical across 100 epochs, plus final inference.
  Actual I/O depends on caching/chunks. Keep the existing dataset on VM storage;
  do not copy X into RAM or create a feature dataset.
- New checkpoint is roughly 6 MB including Adam states; scalar prediction and
  diagnostic exports are tens of MB or less. Reserve 100 MB per result tag.
  Wall time is **unmeasured**: plan a sustained GPU job, not an interactive CPU
  run. Existing trainer history records epoch timings; after the first full epoch
  the user can estimate the remaining maximum as 99 times that epoch duration.
  No pilot or fabricated hours estimate is supplied.

From the repository root, with `CBC_PE_DATA_ROOT` set to the existing data root:

```bash
# VM, cbc_torch; user runs these commands, in sequence.
conda run -n cbc_torch python scripts/m12/run_m12_5_2_reference.py train \
  --data-root "$CBC_PE_DATA_ROOT" --tag patch_mlp_seed123_v1 --device cuda --workers 2

conda run -n cbc_torch python scripts/m12/run_m12_5_2_reference.py infer \
  --data-root "$CBC_PE_DATA_ROOT" --tag patch_mlp_seed123_val_v1 --device cuda \
  --checkpoint "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_2/patch_mlp_seed123_v1/checkpoint.pt"
```

For local analysis, transfer only the inference directory (including both JSON
receipts) and optionally training `history.json`/receipts using the user's existing
transfer process. Open `notebooks/m12/M12_5_2_X_only_reference.ipynb` in the CPU
environment. Set its tag if a different unique tag was used. It opens only scalar
NPZ/JSON artifacts and computes tables; it never trains or infers. Optional batch
equivalent, not needed if using the notebook:

```bash
python scripts/m12/analyze_m12_5_2_reference.py \
  --inference-dir "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_2/patch_mlp_seed123_val_v1" \
  --output-dir "$CBC_PE_DATA_ROOT/results/M11_final_G1_500k/M12_5_2/patch_mlp_seed123_analysis_v1" \
  --bootstrap 2000
```

## Endpoints, diagnostics and decisions fixed before execution

For identical validation row i and target j, favorable delta is
`(M11_ij-y_ij)^2 - (reference_ij-y_ij)^2`. Report both physical and train-standardized
MSE and delta. No comparison of MSE against likelihood or weighted training loss.
Archive schema retains `idx_val`, `pred_val`, `y_val`, `y_mean`, `y_std`, label order,
and adds aligned M11 predictions, source/environment/SNR diagnostics and train-only
region references. Predictions/targets are standardized, as in historical exports.

The analysis returns four tables: endpoint, physical metrics, stratified errors,
and paired observation-level errors. Metrics reuse the existing regression
definitions, adding prediction-on-truth OLS slope/intercept and signed bias below
train Q10 / above train Q90. Strata: train-Mahalanobis percentiles 0–80, 80–95,
95–99, 99–100; target SNR [10,15), [15,20), [20,25]; each held-out environment;
each target's lower/upper tail and the mass×spin tail combinations. Squared
Mahalanobis distance uses train-centered standardized labels and empirical
ddof=1 covariance pseudoinverse. Its percentile order equals distance order.
These diagnostics do not enter model selection; subgroup differences are descriptive.

Use 2000 paired bootstrap replicates (seed 125202), resampling rows within each
observed environment while fixing its count. The three target intervals use
Bonferroni marginal 98.333...% intervals (one family; unit conversions are the same
contrast). These describe source sampling conditional on the finite environment
mixture, not uncertainty over new environments or independent training seeds.
The 40k sources are unique according to the frozen source-ID contract. Reused
environment dependence beyond that conditioning is not estimated by this bootstrap.
Validation selected the checkpoint: **all intervals and decisions here are
exploratory**, not post-selection-corrected confirmatory evidence.

- Predeclared development margin: tau_j = 1% of M11 validation MSE per target.
  Practical improvement for target j: favorable delta lower interval bound > tau_j.
  Equivalence: the entire interval lies inside [-tau_j,+tau_j]. Neither zero
  crossing nor lack of improvement establishes equivalence or a Bayes limit.
- A candidate for confirmation needs at least one practical target improvement,
  all three empirical RMSE ratios ≤1.02, and explicit review of spin, low SNR,
  extreme peripherality and cross-tail/environment failures. The aggregate loss
  alone cannot pass it. Subgroup review has no invented universal numeric threshold;
  document concerns before freezing any candidate.
- Invalid provenance/alignment/scaling, nonfinite values or an incomplete run
  blocks comparison. Non-improvement, unresolved intervals, worsening guardrails
  or a budget-limited fit yields no supported improvement under this run's budget.
  No automatic extension or alternate architecture follows.

Before any confirmatory test: freeze model/checkpoint, code and preprocessing,
target risk domain, primary contrasts, family/margins, source/environment sampling
and required precision using validation. Test is untouched by these scripts; a
separately authorized confirmatory evaluation must account for its own environment
mixture. Do not tune on calibration/test or real events. If conformal intervals are
later required, refit quantiles on calibration for this selected predictor; never
reuse M11 residual quantiles. More training seeds and independent held-out evidence
are needed before claiming reproducibility.

## What would estimate R*_X?

A confirmed reduction on identical X demonstrates attainable improvement of this
specific M11 predictor. The best validated estimator risk is an **upper bound** on
Bayes risk under its declared evaluation law (empirical risk has sampling error).
Agreement between two estimators, including this MLP, supplies no lower bound.
No posterior variance or Bayes-risk number is produced here.

Estimating `R*_X = E[Var(Y|X)]` additionally requires a sufficiently accurate
conditional posterior or a controlled bound on posterior-mean approximation
error under the same joint law and X-only information. Train-versus-held-out
environment mixtures must be resolved explicitly. For a future density method:
joint support/prior checks; independent held-out simulation or untouched suitable
population draws; SBC/PIT and 50/80/90/95% coverage with uncertainty and declared
multiplicity; stratification by SNR, masses, spin and environment; informativeness
relative to the prior; and posterior predictive checks through a validated
nuisance-aware observation model (a density over three labels alone cannot
generate X) are needed. Marginal calibration alone is insufficient. Means must
agree with independent references in verifiable cases, remain stable with capacity,
budget and training seed, and have numerical/MC error small relative to risk
contrast uncertainty. Approximate posterior variance can be severely biased even
when average coverage is good. The present implementation deliberately makes only
the attainable-point-estimator comparison.
