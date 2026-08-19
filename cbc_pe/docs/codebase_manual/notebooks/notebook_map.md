# Notebook map

> **Codebase manual — closed M10 reference**
>
> Reference snapshot:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Purpose

This document classifies notebooks by role.

The repository contains:

```text
final analysis notebooks
demo notebooks
audit notebooks
historical/archive notebooks
```

These categories should not be mixed.

---

# 2. Policy

Preferred division of responsibility:

```text
scripts
=
produce reusable artifacts

notebooks
=
inspect
evaluate
compare
visualize
report
```

A notebook should generally not be the only reproducible path to a core dataset, split, checkpoint, or prediction artifact.

---

# 3. Final M10 notebooks

The final closed-M10 analysis notebooks are:

```text
10_m10_inputzscore_500k_evaluation.ipynb
11_mondrian_m10_inputzscore_500k_final.ipynb
12_real_event_inference_m10_500k_clean.ipynb
13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

Classification:

```text
ACTIVE FINAL ANALYSIS
```

---

# 4. Notebook 10

```text
10_m10_inputzscore_500k_evaluation.ipynb
```

Role:

```text
evaluate closed M10 synthetic cal/test predictions
inspect physical and standardized regression performance
generate diagnostic plots/tables
```

Input artifacts include:

```text
M10 prediction/embedding NPZ
train-only label statistics
dataset metadata as needed
```

It should not retrain the model.

---

# 5. Notebook 11

```text
11_mondrian_m10_inputzscore_500k_final.ipynb
```

Role:

```text
evaluate Mondrian candidate configurations
compare prediction/difficulty taxonomies
compare symmetric/asymmetric intervals
inspect local validity
apply conservative/efficient selection rules
produce selected configuration table
```

This is the final synthetic conformal analysis notebook.

It depends on reusable predictions/embeddings rather than rerunning CNN inference.

---

# 6. Notebook 12

```text
12_real_event_inference_m10_500k_clean.ipynb
```

Role:

```text
single-event real-data sanity/inference workflow
inspect GWOSC processing
inspect PSD selection
inspect X_real_raw vs X_real_z
inspect CNN prediction
inspect embedding
apply selected Mondrian intervals
```

Classification:

```text
ACTIVE FINAL REAL-EVENT SANITY / APPLICATION
```

It is the detailed event-level notebook.

---

# 7. Notebook 13

```text
13_real_events_m10_500k_lvk_comparison_clean.ipynb
```

Role:

```text
multi-event execution
aggregate point/interval predictions
construct LVK detector-frame reference table
compare CNN vs LVK
produce final multi-event diagnostics
```

Classification:

```text
ACTIVE FINAL MULTI-EVENT ANALYSIS
```

This is the broad real-data comparison notebook.

---

# 8. Demo notebooks

Located under:

```text
notebooks/demos/
```

Typical set:

```text
01_signal_generation_demo.ipynb
02_cnn_training_hdf5_demo.ipynb
03_cnn_evaluation_hdf5_demo.ipynb
04_mondrian_hdf5_demo.ipynb
```

Classification:

```text
DEMO / EDUCATIONAL / SANITY
```

Purpose:

```text
show simplified workflow
inspect APIs
demonstrate small-scale behavior
serve as onboarding examples
```

They are not authoritative M10 final analysis.

---

# 9. Audit notebooks

Located under:

```text
notebooks/audits/
```

Examples include:

```text
03_cnn_evaluation_with_500k_draft.ipynb
04_mondrian_hdf5_local_validity_audit.ipynb
05_foundation_dataset_audit.ipynb
06_control_generation_audit.ipynb
07_processing_audit.ipynb
08_cnn_preprocessing_benchmark.ipynb
```

Classification:

```text
AUDIT
```

Purpose:

```text
justify a methodological choice
compare alternatives
investigate a suspected mismatch
benchmark processing
document one-off validation
```

Audit notebooks are valuable evidence but should not be mistaken for active production workflow.

---

# 10. Archive notebooks

Located under:

```text
notebooks/_archive/
```

Subfamilies include:

```text
architecture_search/
m08_baseline/
m10_development/
older historical notebooks
```

Classification:

```text
HISTORICAL
```

They preserve:

```text
architecture search
M00–M09 comparisons
M08 baseline analyses
M10 100k development
early real-event attempts
discarded approaches
```

---

# 11. Architecture-search notebooks

These document experiments such as:

```text
M00 baseline
M01/M02 pooling variants
M04 deeper dense head
M06 wide encoder
M07 multi-head
M08 residual-dilated
M09 multi-attention
```

Their value is historical traceability.

They should not be imported as current pipeline logic.

---

# 12. M10 development notebooks

These are intermediate notebooks used while establishing:

```text
input z-score
real-data scale correction
time-center sensitivity
PSD-window sensitivity
100k vs 500k transition
```

Classification:

```text
HISTORICAL DEVELOPMENT
```

The clean final notebooks supersede them for reporting.

---

# 13. Notebook dependency philosophy

Preferred:

```text
script-produced artifact
    ↓
notebook analysis
```

Examples:

```text
train script
→ checkpoint

predict script
→ cal/test NPZ

Notebook 10
→ synthetic evaluation

Notebook 11
→ conformal model selection
```

This keeps notebook reruns lighter and more deterministic.

---

# 14. Final-vs-audit distinction

## Final notebook

Answers:

```text
"What are the closed baseline results?"
```

## Audit notebook

Answers:

```text
"Why did we choose this method?"
"What did we test?"
"Which alternative failed?"
```

Both are useful, but they serve different purposes.

---

# 15. Final-vs-demo distinction

## Demo

Optimized for:

```text
clarity
small scale
API understanding
```

## Final

Optimized for:

```text
authoritative baseline analysis
artifact traceability
scientific reporting
```

A demo should never be cited as the definitive M10 result source.

---

# 16. Notebook status table

| Notebook/folder | Role | Status |
|---|---|---|
| `10_m10_inputzscore_500k_evaluation.ipynb` | synthetic M10 evaluation | active final |
| `11_mondrian_m10_inputzscore_500k_final.ipynb` | final conformal analysis | active final |
| `12_real_event_inference_m10_500k_clean.ipynb` | single-event real inference | active final |
| `13_real_events_m10_500k_lvk_comparison_clean.ipynb` | multi-event LVK comparison | active final |
| `notebooks/demos/*` | educational/sanity | demo |
| `notebooks/audits/*` | validation/benchmark | audit |
| `notebooks/_archive/architecture_search/*` | historical architecture work | historical |
| `notebooks/_archive/m08_baseline/*` | M08 history | historical |
| `notebooks/_archive/m10_development/*` | M10 development | historical |

---

# 17. Notebook change-impact guide

## Changing CNN checkpoint

Affected:

```text
Notebook 10
Notebook 11
Notebook 12
Notebook 13
```

because predictions/embeddings and real-event inference change.

## Changing prediction artifact

Affected:

```text
Notebook 10
Notebook 11
```

## Changing selected conformal configurations

Affected:

```text
Notebook 12
Notebook 13
```

## Changing LVK conversion/evaluation

Primarily:

```text
Notebook 13
```

## Changing synthetic processing only

Requires:

```text
regenerate dataset
retrain model
regenerate predictions
rerun notebooks 10–13
```

---

# 18. What should remain outside notebooks

Prefer reusable modules/scripts for:

```text
dataset generation
split creation
training
prediction extraction
real-event reusable processing
PSD selection
conformal fit/apply logic
LVK numerical metrics
```

Keep notebooks focused on analysis orchestration and presentation.

---

# 19. Mental model

```text
final/
    = authoritative baseline analysis

demos/
    = how the pipeline works

audits/
    = why methodological choices were made

_archive/
    = how the project evolved
```

The current repository structure is already much clearer because these roles are physically separated.
