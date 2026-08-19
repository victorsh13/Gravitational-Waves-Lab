# Test protection map

> **Codebase manual — closed M10 reference**
>
> Reference snapshot:
>
> ```text
> tag:    m10-closed-baseline
> commit: dadf32f77f5c344c3519843e6bd9f0ee0c5baed0
> ```

## 1. Scope

This document maps automated tests to the scientific and infrastructure contracts they protect.

Inventory summary:

```text
test files: 13
AST-discovered tests: 102
```

The suite is strongest in recently refactored conformal, real-data, path, LVK, and normalization code.

Direct unit-test coverage of the older synthetic scientific core is substantially weaker.

---

# 2. Test files

```text
tests/
├── test_conformal_pipeline.py
├── test_conformal_selection.py
├── test_input_normalization_equivalence.py
├── test_lvk_evaluation.py
├── test_lvk_reference.py
├── test_paths.py
├── test_real_data_catalog.py
├── test_real_data_event_runner.py
├── test_real_data_gwosc.py
├── test_real_data_inference.py
├── test_real_data_psd.py
├── test_real_data_signal_processing.py
└── test_selected_calibrators.py
```

---

# 3. Protection by subsystem

## Conformal

Protected by:

```text
test_conformal_pipeline.py
test_conformal_selection.py
test_selected_calibrators.py
```

Contracts include:

```text
fit/apply/evaluate separation
truth-free application
taxonomy handling
selected configuration application
selection-policy behavior
```

---

# 4. M10 normalization

Protected by:

```text
test_input_normalization_equivalence.py
```

This is especially important because it checks equivalence across:

```text
sample-wise normalization
batch-wise normalization
closed M10 notebook formulation
```

### Classification

```text
HIGH-VALUE REGRESSION TEST
```

Input normalization is the defining methodological change from M08 to M10.

---

# 5. Path resolution

Protected by:

```text
test_paths.py
```

Contracts include:

```text
CLI precedence
environment-variable precedence
config fallback
project-root fallback
dataset-specific processed layout
legacy flat-layout fallback
```

A path regression can masquerade as a missing dataset/checkpoint, so these tests protect a critical infrastructure boundary.

---

# 6. Real-data acquisition

Protected by:

```text
test_real_data_catalog.py
test_real_data_gwosc.py
```

Likely contract areas include:

```text
catalog flattening
URL selection
HDF5 validation
strain reading
time-window validity
```

---

# 7. Real-data PSD

Protected by:

```text
test_real_data_psd.py
```

Contract areas include:

```text
PSD window availability
preferred/fallback selection
finite-data validation
PSD construction behavior
```

This is particularly important because fallback-window correctness was historically a sensitive area.

---

# 8. Real signal processing

Protected by:

```text
test_real_data_signal_processing.py
```

Contract areas include:

```text
event processing window
processing context
detector ordering
final X shape
training-compatible processing
```

---

# 9. Real inference and runner

Protected by:

```text
test_real_data_inference.py
test_real_data_event_runner.py
```

Contract areas include:

```text
real CNN inference shapes
dummy-target structural behavior
standardized/physical prediction conversion
single-event orchestration
normalization application
failure classification
```

---

# 10. LVK reference/evaluation

Protected by:

```text
test_lvk_reference.py
test_lvk_evaluation.py
```

Contract areas include:

```text
source→detector mass conversion
first-order uncertainty propagation
chi_eff pass-through
wide-table construction
delta definitions
interval-overlap definitions
visualization-only clipping
```

---

# 11. Major direct-test gap: synthetic core

No dedicated test files are present for:

```text
parameters.py
sampling.py
waveform.py
detectors.py
windowing.py
injection.py
noise.py
processing.py
snr.py
dataset.py
```

There are also no obvious dedicated files:

```text
test_parameters.py
test_sampling.py
test_waveform.py
test_detectors.py
test_windowing.py
test_injection.py
test_noise.py
test_processing.py
test_snr.py
test_dataset.py
```

### Interpretation

```text
[TEST COVERAGE GAP]

The original synthetic scientific core is less directly protected
against regression than newer conformal/real-data infrastructure.
```

This does **not** imply the code is incorrect.

It implies that future refactors could alter behavior without being caught automatically.

---

# 12. Why this matters before a future pipeline revision

A new pipeline may intentionally change:

```text
signal support in processing context
noise model
PSD behavior
waveform assumptions
placement
SNR targeting
```

Without characterization tests, it is harder to distinguish:

```text
intended M11 change
```

from:

```text
accidental unrelated regression
```

---

# 13. Recommended characterization tests before major synthetic refactor

These can be added without modifying M10 behavior.

## `parameters.py`

Test:

```text
m1 >= m2 invariant
spin swap follows mass swap
RA/polarization wrapping
chi_eff formula
with_distance preserves all non-distance fields
```

## `sampling.py`

Test:

```text
mass ranges
distance ranges
fixed parameter overrides
isotropic inclination construction
isotropic declination construction
spin ranges
```

## `waveform.py`

Test:

```text
h+ / hx same length
delta_t contract
finite output
metadata consistency
```

## `detectors.py`

Test:

```text
detector set
absolute epoch
LAL projection invocation contract
arrival-time/delay metadata consistency
```

## `windowing.py`

Test:

```text
<= duration passes untruncated
long waveform retains late segment
common network end logic
required final duration validation
```

## `injection.py`

Test:

```text
absolute-time index calculation
full overlap
partial clipping
no-overlap error
network placement containment
```

## `noise.py`

Test:

```text
PSD dimensions
delta_f contract
distinct detector seeds
repeatability with fixed seed
```

## `snr.py`

Test:

```text
network norm
distance rescaling formula
target-in-range no-op
post-rescale tolerance
```

## `processing.py`

Test:

```text
input-length validation
processing-context crop
final start_time
final length
finite output
```

## `dataset.py`

Test:

```text
X shape
y order
metadata completeness
detector order
gw_only mode
in_noise mode
distance-rescale rebuild path
```

---

# 14. Test taxonomy

Use three conceptual categories.

## Unit contract tests

Protect one class/function.

Example:

```text
CBCParameters mass ordering
```

## Integration tests

Protect module interaction.

Example:

```text
real-data processing → final X shape
```

## Characterization tests

Freeze current closed behavior before refactoring.

Example:

```text
windowing retains last 4 s under current M10 rules
```

Characterization tests are particularly valuable when code may later be intentionally redesigned.

---

# 15. Tests vs scientific validation

Passing unit tests does not prove:

```text
waveform fidelity
real-data calibration validity
domain transfer
scientific optimality
```

Automated tests protect implementation contracts.

Scientific notebooks/audits validate methodological behavior.

The two should remain separate.

---

# 16. Proposed protection matrix

| Subsystem | Direct protection | Current status |
|---|---|---|
| paths | strong | protected |
| M10 input normalization | strong | protected |
| conformal fit/apply | strong | protected |
| conformal selection | strong | protected |
| selected calibrators | strong | protected |
| real catalog/GWOSC | strong | protected |
| real PSD | strong | protected |
| real processing | strong | protected |
| real inference | strong | protected |
| LVK conversion/evaluation | strong | protected |
| synthetic parameters/sampling | weak | gap |
| waveform/projection | weak | gap |
| windowing/injection | weak | gap |
| synthetic noise/SNR | weak | gap |
| synthetic processor | weak | gap |
| DatasetBuilder | weak | gap |

---

# 17. Test-change impact

## Changing normalization

Must run:

```text
test_input_normalization_equivalence.py
real inference tests
event-runner tests
```

## Changing conformal

Must run:

```text
test_conformal_pipeline.py
test_conformal_selection.py
test_selected_calibrators.py
```

## Changing paths

Must run:

```text
test_paths.py
```

and then at least one script smoke test.

## Changing real PSD

Must run:

```text
test_real_data_psd.py
test_real_data_signal_processing.py
test_real_data_event_runner.py
```

## Changing LVK conversion

Must run:

```text
test_lvk_reference.py
test_lvk_evaluation.py
```

---

# 18. Mental model

```text
tests/
    = implementation contract guards

audits/
    = methodological diagnostics

final notebooks/
    = scientific analysis/reporting
```

The current highest-priority protection gap is:

```text
synthetic scientific core
```

especially before introducing future pipeline changes.
