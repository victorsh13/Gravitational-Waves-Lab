# CBC PE — Instructions for coding agents

## 1. Scientific contract

- This repository supports gravitational-wave CBC parameter-estimation research.
- M10 and M11 are **frozen scientific baselines**. Do not retrospectively modify their code snapshots, configurations, notebooks, checkpoints, datasets, or recorded results.
- M12 investigates improvements to the **point estimator**, especially regression-to-the-mean. Unless explicitly authorized, freeze final M11 G1 data, source IDs, train/val/cal/test membership, signal processing, input normalization, train-only label scaling, model input contract, and real-event evaluation protocol.
- Change **one experimental factor at a time**. An experiment needs a stated hypothesis, controlled comparison, and predefined assessment criteria. Never interpret a successful code change as evidence of scientific improvement.
- Labels are ordered **[chirp_mass, total_mass, chi_eff]**; preserve this order in every artifact and loss.
- Do not fit training weights, distance/CDF references, scalers, or other learned training statistics on validation, calibration, test, or real-event labels.

## 2. M12 experiment map and evaluation policy

- **M12.0:** frozen M11 baseline, metrics, provenance, and acceptance contract.
- **M12.1:** global label-space Mahalanobis sample-weighted MSE; completed as a validation-stage comparison. Preserve its results and implementation.
- **M12.2:** *marginal per-label weighted MSE*, with separate train-only percentile weights for chirp_mass, total_mass, and chi_eff; evaluate against M11 and M12.1.
- **M12.3 (proposed):** balanced sampling as an independent intervention. Do not mix it with weighted losses in its first comparison.
- Later hypotheses (not approved by default): task-gradient conflicts, separate mass/spin regression heads, other architectural changes, and specialist correction models.
- Use **train** for fitting model parameters and train-derived weighting references; **validation** for checkpoint selection, experimental comparison, and candidate selection; **calibration** later to fit new conformal/Mondrian quantiles for the selected predictor; **test** for final held-out evaluation. Do not select variants on test or real events.
- Keep the historical Mondrian method and real-event evaluation protocol unchanged while developing the M12 estimator. Do not reuse M11 conformal residual quantiles for a changed predictor.
- Primary diagnostics: per-label RMSE/MAE/R², prediction-versus-truth OLS slope and intercept, signed Q10/Q90 bias, global Mahalanobis-conditioned errors, SNR-conditioned errors, and relevant marginal/cross-label diagnostics.
- The agreed **provisional** global RMSE guardrail is at most **2% degradation per label relative to M11 on validation**. Also assess slopes, tail bias, extreme peripherality, low SNR, and chi_eff separately. A threshold is not a statistical confidence statement.
- Use paired comparison on identical row IDs when possible. Paired bootstrap over validation samples does **not** estimate variation between independently trained seeds. Do not claim cross-seed reproducibility without testing it.

## 3. Repository and artifact conventions

- `notebooks/m11/` and historical M11-specific scripts are reference material; do not reorganize them opportunistically.
- `notebooks/m12/` contains experiment-stage notebooks and controlled comparisons. Preserve completed notebooks; create a new analysis notebook rather than rewriting a closed experiment.
- `scripts/` holds shared command-line entry points; `scripts/m12/` is optional for genuinely reusable M12-specific functionality. Avoid copying shared trainers or prediction scripts.
- Prefer `src.paths`, existing HDF5 loaders, configuration conventions, and established prediction/archive schemas. Keep original HDF5 row IDs for sample alignment.
- Keep large datasets, checkpoints, predictions, logs, caches, and generated artifacts outside Git. Never silently overwrite scientific artifacts; use distinct experiment tags and inspect output destinations.
- Avoid introducing machine-specific absolute paths into reusable code. Resolve the data root using the existing project mechanisms; do not invent a parallel convention.

## 4. Codex scope, permissions, and resource discipline

**Read-only inspection is permitted without additional confirmation when necessary for an authorized task.** This includes targeted commands to read `AGENTS.md`, inspect relevant code/notebook cells and JSON files, and view diffs (`cat`, `sed`, `grep`, targeted Python/JSON inspection, `git diff`, `git status`). Do not ask permission again merely to inspect these files.

- Inspect only files directly relevant to the request. Prefer targeted notebook-cell extraction over printing entire `.ipynb` files; avoid repository-wide recursive searches, repeated listings, and redundant probes.
- File creation or edits are permitted **only within the user-approved deliverable scope**. Do not alter unrelated files, historical results, or existing configurations to make checks pass.
- A small, targeted syntax or deterministic CPU unit check may be run **only when explicitly included in the task**, or proposed for the user to run manually. Otherwise provide the minimal command instead of executing it.
- **Never autonomously run** training, inference, dataset generation, GPU jobs, large HDF5 passes, complete notebooks, benchmarks, full test suites, downloads, package installation, or multi-hour processing.
- Never autonomously launch follow-on experiments based on outcomes, expand a one-file request into a refactor, or create unrequested diagnostics, tests, reports, figures, or helper scripts.
- Do not waste tool calls on trivial terminal operations that the user can run directly. Do not repeatedly request authorization for already-permitted read-only inspection.
- Finish after delivering the requested implementation and a concise summary. If blocked by a genuine ambiguity affecting scientific correctness, identify that specific blocker; do not perform speculative work.

## 5. Coding and notebook workflow

1. Read this file and inspect the minimum relevant current implementation.
2. Identify existing contracts, train-only references, and potential historical compatibility risks.
3. Make the smallest change needed for the authorized hypothesis or artifact.
4. Preserve default behavior for old JSON configurations, MSE losses, prediction formats, and train/validation loader contracts.
5. Where requested, prepare only a **small, targeted** deterministic check for altered behavior (e.g. index alignment, loss and gradient equivalence, frozen configuration comparison).
6. Report files changed, important scientific assumptions, compatibility concerns, and minimal manual validation instructions. Do not commit or start another experiment.

For notebooks: keep cell order reproducible, avoid hidden state, reuse existing metric definitions, do not fabricate outputs, do not execute `Run All` autonomously, and document checkpoint/config/dataset/split provenance. Test/calibration/real-event data must not enter variant selection.

## 6. Local and VM execution

- Local workstation: VS Code, Codex, notebooks, light CPU diagnostics, and analysis. Assume limited CPU/RAM.
- Shared GPU VM: Fedora-based environment, `cbc_torch`, historically Python 3.10-compatible code and **PyTorch 1.9.1+cu111 on Tesla P100**. Check API compatibility with this older PyTorch version. Avoid GPU-intensive work unless the user explicitly launches it.
- GPU training/inference use existing CLI scripts and JSON configs; notebooks are **not** the VM execution interface. The user runs GPU commands and inspects logs.
- Synchronization is user-operated. Treat local-to-VM `rsync --delete` as hazardous: never execute destructive synchronization or deletions autonomously. Prefer dry-run review and no-delete transfers for experiment code. Keep large artifacts outside code synchronization.
- Prefer explicit, unique logs and result paths for future runs. Do not add training, `tmux`, logging, or deployment machinery unless requested. An active job must not be interrupted merely to change logging.

## 7. Git and output safety

- M12 development branch convention: `m12-estimator-regression-bias`; verify actual branch when relevant rather than assuming it.
- Do not commit, push, checkout/switch branches, merge, rebase, reset, clean, delete, or force-overwrite unless explicitly instructed.
- Read-only Git checks are allowed when relevant.
- Preserve frozen M10/M11 material and closed M12 experiments. Summarize the diff rather than producing lengthy command transcripts.
- Use distinct config names, checkpoint tags, and output filenames; never reuse a historical output location for a new model.

## 8. Style and final response

- Keep Python 3.10 compatibility and avoid new dependencies without permission.
- Prefer simple, local, reviewable changes over new abstractions. Preserve shared APIs and avoid unrelated formatting.
- Comment non-obvious scientific decisions and data contracts, not self-evident code.
- Final response should state: **files created/modified; what changed; any scientific assumptions or unresolved blockers; and minimal user-run verification commands if necessary**.
- Do not include lengthy transcripts, additional recommendations unrelated to the assigned deliverable, or claims of scientific validation without results.

## 9. Claude Code in this repository
- Las decisiones científicas se toman fuera (chat del Proyecto). Implementa solo lo que pide el prompt; si algo afecta a la corrección científica, para y pregunta.
- Economía de tokens: lee solo los ficheros indicados; nada de búsquedas amplias ni de abrir notebooks completos (extrae celdas concretas).
- No ejecutes tests, entrenamiento, inferencia, notebooks ni comandos sobre datos reales; propón el comando mínimo para que lo ejecute el usuario.
- La VM (Fedora 16, PyTorch 1.9.1, P100) solo ejecuta scripts: mantén compatibilidad y no añadas dependencias.
- Respuesta final: ficheros cambiados, resumen del diff, supuestos y comando de verificación. Sin commits.
