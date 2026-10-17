---
name: bayesian-notebook-workflow
description: "Use when editing or running this repository's Bayesian optimization notebooks, explaining GP length scales or slices, sweeping alpha/xi, selecting EI candidates, or exporting weekly inputs and predictions."
---

# Bayesian notebook workflow

## Establish current facts

1. Read the relevant notebook's current source cells without dumping image outputs. Preserve user edits, cell IDs and historical results unless intentionally rerunning.
2. Read the relevant weekly workflow under `doc/` and package APIs under `src/bayesian_bbo/`. Documentation is the analysis record, not a substitute for checking current code.
3. Debug from the exact traceback, function signature and training-week cutoff. If disk accepts an argument but the live kernel rejects it, check for stale imports or a different module before changing the API.
4. Use Python fact-grounded tooling when available: verify relevant diagnostics/signatures or runtime values before drawing conclusions. Do not assume tools used in a previous session remain installed.

## Fit, predict and select

1. Use `load_observations(..., up_to_week=...)` explicitly to avoid outcome leakage into historical predictions.
2. Use `CONFIG["length_caps"]` in week-2 notebooks; keep week-1 hardcoded caps unchanged. Candidate count comes from the shared configuration when appropriate; verify its actual value.
3. Construct `BayesianOptimizer(observations, OptimisationConfig(...))`. Use `predict()` for specified inputs and `suggest_with_prediction()` for a normal suggestion.
4. For alpha sensitivity, use `suggest_ei_sweep(alpha_values * std(y), batch_size=...)`: one pool and one fitted GP per kernel, batched predictions, optimizer-owned selection. Repeated normal suggestions advance the RNG and do not compare the same pool.
5. Keep seed, kernel, observations and pool fixed when comparing alpha. EI is maximized over the finite pool, not certified globally. Record candidate, mean, std, xi, score and selection method.
6. Treat per-function alpha choices and heuristic overrides as explicit decisions documented in the weekly workflow, not universal optimization rules.

## Interpret diagnostics

- Length scales much larger than the input range indicate a nearly flat fitted direction, not proof of objective independence. Axis switching after one observation suggests weak identification.
- Xi changes EI, not GP mean/std at fixed inputs under the same fit. Candidate-specific mean/std can change because the selected point changes.
- `plot_gp_slices` fixes other coordinates at the candidate when provided, otherwise at the best observed input. Moving the candidate changes the slice and can change top-row curves without changing the GP.
- Optional `expected_mean`, `expected_std` and `expected_ei` describe candidate markers; they do not define the slice curves. Check current helper validation when supplying them.
- Larger xi raises the improvement threshold. Falling maximum EI is expected and cannot establish that smaller alpha optimizes better. Compare decisions and eventual measured outcomes.
- A coordinate-wise diagnostic keeps other inputs fixed. Label information-seeking queries separately from EI maxima; a negative feature is not the incumbent merely because it is the only substantial signal in a maximization problem.

## Export and execute

1. Use `format_submission()` for six-decimal strings and verify predictions correspond to the exact exported coordinates.
2. Use `save_predictions()` for `weekly_updates/week_N/inputs.json` and `predictions.json`, ordered by function ID. Use the next week after the training cutoff.
3. Save actual chosen-point mean/std, EI with selected xi, and acquisition settings including incumbent `best`. Expected best-so-far uses incumbent plus EI at xi = 0; do not confuse it with the selected acquisition score.
4. Do not overwrite historical inputs, outcomes or predictions. Default export protection refuses different existing files; obtain explicit approval before intentional replacement.
5. Find the existing repository `.venv`; never create one. Use its interpreter/kernel. Do not infer the editor-selected interpreter from terminal Python alone.
6. Validate notebook JSON/schema and code syntax, then run the smallest meaningful execution. Check reduced-pool results against direct calculations or focused tests when changing logic.
7. When asked to run/save a notebook, persist refreshed outputs after successful execution. Temporary validation cells must not remain in the saved notebook. A script-only run does not refresh notebook outputs.
8. Report whether validation used the full configured pool, whether outputs were saved and whether exports were written. Keep convergence warnings distinct from execution failures.
