# Week 5 workflow: results, diagnostics, and Week 6 proposal

## Scope and execution order

This workflow reviews measured outcomes through Week 5 and prepares the Week 6 proposal. Week 6 currently contains inputs and pre-evaluation predictions, not measured outputs. All current fits use observations with `weeks <= 5`; historical backtests use only observations available before the week being predicted.

1. Review the imported Week 5 outcomes in [Notebook 00](../notebooks/week_5/00_optimum_progress.ipynb), comparing saved pre-evaluation predictions with measured outputs and incumbent progress.
2. Investigate Function 5 in [Notebook 01](../notebooks/week_5/01_function_5_kernel_diagnostics.ipynb), then compare all functions and kernels in [Notebook 02](../notebooks/week_5/02_all_functions_surrogate_diagnostics.ipynb).
3. Use [Notebook 03](../notebooks/week_5/03_length_scale_sensitivity.ipynb) and [Notebook 04](../notebooks/week_5/04_length_scale_lower_bound_sensitivity.ipynb) to assess length-scale and noise-bound sensitivity. These experiments do not update shared defaults.
4. Inspect the separate Function 2 local-fit experiment in [Notebook 05](../notebooks/week_5/05_function_2_local_gp.ipynb). It does not replace the global proposal model.
5. Generate candidates and inspect their GP diagnostics in [Notebook 06](../notebooks/week_5/06_gaussian_estimations_bckp.ipynb). Review coordinates and predictions before running its export cell.

Notebooks 00-05 are diagnostic consumers; Notebook 06 writes the Week 6 proposal. These are analysis steps, not automatic kernel or policy selection.

## Evaluation and data import

The Week 5 proposal was generated from observations through Week 4, with candidates, predictions, and acquisition settings recorded in [`data/weekly_updates/week_5/`](../data/weekly_updates/week_5/). The cumulative evaluator text files contain five weekly batches. The final batch of `outputs.txt` matches the Week 5 inputs and was recorded as `outputs.json`; `append_weekly_batch` then imported the eight results into the observation histories. Earlier weekly files and predictions were left unchanged.

## Measured outcomes

The table compares each Week 5 output with its pre-evaluation mean and standard deviation. The standardized residual is $z=(y-\mu)/\sigma$. Expected best-so-far is the stored model expectation; observed best-so-far uses the measured outcome.

| Function | Previous best | Predicted mean | Predicted std | Actual output | Residual z | Improved incumbent? | Observed best |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| F1 | 0.000001 | 0.000019 | 0.000417 | 0.000006 | -0.031 | Yes | 0.000006 |
| F2 | 0.611205 | 0.605378 | 0.026900 | 0.610531 | 0.192 | No | 0.611205 |
| F3 | -0.005895 | -0.060350 | 0.080565 | -0.096282 | -0.446 | No | -0.005895 |
| F4 | 0.687027 | -0.194162 | 0.773523 | 0.582951 | 1.005 | No | 0.687027 |
| F5 | 3949.359249 | 5312.777889 | 654.553522 | 7156.856542 | 2.817 | Yes | 7156.856542 |
| F6 | -0.466997 | -0.433969 | 0.105457 | -0.723566 | -2.746 | No | -0.466997 |
| F7 | 2.367723 | 2.483301 | 0.095127 | 1.687241 | -8.369 | No | 2.367723 |
| F8 | 9.892962 | 9.960552 | 0.083841 | 9.854370 | -1.267 | No | 9.892962 |

Two of eight evaluations improved their incumbent: F1 and F5. F5 exceeded its saved mean by about 2.82 predictive standard deviations; F6 and F7 fell below their saved means by about 2.75 and 8.37 standard deviations. These outcomes are strong warnings against reading the nominal Gaussian bands or model-based improvement probabilities as calibrated guarantees. One result per function is not enough to estimate stable coverage.

The selected alpha, xi, kernel, and candidate rationale are recorded in the [Week 4 workflow](week-4-workflow.md). Expected-best values and acquisition scores remain model-based quantities and should not be substituted for measured outcomes.

## Week 6 proposal

The [suggestion notebook](../notebooks/week_5/06_gaussian_estimations_bckp.ipynb) fits on all observations through Week 5, using shared length caps, seed 42, and a one-million-candidate pool per non-heuristic function. Each exploration threshold is `xi = alpha * np.std(training outputs)` with NumPy's default population standard deviation.

| Function | Kernel | Alpha | xi | Six-decimal submission |
| --- | --- | ---: | ---: | --- |
| F1 | Rational Quadratic | 0.1 | 0.000089909 | `0.773730-0.763077` |
| F2 | RBF | 0.1 | 0.023997750 | `0.712637-0.926564` |
| F3 | Rational Quadratic | 0.2 | 0.015735971 | `0.392553-0.473772-0.496042` |
| F4 | Matérn 3/2 | 0.0 | 0 | `0.372020-0.381747-0.406424-0.396711` |
| F5 | Rational Quadratic | 0.1 | 158.103381884 | `0.999317-0.988415-0.993298-0.955076` |
| F6 | Rational Quadratic | 0.1 | 0.051536940 | `0.483903-0.337689-0.522437-0.776108-0.145322` |
| F7 | Matérn 3/2 | 0.1 | 0.066165757 | `0.013554-0.001575-0.523280-0.310908-0.295869-0.730409` |
| F8 | Rational Quadratic | 0.0 | 0 | `0.060407-0.148811-0.140889-0.174148-0.664420-0.450375-0.212793-0.322075` |

The kernel plan remains provisional: F6 is a prospective Rational Quadratic test, and F7 remains unresolved rather than a validated Matérn choice. The historical recommendation table in Notebook 02 belongs to its earlier `1e-8` noise-floor analysis, not the current `1e-10` rerun.

F2 uses the explicit override:

```python
HEURISTIC_CANDIDATES = {
    2: (0.712637, 0.926564),
}
```

This increases the incumbent's first coordinate by approximately 0.01 while holding the second fixed to six decimals. Its global RBF prediction is approximately 0.572913 with standard deviation 0.026643; EI at xi approximately 0.023998 is 0.000087, a diagnostic at the heuristic point rather than a pool maximum. The other functions use `suggest_with_prediction()` to select finite-pool EI maxima. The override path uses `predict()` at the specified coordinates and rejects unknown function IDs, invalid coordinates, and previously evaluated points.

### Candidate markers versus plotting maxima

Notebook 06 passes each selected candidate and its mean, standard deviation, and EI to `plot_gp_2d` or `plot_gp_slices`, so the marked candidates correspond to `next_candidates`. The 2D heatmaps evaluate a separate 100-by-100 regular grid; slice curves hold other coordinates at the selected candidate. Neither grid nor slices reproduce the full random selection pool.

Without a supplied candidate, `plot_gp_2d` marks the plotting-grid EI maximum. With a candidate, it marks that candidate instead. Notebook 06's conditional subtitle annotation only adds grid-max coordinates if a `Grid EI max` artist exists; its current candidate-marking calls do not produce that artist.

### Function 2 local-fit diagnostic

Notebook 05 selects the nine nearest observations out of 15 by Euclidean distance to the incumbent `(0.70263656, 0.92656420)` in normalized input space. The incumbent itself is included, and tied distances retain their original observation order. The ninth distance is approximately 0.426564 and the tenth 0.682740: this noticeable gap motivates the exploratory count of nine, but does not prove a shared basin or an optimal neighborhood size.

The local GP uses the same RBF construction and configured length cap as the global model. Its xi remains `0.1 * std(all Week 5 outputs)` to preserve the proposal's improvement threshold; target normalization and kernel fitting use only the selected observations. The saved `plot_gp_2d(..., n=100)` heatmaps cover the full domain and mark the plotting-grid EI maximum. Predictions far from the local observations are extrapolations, not evidence of a better global optimum.

The notebook was executed with the existing repository environment and saved with rendered outputs. A noise-variance lower-bound convergence warning was emitted. No observations or proposal exports were changed; this fit has not been validated as a replacement for the global model. A time-ordered comparison of neighborhood sizes would need to recompute each historical incumbent and neighborhood using only past data.

### Export and verification

The export cell sorts `next_candidates` by function ID, requires exactly one row per function, uses `format_submission()` for six-decimal strings, and checks that the parsed export coordinates equal the points used for predictions. It calls `save_predictions()` for:

- [`week_6/inputs.json`](../data/weekly_updates/week_6/inputs.json): the eight candidate vectors in function order.
- [`week_6/predictions.json`](../data/weekly_updates/week_6/predictions.json): chosen-point means, standard deviations, expected-best values, EI scores, and settings containing `name`, `xi`, and incumbent `best`.

Expected best-so-far is the incumbent plus EI at xi = 0, not the selected acquisition score when xi is positive. These values remain pre-evaluation expectations.

The saved Week 6 proposal was checked against a full one-million-candidate recomputation: all eight coordinates and prediction/acquisition fields matched, and expected-best values were checked independently. This source-based verification did not overwrite exports or refresh notebook outputs. It is distinct from the executed local diagnostic and is not a reduced-pool check.

The Week 6 inputs and predictions were intentionally replaced after approval earlier in this analysis. **The current Notebook 06 export cell contains `overwrite=True`, so rerunning it can replace differing files.** The package default refuses differing replacements, but that protection is bypassed by this explicit argument. Obtain approval before any further intentional replacement; leave historical inputs, outcomes, and predictions unchanged. See [data flow](data-flow.md) for persistence semantics.

## Notebooks

- [`00_optimum_progress.ipynb`](../notebooks/week_5/00_optimum_progress.ipynb) compares actual outputs, expected and observed best-so-far, and pre-evaluation residual calibration across Weeks 1-5 and all eight functions.
- [`01_function_5_kernel_diagnostics.ipynb`](../notebooks/week_5/01_function_5_kernel_diagnostics.ipynb) assesses the Week 5 Function 5 result, shared-fold kernel accuracy, and sequential historical predictions through Week 5. All backtests train strictly before the evaluated week.
- [`02_all_functions_surrogate_diagnostics.ipynb`](../notebooks/week_5/02_all_functions_surrogate_diagnostics.ipynb) compares five kernels and a training-mean baseline across all functions with shared shuffled folds and a leakage-free rolling-origin backtest through Week 5. The current rerun uses a notebook-local WhiteKernel noise-variance lower bound of `1e-10` for all GP fits, without changing shared defaults. Its final static recommendation table is labelled as the historical `1e-8` analysis; freshly executed tables above it contain the current scores.
- [`03_length_scale_sensitivity.ipynb`](../notebooks/week_5/03_length_scale_sensitivity.ipynb) reproduces the Week 1 likelihood-versus-length-cap plot for all eight functions and five shared package kernels, using observations through Week 5. It sweeps caps from `0.01` to `1,000` at a notebook-local noise-variance floor of `1e-12`, marks the configured caps, and compares noise floors `1e-8`, `1e-10`, and `1e-12` at those configured caps. Tables record fitted noise, likelihood changes, boundary hits, and warning messages by function, kernel, cap, and noise floor. It does not change shared kernel defaults, configuration, or candidate exports; likelihood is an in-sample diagnostic, not predictive validation.

- [`04_length_scale_lower_bound_sensitivity.ipynb`](../notebooks/week_5/04_length_scale_lower_bound_sensitivity.ipynb) compares length-scale lower bounds `0.001`, `0.0001`, and `0.00001` across all eight functions and five kernels using observations through Week 5. Maximum length caps stay at their configured values and noise bounds stay at `(1e-8, 1e1)`. It records fitted scales, lower-bound hits, likelihood gains relative to `0.001`, and warnings, without changing defaults or exports.
- [`05_function_2_local_gp.ipynb`](../notebooks/week_5/05_function_2_local_gp.ipynb) fits the Function 2 RBF GP to the nine observations nearest the incumbent. This exploratory count follows the distance gap after the ninth point; the full-domain heatmaps can extrapolate far beyond the local data.

These diagnostic notebooks are read-only. Shuffled cross-validation, historical outcomes, and in-sample likelihood sweeps are descriptive diagnostics; none selects a kernel or changes the optimization policy automatically.
