# Week 3 workflow

## Analysis and candidate selection

[Notebook 01](../notebooks/week_3/01_optimum_progress_before_optimizer.ipynb) reconstructs the models trained through week 2 and diagnoses the saved week-3 candidates using their original predictions and acquisition settings. Week-3 outcomes are excluded from these fits. It also includes a separate live EI suggestion example for Function 1; that diagnostic is not exported.

[Notebook 03](../notebooks/week_3/03_optimum_progress.ipynb) checks the saved predictions and measured outcomes for weeks 1-3, then fits Matérn 3/2 Gaussian processes using observations through week 3.

[Notebook 02](../notebooks/week_3/02_xi_sensitivity.ipynb) provides read-only xi sensitivity diagnostics using the same week-3 training cutoff, length caps and Matérn 3/2 kernel. It compares alpha values 0, 0.025, 0.05, 0.1, 0.2 and 0.4 on one seeded configured candidate pool per function. Plots split expected improvement at xi = 0 into positive mean gain and uncertainty premium, and show model-based probabilities of exceeding the incumbent and incumbent plus xi. An adjustable 20% probability cutoff flags selected candidates without replacing them. These are expected, not realized gains; historical calibration residuals are included as a reliability warning. The notebook does not export suggestions or change Notebook 03's alpha policy.

The Week 2 F1-F2 diagnostic points have now been evaluated, so they are not repeated. The original week-4 export used maximum expected improvement for all eight functions over the configured, seeded 1,000,000-candidate pool, carrying forward the Week 2 alpha proportions (0.1 for Functions 1-5; 0.2 for Functions 6-8). The table below records that earlier export, not the revised decisions recorded next. These are finite-pool suggestions, not guaranteed global maxima or measured improvements.

| Function | Alpha | Xi | Week 4 input |
| --- | ---: | ---: | --- |
| 1 | 0.1 | 0.0000960919 | `(0.370663, 0.013678)` |
| 2 | 0.1 | 0.0241363 | `(0.981927, 0.156420)` |
| 3 | 0.1 | 0.00819607 | `(0.261123, 0.854114, 0.486273)` |
| 4 | 0.1 | 0.812393 | `(0.441102, 0.413939, 0.257826, 0.439004)` |
| 5 | 0.1 | 87.0385 | `(0.591628, 0.999967, 0.999971, 0.503519)` |
| 6 | 0.2 | 0.100220 | `(0.404544, 0.503038, 0.326317, 0.999282, 0.014118)` |
| 7 | 0.2 | 0.116837 | `(0.013554, 0.001575, 0.523280, 0.310908, 0.295869, 0.730409)` |
| 8 | 0.2 | 0.206167 | `(0.034951, 0.173596, 0.172811, 0.046456, 0.986580, 0.812290, 0.040115, 0.448451)` |

Candidate means, standard deviations, EI scores, and the resolved incumbent settings for that export are saved in `data/weekly_updates/week_4/predictions.json`. Revised notebook settings do not automatically replace these files.

## Current decisions from Notebook 02

Compare each alpha on the same fitted GP and seeded candidate pool using improvement probability, expected gain at xi = 0, uncertainty share, candidate stability and GP shape. Set `xi = alpha * std(y)` from observations through week 3. The 20% probability flag is advisory, not a selection constraint; differently thresholded EI scores are not a performance ranking.

| Function | Alpha or method | Rationale |
| --- | --- | --- |
| F1 | Heuristic `(0.550000, 0.650000)` | Keep incumbent x1 = 0.55 and move x2 from 0.681526 to 0.65. Test whether the newly found small positive response persists nearby along x2; last week varied x1. |
| F2 | Heuristic `(0.500000, 0.926564)` | Hold the incumbent x2 to six decimals and vary x1 from about 0.702637 to 0.5; last week varied x2. Probe a different x1 region at the promising x2. Predicted mean/std about 0.119/0.131, versus incumbent 0.611: information-seeking, not predicted exploitation. |
| F3 | 0.1, unchanged | Keep xi about 0.008196. Reducing alpha to 0.05 barely changed the joint candidate's x3 (0.486273 to 0.485803) and did not remove the boundary maximum in the incumbent-centred slice. Trial not adopted. |
| F4 | 0.025 | Refine the apparent GP local peak rather than explore further. Improvement probability rises from 11.2% to 21.6%; expected gain is about 0.062, small relative to historical output spread. This is a modest, not high, absolute success probability. |
| F5 | 0.1, unchanged | Same point for alpha 0.05-0.2; 80.5% improvement probability and about 89% mean contribution to EI0. Promising next evaluation, subject to calibration. |
| F6 | 0.05 | Keep low alpha: 36.5% improvement probability, versus 31.4% at 0.2. All EI0 is uncertainty premium, so the outcome remains uncertain. |
| F7 | 0.1 | Keep low alpha: 36.7% improvement probability, versus 14.4% at 0.2. All EI0 is uncertainty premium; alpha 0-0.1 selects the same point. |
| F8 | 0.1 | Same point for alpha 0-0.1; 93.1% improvement probability and about 98% mean contribution. Favour local refinement; expected gain about 0.135 is small relative to the incumbent 9.881. |

These are model-based expectations, not observed gains or guaranteed optima. Uncertainty premium measures reliance on the predictive distribution, not information gain; calibration failures, especially F1/F5, limit probability reliability. Notebook 03 applies the approved F1/F2 overrides and recomputes predictions at their exact coordinates. Alpha 0.1 controls their reported EI scores, not their chosen locations; F3-F8 remain optimizer-pool EI maxima.

The latest selections have now been regenerated using the full configured candidate pools and saved with explicitly approved `overwrite=True`. The week-4 inputs and predictions contain F1 `(0.550000, 0.650000)`, F2 `(0.500000, 0.926564)` and the revised F3-F8 selections. Saved coordinates, acquisition settings, EI scores and expected-best values were read back and verified; candidate/export notebook outputs were refreshed. The earlier export table above is superseded. Earlier weeks and measured observations were not changed.

## Shared notebook helpers

All three notebooks use `load_weekly_history` from [history.py](../src/bayesian_bbo/history.py) and `make_matern_32_kernel` from [kernels.py](../src/bayesian_bbo/kernels.py). The factory reproduces the previous notebook kernels exactly; it does not change the default surrogate kernel.

```python
from bayesian_bbo.history import load_weekly_history
from bayesian_bbo.kernels import make_matern_32_kernel

history = load_weekly_history(data_root, current_week=3)
historical_results = history.records.loc[history.records.week.isin(WEEKS)].copy()
stored_rows = history.records.loc[history.records.week == 3]
```

The collector reads each function's observations once, collects evaluated weeks 1 through `current_week`, and validates saved inputs, means and standard deviations against the observation records. Week 0 supplies the previous incumbent. History rows include candidate coordinates, previous/expected/observed best, predicted mean/std, observed output, standardized residual, and optional acquisition score/settings. Standard deviations must be positive for standardized residuals. The shared prediction reader accepts both UCB and EI metadata; notebooks requiring EI check the selected rows separately.

`history.observations` is capped at `current_week`. Notebook 01 explicitly applies `.through_week(WEEK)` with `WEEK = 2` before fitting; notebooks 02 and 03 apply `.through_week(WEEKS[-1])` with a cutoff of 3. Filtering a history table does not set the model's training cutoff. No helper discovers future week folders or writes observations or predictions.

Refactor verification compared all 24 history rows against the previous calculations, ran the full configured million-candidate sweep in Notebook 02, and reproduced Notebook 03's saved week-4 coordinates and predictions using temporary exports. Repository data hashes were unchanged. Notebook 02's executed outputs were refreshed; Notebook 01 and Notebook 03 retain their existing saved outputs.

## Progress and exports

The notebook compares previous, expected, and observed best-so-far for weeks 1-3 and plots standardized residuals for the 24 evaluated suggestions. The GP surfaces use observations through week 3; their candidate markers show the saved week-3 predictions made before evaluation, so marker values need not match the refitted curves. GP convergence warnings at kernel parameter bounds are diagnostics; notebook execution and export completed successfully.

The notebook writes the eight six-decimal inputs and their predictions to:

- [week 4 inputs](../data/weekly_updates/week_4/inputs.json)
- [week 4 predictions](../data/weekly_updates/week_4/predictions.json)

No objective was evaluated. Expected best-so-far uses the incumbent through week 3 plus EI with `xi = 0`; the acquisition score in the prediction file uses the selected function-specific `xi`.
