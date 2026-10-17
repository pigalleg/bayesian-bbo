# Week 3 workflow

## Main conclusions and candidate choices

Week 3 reviews observations through week 3 and selects the next unevaluated candidates for week 4:

- **Function 1: `(0.550000, 0.681526)`**, a diagnostic heuristic. Almost all outputs are effectively zero except a negative response near `(0.650114, 0.681526)`. Keep x2 fixed to six decimals and change x1 to test whether the fitted broad negative band persists. The GP's short length-scale axis switched after week 3, so its apparent axis relevance is unreliable. This query seeks information, not exploitation of a negative value in a maximization problem.
- **Function 2: `(0.702637, 0.500000)`**, a diagnostic heuristic with a promising x1. Outputs near 0.61 and 0.60 occur around x1 = 0.70 at widely separated x2 values. Hold the best observed x1 to six decimals and test an intermediate x2 to check whether the promising region persists.
- **Functions 3-8:** use the Bayesian optimizer's maximum EI over a seeded 1,000,000-candidate pool, with Matérn 3/2, configured length caps and the selected xi values below. These are pool maxima, not guaranteed global maxima or measured improvements.

Flat GP predictions are not proof that an input is irrelevant. Controlled sampling motivates the F1-F2 overrides; F3 also has capped axes, but no heuristic override was selected.

## Notebook order

| Notebook | Purpose |
| --- | --- |
| [01_optimum_progress_before_optimizer.ipynb](../notebooks/week_3/01_optimum_progress_before_optimizer.ipynb) | Reconstructs the pre-result model using observations through week 2 and diagnoses the stored week-3 suggestion. |
| [02_all_functions_ei_alpha_sweep.ipynb](../notebooks/week_3/02_all_functions_ei_alpha_sweep.ipynb) | Uses observations through week 3 to compare alpha values, then concludes with recommended xi values. |
| [03_optimum_progress.ipynb](../notebooks/week_3/03_optimum_progress.ipynb) | Checks week-2/3 progress and calibration, inspects current GPs, applies the F1-F2 overrides, selects F3-F8 by EI and exports week-4 candidates. |

Notebook 02 also provides Matérn-only dashboards of candidate coordinates, predicted mean, standard deviation, best observed, nearest-observation distance, xi and maximum EI versus alpha.

## Selected exploration settings

Selected by visual inspection: alpha = 0.1 for Functions 1-5 and 0.2 for Functions 6-8, with `xi = alpha * std(y)` using observations through week 3. Notebook 02 marks these choices and calculates the values transferred to Notebook 03:

| Function | Alpha | xi (approximately) |
| --- | --- | --- |
| 1 | 0.1 | 0.0000996663 |
| 2 | 0.1 | 0.0232669 |
| 3 | 0.1 | 0.00839155 |
| 4 | 0.1 | 0.770138 |
| 5 | 0.1 | 45.2162 |
| 6 | 0.2 | 0.100055 |
| 7 | 0.2 | 0.0988600 |
| 8 | 0.2 | 0.200978 |

These are heuristics, not validated optima. Lower maximum EI at higher alpha is expected from the higher improvement threshold, not evidence of worse optimization. F1-F2 coordinates are fixed heuristics; their xi values determine the reported EI scores, not the chosen locations.

## Candidate pool

The week-3 notebooks use 1,000,000 candidates, increased from 2,048 via 100,000: the estimated EI maximum and selected point were observed to change substantially with the candidate pool. The larger pool improves search coverage but does not guarantee the global EI maximum.

Notebook 02 uses `BayesianOptimizer.suggest_ei_sweep()` to generate and select candidates on one seeded pool per kernel, reusing batched GP predictions across all alpha values.

## Length-scale observations

Week-3 notebooks read per-function upper caps from `length_caps` in [config.yaml](../src/bayesian_bbo/config.yaml). Week-2 notebooks retain their hardcoded caps for historical comparison.

With the Matérn 3/2 model fitted through week 3, length scales are approximately:

| Function | Length scales | At the upper bound (31.6) |
| --- | --- | --- |
| 1 | `[31.6, 0.0377]` | x1 |
| 2 | `[0.0776, 31.6]` | x2 |
| 3 | `[31.6, 31.6, 0.0360]` | x1, x2 |

The cap indicates the fit is nearly flat along that axis over `[0, 1]`; it does not establish that the objective is flat. Function 1's fitted short axis switched between weeks 2 and 3, so its axis dependence is especially uncertain.

The selected F1 and F2 queries test x1 and x2 respectively. Further coordinate-wise sampling, including F1's x2 and F3's x1/x2, remains a possible follow-up rather than part of this submission.

## Final table and export

Notebook 03 ends with one candidate per function, its selection method, six-decimal submission string, predicted mean, standard deviation and EI. Its final cell saves:

- [week_4/inputs.json](../data/weekly_updates/week_4/inputs.json): the eight selected input vectors in function order.
- [week_4/predictions.json](../data/weekly_updates/week_4/predictions.json): predicted means, standard deviations, expected best-so-far, acquisition scores and settings (`name`, `xi`, incumbent `best`).

Expected best-so-far is the incumbent through week 3 plus EI with xi = 0. The acquisition score instead uses each function's selected xi; for F1-F2 it is EI at the heuristic point, not maximum EI. No objective is evaluated, historical week-2/3 files are unchanged, and different existing exports are refused rather than overwritten.
