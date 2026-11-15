# Week 4 analysis and Week 5 proposal

## Training data and notebook roles

The week-4 observations are the training cutoff for the next proposal. [Notebook 06](../notebooks/week_4/06_gaussian_estimations.ipynb) fits on observations through week 4, predicts the F1/F2 local heuristic candidates and selects F3-F8 by expected improvement over the configured one-million-point pool. [Notebook 07](../notebooks/week_4/07_xi_sensitivity.ipynb) is a read-only alpha comparison using one seeded pool per function and the same kernel map: Matérn 3/2 for F1-F4 and F6-F8, Rational Quadratic for F5.

Notebook 07 compares alpha values 0, 0.025, 0.05, 0.1, 0.2, 0.3 and 0.4. The F3 value 0.3 was added to the grid to directly examine the requested more exploratory setting. For both notebooks, `xi = alpha * std(training outputs)`. The selected alpha is a policy choice; candidates are finite-pool model suggestions, not guaranteed global optima or observed gains.

## Week 5 alpha decisions

| Function | Alpha | Decision |
| --- | ---: | --- |
| F1 | 0.1 | Keep the local probe near the observed maximum; this alpha controls its EI diagnostic, not its location. |
| F2 | 0.1 | Probe locally by adjusting x1 near the incumbent; this alpha controls its EI diagnostic, not its location. |
| F3 | 0.3 | Increase exploration to investigate possible local minima. |
| F4 | 0.025 | Use a conservative improvement threshold for refinement. |
| F5 | 0.1 | Retain the selected setting and use the Rational Quadratic kernel. |
| F6 | 0.0 | Choose the low-alpha candidate with the stronger model-based probability of improvement. |
| F7 | 0.0 | Choose the stable low-alpha candidate with the stronger model-based improvement diagnostics. |
| F8 | 0.0 | Favor the alpha-0 candidate's higher predicted mean and improvement probability. |

These are model-based decisions. In particular, past week-4 residuals caution against treating predicted probabilities as calibrated outcome guarantees.

At F3 alpha 0.3, the NB07 million-point sweep selected a candidate with predicted mean -0.060350 and standard deviation 0.080565, compared with incumbent -0.005895. Its modeled probability of exceeding the incumbent is 24.95%, and probability of exceeding incumbent plus xi is 16.44%; EI at xi = 0 is 0.011988 and is entirely uncertainty premium. This supports describing the query as exploratory rather than as a predicted mean improvement.

## Exported Week 5 proposal

The six-decimal inputs, predictions, and acquisition settings were generated from the week-4 cutoff and saved in `data/weekly_updates/week_5/`. `EI` below is the configured score at the selected xi; `expected_best_outputs` in `predictions.json` instead use xi = 0.

| Function | Alpha | Xi | Selection method | Week 5 input | Predicted mean | Predicted std | EI at xi |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| F1 | 0.1 | 0.00009281113 | Local heuristic, vary x1 | `(0.560000, 0.681526)` | 0.0000186741 | 0.000416746 | 0.000131554 |
| F2 | 0.1 | 0.023288059 | Local heuristic, vary x1 | `(0.692637, 0.926564)` | 0.60537824 | 0.0269002 | 0.00191127 |
| F3 | 0.3 | 0.024217148 | Optimizer pool EI max | `(0.230343, 0.002456, 0.525039)` | -0.060349712 | 0.0805653 | 0.00701812 |
| F4 | 0.025 | 0.20901014 | Optimizer pool EI max | `(0.377087, 0.375898, 0.419683, 0.416322)` | -0.1941617 | 0.773523 | 0.0277819 |
| F5 | 0.1 | 91.200017 | Optimizer pool EI max | `(0.971643, 0.960706, 0.984862, 0.997630)` | 5312.7779 | 654.554 | 1278.673735 |
| F6 | 0.0 | 0 | Optimizer pool EI max | `(0.156038, 0.305387, 0.577780, 0.998556, 0.132981)` | -0.43396856 | 0.105457 | 0.0606324 |
| F7 | 0.0 | 0 | Optimizer pool EI max | `(0.010124, 0.000341, 0.171968, 0.116063, 0.340963, 0.665352)` | 2.4833006 | 0.0951275 | 0.120752 |
| F8 | 0.0 | 0 | Optimizer pool EI max | `(0.055054, 0.282234, 0.245516, 0.052966, 0.993069, 0.708465, 0.202215, 0.245714)` | 9.9605519 | 0.0838408 | 0.0775598 |

The export's incumbent, xi, predictions, and acquisition scores were checked against the week-4 observations and the alpha table. Week 5 has no `outputs.json` yet because these candidates have not been evaluated.

## Historical Week 4 files

`data/weekly_updates/week_4/inputs.json` remains the evaluated Week 4 submission, not the new proposal. Its eight rows match the week-4 observations and its saved predictions; the corresponding `outputs.json` and `predictions.json` were not changed. The new alpha table and week-5 proposal must not be written back into this historical folder.

## Validation notes

Both notebooks were executed with the full configured one-million-candidate pool per function. Execution completed without cell errors. Scikit-learn emitted convergence warnings for some fitted kernel parameters at their configured bounds; these warnings are retained in the notebook outputs and limit how strongly to interpret the fitted model diagnostics.
