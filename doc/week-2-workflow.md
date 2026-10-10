# Week 2 workflow

Week 2 compares Gaussian-process (GP) kernels on the eight supplied functions, selects one kernel for the Expected Improvement (EI) suggestions, and produces one unevaluated candidate per function for the submission table.

## Kernel policy

- Benchmark kernels: RBF, Matérn 3/2 and Matérn 5/2. These are the kernels the notebooks use to make decisions.
- Comparison-only kernels: Rational Quadratic (RQ) and ExpSineSquared (ESS). They are reported for context and do not drive decisions.
- A kernel choice is made only when the difference is substantial. Small differences are reported but not acted on.
- Kernel rankings describe predictive performance on the supplied observations, not optimisation performance. No objective is evaluated in Week 2.

## Notebook order

| Notebook | Purpose |
| --- | --- |
| [01_output_comparison.ipynb](../notebooks/week_2/01_output_comparison.ipynb) | Compares expected and observed outputs for each function whose `all_observations.npz` exists. The RBF reference model is a fixed comparison. |
| [02_all_functions_length_scale_sensitivity.ipynb](../notebooks/week_2/02_all_functions_length_scale_sensitivity.ipynb) | Measures how the maximised GP log marginal likelihood (LML) changes with the maximum length-scale cap. |
| [03_function_3_kernel_comparison.ipynb](../notebooks/week_2/03_function_3_kernel_comparison.ipynb) | Compares the four candidate kernels on Function 3 with fitted parameters, kernel similarity and cross-validation. |
| [04_all_functions_cross_validation.ipynb](../notebooks/week_2/04_all_functions_cross_validation.ipynb) | Compares the same kernels on Functions 1 to 8 with shared, shuffled 5-fold cross-validation (NRMSE and 90% coverage). |
| [05_all_functions_ei_alpha_sweep.ipynb](../notebooks/week_2/05_all_functions_ei_alpha_sweep.ipynb) | Shows how the maximum EI changes with `alpha`, where `xi = alpha * std(y)`. The selected `alpha` is 0.1. |
| [06_all_functions_ei_matern32_suggestions.ipynb](../notebooks/week_2/06_all_functions_ei_matern32_suggestions.ipynb) | Produces one Matérn 3/2 EI candidate per function and formats the submission table. |

## Kernel decision for Notebook 06

Matérn 3/2 is the only kernel used in Notebook 06. The choice rests on held-out cross-validation in Notebook 04, judged within each function:

- Among the benchmark kernels, Matérn 3/2 has the lowest mean RMSE on Functions 2, 3, 5, 6, 7 and 8.
- Matérn 5/2 has the lowest mean RMSE on Function 4.
- Function 1 is a tie. Its outputs have a standard deviation of about 0.001, so the differences are negligible.
- Notebook 02 in-sample LML favours RBF on most functions. This conflicts with the held-out result. The held-out result is weighted more heavily because it measures predictive performance on unseen points. The conflict is recorded rather than resolved.
- Averages of NRMSE across functions are not used for the decision, as Notebook 04 cautions.

## Notebook 06 outputs

- One EI candidate per function (Functions 1 to 8), with Matérn 3/2 as the only kernel.
- Random state 42 (`CONFIG["random_state"]`), candidate pool of 2048 (`candidates` in `src/bayesian_bbo/config.yaml`), and a one-suggestion budget.
- Exploration parameter `xi` = 0.1 times the standard deviation of each function's observed outputs (`XI_VALUES` in section 1).
- Length-scale upper caps (`LENGTH_CAPS` in section 2): 31.62 for Functions 1, 2, 3, 6 and 7; 10 for Function 4; 100 for Functions 5 and 8.
- Section 9 gives the submission table in six-decimal format, with coordinates joined by "-".

## Caveats for next week

1. **Rational Quadratic has lower per-function RMSE.** In Notebook 04, RQ has a lower mean RMSE than Matérn 3/2 on Functions 3, 4, 6 and 7 (for example, Function 4: about 1.27 against 2.29). RQ is comparison-only, so it does not drive the choice. A rule is needed before deciding whether RQ should become a benchmark kernel.
2. **Per-function kernel selection.** Selecting a different kernel for each function is possible, but the within-function evidence is mixed: Matérn 5/2 leads on Function 4, Function 1 is a tie, and RQ leads on four functions. Adopt per-function selection only with an explicit rule, a stated margin and more data.
3. **Candidate pool may be too small for higher dimensions.** The pool holds 2048 candidates. Functions 6 (5-D), 7 (6-D) and 8 (8-D) may not be covered densely enough. Test the sensitivity of the EI maximum to the pool size before relying on it.
4. **Extrapolation on Function 5.** The Matérn 3/2 candidate for Function 5 has a predicted mean of about 1720, while the observed maximum is about 1398 (standard deviation about 358). The candidate is an extrapolation and should be checked before submission.
5. **Likelihood and cross-validation disagree.** Notebook 02 in-sample LML favours RBF, while Notebook 04 held-out CV favours Matérn 3/2. Neither result has been reconciled.
6. **Small samples.** Observation counts are 11 (Functions 1 and 2), 16 (Function 3), 31 (Function 4), 21 (Function 5), 21 (Function 6), 31 (Function 7) and 41 (Function 8). Five-fold CV on 11 to 41 points gives noisy estimates. Function 1 differences are negligible.
7. **Bounds are being hit.** Several fits reach the length-scale cap or noise bounds, and Notebook 06 reports ConvergenceWarnings. The Function 8 Matérn 5/2 advantage in Notebook 02 sits at the cap of 100 and has not converged, so it is discounted.
8. **Undocumented XI rule.** `xi` is set to 0.1 times the observed output standard deviation. The rationale for 0.1 is not recorded in the notebooks.
9. **Single seed.** Candidate pools and GP fits use seed 42 only. Sensitivity to the seed is untested.
10. **One point per function.** Notebook 06 proposes one EI point per function with no optimisation loop, so no regret or convergence is measured.
11. **Data hygiene.** The data folders contain `:Zone.Identifier` files, which are download metadata. They are not data and should be removed or ignored.
