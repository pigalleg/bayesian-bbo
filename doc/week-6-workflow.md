# Week 6 workflow: measured outcomes, strategy audit, and Week 7 proposal

## Scope and status

This workflow reviews measured outcomes through Week 6 (`weeks <= 6`), summarizes which weekly parameter choices coincided with incumbent improvements across Weeks 1-6, and records the verified Week 7 proposal exported to [`data/weekly_updates/week_7/`](../data/weekly_updates/week_7/). Week 7 currently contains pre-evaluation inputs and predictions, not measured outputs. Earlier weekly inputs, outcomes, and predictions remain unchanged.

1. Review the imported Week 6 outcomes, progress curves, calibration, and Weeks 1-6 per-function strategy summary in [Notebook 00](../notebooks/week_6/00_optimum_progress.ipynb).
2. Run alpha/xi sensitivity diagnostics in [Notebook 01](../notebooks/week_6/01_xi_sensitivity.ipynb) and inspect the single-function space-partitioning walkthrough in [Notebook 05](../notebooks/week_6/05_space_partitioning_single_function.ipynb).
3. Generate, inspect, and export the Week 7 candidates and pre-evaluation predictions in [Notebook 06](../notebooks/week_6/06_gaussian_estimations.ipynb).

## Evaluation and data import

The Week 6 proposal was generated from observations through Week 5 and recorded in [`data/weekly_updates/week_6/`](../data/weekly_updates/week_6/) (see [Week 5 workflow](week-5-workflow.md)). The cumulative evaluator exports (`inputs.txt` and `outputs.txt`) were converted to `outputs.json` and imported into each function's `all_observations.npz` while leaving earlier weekly batches unchanged.

## Measured Week 6 outcomes

The table compares each Week 6 measured output against its saved pre-evaluation mean and standard deviation. Standardized residuals are $z = (y - \mu)/\sigma$. Expected best-so-far is the stored model expectation at $\xi = 0$, not an observed result.

| Function | Previous best | Predicted mean | Predicted std | Actual output | Residual z | Improved incumbent? | Observed best |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| F1 | 0.000006 | -0.000039 | 0.000806 | 0.000000 | 0.048 | No | 0.000006 |
| F2 | 0.611205 | 0.572913 | 0.026643 | 0.638942 | 2.478 | Yes | 0.638942 |
| F3 | -0.005895 | 0.012634 | 0.047544 | -0.011536 | -0.508 | No | -0.005895 |
| F4 | 0.687027 | 0.434003 | 0.470675 | 0.269984 | -0.348 | No | 0.687027 |
| F5 | 7156.856542 | 7249.598409 | 101.275657 | 7525.788598 | 2.727 | Yes | 7525.788598 |
| F6 | -0.466997 | -0.389958 | 0.106735 | -0.332793 | 0.536 | Yes | -0.332793 |
| F7 | 2.367723 | 2.009686 | 0.242663 | 2.515628 | 2.085 | Yes | 2.515628 |
| F8 | 9.892961 | 10.094976 | 0.081316 | 9.976219 | -1.460 | Yes | 9.976219 |

Five of eight evaluations improved their incumbent in Week 6: F2, F5, F6, F7, and F8 (tying Week 2 as the strongest week so far). Week 6 had the smallest maximum absolute residual across Weeks 1-6 (`max |z| = 2.73`, `4/8` within $\pm 1\sigma$, `5/8` within $\pm 1.96\sigma$), and all three residuals outside $\pm 1.96\sigma$ (F2, F5, F7) were upside surprises that produced new incumbents. Across all 48 evaluations (Weeks 1-6), empirical coverage remains **37.5%** (`18/48`) within $\pm 1\sigma$ and **64.6%** (`31/48`) within $\pm 1.96\sigma$, confirming that nominal Gaussian bands remain diagnostics rather than calibrated guarantees.

## Weeks 1-6 per-function strategy summary

Here is the consolidated per-function summary across **Weeks 1–6** (with **Week 1 UCB** kept distinct from **Weeks 2–6 EI**):

| Fn (Dim) | Progress (W0 → W6 Best) | Winning Weeks | What Produced Improvements | What Failed / Stalled | Length-Scale & Rolling Backtest Note |
| :-: | --- | :---: | --- | --- | --- |
| **F1** (2D) | `7.71e-16` → `5.85e-06` *(W5)* | **W3, W5** (`2/6`) | **Local heuristics** near `[0.55–0.56, 0.6815]` | W1 UCB; EI pool in W2 (Matérn 3/2, $\alpha=0.1$) & W6 (RQ, $\alpha=0.1$); W4 heuristic (`x1=0.65`) | ARD Matérn 3/2 flipped its `31.6` cap between `x0` and `x1`; RQ (`l=0.038`) has `6/6` coverage, but `std(y)`-scaled $\xi$ is dominated by the `-3.6e-3` outlier. |
| **F2** (2D) | `0.6112` → `0.6389` *(W6)* | **W6** (`1/6`) | **Local heuristic** (`x0 + 0.01` at `x1=0.9266`) | W1 UCB; W2 EI (Matérn 3/2, $\alpha=0.1$); W3–W5 heuristics (`x1=0.5`, `x0=0.5`, `x0-0.01`) | ARD Matérn 3/2 & RBF both pinned `x1=31.6` every week and underpredicted W6 (`~0.573` vs `0.639`); Matérn 3/2 (`6/6`) now edges RBF (`5/6`) on rolling NLL. |
| **F3** (3D) | `-0.0348` → `-0.0059` *(W1)* | **W1** (`1/6`) *(W6 rank 2)* | **W1 UCB** ($\kappa=1.96$, fixed RBF `l=0.1`); in EI era, **W6 RQ ($\alpha=0.2$)** came closest (`-0.0115`) | W2–W5 ARD Matérn 3/2 EI ($\alpha=0.1$ in W2–W4; $\alpha=0.3$ in W5) | ARD Matérn 3/2 pinned **both `x0, x1 = 31.6`** in W2, W3, W5; isotropic RQ (`l=0.22`, no caps) leads rolling RMSE, NLL, and `6/6` coverage. |
| **F4** (4D) | `-4.0255` → `+0.6870` *(W3)* | **W1, W2, W3** (`3/6`) | **W1 UCB**; **ARD Matérn 3/2 + EI $\alpha=0.1$** (W2, W3) | Lowering EI $\alpha$ under Matérn 3/2 to `0.025` (W4, W5) and `0.0` (W6) | Never hit cap `10` (`l ≈ 3.0–4.2` across all 4 axes); Matérn 3/2 still leads rolling NLL (`2.32`) and predicted W6 cleanly (`z=-0.35`). |
| **F5** (4D) | `1088.86` → `7525.79` *(W6)* | **W1, W2, W3, W5, W6** (`5/6`) | **W1 UCB**; **EI $\alpha=0.1$** every winning EI week — Matérn 3/2 (W2, W3) then **isotropic RQ** (W5, W6) | W4 only (ARD Matérn 3/2, $\alpha=0.1$): crashed `3949 → 2060` (`z=-12.1`) | ARD Matérn 3/2 pinned `x0=100` (W2–W4) and blew `x3` to `32` in W4; isotropic RQ (`l=1.1–1.5`, no caps) leads rolling RMSE/NLL but still underpredicts the corner slope (`z≈+2.7`). |
| **F6** (5D) | `-0.7143` → `-0.3328` *(W6)* | **W2, W6** (`2/6`) | **EI $\alpha=0.1$** in both wins — W2 Matérn 3/2 (`-0.467`), W6 **isotropic RQ** (`-0.333`) | W1 UCB; W3 Matérn 3/2 ($\alpha=0.2$); low/zero $\alpha$ under Matérn 3/2 (W4 `0.05`, W5 `0.0`, `z=-2.75`) | ARD Matérn 3/2 abruptly pinned `x0=31.6` in W5; isotropic RQ (`l=1.1`, no caps) wins rolling RMSE, NLL, and `6/6` coverage. |
| **F7** (6D) | `1.3650` → `2.5156` *(W6)* | **W2, W6** (`2/6`) | **ARD Matérn 3/2 + EI $\alpha=0.1$** produced both wins | W1 UCB; W3 ($\alpha=0.2$); W4 ($\alpha=0.1$); W5 ($\alpha=0.0$, `z=-8.37`) | Both wins had `x2=31.6` capped; W5 dropped `x2` to `2.9` and crashed. Rolling backtest is split: Matérn 3/2–5/2 lead RMSE, while **RQ** cuts NLL in half (`4.87` vs `10.20`). |
| **F8** (8D) | `9.5985` → `9.9762` *(W6)* | **W2, W4, W6** (`3/6`) | **ARD Matérn 3/2 + EI $\alpha=0.1$** (W2, W4); **isotropic RQ + EI $\alpha=0.0$** (W6) | W1 UCB; W3 Matérn 3/2 ($\alpha=0.2$); W5 Matérn 3/2 ($\alpha=0.0$) | ARD Matérn 3/2 pinned `x7=100` in W2–W5; isotropic RQ (`l=7.7`, no caps) leads rolling RMSE, NLL, and `6/6` coverage (mild overprediction near `10.0`). |

### Key patterns and caveats (Weeks 2-6 EI era)

- **Acquisition settings (Weeks 2-6 EI pool, `n = 33`):** $\alpha = 0.1$ coincided with `12/20` (60.0%) incumbent improvements (`7/12` in Weeks 3-6), compared with `1/5` at $\alpha = 0.0$ (only F8 Week 6 under RQ), `0/3` at $\alpha \in \{0.025, 0.05\}$, and `0/5` at $\alpha \in \{0.2, 0.3\}$. These pooled counts are descriptive, not an apples-to-apples alpha comparison: each week's $\xi=\alpha\,\mathrm{std}(y)$ had a different output scale, and kernels, observations, and heuristics also changed. Week 1 was the only round using UCB ($\kappa = 1.96$, fixed isotropic RBF `l = 0.1`, `3/8` improved).
- **Kernel and length-scale behavior:** Isotropic Rational Quadratic achieved `4/6` (66.7%) EI-pool improvements with `0/6` upper-cap hits (plus rank 2/21 on F3), whereas ARD Matérn 3/2 hit at least one upper length cap in `17/27` EI-pool fits (`9/27` improved overall; `4/19` in Weeks 3-6). Sudden ARD axis pinning coincided with major misses on F5 (W4) and F6 (W5) and repeated axis blindness on F1-F3.
- **Local heuristics on F1 and F2:** Controlled coordinate-wise overrides produced `3/7` improvements (100% of all gains on F1 and F2), compared with `0/3` for EI pool and `0/2` for Week 1 UCB on those two functions.
- **Methodological caveat:** With one new observation per function per week and multiple settings changing across rounds (new data, kernel, $\alpha$, and heuristic overrides), these associations are descriptive diagnostics to inform Week 7 modeling — not controlled causal proofs or universal optimization rules.

## Random versus Sobol/L-BFGS acquisition search

To assess the opt-in search strategy for Week 7, both methods were compared on the same observations through Week 6, fitted GP, per-function kernel and $\xi$, and seed (`42`). Each used the configured 1,000,000 screening candidates. The random method selected the maximum EI from its uniform integer-grid pool. The alternative selected from a scrambled Sobol pool plus 1,024 local perturbations around the incumbent, then polished up to 20 top starts with L-BFGS-B. All reported EI values are recomputed at the returned six-decimal candidate using the same fitted GP; the comparison did not evaluate the real objective or write predictions. The alpha values in this comparison were the notebook settings at the time and are not the Week 7 selections recorded below.

| Function | Kernel | $\alpha$ | Random EI | Sobol/local screen EI | Sobol/local + L-BFGS EI | Change vs. random |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F1 | Rational Quadratic | 0.1 | 0.000248428 | 0.000248416 | 0.000248432 | +0.000000004 |
| F2 | RBF | 0.0 | 0.009197200 | 0.009197199 | 0.009197201 | +0.000000001 |
| F3 | Rational Quadratic | 0.2 | 0.009076372 | 0.009082769 | 0.009083328 | +0.000006956 |
| F4 | Matérn 3/2 | 0.0 | 0.064946199 | 0.076405451 | 0.086231468 | +0.021285269 |
| F5 | Rational Quadratic | 0.1 | 0.001972290 | 105.258259348 | 105.258259348 | +105.256287058 |
| F6 | Rational Quadratic | 0.1 | 0.004088280 | 0.004126470 | 0.004516741 | +0.000428461 |
| F7 | Matérn 3/2 | 0.1 | 0.013077343 | 0.013940749 | 0.015008236 | +0.001930893 |
| F8 | Rational Quadratic | 0.0 | 0.024212300 | 0.024224482 | 0.030263497 | +0.006051197 |

For F8, screening alone was nearly tied with random (+0.000012182 EI); polishing raised EI by another 0.006039015, yielding about a **25.0%** increase over random. Its selected point moved from `[0.118088, 0.413937, 0.039105, 0.341230, 0.759918, 0.682721, 0.043843, 0.008288]` to `[0.078306, 0.248985, 0.058345, 0.354498, 0.725652, 0.655225, 0.033302, 0.000000]`.

The alternative also raised the reported EI for the other seven fitted models, but the F5 result deserves particular caution: its optimizer selected the upper cube boundary `[0.999999, 0.999999, 0.999999, 0.999999]`. Its very large predicted EI is a model-based boundary result, not evidence of a similarly large real-world gain. Across the comparison, runtimes were about 47 seconds total for all eight functions, including fitting each GP once. The result supports trying the strategy, not treating EI improvement as a measured outcome or a guarantee.

## Proposed Week 7 alpha settings

The Week 7 settings below are recorded in [Notebook 06](../notebooks/week_6/06_gaussian_estimations.ipynb). The requested policy is to favor exploration for Functions 1-5 and use $\alpha=0$ for Functions 6-8. These choices do not follow the pooled historical win counts alone; those counts combine different $\xi$ values and modeling setups.

| Function | Alpha | Week 6 training `std(y)` | Week 7 `xi` | Selection intent |
| --- | ---: | ---: | ---: | --- |
| F1 | 0.1 | 0.000872508 | 0.000087251 | Moderate exploration |
| F2 | 0.1 | 0.246568546 | 0.024656855 | Moderate exploration |
| F3 | 0.1 | 0.078873119 | 0.007887312 | Moderate exploration; below the prior 0.2 after no Week 6 gain |
| F4 | 0.1 | 8.894177978 | 0.889417798 | Exploration; accepts lower model-predicted improvement probability |
| F5 | 0.1 | 2023.081227 | 202.308123 | Exploration priority |
| F6 | 0.0 | 0.540984820 | 0.000000000 | Exploitation |
| F7 | 0.0 | 0.735104506 | 0.000000000 | Exploitation |
| F8 | 0.0 | 1.108264450 | 0.000000000 | Exploitation |

Here $\xi=\alpha\,\mathrm{std}(y)$ uses the observations through Week 6, so the same alpha does not imply the same threshold across functions or weeks. In the Week 6 sensitivity sweep, F5 selected the same upper-corner candidate across the tested alpha grid; its $\alpha=0.1$ setting records the exploration policy but did not change the selected point in that fitted model. The current model's EI and probability diagnostics remain model-based and should not be read as outcome guarantees.

## Week 7 proposal and verified exports

[Notebook 06](../notebooks/week_6/06_gaussian_estimations.ipynb) was executed on observations through Week 6 (`weeks <= 6`) with shared length caps, seed `42`, and `search_strategy='sobol_lbfgs'` over the full configured `1,000,000`-candidate screening pool for F4-F8.

### Space-partitioning heuristic overrides (F1-F3)

Functions 1, 2, and 3 use explicit coordinates in `HEURISTIC_CANDIDATES` generated via [`SpacePartitioningOptimizer`](../src/bayesian_bbo/space_partitioning.py) using each function's NB06 kernel (`Rational Quadratic`, `RBF`, `Rational Quadratic`) with `min_region_points=4`, `max_depth=3`, `trust_length=0.4`, `candidates=2048`, `acquisition='ei'`, `xi=0.1`, `use_rank_transform=True`, and `seed=42`:

- **F1 (`Rational Quadratic`):** 2 splits (`9/7 -> 5/4`, incumbent retained in active leaf) → `(0.564405, 0.685288)`.
- **F2 (`RBF`):** 1 split (`12/4`, incumbent retained in active leaf) → `(0.729659, 0.995987)`.
- **F3 (`Rational Quadratic`):** 1 split (`10/11`, incumbent retained in active leaf) → `(0.245071, 0.447022, 0.484866)`.

In NB06, these three points are scored via `optimizer.predict()` under the global GP so their saved `expected_outputs`, `stds`, `expected_best_outputs`, and `acquisition_scores` remain on the same global reference scale as F4-F8.

### Exported Week 7 submissions and predictions

| Function | Kernel | Selection method | $\alpha$ | Six-decimal submission | Predicted mean | Predicted std | Expected best ($\xi=0$) | Selection EI ($\xi$) | Previous best |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| F1 | Rational Quadratic | Local heuristic (SP) | 0.1 | `0.564405-0.685288` | -0.000027 | 0.000060 | 0.000017 | 5.10e-7 | 0.000006 |
| F2 | RBF | Local heuristic (SP) | 0.1 | `0.729659-0.995987` | 0.579979 | 0.047652 | 0.641417 | 0.000762 | 0.638942 |
| F3 | Rational Quadratic | Local heuristic (SP) | 0.1 | `0.245071-0.447022-0.484866` | -0.004825 | 0.013883 | 0.000195 | 0.002784 | -0.005895 |
| F4 | Matérn 3/2 | EI Sobol/L-BFGS | 0.1 | `0.388077-0.398319-0.448079-0.476222` | 0.037588 | 0.632607 | 0.737116 | 0.001559 | 0.687027 |
| F5 | Rational Quadratic | EI Sobol/L-BFGS | 0.1 | `0.999999-0.999999-0.999999-0.999999` | 7817.210410 | 120.668359 | 7817.524084 | 105.258259 | 7525.788598 |
| F6 | Rational Quadratic | EI Sobol/L-BFGS | 0.0 | `0.462421-0.326523-0.541069-0.826821-0.242243` | -0.348088 | 0.058050 | -0.316483 | 0.016310 | -0.332793 |
| F7 | Matérn 3/2 | EI Sobol/L-BFGS | 0.0 | `0.000000-0.000000-0.266239-0.307549-0.309511-0.794550` | 2.368466 | 0.205407 | 2.544168 | 0.028540 | 2.515628 |
| F8 | Rational Quadratic | EI Sobol/L-BFGS | 0.0 | `0.078306-0.248985-0.058345-0.354498-0.725652-0.655225-0.033302-0.000000` | 9.900741 | 0.152105 | 10.006483 | 0.030263 | 9.976219 |

### Export verification and safeguards

- Exported to [`data/weekly_updates/week_7/inputs.json`](../data/weekly_updates/week_7/inputs.json) and [`data/weekly_updates/week_7/predictions.json`](../data/weekly_updates/week_7/predictions.json) via `save_predictions()` **without** `overwrite=True`, restoring default protection against accidental replacement.
- Both JSON files were read back and independently verified: eight rows in function order, exact six-decimal coordinates matching the prediction points, `expected_best_outputs == best + EI_0` (`rtol=1e-10`), and earlier weekly records (`week_1` through `week_6`) untouched.
- Refreshed candidate tables and all eight GP diagnostic figures are saved in [Notebook 06](../notebooks/week_6/06_gaussian_estimations.ipynb).

## Notebooks

- [`00_optimum_progress.ipynb`](../notebooks/week_6/00_optimum_progress.ipynb) compares actual outputs, expected and observed best-so-far, saved residual calibration across Weeks 1-6 (48 evaluations), and records the Weeks 1-6 per-function strategy summary table.
- [`01_xi_sensitivity.ipynb`](../notebooks/week_6/01_xi_sensitivity.ipynb) uses a shared Sobol/local screening pool and fits each xi-specific EI objective for alpha/xi sensitivity diagnostics.
- [`05_space_partitioning_single_function.ipynb`](../notebooks/week_6/05_space_partitioning_single_function.ipynb) is a read-only pedagogical single-function walkthrough of [`space_partitioning.py`](../src/bayesian_bbo/space_partitioning.py) (K-Means + SVM space partitioning, rank-Gaussian target transform, local GP, ARD-shaped trust region, and Thompson sampling / local EI) compared against the global `sobol_lbfgs` GP. It does not export weekly candidates.
- [`06_gaussian_estimations.ipynb`](../notebooks/week_6/06_gaussian_estimations.ipynb) fits on observations through Week 6, scores the F1-F3 space-partitioning heuristic overrides via `predict()`, selects F4-F8 via `sobol_lbfgs` EI over the full 1,000,000-candidate screening pool, renders GP diagnostic plots, and exports the verified Week 7 proposal with default overwrite protection.
