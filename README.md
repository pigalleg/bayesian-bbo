# Bayesian Black-Box Optimisation (BBO)

## 1. Project overview

This BBO capstone explores how to find high-performing inputs for unknown functions when evaluations are limited and their formulas and gradients are unavailable. The main challenge covers eight scalar-valued functions with two to eight input dimensions.

The high-level loop is **observe, model, propose, evaluate and update**. A Gaussian Process (GP) learns from previous input/output pairs, estimates both performance and uncertainty, and guides the next query. The aim is to spend scarce evaluations more effectively than an exhaustive search.

This resembles real-world ML hyperparameter tuning, where training a model can be expensive and the relationship between settings and validation performance is unknown. Similar problems arise in experimental design, chemical-process tuning and engineering simulation; see the [benchmark descriptions](doc/benchmark-functions.md).

### Career relevance

The project develops practical skills for ML engineering and applied data science: sample-efficient experimentation, uncertainty-aware decisions, model validation, reproducible Python workflows and communicating the limits of an analysis. It provides a concrete portfolio example of making defensible decisions with sparse data rather than relying only on predictive accuracy.

## 2. Inputs and outputs

### Inputs

Each query is a numeric vector with coordinates in **`[0, 1)`**, submitted to exactly **six decimal places**.

| Function | Input dimensions |
| --- | ---: |
| 1 | 2 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 4 |
| 6 | 5 |
| 7 | 6 |
| 8 | 8 |

For example, the two-dimensional candidate `[0.702637, 0.500000]` is submitted as:

```text
0.702637-0.500000
```

Initial observations are supplied as input arrays of shape `(n, d)` and output arrays of shape `(n,)`. Weekly input files contain one vector per function in function-ID order. The package also defines a one-dimensional Function 9 for separate experimentation; it is outside the eight-function capstone workflow.

### Outputs

- **The evaluator (external)** returns one real scalar score for each query. Higher is better; the score's scale differs across functions.
- **The GP surrogate (internal to this repository)** returns a predicted mean and standard deviation at an input.
- **The optimizer (internal to this repository)** returns a suggested candidate, its predicted mean and uncertainty, and an acquisition score such as Expected Improvement (EI). Predictions are not measured results.

The workflow exports [inputs.json](data/weekly_updates/week_3/inputs.json) and [predictions.json](data/weekly_updates/week_3/predictions.json). Predictions include expected outputs, standard deviations, expected best-so-far, acquisition scores and the settings used to calculate them. Once the external evaluator returns the actual scores, these are recorded separately in `outputs.json` for the corresponding week (for example, [week-2 outputs.json](data/weekly_updates/week_2/outputs.json)), in the same function order as the inputs. These are measured outcomes, not surrogate predictions. See [data placement and provenance](data/README.md) and [data flow](doc/data-flow.md) for schemas and validation rules.

## 3. Challenge objectives

The objective is to **maximise each function**, tracking the best *measured* output so far. For tasks naturally expressed as minimisation, the supplied objective is already framed so that higher is better; the notebooks do not invent additional transformations.

The main constraints are:

- **Limited, delayed feedback:** the capstone allows a total of **13 queries**. The current workflow selects one query per function per submission round and waits for external results. Notebooks do not evaluate the real objectives automatically.
- **Unknown structure:** formulas and gradients are unavailable. Nonlinearity, local maxima, sparse signals and possible observation noise make extrapolation uncertain.
- **Growing dimensionality:** even a large random candidate pool sparsely covers an eight-dimensional space.
- **Small samples:** fitted length scales, uncertainty estimates and cross-validation rankings can be unstable.
- **Valid queries and honest records:** candidates must satisfy input constraints; only confirmed evaluator outputs are recorded as observations.

Success is judged by actual best-so-far improvement and useful learning from evaluations, not by high predicted means or acquisition scores alone. **No global-optimality claim can be made**.

## 4. Technical approach

### Surrogate and acquisition strategy

The implemented approach is **Gaussian Process regression with Bayesian optimisation**. A GP provides both a predicted response and uncertainty, allowing acquisition functions to balance:

- **Exploitation:** query where the predicted response is promising.
- **Exploration:** query where uncertainty could conceal an improvement.

The early baseline uses Upper Confidence Bound (UCB). Later analysis uses EI, with an improvement threshold `xi = alpha * std(y)`, with `std(y)` corresponding to the standard deviation of the observed values. This expression allows to account for different output scales. Increasing xi changes acquisition preferences, not GP predictions at fixed inputs. Alpha so far is heuristic choice, not an independently validated optimum.

### Living record of the first three submission decisions

The repository labels exported rounds by their **evaluation week**: week-0 analysis prepares week-1 inputs, week-1 analysis prepares week-2 inputs, and week-2 analysis prepares week-3 inputs.

| Submission decision | Strategy and rationale |
| --- | --- |
| **Week-1** | Use the supplied initial observations, an RBF GP with fixed length scale 0.1, and UCB with `kappa = 1.96`. Inspect the low-dimensional plots to understand predicted peaks and uncertainty. See the [week-0 notebook](notebooks/week_0/week_0.ipynb). |
| **Week-2** | Compare kernels, length-scale caps and alpha sensitivity. Select Matérn 3/2 for EI suggestions, prioritising within-function held-out cross-validation over in-sample likelihood when they disagree. Use alpha 0.1 and per-function caps. The decision record reports a 2,048-candidate pool; scripts reading shared settings may now use a different pool. See the [week-1 workflow](doc/week-1-workflow.md). |
| **Week-3 candidates, exported but not yet evaluated** | Review best-so-far progress and calibration, increase the search pool to 1,000,000 after observing candidate sensitivity, and choose alpha 0.1 for F1-F5 and 0.2 for F6-F8 by visual inspection. Override EI with diagnostic queries for F1-F2; use optimizer-selected pool EI maxima for F3-F8. See the [week-2 workflow](doc/week-2-workflow.md). |

The kernel comparison is evidence about predictive performance, not proof of optimisation performance. Matérn 3/2 is retained as a consistent suggestion model despite per-function exceptions and disagreement between likelihood and cross-validation.

### Why combine Bayesian suggestions with heuristics?

Functions 1-3 have fitted directions that are nearly flat because some length scales reach their upper bounds. That does not establish that the real functions ignore those inputs; Function 1's fitted short axis even switches after adding an observation.

For the next round:

- **F1: `(0.550000, 0.681526)`** tests whether the isolated negative feature persists along x1. This is an information-seeking query, not exploitation of a negative value in a maximisation task.
- **F2: `(0.702637, 0.500000)`** holds a promising x1 fixed and tests an intermediate x2 between widely separated high-response observations.
- **F3-F8:** retain the optimizer's EI-maximising candidates from the finite pool.

The distinguishing feature is not a claim of a novel optimizer, but a combination of reproducible Bayesian search, sensitivity checks and explicit diagnostic overrides when the surrogate's interpretation is weak.

### Reproduce and continue

The reusable code is in [src/bayesian_bbo](src/bayesian_bbo); shared defaults are in [config.yaml](src/bayesian_bbo/config.yaml). Week-2 notebooks read shared length caps, while week-1 notebooks retain historical hardcoded caps.

Using an existing suitable Python 3.10+ environment, install and validate with:

```bash
python -m pip install -e '.[dev]'
python -m pytest
jupyter notebook
```

Place supplied observations as described in [data/README.md](data/README.md). Follow the [week-2 notebook order](doc/week-2-workflow.md#notebook-order) for diagnostics, alpha selection and final exports. The evaluator is external; do not substitute predictions for actual outcomes or overwrite historical submissions when regenerating candidates.

Continue updating this section as evaluations arrive. Record changes in best measured output, calibration, unsuccessful choices and the reasons for subsequent strategy changes. The [documentation index](doc/README.md) links detailed analysis; [reports](reports/) contains the developing reporting artefacts.
